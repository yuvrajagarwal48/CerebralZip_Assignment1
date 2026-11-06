#!/usr/bin/env python3
"""
pipeline/vlm.py: Qwen2-VL vision-language verification stage.
Screens candidate event proposals using grounded physical bodily action queries.
"""

import logging
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Optional
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PipelineConfig
from .metadata import EventProposal

logger = logging.getLogger("QwenVLVerifier")


class QwenVLVerifier:
    """Zero-shot vision-language verification stage using Qwen2-VL."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Qwen-VL model '{config.vlm_model_name}' on {self.device}...")

        try:
            from transformers import Qwen2VLForConditionalGeneration
            model_cls = Qwen2VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(config.vlm_model_name)
        dtype = torch.float16 if (config.use_fp16 and torch.cuda.is_available()) else torch.float32
        self.model = model_cls.from_pretrained(
            config.vlm_model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None
        ).eval()

    def verify(
        self,
        video_path: str,
        proposals: List[EventProposal],
        fps: float,
        total_frames: int,
        duration: float
    ):
        """Runs grounded physical action verification on each event proposal."""
        for prop in proposals:
            c_start = max(0.0, prop.start_time - self.config.vlm_context_sec)
            c_end = min(duration, prop.end_time + self.config.vlm_context_sec)
            proposed_classes = list(set(c.activity_class for c in prop.candidates))

            # Sample frames at target FPS (4 FPS, max 16 frames)
            num_frames = min(self.config.vlm_max_frames, max(4, int((c_end - c_start) * self.config.vlm_fps)))
            sample_times = np.linspace(c_start, c_end, num_frames)
            frame_indices = [min(total_frames - 1, max(0, int(round(st * fps)))) for st in sample_times]

            cap = cv2.VideoCapture(video_path)
            pil_frames = []
            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    if max(h, w) > self.config.vlm_max_img_dim:
                        scale = self.config.vlm_max_img_dim / max(h, w)
                        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    pil_frames.append(Image.fromarray(rgb))
            cap.release()

            if not pil_frames:
                for c in prop.candidates:
                    c.vlm_decision = "UNCERTAIN"
                    c.vlm_verified = True
                continue

            prompt_text = (
                f"You are an objective video action verifier. Examine these sequential frames ({c_start:.1f}s to {c_end:.1f}s).\n"
                f"Candidates: {', '.join(proposed_classes)}.\n"
                "For EACH activity, determine if visible physical bodily movement or actions are observed.\n"
                "Rules: Do NOT infer intent. Output format for each class:\n"
                "Class: <class_name> | Status: <SUPPORTED, REJECTED, or UNCERTAIN> | Evidence: <brief description>"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in pil_frames],
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]

            try:
                chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[chat_prompt], images=pil_frames, padding=True, return_tensors="pt")
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
                vlm_out = self.processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

                for cand in prop.candidates:
                    cand_lower = cand.activity_class.lower()
                    matched_line = next((line for line in vlm_out.split("\n") if cand_lower.split("/")[0] in line.lower()), "")
                    line_lower = matched_line.lower()

                    if any(w in line_lower for w in ["rejected", "not supported", "false", "disproven"]):
                        cand.vlm_decision = "REJECTED"
                        cand.vlm_verified = False
                    elif any(w in line_lower for w in ["supported", "confirmed", "verified", "observed"]):
                        cand.vlm_decision = "SUPPORTED"
                        cand.vlm_verified = True
                    else:
                        cand.vlm_decision = "UNCERTAIN"
                        cand.vlm_verified = True

                    cand.vlm_evidence = matched_line.strip() if matched_line else vlm_out.strip()

            except Exception as e:
                logger.warning(f"VLM verification exception: {e}")
                for c in prop.candidates:
                    c.vlm_decision = "UNCERTAIN"
                    c.vlm_verified = True
                    c.vlm_evidence = f"VLM exception: {e}"

