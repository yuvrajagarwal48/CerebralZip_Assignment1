#!/usr/bin/env python3
"""
pipeline/inference.py: X-CLIP video-text contrastive inference engine.
Imports prompts exclusively from prompts.py and configuration from config.py.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PipelineConfig
from prompts import ACTIVITY_PROMPTS
from .metadata import extract_tensor_features

logger = logging.getLogger("XCLIPEngine")


class XCLIPInferenceEngine:
    """Pretrained, frozen X-CLIP inference engine for zero-shot activity detection."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.classes = list(ACTIVITY_PROMPTS.keys())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading X-CLIP backbone '{config.model_name}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).eval().to(self.device)
        self._preencode_text_prompts()

    def _preencode_text_prompts(self):
        """Pre-encodes all text prompts for all 10 canonical classes once."""
        flat_prompts, self.prompt_to_class = [], []
        for c_idx, c_name in enumerate(self.classes):
            for p in ACTIVITY_PROMPTS[c_name]:
                flat_prompts.append(p)
                self.prompt_to_class.append(c_idx)

        self.prompt_to_class = torch.tensor(self.prompt_to_class, device=self.device)
        inputs = self.processor(text=flat_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            if self.config.use_fp16 and self.device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    feats = self.model.get_text_features(**inputs)
            else:
                feats = self.model.get_text_features(**inputs)
            self.text_features = F.normalize(extract_tensor_features(feats), dim=-1)

        logger.info(f"Pre-encoded {len(flat_prompts)} prompts across {len(self.classes)} classes. Shape: {self.text_features.shape}")

    def infer_windows(self, clips: List[List[np.ndarray]]) -> np.ndarray:
        """Runs batched video feature extraction and computes class similarity scores."""
        if not clips:
            return np.empty((0, len(self.classes)), dtype=np.float32)

        batch_size = self.config.batch_size
        all_scores = []

        for idx in range(0, len(clips), batch_size):
            batch = clips[idx:idx + batch_size]
            try:
                inputs = self.processor(videos=batch, return_tensors="pt")
            except Exception:
                inputs = self.processor(text=[""] * len(batch), videos=batch, return_tensors="pt", padding=True)

            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.inference_mode():
                if self.config.use_fp16 and self.device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        v_feats = self.model.get_video_features(pixel_values=pixel_values)
                else:
                    v_feats = self.model.get_video_features(pixel_values=pixel_values)

                v_norm = F.normalize(extract_tensor_features(v_feats), dim=-1)
                t_norm = self.text_features.to(dtype=v_norm.dtype, device=v_norm.device)
                sims = torch.matmul(v_norm, t_norm.T)

                # Prompt aggregation: mean across class prompts
                batch_scores = torch.zeros((sims.shape[0], len(self.classes)), device=self.device)
                for c_idx in range(len(self.classes)):
                    batch_scores[:, c_idx] = torch.mean(sims[:, self.prompt_to_class == c_idx], dim=1)
                all_scores.append(batch_scores.cpu())

        return torch.cat(all_scores, dim=0).float().numpy()

