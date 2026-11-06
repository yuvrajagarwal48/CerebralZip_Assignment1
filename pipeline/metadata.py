#!/usr/bin/env python3
"""
pipeline/metadata.py: Video metadata structures, ActivityInterval, and EventProposal dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import cv2
import torch


@dataclass
class VideoMetadata:
    video_path: str
    video_name: str
    duration_sec: float
    fps: float
    resolution: Tuple[int, int]
    total_frames: int

    @staticmethod
    def extract(video_path: Union[str, Path]) -> "VideoMetadata":
        vpath = str(video_path)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: '{vpath}'")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = float(total_frames / max(1.0, fps))
        return VideoMetadata(
            video_path=vpath,
            video_name=Path(vpath).name,
            duration_sec=duration,
            fps=fps,
            resolution=(w, h),
            total_frames=total_frames
        )


@dataclass
class ActivityInterval:
    activity_class: str
    start_time: float
    end_time: float
    confidence: float
    ranking_score: float = 0.0
    mean_similarity: float = 0.0
    max_similarity: float = 0.0
    relative_margin: float = 0.0
    logoic_score: float = 0.0
    vlm_decision: str = "PENDING"
    vlm_verified: Optional[bool] = None
    vlm_evidence: Optional[str] = None

    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def center_time(self) -> float:
        return (self.start_time + self.end_time) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "start_time": round(float(self.start_time), 2),
            "end_time": round(float(self.end_time), 2),
            "class": self.activity_class,
            "confidence": round(float(self.confidence), 4),
            "vlm_decision": self.vlm_decision,
            "vlm_verified": self.vlm_verified,
            "vlm_evidence": self.vlm_evidence
        }
        return d


@dataclass
class EventProposal:
    proposal_id: int
    start_time: float
    end_time: float
    candidates: List[ActivityInterval]


def extract_tensor_features(output: Any) -> torch.Tensor:
    """Extracts 2D feature tensor from diverse HuggingFace model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "video_embeds") and output.video_embeds is not None:
        return output.video_embeds
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return output[0]
    raise TypeError(f"Cannot extract tensor from {type(output)}")

