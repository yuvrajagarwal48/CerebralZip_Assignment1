#!/usr/bin/env python3
"""
Zero-Shot Temporal Action Localization Pipeline Package.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from .runner import ZeroShotActivityPipeline, predict_video
from .metadata import VideoMetadata, ActivityInterval, EventProposal
from .chunking import TemporalWindow, TemporalChunker
from .inference import XCLIPInferenceEngine
from .scoring import compute_relative_margins, smooth_timelines, generate_candidate_intervals
from .nms import compute_temporal_iou, apply_class_wise_nms
from .grouping import group_cross_class_candidates
from .vlm import QwenVLVerifier
from .calibration import CalibratedConfidenceEstimator

__all__ = [
    "ZeroShotActivityPipeline",
    "predict_video",
    "VideoMetadata",
    "ActivityInterval",
    "EventProposal",
    "TemporalWindow",
    "TemporalChunker",
    "XCLIPInferenceEngine",
    "compute_relative_margins",
    "smooth_timelines",
    "generate_candidate_intervals",
    "compute_temporal_iou",
    "apply_class_wise_nms",
    "group_cross_class_candidates",
    "QwenVLVerifier",
    "CalibratedConfidenceEstimator"
]

