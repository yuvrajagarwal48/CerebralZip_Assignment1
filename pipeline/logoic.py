#!/usr/bin/env python3
"""
pipeline/logoic.py: Log-Outer-Inner Contrast (LogOIC) temporal contrast estimation.
Measures the saliency of an action candidate relative to its local outer temporal context.
"""

import math
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PipelineConfig


class LogOICContrastEstimator:
    """Computes FreeZAD-inspired temporal contrast between inner interval and outer context."""
    @staticmethod
    def compute_contrast(
        timestamps: np.ndarray,
        smoothed_scores: np.ndarray,
        start_time: float,
        end_time: float,
        duration: float,
        s_inner: float,
        s_max: float,
        config: PipelineConfig
    ) -> float:
        if not config.enable_logoic:
            return 0.0

        c_delta = config.logoic_context_ratio * duration
        out_mask = (
            ((timestamps >= start_time - c_delta) & (timestamps < start_time)) |
            ((timestamps > end_time) & (timestamps <= end_time + c_delta))
        )

        if np.any(out_mask):
            s_outer = float(np.mean(smoothed_scores[out_mask]))
        else:
            s_outer = s_inner

        ratio = (s_inner + 1e-4) / (s_outer + 1e-4)
        logoic = math.log(max(1e-6, ratio)) + config.logoic_alpha * s_max
        return float(logoic)

