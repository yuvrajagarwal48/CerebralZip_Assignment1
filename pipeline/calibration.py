#!/usr/bin/env python3
"""
pipeline/calibration.py: Multi-Modal Dynamic Range Confidence Calibration.
Resolves numerical scale compression between raw CLIP dot-products and [0.0, 1.0] confidence.
Based on the exact formulation in calibration_fix.txt.
"""

import math
import numpy as np


class CalibratedConfidenceEstimator:
    """
    Decouples raw cosine similarity, relative margin, and LogOIC contrast from arbitrary
    scales, normalizing each independently into [0.0, 1.0] and combining them with
    balanced weights:
        Conf = 0.45 * Norm_Abs + 0.35 * Norm_Margin + 0.20 * Norm_LogOIC
    """
    SIM_MIN: float = 0.16
    SIM_MAX: float = 0.32
    MARGIN_SCALE: float = 0.04
    MARGIN_OFFSET: float = 0.01

    @classmethod
    def compute_candidate_confidence(
        cls,
        max_similarity: float,
        mean_similarity: float,
        relative_margin: float,
        logoic_score: float,
        enable_logoic: bool = True
    ) -> float:
        """Computes calibrated confidence score prior to VLM verification."""
        # 1. Dynamic min-max scaling on absolute similarity (~0.16 to ~0.32 operational range)
        raw_abs = 0.5 * max_similarity + 0.5 * mean_similarity
        norm_abs = float(np.clip((raw_abs - cls.SIM_MIN) / (cls.SIM_MAX - cls.SIM_MIN), 0.0, 1.0))

        # 2. Dynamic margin ratio (advantage of +0.04 over competing classes represents strong distinctiveness)
        norm_margin = float(np.clip((relative_margin + cls.MARGIN_OFFSET) / cls.MARGIN_SCALE, 0.0, 1.0))

        # 3. Centered sigmoid on LogOIC contrast (0 -> 0.5, >0 -> scales to 1.0)
        if enable_logoic:
            norm_logoic = float(1.0 / (1.0 + math.exp(-logoic_score)))
        else:
            norm_logoic = 0.5

        # 4. Balanced linear combination
        raw_confidence = 0.45 * norm_abs + 0.35 * norm_margin + 0.20 * norm_logoic
        return float(np.clip(raw_confidence, 0.0, 1.0))

    @classmethod
    def apply_vlm_attenuation(
        cls,
        raw_confidence: float,
        vlm_decision: str
    ) -> float:
        """
        Attenuates confidence if the vision-language model marked the proposal as UNCERTAIN.
        - SUPPORTED -> 1.0x (full confidence)
        - UNCERTAIN -> 0.85x (uncertainty penalty)
        - REJECTED -> pruned / suppressed
        """
        if vlm_decision == "UNCERTAIN":
            calibrated = raw_confidence * 0.85
        else:
            calibrated = raw_confidence
        return float(np.clip(calibrated, 0.0, 1.0))

