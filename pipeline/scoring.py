#!/usr/bin/env python3
"""
pipeline/scoring.py: Relative class evidence calculation, Gaussian temporal smoothing,
and dual-evidence candidate interval proposal generation.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PipelineConfig
from .metadata import ActivityInterval
from .chunking import TemporalWindow
from .logoic import LogOICContrastEstimator
from .calibration import CalibratedConfidenceEstimator


def compute_relative_margins(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Computes relative margin over competing classes:
        R_c(t) = S_c(t) - max_{j != c} S_j(t)
    """
    relative_margins = np.zeros_like(similarity_matrix)
    for c in range(similarity_matrix.shape[1]):
        other_max = np.max(np.delete(similarity_matrix, c, axis=1), axis=1)
        relative_margins[:, c] = similarity_matrix[:, c] - other_max
    return relative_margins


def smooth_timelines(
    similarity_matrix: np.ndarray,
    relative_margins: np.ndarray,
    sigma: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies 1D Gaussian temporal smoothing to similarity and relative margin series."""
    smoothed_sim = gaussian_filter1d(similarity_matrix, sigma=sigma, axis=0)
    smoothed_margin = gaussian_filter1d(relative_margins, sigma=sigma, axis=0)
    return smoothed_sim, smoothed_margin


def generate_candidate_intervals(
    classes: List[str],
    windows: List[TemporalWindow],
    smoothed_sim: np.ndarray,
    smoothed_margin: np.ndarray,
    config: PipelineConfig
) -> List[ActivityInterval]:
    """
    Extracts candidate intervals that simultaneously satisfy:
    1. Smoothed class similarity >= class_thresholds[c] (or default_threshold)
    2. Smoothed relative margin >= rel_margin_threshold
    Consecutive window support and gap merging applied.
    """
    timestamps = np.array([w.center_time for w in windows])
    candidates: List[ActivityInterval] = []

    for c_idx, c_name in enumerate(classes):
        threshold = config.class_thresholds.get(c_name, config.default_threshold)
        mask = (smoothed_sim[:, c_idx] >= threshold) & (smoothed_margin[:, c_idx] >= config.rel_margin_threshold)

        raw_intervals = []
        in_int, start_i = False, 0
        for i, val in enumerate(mask):
            if val and not in_int:
                in_int, start_i = True, i
            elif not val and in_int:
                in_int = False
                if (i - start_i) >= config.min_consecutive_windows:
                    raw_intervals.append((start_i, i - 1))
        if in_int and (len(mask) - start_i) >= config.min_consecutive_windows:
            raw_intervals.append((start_i, len(mask) - 1))

        # Merge intervals within max_merge_gap
        merged = []
        if raw_intervals:
            merged.append(raw_intervals[0])
            for cur in raw_intervals[1:]:
                prev = merged[-1]
                if windows[cur[0]].start_time - windows[prev[1]].end_time <= config.max_merge_gap:
                    merged[-1] = (prev[0], cur[1])
                else:
                    merged.append(cur)

        for s_idx, e_idx in merged:
            start_t = windows[s_idx].start_time
            end_t = windows[e_idx].end_time
            dur = end_t - start_t

            if config.min_duration <= dur <= config.max_duration:
                sub_sims = smoothed_sim[s_idx:e_idx + 1, c_idx]
                sub_margins = smoothed_margin[s_idx:e_idx + 1, c_idx]

                in_mask = (timestamps >= start_t) & (timestamps <= end_t)
                s_inner = float(np.mean(smoothed_sim[in_mask, c_idx])) if np.any(in_mask) else float(np.mean(sub_sims))
                s_max = float(np.max(sub_sims))
                mean_margin = float(np.mean(sub_margins))

                # LogOIC Contrast computation
                logoic = LogOICContrastEstimator.compute_contrast(
                    timestamps=timestamps,
                    smoothed_scores=smoothed_sim[:, c_idx],
                    start_time=start_t,
                    end_time=end_t,
                    duration=dur,
                    s_inner=s_inner,
                    s_max=s_max,
                    config=config
                )

                # Calibrated confidence scoring (from calibration_fix.txt)
                conf = CalibratedConfidenceEstimator.compute_candidate_confidence(
                    max_similarity=s_max,
                    mean_similarity=s_inner,
                    relative_margin=mean_margin,
                    logoic_score=logoic,
                    enable_logoic=config.enable_logoic
                )

                candidates.append(ActivityInterval(
                    activity_class=c_name,
                    start_time=start_t,
                    end_time=end_t,
                    confidence=conf,
                    ranking_score=conf,
                    mean_similarity=s_inner,
                    max_similarity=s_max,
                    relative_margin=mean_margin,
                    logoic_score=logoic
                ))

    return candidates

