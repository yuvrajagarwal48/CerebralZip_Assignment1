#!/usr/bin/env python3
"""
pipeline/nms.py: 1D Temporal IoU and Class-Wise Non-Maximum Suppression (NMS).
"""

from typing import List, Dict
from .metadata import ActivityInterval


def compute_temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    """Computes 1D temporal Intersection over Union between two intervals."""
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(1e-6, max(e1, e2) - min(s1, s2))
    return float(inter / union)


def apply_class_wise_nms(
    candidates: List[ActivityInterval],
    iou_threshold: float = 0.3
) -> List[ActivityInterval]:
    """Applies greedy Non-Maximum Suppression independently per activity class."""
    by_class: Dict[str, List[ActivityInterval]] = {}
    for c in candidates:
        by_class.setdefault(c.activity_class, []).append(c)

    nms_candidates: List[ActivityInterval] = []
    for c_name, items in by_class.items():
        # Sort descending by ranking score / confidence
        items.sort(key=lambda x: (x.ranking_score, x.confidence), reverse=True)
        kept: List[ActivityInterval] = []
        for it in items:
            if not any(compute_temporal_iou(it.start_time, it.end_time, k.start_time, k.end_time) > iou_threshold for k in kept):
                kept.append(it)
        nms_candidates.extend(kept)

    # Sort final kept candidates chronologically
    nms_candidates.sort(key=lambda x: x.start_time)
    return nms_candidates

