#!/usr/bin/env python3
"""
pipeline/grouping.py: Multi-hypothesis cross-class event proposal grouping.
Clusters overlapping candidates of competing classes into multi-hypothesis proposals.
"""

from typing import List
from .metadata import ActivityInterval, EventProposal
from .nms import compute_temporal_iou


def group_cross_class_candidates(
    candidates: List[ActivityInterval],
    iou_threshold: float = 0.3,
    overlap_ratio: float = 0.5,
    enable_grouping: bool = True
) -> List[EventProposal]:
    """Groups overlapping multi-class candidates into unified multi-hypothesis event proposals."""
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda x: x.start_time)

    if not enable_grouping:
        return [
            EventProposal(proposal_id=idx, start_time=c.start_time, end_time=c.end_time, candidates=[c])
            for idx, c in enumerate(sorted_cands)
        ]

    proposals: List[EventProposal] = []
    visited = set()

    for i, c_i in enumerate(sorted_cands):
        if i in visited:
            continue

        group_indices = [i]
        visited.add(i)

        for j in range(i + 1, len(sorted_cands)):
            if j in visited:
                continue

            c_j = sorted_cands[j]
            # Prune comparison if beyond candidate window
            if c_j.start_time > sorted_cands[group_indices[-1]].end_time:
                break

            # Disallow grouping candidates of the identical class
            if any(sorted_cands[idx].activity_class == c_j.activity_class for idx in group_indices):
                continue

            tiou = compute_temporal_iou(c_i.start_time, c_i.end_time, c_j.start_time, c_j.end_time)
            inter = max(0.0, min(c_i.end_time, c_j.end_time) - max(c_i.start_time, c_j.start_time))
            min_dur = min(c_i.duration(), c_j.duration())
            overlap = inter / max(1e-4, min_dur)

            if tiou >= iou_threshold or overlap >= overlap_ratio:
                group_indices.append(j)
                visited.add(j)

        grouped_cands = [sorted_cands[idx] for idx in group_indices]
        p_start = min(c.start_time for c in grouped_cands)
        p_end = max(c.end_time for c in grouped_cands)
        proposals.append(EventProposal(
            proposal_id=len(proposals),
            start_time=p_start,
            end_time=p_end,
            candidates=grouped_cands
        ))

    return proposals

