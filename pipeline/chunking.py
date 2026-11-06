#!/usr/bin/env python3
"""
pipeline/chunking.py: Sliding temporal window chunking and optimized block frame extraction.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .metadata import VideoMetadata


@dataclass
class TemporalWindow:
    window_id: int
    start_time: float
    end_time: float
    center_time: float
    frame_indices: List[int]


def generate_windows(
    metadata: VideoMetadata,
    window_duration: float = 4.0,
    window_stride: float = 1.0,
    frames_per_window: int = 8
) -> List[TemporalWindow]:
    """Generates overlapping temporal windows across the full video duration."""
    duration = metadata.duration_sec
    fps = metadata.fps
    total_frames = metadata.total_frames

    windows: List[TemporalWindow] = []
    curr_t, w_idx = 0.0, 0

    while curr_t + window_duration <= duration:
        start_f = int(round(curr_t * fps))
        end_f = min(total_frames - 1, int(round((curr_t + window_duration) * fps)))
        f_indices = np.linspace(start_f, end_f, frames_per_window, dtype=int).tolist()
        windows.append(TemporalWindow(
            window_id=w_idx,
            start_time=curr_t,
            end_time=curr_t + window_duration,
            center_time=curr_t + window_duration / 2.0,
            frame_indices=f_indices
        ))
        curr_t += window_stride
        w_idx += 1

    if not windows:
        f_indices = np.linspace(0, max(0, total_frames - 1), frames_per_window, dtype=int).tolist()
        windows.append(TemporalWindow(
            window_id=0,
            start_time=0.0,
            end_time=duration,
            center_time=duration / 2.0,
            frame_indices=f_indices
        ))

    return windows


def extract_block_clips(
    video_path: str,
    block_windows: List[TemporalWindow]
) -> List[List[np.ndarray]]:
    """
    Extracts frames for a block of windows using sequential single-pass video decoding.
    Eliminates redundant decodes across overlapping temporal windows.
    """
    indices = sorted(list(set(idx for w in block_windows for idx in w.frame_indices)))
    frame_cache: Dict[int, np.ndarray] = {}

    cap = cv2.VideoCapture(str(video_path))
    if indices and cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        cur_idx = indices[0]
        target_ptr = 0
        while cap.isOpened() and target_ptr < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx == indices[target_ptr]:
                frame_cache[cur_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                while target_ptr < len(indices) and indices[target_ptr] == cur_idx:
                    target_ptr += 1
            cur_idx += 1
    cap.release()

    clips = [
        [frame_cache.get(f_idx, np.zeros((224, 224, 3), dtype=np.uint8)) for f_idx in w.frame_indices]
        for w in block_windows
    ]
    return clips


class TemporalChunker:
    """Wrapper class providing chunking utilities."""
    generate_windows = staticmethod(generate_windows)
    extract_block_clips = staticmethod(extract_block_clips)

