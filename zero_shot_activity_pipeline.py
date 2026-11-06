#!/usr/bin/env python3
"""
================================================================================
Zero-Shot Activity Detection & Temporal Localization Pipeline (X-CLIP + Relative Evidence)
================================================================================

Objective:
Detect and classify 10 predefined activities in long, untrimmed videos, determine
their start and end timestamps, and generate confidence scores without fine-tuning
or training any model.

Key Revised Architectural Highlights:
1. Prompt Configuration & Aggregation: 10 canonical classes, balanced action prompts,
   default aggregation changed to 'mean' (with 'top_k_mean' and 'max' alternatives).
2. X-CLIP Inference: Pretrained microsoft/xclip-base-patch32, batched GPU, FP16,
   pre-encoded text embeddings.
3. Relative Class Evidence: Computes R_c(t) = S_c(t) - max_{j != c} S_j(t) to measure
   how strongly each class stands out against competing classes alongside absolute similarity A_c(t).
4. Improved Temporal Smoothing: 1D Gaussian smoothing (sigma=0.5 default) applied to
   both absolute similarity and relative class evidence timelines.
5. Temporal Consistency Checks: Distinguishes sustained activities from isolated spikes
   using neighboring-window temporal support.
6. Permissive Candidate Generation: Uses both relative margin and absolute similarity to
   prevent premature rejection of valid events.
7. Class-Specific Interval Merging: Merges only intervals belonging to the same class within max_merge_gap.
8. Distance-Weighted LogOIC Ranking: FreeZAD-inspired window-level Outer-Inner Contrast
   with distance decay and proper video boundary context handling.
9. Class-Wise NMS & Ranking Score: Separate ranking_score calculated before NMS,
   preserving distinct classes without cross-class suppression.
10. Cross-Class Overlap Grouping: Groups concurrent overlapping detections into multi-hypothesis
    event proposals (e.g. concurrent Intrusion, Theft, Vandalism).
11. Optional Qwen3-VL Verification: Plug-and-play visual language model verification stage
    for grouped event proposals (checking visible action evidence without inferring intent).
12. Localized Boundary Refinement: Sub-window search evaluated ONLY within local candidate range.
13. Explicit Metric Separation: Separates mean_similarity, max_similarity, relative_margin,
    logoic_score, ranking_score, and uncalibrated confidence score.
14. Comprehensive Timeline & Candidate Export: Exports raw/smoothed similarities, relative margins,
    candidates, and ranking scores.
15. Scalable Long-Video Processing: Block-based temporal chunking to prevent out-of-memory errors.

Target Activity Classes (10 Predefined Categories):
1. Fighting / Physical Assault
2. Falling
3. Fire / Smoke
4. Intrusion / Trespassing
5. Loitering
6. Vandalism
7. Theft / Chain Snatching
8. Crowd Formation / Unusual Gathering
9. Running in a Restricted Zone
10. Abandoned Object

Execution:
Self-contained, designed for Kaggle (2x NVIDIA T4 GPUs) or single-GPU/CPU execution.
================================================================================
"""

import os
import sys
import gc
import math
import time
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union

# Optimize CUDA memory allocation on NVIDIA GPUs (e.g., Tesla T4)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np

# Optional/conditional imports with helpful fallbacks
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

try:
    from transformers import AutoProcessor, AutoModel
except ImportError:
    AutoProcessor = None
    AutoModel = None

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    gaussian_filter1d = None

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ZeroShotTALPipeline")


# ==============================================================================
# 1. Activity Text Prompt Definitions (10 Canonical Classes & Balanced Variations)
# ==============================================================================
ACTIVITY_PROMPTS: Dict[str, List[str]] = {
    "Fighting/Physical Assault": [
        "Two or more people engage in physical aggression or confrontation involving deliberate physical contact. The activity may include punching, kicking, slapping, pushing, hitting, wrestling, grappling, or repeatedly attacking another person.",
        "A person punching another person during a physical fight.",
        "A person kicking another person with their legs.",
        "Two people repeatedly hitting and attacking each other.",
        "Two people wrestling, grappling or struggling physically.",
        "A person pushing another person aggressively to the ground.",
        "A person slapping or striking another person.",
        "One person physically assaulting another person while the other attempts to defend themselves.",
        "Two or more people engaging in a violent physical confrontation.",
        "A person repeatedly striking another individual who is on the ground.",
        "A person chasing and physically attacking another person."
    ],
    "Falling": [
        "A person unexpectedly loses balance, slips, trips, collapses or falls from an upright or elevated position onto the ground or another surface. The activity involves a noticeable downward movement of the person's body.",
        "A person slipping and falling onto the ground.",
        "A person tripping while walking and falling forward.",
        "A person losing balance and falling backward.",
        "A person suddenly collapsing onto the floor.",
        "A person falling down while running.",
        "A person stumbling and falling after losing balance.",
        "A person falling from a chair, staircase or elevated surface.",
        "A person falling onto the ground after being pushed.",
        "A person dropping from a standing position to the ground.",
        "A person struggling to maintain balance before falling."
    ],
    "Fire/Smoke": [
    # General fire and smoke
    "A video showing an active fire or visible smoke in the environment.",
    "A scene containing visible flames, burning materials or thick smoke.",
    "A fire incident occurring in an indoor or outdoor environment.",

    # Visible flames
    "Bright orange and yellow flames burning and flickering in a scene.",
    "Large flames rising from a burning object or structure.",
    "A building engulfed in fire with flames coming out of windows.",
    "Flames spreading across the surface of an object.",
    "A pile of materials actively burning with visible flames.",
    "A fire burning on the ground and spreading to nearby objects.",
    "A vehicle on fire with flames coming from its body or engine.",
    "A trash bin or garbage pile burning with visible flames.",
    "A small fire producing continuous flickering flames.",
    "An intense fire with flames rising high into the air.",

    # Smoke
    "Thick dark smoke rising into the air from a particular location.",
    "Large clouds of black smoke emerging from a burning building.",
    "Dense grey smoke billowing from a vehicle or other object.",
    "A continuous plume of smoke rising from a fire.",
    "Smoke spreading through an indoor room or enclosed space.",
    "A large area covered by thick smoke from a fire incident.",
    "White smoke rising from an object that is burning.",
    "Black smoke coming out of windows, doors or openings of a building.",
    "A distant fire producing a large visible smoke column.",

    # Fire spreading and developing
    "A fire starting and gradually producing visible flames.",
    "Flames growing larger and spreading across nearby materials.",
    "A burning object producing increasingly thick smoke.",
    "An area becoming engulfed in flames and smoke.",
    "Fire spreading from one object to another.",

    # Small or partially visible fires
    "A small flame burning on an object in an otherwise normal environment.",
    "A small fire burning near a wall or building.",
    "Visible flames emerging from behind an object.",
    "A partially obscured fire producing visible smoke.",
    "A distant burning object with visible flames or smoke."
],
    "Intrusion/Trespassing": [
        "A person enters, attempts to enter or moves into an area that appears restricted, enclosed or protected without an obvious authorized entry process. Visible actions may include climbing barriers, bypassing gates, forcing doors, entering through windows or accessing restricted premises.",
        "A person climbing over a wall or fence to enter an enclosed property.",
        "A person entering a building through a window.",
        "A person forcing open a closed or locked door.",
        "A person attempting to enter a restricted area through a gate.",
        "A person bypassing a security barrier to access a property.",
        "A person entering an enclosed area without using its designated entrance.",
        "A person breaking into a building through an alternative entry point.",
        "A person attempting to open a secured entrance.",
        "A person entering a restricted compound by climbing a boundary wall.",
        "A person moving into a clearly marked restricted-access area."
    ],
    "Loitering": [
        "A person remains in the same general location for an unusually extended period without apparent progression toward a destination or clearly visible activity. The person may repeatedly stand, sit, wait, wander around a small area or move back and forth.",
        "A person standing in one location for an extended period.",
        "A person repeatedly pacing around the same small area.",
        "A person sitting in an area for a prolonged duration.",
        "A person wandering around a building entrance without leaving.",
        "A person repeatedly approaching and moving away from the same location.",
        "A person standing near a doorway for an unusually long time.",
        "A person lingering around a vehicle or property.",
        "A person walking back and forth in a confined area.",
        "A person remaining near an entrance without entering or leaving.",
        "A person repeatedly stopping and waiting in the same location."
    ],
    "Vandalism": [
        "A person deliberately damages, destroys, defaces or interferes with public or private property. Visible actions may include smashing, breaking, kicking, hitting, spraying, scratching, overturning or damaging objects and structures.",
        "A person smashing a window with an object.",
        "A person kicking and damaging a door or other property.",
        "A person breaking objects by hitting them repeatedly.",
        "A person spray-painting graffiti on a wall.",
        "A person scratching or defacing a public surface.",
        "A person deliberately overturning bins, furniture or other objects.",
        "A person damaging a parked vehicle.",
        "A person throwing objects at windows or buildings.",
        "A person breaking street furniture or public infrastructure.",
        "A person repeatedly striking an object until it breaks."
    ],
    "Theft/Chain Snatching": [
        "A person takes, attempts to take or forcibly removes an object or personal belonging that appears to belong to another person. The activity may involve secretly picking up unattended belongings, grabbing objects from someone, snatching bags or accessories, or taking items and quickly leaving the scene.",
        "A person stealing a bag belonging to another person.",
        "A person forcibly snatching an object from someone's hand.",
        "A person grabbing a handbag and running away.",
        "A person taking a mobile phone from another person's pocket or hand.",
        "A person stealing belongings from a table or unattended location.",
        "A person removing an object from another person's possession.",
        "A person grabbing a necklace or chain from someone's neck.",
        "A person taking an object and quickly fleeing the scene.",
        "A person reaching into another person's bag and removing an item.",
        "A person stealing an object from a parked vehicle."
    ],
    "Crowd Formation/Unusual Gathering": [
        "Multiple people gather or converge in the same general location, forming a noticeable group or crowd. The activity may involve people arriving from different directions, clustering around a particular location, remaining together or rapidly increasing in number.",
        "A group of people gathering in an open area.",
        "Multiple people converging toward the same location.",
        "A crowd gradually forming near a building entrance.",
        "Several people gathering around a particular object or incident.",
        "A large group of people clustering together in a public area.",
        "Multiple individuals approaching and joining an existing group.",
        "People forming a dense gathering around a particular location.",
        "A group of people suddenly assembling in the street.",
        "A crowd growing as more people arrive.",
        "Several people surrounding another person or an object."
    ],
    "Running in a Restricted Zone": [
        "A person moves rapidly through an area by running or sprinting rather than walking. The activity involves clear running motion, such as alternating strides, rapid leg movement and forward displacement, within an area where running is not permitted.",
        "A person sprinting through a designated restricted area.",
        "A person running rapidly through a building corridor.",
        "A person running inside an area marked as a no-running zone.",
        "A person sprinting through a restricted public facility.",
        "A person running through a designated pedestrian-only area.",
        "A person rapidly running across an indoor facility.",
        "A person running through a clearly marked prohibited area.",
        "A person running inside a facility where walking is required."
    ],
    "Abandoned Object": [
        "A person places, drops or leaves an object in a location and subsequently moves away without retrieving it. The object remains unattended for a noticeable duration after the person has departed.",
        "A person leaving a bag on the ground and walking away.",
        "A person placing a suitcase in a public area and departing.",
        "A person leaving a package near a building entrance.",
        "A person dropping an object and walking away without retrieving it.",
        "A person placing a backpack on a bench and leaving the area.",
        "A person leaving a box unattended in a corridor.",
        "A person placing an object in a public space and moving out of view.",
        "A person leaving belongings behind while walking away.",
        "An object remaining unattended after the person who brought it has departed.",
        "A person leaving a bag near a crowded area and walking away."
    ]
}


# ==============================================================================
# Configuration Data Structures
# ==============================================================================
@dataclass
class PipelineConfig:
    """Master Configuration for the Zero-Shot Activity Detection Pipeline."""
    model_name: str = "microsoft/xclip-base-patch32"
    window_duration: float = 4.0        # Temporal window duration in seconds
    window_stride: float = 1.0          # Stride between windows in seconds
    frames_per_window: int = 8          # Uniformly sampled frames per window
    batch_size: int = 16                # Batch size for GPU inference
    use_fp16: bool = True               # Use FP16 mixed precision on GPU
    temporal_block_windows: int = 120   # Process long videos in blocks of 120 windows (~2 min)
    num_workers: int = 4                # Number of CPU workers for video decoding
    
    # 1. Prompt Aggregation Method: 'mean' as default (was 'max'), 'top_k_mean', or 'max'
    prompt_aggregation: str = "mean"
    top_k_prompts: int = 3
    
    # 3 & 4. Relative Class Evidence & Temporal Smoothing
    smoothing_type: str = "gaussian"    # "gaussian", "moving_average", or "none"
    smoothing_window: int = 3           # Window length for moving average
    smoothing_sigma: float = 0.5        # Gaussian sigma (default 0.5 to prevent over-smoothing)
    
    # 5 & 6. Candidate Interval Thresholding & Temporal Consistency
    threshold_type: str = "fixed"       # "fixed", "dynamic_percentile", "dynamic_mean_std"
    default_threshold: float = 0.21     # Permissive absolute similarity threshold
    class_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "Fighting/Physical Assault": 0.20,
        "Falling": 0.10,
        "Fire/Smoke": 0.10,
        "Intrusion/Trespassing": 0.20,
        "Loitering": 0.20,
        "Vandalism": 0.20,
        "Theft/Chain Snatching": 0.20,
        "Crowd Formation/Unusual Gathering": 0.21,
        "Running in a Restricted Zone": 0.20,
        "Abandoned Object": 0.20
    })
    rel_margin_threshold: float = -0.015 # Permissive relative margin threshold
    min_consecutive_windows: int = 2    # Require support across consecutive windows (2 or 3)
    allow_brief_spikes: bool = True     # Allow high-confidence isolated peaks for brief activities
    brief_spike_threshold: float = 0.26 # Absolute threshold for isolated spike
    max_merge_gap: float = 2.0          # Max gap (in seconds) to merge nearby candidates of same class
    min_duration: float = 1.5           # Minimum candidate interval duration
    max_duration: float = 300.0         # Maximum candidate interval duration
    
    # 8. LogOIC-based Temporal Localization (FreeZAD-inspired)
    enable_logoic: bool = True
    logoic_context_ratio: float = 0.25  # Ratio of candidate duration for outer context
    logoic_distance_weighting: bool = True # Distance-weighted decay for outer context
    logoic_tau: float = 2.0             # Temporal decay constant in seconds for outer context
    logoic_alpha: float = 0.3           # Weight for maximum activation term
    logoic_epsilon: float = 1e-4        # Stability epsilon for contrast ratio
    
    # 9. Class-wise NMS
    nms_iou_threshold: float = 0.3      # 1D Temporal IoU threshold for class-wise NMS
    
    # 10. Cross-Class Overlap Grouping
    enable_cross_class_grouping: bool = True
    cross_class_iou_threshold: float = 0.3      # IoU threshold to group cross-class hypotheses
    cross_class_overlap_ratio: float = 0.5      # Overlap ratio relative to shorter interval
    
    # 11. Qwen-VL Verification (Optional)
    enable_vlm_verification: bool = True
    vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"  # or "Qwen/Qwen2-VL-7B-Instruct"
    vlm_fps: float = 4.0               # Sample frames at 4 FPS from candidate interval
    vlm_context_sec: float = 1.0       # Approximately 1 second context before and after
    vlm_max_frames: int = 16           # Max frames per VLM prompt to protect GPU memory
    vlm_max_img_dim: int = 768         # Max image dimension in pixels (768px)
    
    # 12. Boundary Refinement
    enable_refinement: bool = True
    refinement_window_duration: float = 2.0  # Smaller window duration for refinement
    refinement_stride: float = 0.5           # Smaller stride for refinement
    refinement_frames: int = 8               # Frames per refinement window
    
    # Hardware & Execution
    device: str = "auto"                # "auto", "cuda", "cpu"
    export_timeline: bool = True        # Export CSV timelines & candidates for inspection
    detailed_output: bool = True        # Include separate metrics in JSON output


@dataclass
class VideoMetadata:
    """Metadata extracted from the original input video."""
    video_path: str
    video_name: str
    duration_sec: float
    fps: float
    resolution: Tuple[int, int]         # (width, height)
    total_frames: int
    num_windows: int


@dataclass
class TemporalWindow:
    """Representation of an individual overlapping temporal video window."""
    window_id: int
    start_time: float
    end_time: float
    center_time: float
    start_frame: int
    end_frame: int
    frame_indices: List[int] = field(default_factory=list)


@dataclass
class ActivityInterval:
    """Candidate or refined activity detection interval with separated metrics."""
    activity_class: str
    start_time: float
    end_time: float
    confidence: float                  # Final reported uncalibrated detection/ranking score
    ranking_score: float = 0.0         # Pre-NMS ranking score
    mean_similarity: float = 0.0       # Average class similarity
    max_similarity: float = 0.0        # Peak class similarity
    relative_margin: float = 0.0       # Difference from competing classes (smoothed)
    max_relative_margin: float = 0.0   # Peak difference from competing classes (raw)
    logoic_score: float = 0.0          # Inner-vs-outer temporal contrast
    refined: bool = False
    vlm_verified: Optional[bool] = None
    vlm_decision: str = "PENDING"      # "SUPPORTED", "REJECTED", "UNCERTAIN"
    vlm_evidence: Optional[str] = None
    window_indices: List[int] = field(default_factory=list)

    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "start_time": round(float(self.start_time), 2),
            "end_time": round(float(self.end_time), 2),
            "class": self.activity_class,
            "confidence": round(float(self.confidence), 4)
        }
        if self.vlm_decision != "PENDING":
            d["vlm_decision"] = self.vlm_decision
        if self.vlm_verified is not None:
            d["vlm_verified"] = self.vlm_verified
            if self.vlm_evidence:
                d["vlm_evidence"] = self.vlm_evidence
        return d


@dataclass
class EventProposal:
    """Grouped multi-hypothesis event proposal for cross-class overlapping activities."""
    proposal_id: int
    start_time: float
    end_time: float
    candidates: List[ActivityInterval] = field(default_factory=list)
    vlm_verified: bool = False
    vlm_notes: str = ""

    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


# ==============================================================================
# Feature Extraction Helper (Handles BaseModelOutputWithPooling & Raw Tensors)
# ==============================================================================
def extract_tensor_features(output: Any) -> torch.Tensor:
    """
    Extracts the underlying feature tensor from model outputs.
    Handles raw torch.Tensor, HuggingFace BaseModelOutputWithPooling,
    and tuple/list structures across transformers versions.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "video_embeds") and output.video_embeds is not None:
        return output.video_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() == 2:
                return item
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"Unable to extract feature tensor from output of type: {type(output)}")


# ==============================================================================
# 2. Video Input and Metadata Extraction
# ==============================================================================
class VideoMetadataExtractor:
    """Extracts essential video metadata and calculates required temporal windows."""
    @staticmethod
    def extract(video_path: Union[str, Path], config: PipelineConfig) -> VideoMetadata:
        video_path_str = str(video_path)
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for video reading.")

        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path_str}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 25.0
            logger.warning(f"Could not determine valid FPS. Defaulting to {fps} FPS.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        cap.release()

        # Calculate estimated temporal windows
        if duration_sec <= config.window_duration:
            num_windows = 1
        else:
            num_windows = int(math.ceil((duration_sec - config.window_duration) / config.window_stride)) + 1

        meta = VideoMetadata(
            video_path=video_path_str,
            video_name=Path(video_path_str).name,
            duration_sec=float(duration_sec),
            fps=float(fps),
            resolution=(width, height),
            total_frames=total_frames,
            num_windows=num_windows
        )
        logger.info(
            f"Extracted Metadata: '{meta.video_name}' | Duration: {meta.duration_sec:.2f}s | "
            f"FPS: {meta.fps:.2f} | Res: {meta.resolution[0]}x{meta.resolution[1]} | "
            f"Frames: {meta.total_frames} | Windows: {meta.num_windows}"
        )
        return meta


# ==============================================================================
# Temporal Video Chunking
# ==============================================================================
class TemporalChunker:
    """Divides the video into overlapping short temporal windows."""
    @staticmethod
    def generate_windows(
        metadata: VideoMetadata,
        window_duration: float,
        window_stride: float,
        frames_per_window: int
    ) -> List[TemporalWindow]:
        duration = metadata.duration_sec
        fps = metadata.fps
        win_dur = window_duration
        stride = window_stride
        num_frames = frames_per_window

        windows: List[TemporalWindow] = []

        if duration <= win_dur:
            start_t = 0.0
            end_t = duration
            start_f = 0
            end_f = max(0, metadata.total_frames - 1)
            frame_indices = np.linspace(start_f, end_f, num_frames, dtype=int).tolist()
            windows.append(TemporalWindow(
                window_id=0,
                start_time=start_t,
                end_time=end_t,
                center_time=(start_t + end_t) / 2.0,
                start_frame=start_f,
                end_frame=end_f,
                frame_indices=frame_indices
            ))
            return windows

        curr_start = 0.0
        win_id = 0

        while curr_start + win_dur <= duration:
            curr_end = curr_start + win_dur
            start_f = int(round(curr_start * fps))
            end_f = min(metadata.total_frames - 1, int(round(curr_end * fps)))
            frame_indices = np.linspace(start_f, end_f, num_frames, dtype=int).tolist()

            windows.append(TemporalWindow(
                window_id=win_id,
                start_time=round(curr_start, 3),
                end_time=round(curr_end, 3),
                center_time=round((curr_start + curr_end) / 2.0, 3),
                start_frame=start_f,
                end_frame=end_f,
                frame_indices=frame_indices
            ))
            win_id += 1
            curr_start += stride

        # Ensure the final window covers the end of the video
        if windows and windows[-1].end_time < duration:
            final_start = max(0.0, duration - win_dur)
            final_end = duration
            start_f = int(round(final_start * fps))
            end_f = max(0, metadata.total_frames - 1)
            frame_indices = np.linspace(start_f, end_f, num_frames, dtype=int).tolist()

            windows.append(TemporalWindow(
                window_id=win_id,
                start_time=round(final_start, 3),
                end_time=round(final_end, 3),
                center_time=round((final_start + final_end) / 2.0, 3),
                start_frame=start_f,
                end_frame=end_f,
                frame_indices=frame_indices
            ))

        logger.info(f"Generated {len(windows)} temporal windows (duration={win_dur}s, stride={stride}s).")
        return windows


# ==============================================================================
# 15. Scalable Frame Extraction (Block-Based Memory Management)
# ==============================================================================
class CachedFrameExtractor:
    """
    Extracts frames with sequential caching and temporal block chunking to avoid OOM
    on long videos. Releases decoded frame caches per block.
    """
    @staticmethod
    def extract_window_frames_block(
        video_path: str,
        windows: List[TemporalWindow]
    ) -> List[List[np.ndarray]]:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for video reading.")

        if not windows:
            return []

        unique_frame_indices = set()
        for w in windows:
            unique_frame_indices.update(w.frame_indices)
        sorted_indices = sorted(list(unique_frame_indices))

        if not sorted_indices:
            return []

        frame_cache: Dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Seek close to first frame if possible
        min_target = sorted_indices[0]
        if min_target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min_target)
            current_frame_idx = min_target
        else:
            current_frame_idx = 0

        target_idx_ptr = 0
        total_targets = len(sorted_indices)

        while cap.isOpened() and target_idx_ptr < total_targets:
            while target_idx_ptr < total_targets and sorted_indices[target_idx_ptr] < current_frame_idx:
                target_idx_ptr += 1

            if target_idx_ptr >= total_targets:
                break

            ret, frame = cap.read()
            if not ret:
                break

            target_frame_idx = sorted_indices[target_idx_ptr]
            if current_frame_idx == target_frame_idx:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_cache[target_frame_idx] = rgb_frame
                target_idx_ptr += 1
                while target_idx_ptr < total_targets and sorted_indices[target_idx_ptr] == target_frame_idx:
                    target_idx_ptr += 1

            current_frame_idx += 1

        cap.release()

        # Handle any trailing indices
        if frame_cache:
            last_valid_idx = max(frame_cache.keys())
            last_valid_frame = frame_cache[last_valid_idx]
            for idx in sorted_indices:
                if idx not in frame_cache:
                    frame_cache[idx] = last_valid_frame.copy()
        else:
            raise RuntimeError(f"Failed to decode any frames from video {video_path}")

        # Assemble clips
        window_clips: List[List[np.ndarray]] = []
        for w in windows:
            clip = [frame_cache[f_idx] for f_idx in w.frame_indices]
            window_clips.append(clip)

        return window_clips


# ==============================================================================
# Multi-GPU Video Encoder Wrapper (for DataParallel on Kaggle 2x T4 GPUs)
# ==============================================================================
class XCLIPVideoEncoderWrapper(nn.Module):
    """Wraps X-CLIP vision backbone with internal FP16 autocasting for multi-GPU replicas."""
    def __init__(self, xclip_model: nn.Module, use_fp16: bool = True):
        super().__init__()
        self.model = xclip_model
        self.use_fp16 = use_fp16

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.is_cuda and self.use_fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                raw_feats = self.model.get_video_features(pixel_values=pixel_values)
        else:
            raw_feats = self.model.get_video_features(pixel_values=pixel_values)

        feats = extract_tensor_features(raw_feats)
        return F.normalize(feats, dim=-1)


# ==============================================================================
# X-CLIP Zero-Shot Inference Engine
# ==============================================================================
class XCLIPInferenceEngine:
    """Pretrained X-CLIP engine with prompt pre-encoding and configurable aggregation."""
    def __init__(self, config: PipelineConfig, prompts_dict: Optional[Dict[str, List[str]]] = None):
        self.config = config
        self.prompts_dict = prompts_dict if prompts_dict is not None else ACTIVITY_PROMPTS
        self.classes = list(self.prompts_dict.keys())
        
        if self.config.device == "auto":
            self.device = torch.device("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(f"Initializing XCLIPInferenceEngine on device: {self.device}")

        if torch is None or AutoProcessor is None or AutoModel is None:
            raise ImportError("PyTorch and HuggingFace Transformers are required for X-CLIP inference.")

        logger.info(f"Loading pretrained model: '{self.config.model_name}'...")
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.raw_model = AutoModel.from_pretrained(self.config.model_name)
        self.raw_model.eval()

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.video_encoder = XCLIPVideoEncoderWrapper(self.raw_model, use_fp16=self.config.use_fp16)
        
        if self.device.type == "cuda" and num_gpus > 1:
            logger.info(f"Detected {num_gpus} CUDA GPUs. Enabling torch.nn.DataParallel across all GPUs.")
            self.video_encoder = nn.DataParallel(self.video_encoder)

        self.video_encoder.to(self.device)
        self.raw_model.to(self.device)

        # Pre-encode text prompts
        self._preencode_text_prompts()

    def _preencode_text_prompts(self):
        """Pre-encodes normalized text embeddings for all class prompt variations."""
        logger.info("Pre-encoding text prompts for all 10 activity classes...")
        self.flat_prompts: List[str] = []
        self.prompt_class_indices: List[int] = []

        for class_idx, class_name in enumerate(self.classes):
            variations = self.prompts_dict[class_name]
            for prompt in variations:
                self.flat_prompts.append(prompt)
                self.prompt_class_indices.append(class_idx)

        self.prompt_class_indices = torch.tensor(self.prompt_class_indices, dtype=torch.long, device=self.device)

        text_inputs = self.processor(
            text=self.flat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.inference_mode():
            if self.device.type == "cuda" and self.config.use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    raw_text_features = self.raw_model.get_text_features(**text_inputs)
            else:
                raw_text_features = self.raw_model.get_text_features(**text_inputs)

            text_features = extract_tensor_features(raw_text_features)
            self.text_features = F.normalize(text_features, dim=-1)

        logger.info(
            f"Successfully pre-encoded {len(self.flat_prompts)} prompts across {len(self.classes)} classes. "
            f"Text features shape: {self.text_features.shape}"
        )

    def infer_windows(self, window_clips: List[List[np.ndarray]]) -> np.ndarray:
        """Runs batched zero-shot inference on a list of temporal window clips."""
        num_windows = len(window_clips)
        if num_windows == 0:
            return np.empty((0, len(self.classes)), dtype=np.float32)

        batch_size = self.config.batch_size
        all_class_scores: List[torch.Tensor] = []

        for start_idx in range(0, num_windows, batch_size):
            end_idx = min(start_idx + batch_size, num_windows)
            batch_clips = window_clips[start_idx:end_idx]

            try:
                inputs = self.processor(videos=batch_clips, return_tensors="pt")
            except Exception:
                inputs = self.processor(
                    text=[""] * len(batch_clips),
                    videos=batch_clips,
                    return_tensors="pt",
                    padding=True
                )

            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.inference_mode():
                if self.device.type == "cuda" and self.config.use_fp16:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        video_features = self.video_encoder(pixel_values)
                else:
                    video_features = self.video_encoder(pixel_values)

                # Match dtype and device
                text_feats = self.text_features.to(dtype=video_features.dtype, device=video_features.device)
                prompt_similarities = torch.matmul(video_features, text_feats.T)
                batch_class_scores = self._aggregate_prompts_to_classes(prompt_similarities)
                all_class_scores.append(batch_class_scores.cpu())

        similarity_matrix = torch.cat(all_class_scores, dim=0).float().numpy()
        return similarity_matrix

    def _aggregate_prompts_to_classes(self, prompt_sims: torch.Tensor) -> torch.Tensor:
        """Aggregates prompt similarities into 10 class-level scores (default: 'mean')."""
        batch_size = prompt_sims.shape[0]
        num_classes = len(self.classes)
        class_scores = torch.zeros((batch_size, num_classes), device=prompt_sims.device, dtype=prompt_sims.dtype)

        for c_idx in range(num_classes):
            mask = (self.prompt_class_indices == c_idx)
            c_sims = prompt_sims[:, mask]

            if self.config.prompt_aggregation == "mean":
                class_scores[:, c_idx] = torch.mean(c_sims, dim=1)
            elif self.config.prompt_aggregation == "top_k_mean":
                k = min(self.config.top_k_prompts, c_sims.shape[1])
                topk_vals = torch.topk(c_sims, k=k, dim=1).values
                class_scores[:, c_idx] = torch.mean(topk_vals, dim=1)
            elif self.config.prompt_aggregation == "max":
                class_scores[:, c_idx] = torch.max(c_sims, dim=1).values
            else:
                # Default fallback is 'mean'
                class_scores[:, c_idx] = torch.mean(c_sims, dim=1)

        return class_scores


# ==============================================================================
# 3. Relative Class Evidence Computation
# ==============================================================================
def compute_relative_margins(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Computes relative class evidence for each window and class:
    R_c(t) = S_c(t) - max_{j != c} S_j(t)
    
    Args:
        similarity_matrix: np.ndarray of shape (num_windows, num_classes)
    Returns:
        relative_margins: np.ndarray of shape (num_windows, num_classes)
    """
    W, C = similarity_matrix.shape
    margins = np.zeros_like(similarity_matrix, dtype=np.float32)
    for c in range(C):
        other_cols = np.delete(similarity_matrix, c, axis=1)
        max_other = np.max(other_cols, axis=1)
        margins[:, c] = similarity_matrix[:, c] - max_other
    return margins


# ==============================================================================
# 4. Temporal Score Aggregation & Smoothing (Gaussian sigma=0.5 default)
# ==============================================================================
class TemporalScoreAggregator:
    """
    Arranges both absolute similarity and relative margin scores chronologically.
    Applies Gaussian smoothing (default sigma=0.5) to both signals while preserving raw values.
    """
    @staticmethod
    def process_timelines(
        similarity_matrix: np.ndarray,
        relative_margins: np.ndarray,
        windows: List[TemporalWindow],
        classes: List[str],
        config: PipelineConfig
    ) -> Dict[str, Dict[str, np.ndarray]]:
        timestamps = np.array([w.center_time for w in windows], dtype=np.float32)
        timelines: Dict[str, Dict[str, np.ndarray]] = {}

        for c_idx, class_name in enumerate(classes):
            raw_scores = similarity_matrix[:, c_idx].astype(np.float32)
            raw_margins = relative_margins[:, c_idx].astype(np.float32)

            if config.smoothing_type == "gaussian" and len(raw_scores) > 1:
                if gaussian_filter1d is not None:
                    smoothed_scores = gaussian_filter1d(raw_scores, sigma=config.smoothing_sigma)
                    smoothed_margins = gaussian_filter1d(raw_margins, sigma=config.smoothing_sigma)
                else:
                    smoothed_scores = TemporalScoreAggregator._moving_average(raw_scores, config.smoothing_window)
                    smoothed_margins = TemporalScoreAggregator._moving_average(raw_margins, config.smoothing_window)
            elif config.smoothing_type == "moving_average":
                smoothed_scores = TemporalScoreAggregator._moving_average(raw_scores, config.smoothing_window)
                smoothed_margins = TemporalScoreAggregator._moving_average(raw_margins, config.smoothing_window)
            else:
                smoothed_scores = raw_scores.copy()
                smoothed_margins = raw_margins.copy()

            timelines[class_name] = {
                "timestamps": timestamps,
                "raw_scores": raw_scores,
                "smoothed_scores": smoothed_scores,
                "raw_relative_margins": raw_margins,
                "smoothed_relative_margins": smoothed_margins
            }

        return timelines

    @staticmethod
    def _moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        if len(data) < window_size or window_size <= 1:
            return data.copy()
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode="same")


# ==============================================================================
# 5 & 6. Candidate Interval Generation with Temporal Consistency
# ==============================================================================
class CandidateIntervalGenerator:
    """
    Generates activity candidates using both relative margin and absolute similarity.
    Applies temporal consistency checks across neighboring windows, while allowing
    prominent brief spikes to be preserved.
    """
    @staticmethod
    def generate_candidates(
        timelines: Dict[str, Dict[str, np.ndarray]],
        windows: List[TemporalWindow],
        config: PipelineConfig
    ) -> List[ActivityInterval]:
        candidates: List[ActivityInterval] = []

        for class_name, data in timelines.items():
            smoothed_scores = data["smoothed_scores"]
            raw_scores = data["raw_scores"]
            smoothed_margins = data["smoothed_relative_margins"]
            raw_margins = data["raw_relative_margins"]

            abs_thresh = config.class_thresholds.get(class_name, config.default_threshold)
            rel_thresh = config.rel_margin_threshold

            # Dual-evidence criteria: absolute similarity AND relative margin
            positive_mask = (smoothed_scores >= abs_thresh) & (smoothed_margins >= rel_thresh)

            # Identify consecutive positive windows
            raw_intervals: List[Tuple[int, int]] = []
            in_interval = False
            start_idx = 0

            for i, is_pos in enumerate(positive_mask):
                if is_pos and not in_interval:
                    in_interval = True
                    start_idx = i
                elif not is_pos and in_interval:
                    in_interval = False
                    if (i - start_idx) >= config.min_consecutive_windows:
                        raw_intervals.append((start_idx, i - 1))
                    elif config.allow_brief_spikes and (i - start_idx) == 1:
                        # Temporal consistency check for brief actions: check if peak is prominent
                        if raw_scores[start_idx] >= config.brief_spike_threshold and raw_margins[start_idx] > 0.0:
                            raw_intervals.append((start_idx, start_idx))

            if in_interval:
                if (len(positive_mask) - start_idx) >= config.min_consecutive_windows:
                    raw_intervals.append((start_idx, len(positive_mask) - 1))
                elif config.allow_brief_spikes and (len(positive_mask) - start_idx) == 1:
                    if raw_scores[start_idx] >= config.brief_spike_threshold and raw_margins[start_idx] > 0.0:
                        raw_intervals.append((start_idx, start_idx))

            if not raw_intervals:
                continue

            # 7. Merge nearby intervals belonging ONLY to the same class
            merged_intervals = CandidateIntervalGenerator._merge_nearby_intervals(
                raw_intervals, windows, config.max_merge_gap
            )

            # Assemble ActivityInterval objects with separated evidence fields
            for s_idx, e_idx in merged_intervals:
                win_indices = list(range(s_idx, e_idx + 1))
                start_t = windows[s_idx].start_time
                end_t = windows[e_idx].end_time

                interval_raw_scores = raw_scores[win_indices]
                interval_smooth_scores = smoothed_scores[win_indices]
                interval_smooth_margins = smoothed_margins[win_indices]
                interval_raw_margins = raw_margins[win_indices]

                mean_sim = float(np.mean(interval_smooth_scores))
                max_sim = float(np.max(interval_raw_scores))
                mean_margin = float(np.mean(interval_smooth_margins))
                max_margin = float(np.max(interval_raw_margins))
                duration = end_t - start_t

                if duration >= config.min_duration and duration <= config.max_duration:
                    candidates.append(ActivityInterval(
                        activity_class=class_name,
                        start_time=start_t,
                        end_time=end_t,
                        confidence=mean_sim,  # Initial placeholder
                        ranking_score=0.0,
                        mean_similarity=mean_sim,
                        max_similarity=max_sim,
                        relative_margin=mean_margin,
                        max_relative_margin=max_margin,
                        window_indices=win_indices
                    ))

        logger.info(f"Generated {len(candidates)} candidate activity intervals across all classes.")
        return candidates

    @staticmethod
    def _merge_nearby_intervals(
        intervals: List[Tuple[int, int]],
        windows: List[TemporalWindow],
        max_gap: float
    ) -> List[Tuple[int, int]]:
        if not intervals:
            return []

        merged: List[Tuple[int, int]] = [intervals[0]]

        for curr_start, curr_end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            gap = windows[curr_start].start_time - windows[prev_end].end_time

            if gap <= max_gap:
                merged[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged.append((curr_start, curr_end))

        return merged


# ==============================================================================
# 8. Distance-Weighted LogOIC Ranking & Candidate Ranking Score
# ==============================================================================
class LogOICTemporalLocalizer:
    """
    Window-level adaptation of LogOIC (Logarithmic Outer-Inner Contrast) from FreeZAD.
    Includes distance-weighted outer context decay and boundary context handling.
    Calculates separate ranking_score combining similarity, relative margin, and LogOIC contrast.
    """
    @staticmethod
    def compute_logoic_and_ranking(
        candidates: List[ActivityInterval],
        timelines: Dict[str, Dict[str, np.ndarray]],
        windows: List[TemporalWindow],
        config: PipelineConfig
    ) -> List[ActivityInterval]:
        if not candidates:
            return candidates

        scored_candidates: List[ActivityInterval] = []

        for cand in candidates:
            class_name = cand.activity_class
            timeline = timelines[class_name]
            scores = timeline["smoothed_scores"]
            timestamps = timeline["timestamps"]

            duration = cand.duration()
            context_delta = config.logoic_context_ratio * duration

            # Inner activation
            inner_mask = (timestamps >= cand.start_time) & (timestamps <= cand.end_time)
            inner_scores = scores[inner_mask] if np.any(inner_mask) else np.array([cand.mean_similarity])
            s_inner = float(np.mean(inner_scores))
            s_max = float(np.max(inner_scores))

            # If LogOIC is disabled, compute fallback ranking score directly using similarity and relative margin
            if not config.enable_logoic:
                rel_margin_contrib = float(np.clip(cand.relative_margin * 5.0, -0.5, 0.5))
                cand.logoic_score = 0.0
                cand.mean_similarity = s_inner
                cand.max_similarity = s_max
                cand.ranking_score = float(
                    0.50 * s_inner +
                    0.30 * s_max +
                    0.20 * max(0.0, rel_margin_contrib)
                )
                scored_candidates.append(cand)
                continue

            # Outer context with distance-weighted contribution and boundary handling
            left_mask = (timestamps >= (cand.start_time - context_delta)) & (timestamps < cand.start_time)
            right_mask = (timestamps > cand.end_time) & (timestamps <= (cand.end_time + context_delta))
            has_left = np.any(left_mask)
            has_right = np.any(right_mask)

            if has_left or has_right:
                # Use available context (left, right, or both)
                outer_mask = left_mask | right_mask
                outer_scores = scores[outer_mask]
                outer_times = timestamps[outer_mask]

                if config.logoic_distance_weighting:
                    # Decay weight by temporal distance from interval boundary
                    dists = np.minimum(
                        np.abs(outer_times - cand.start_time),
                        np.abs(outer_times - cand.end_time)
                    )
                    weights = np.exp(-dists / max(0.5, config.logoic_tau))
                    s_outer = float(np.sum(weights * outer_scores) / max(1e-6, np.sum(weights)))
                else:
                    s_outer = float(np.mean(outer_scores))
            else:
                # Video boundary fallback: if neither left nor right context is available
                # (e.g. interval spans entire video), mark contrast unavailable by setting
                # s_outer = s_inner so contrast ratio is 1.0 (log(1) = 0)
                s_outer = s_inner

            eps = config.logoic_epsilon
            contrast_ratio = (s_inner + eps) / (s_outer + eps)
            log_oic = math.log(max(1e-6, contrast_ratio)) + config.logoic_alpha * s_max

            cand.logoic_score = float(log_oic)
            cand.mean_similarity = s_inner
            cand.max_similarity = s_max

            # 9. Calculate separate ranking_score before NMS
            # Combines absolute similarity, peak response, relative margin, and LogOIC contrast
            logoic_sigmoid = 1.0 / (1.0 + math.exp(-cand.logoic_score))
            rel_margin_contrib = float(np.clip(cand.relative_margin * 5.0, -0.5, 0.5))

            cand.ranking_score = float(
                0.35 * s_inner +
                0.25 * s_max +
                0.20 * rel_margin_contrib +
                0.20 * logoic_sigmoid
            )

            scored_candidates.append(cand)

        return scored_candidates


# ==============================================================================
# 9. Class-Wise Temporal NMS (Restricted to Same Class)
# ==============================================================================
class ClassWiseNMS:
    """
    Performs 1D Temporal Non-Maximum Suppression (NMS) strictly restricted to the same class.
    Uses ranking_score to resolve duplicate candidates without suppressing other classes.
    """
    @staticmethod
    def apply_nms(candidates: List[ActivityInterval], iou_thresh: float) -> List[ActivityInterval]:
        if not candidates:
            return []

        by_class: Dict[str, List[ActivityInterval]] = {}
        for c in candidates:
            by_class.setdefault(c.activity_class, []).append(c)

        preserved: List[ActivityInterval] = []

        for class_name, items in by_class.items():
            # Sort descending by ranking_score
            items.sort(key=lambda x: x.ranking_score, reverse=True)
            kept: List[ActivityInterval] = []

            for item in items:
                overlap = False
                for existing in kept:
                    iou = ClassWiseNMS._temporal_iou(item, existing)
                    if iou > iou_thresh:
                        overlap = True
                        break
                if not overlap:
                    kept.append(item)

            preserved.extend(kept)

        preserved.sort(key=lambda x: x.start_time)
        return preserved

    @staticmethod
    def _temporal_iou(i1: ActivityInterval, i2: ActivityInterval) -> float:
        inter_start = max(i1.start_time, i2.start_time)
        inter_end = min(i1.end_time, i2.end_time)
        intersection = max(0.0, inter_end - inter_start)

        union_start = min(i1.start_time, i2.start_time)
        union_end = max(i1.end_time, i2.end_time)
        union = max(1e-6, union_end - union_start)

        return intersection / union


# ==============================================================================
# 10. Cross-Class Overlap Grouping
# ==============================================================================
class CrossClassGrouper:
    """
    Compares candidate intervals belonging to different classes.
    Groups overlapping candidates into unified EventProposal objects with multiple hypotheses.
    """
    @staticmethod
    def group_cross_class_candidates(
        candidates: List[ActivityInterval],
        config: PipelineConfig
    ) -> List[EventProposal]:
        if not candidates:
            return []

        # Respect cross-class grouping configuration
        if not config.enable_cross_class_grouping:
            logger.info("Cross-class grouping is disabled by configuration. Preserving single-candidate proposals.")
            return [
                EventProposal(
                    proposal_id=idx,
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    candidates=[cand]
                )
                for idx, cand in enumerate(candidates)
            ]

        # Sort chronologically
        candidates_sorted = sorted(candidates, key=lambda x: x.start_time)
        n = len(candidates_sorted)

        # Adjacency matrix for overlap
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = candidates_sorted[i], candidates_sorted[j]

                # Prevent same-class candidates from being cross-class grouped
                if c1.activity_class == c2.activity_class:
                    continue
                inter_start = max(c1.start_time, c2.start_time)
                inter_end = min(c1.end_time, c2.end_time)
                intersection = max(0.0, inter_end - inter_start)

                if intersection > 0:
                    union = max(1e-6, max(c1.end_time, c2.end_time) - min(c1.start_time, c2.start_time))
                    iou = intersection / union
                    min_dur = max(1e-6, min(c1.duration(), c2.duration()))
                    overlap_ratio = intersection / min_dur

                    if iou >= config.cross_class_iou_threshold or overlap_ratio >= config.cross_class_overlap_ratio:
                        adj[i].append(j)
                        adj[j].append(i)

        # Connected components grouping
        visited = [False] * n
        proposals: List[EventProposal] = []
        prop_id = 0

        for i in range(n):
            if not visited[i]:
                group_indices = []
                queue = [i]
                visited[i] = True

                while queue:
                    curr = queue.pop(0)
                    group_indices.append(curr)
                    for neighbor in adj[curr]:
                        if not visited[neighbor]:
                            # Prevent same-class candidates from being merged into the same multi-hypothesis proposal
                            if any(candidates_sorted[idx].activity_class == candidates_sorted[neighbor].activity_class for idx in group_indices):
                                continue
                            visited[neighbor] = True
                            queue.append(neighbor)

                group_candidates = [candidates_sorted[idx] for idx in group_indices]
                start_t = min(c.start_time for c in group_candidates)
                end_t = max(c.end_time for c in group_candidates)

                proposals.append(EventProposal(
                    proposal_id=prop_id,
                    start_time=start_t,
                    end_time=end_t,
                    candidates=group_candidates
                ))
                prop_id += 1

        logger.info(
            f"Cross-Class Grouping: Organized {len(candidates)} candidates into "
            f"{len(proposals)} multi-hypothesis event proposals."
        )
        return proposals


# ==============================================================================
# 11. Real Qwen-VL Verification Stage (HuggingFace Transformers)
# ==============================================================================
class QwenVLVerifier:
    """
    Real vision-language verification stage using Qwen2-VL / Qwen2.5-VL from Hugging Face.
    Extracts keyframes for each grouped event proposal, formats a visual verification query,
    and runs generative inference to confirm or reject candidate activity hypotheses
    based strictly on visible action evidence (without inferring intent).
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_name = config.vlm_model_name
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads Qwen-VL model and processor from Hugging Face."""
        logger.info(f"Loading Qwen-VL model: '{self.model_name}' from HuggingFace...")
        start_t = time.time()
        try:
            try:
                from transformers import Qwen2VLForConditionalGeneration
                model_cls = Qwen2VLForConditionalGeneration
            except ImportError:
                from transformers import AutoModelForVision2Seq
                model_cls = AutoModelForVision2Seq

            self.processor = AutoProcessor.from_pretrained(self.model_name)

            if torch and torch.cuda.is_available():
                dtype = torch.float16 if self.config.use_fp16 else torch.float32
                try:
                    # device_map="auto" distributes across available GPUs (e.g. 2x T4 on Kaggle)
                    self.model = model_cls.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        device_map="auto"
                    )
                except Exception as e:
                    logger.warning(f"device_map='auto' fallback ({e}). Loading directly onto cuda:0...")
                    self.model = model_cls.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype
                    ).to("cuda:0")
            else:
                self.model = model_cls.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                ).to("cpu")

            self.model.eval()
            elapsed = time.time() - start_t
            logger.info(f"Qwen-VL model '{self.model_name}' successfully loaded in {elapsed:.2f}s.")
        except Exception as e:
            logger.error(f"Error loading Qwen-VL model '{self.model_name}': {e}")
            raise

    def verify_proposals(
        self,
        proposals: List[EventProposal],
        video_metadata: VideoMetadata,
        timelines: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        config: Optional[PipelineConfig] = None
    ) -> List[EventProposal]:
        if not proposals:
            return proposals

        cfg = config if config is not None else self.config

        if self.model is None or self.processor is None:
            logger.warning("Qwen-VL model is not loaded. Skipping VLM verification.")
            return proposals

        logger.info(f"Running Qwen-VL inference on {len(proposals)} event proposal(s)...")

        for p_idx, prop in enumerate(proposals, 1):
            clip_start = max(0.0, prop.start_time - cfg.vlm_context_sec)
            clip_end = min(video_metadata.duration_sec, prop.end_time + cfg.vlm_context_sec)
            proposed_classes = list(set(c.activity_class for c in prop.candidates))

            # Determine peak activation timestamps for candidate activities
            peak_timestamps: List[float] = []
            for cand in prop.candidates:
                peak_t = self._get_candidate_peak_timestamp(cand, timelines)
                if "Fire" in cand.activity_class:
                    peak_timestamps.insert(0, peak_t)
                else:
                    peak_timestamps.append(peak_t)

            # 1. Extract frames sampled at 4 FPS with context, capped at max_frames (16) and max_dim (768px)
            pil_frames = self._extract_clip_frames(
                video_path=video_metadata.video_path,
                clip_start=clip_start,
                clip_end=clip_end,
                fps=video_metadata.fps,
                total_frames=video_metadata.total_frames,
                target_fps=cfg.vlm_fps,
                max_frames=cfg.vlm_max_frames,
                max_dim=cfg.vlm_max_img_dim,
                peak_timestamps=peak_timestamps
            )

            if not pil_frames:
                logger.warning(f"Could not extract frames for Proposal #{prop.proposal_id}. Preserving candidates.")
                for cand in prop.candidates:
                    cand.vlm_decision = "UNCERTAIN"
                    cand.vlm_verified = True
                    cand.vlm_evidence = "Frame extraction fallback; preserved as uncertain."
                continue

            # 2. Build structured verification prompt
            class_bullets = "\n".join([f"- {cls_name}" for cls_name in proposed_classes])
            prompt_text = (
                f"You are an objective video action verifier. "
                f"Examine these sequential video frames covering timestamps {clip_start:.1f}s to {clip_end:.1f}s.\n"
                f"The following candidate activities were proposed:\n{class_bullets}\n\n"
                "Task: For EACH proposed activity, determine whether its visible physical actions "
                "(bodily movements, falling, physical struggle, breaking objects, visible flames/smoke, rapid running) "
                "are clearly observable in the frames.\n"
                "Rules:\n"
                "- Do NOT infer criminal intent, legal authorization, or unobservable mental states.\n"
                "- Only verify if the physical actions themselves are visible.\n"
                "- For each class, choose one status: SUPPORTED, REJECTED, or UNCERTAIN.\n"
                "- Output exactly one line per class in this format:\n"
                "Class: <class_name> | Status: <SUPPORTED, REJECTED, or UNCERTAIN> | Evidence: <brief description of visible actions>\n"
            )

            # 3. Construct chat messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in pil_frames],
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]

            # 4. Run model inference
            try:
                try:
                    chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(
                        text=[chat_prompt],
                        images=pil_frames,
                        padding=True,
                        return_tensors="pt"
                    )
                except Exception:
                    inputs = self.processor(
                        text=[f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"],
                        images=pil_frames,
                        padding=True,
                        return_tensors="pt"
                    )

                target_device = self.model.device if hasattr(self.model, "device") else "cuda:0"
                inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False
                    )

                input_len = inputs["input_ids"].shape[1]
                generated_ids = output_ids[:, input_len:]
                vlm_response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                logger.info(f"--- Qwen-VL Response for Proposal #{prop.proposal_id} [{clip_start:.1f}s-{clip_end:.1f}s] ---")
                logger.info(f"{vlm_response}")

                # 5. Parse response for each proposed class and handle disagreements
                for cand in prop.candidates:
                    status, evidence = self._parse_cand_status(vlm_response, cand.activity_class)

                    # If REJECTED, check if candidate has strong X-CLIP evidence requiring targeted re-examination
                    if status == "REJECTED" and self._is_strong_xclip_candidate(cand):
                        logger.info(
                            f"Strong X-CLIP candidate '{cand.activity_class}' (max_sim={cand.max_similarity:.3f}, "
                            f"margin={cand.relative_margin:.3f}, ranking={cand.ranking_score:.3f}) was REJECTED by initial VLM pass. "
                            f"Re-examining disagreement using targeted peak frames..."
                        )
                        cand_peak_t = self._get_candidate_peak_timestamp(cand, timelines)
                        status, evidence = self._reexamine_disagreement(
                            cand=cand,
                            video_metadata=video_metadata,
                            peak_timestamp=cand_peak_t,
                            config=cfg
                        )

                    cand.vlm_decision = status
                    cand.vlm_verified = (status == "SUPPORTED")
                    cand.vlm_evidence = evidence

                prop.vlm_verified = any(c.vlm_decision in ["SUPPORTED", "UNCERTAIN"] for c in prop.candidates)
                prop.vlm_notes = vlm_response

            except Exception as e:
                logger.warning(f"Qwen-VL generation exception for proposal #{prop.proposal_id}: {e}")
                for cand in prop.candidates:
                    cand.vlm_decision = "UNCERTAIN"
                    cand.vlm_verified = True
                    cand.vlm_evidence = f"VLM generation exception: {e}; preserved as uncertain."

        return proposals

    def _extract_clip_frames(
        self,
        video_path: str,
        clip_start: float,
        clip_end: float,
        fps: float,
        total_frames: int,
        target_fps: float = 4.0,
        max_frames: int = 16,
        max_dim: int = 768,
        peak_timestamps: Optional[List[float]] = None
    ) -> List[Any]:
        """
        Samples frames at target_fps (default 4 FPS) from the candidate clip with context.
        Prioritizes frames from peak X-CLIP windows (e.g. Fire/Smoke) while capping total
        frames at max_frames (16) and resizing up to max_dim (768px).
        """
        if cv2 is None:
            return []

        step_sec = 1.0 / max(0.5, target_fps)
        sample_times = []
        t = clip_start
        while t <= clip_end + 1e-4:
            sample_times.append(round(t, 3))
            t += step_sec

        if not sample_times:
            sample_times = [clip_start]

        raw_frame_indices = [
            min(total_frames - 1, max(0, int(round(st * fps))))
            for st in sample_times
        ]
        unique_indices = sorted(list(set(raw_frame_indices)))

        # If exceeding max_frames, prioritize frames near peak windows
        if len(unique_indices) > max_frames:
            selected_set = set()

            # Always preserve start, middle, and end anchor frames
            selected_set.add(unique_indices[0])
            selected_set.add(unique_indices[len(unique_indices) // 2])
            selected_set.add(unique_indices[-1])

            # Prioritize frames near peak windows
            if peak_timestamps:
                for pt in peak_timestamps:
                    peak_frame = min(total_frames - 1, max(0, int(round(pt * fps))))
                    near_peak = sorted(unique_indices, key=lambda f: abs(f - peak_frame))
                    for f in near_peak[:4]:
                        if len(selected_set) < max_frames:
                            selected_set.add(f)

            # Fill remaining slots evenly across clip
            if len(selected_set) < max_frames:
                remaining_slots = max_frames - len(selected_set)
                step = max(1, len(unique_indices) // (remaining_slots + 1))
                for idx in range(0, len(unique_indices), step):
                    selected_set.add(unique_indices[idx])
                    if len(selected_set) >= max_frames:
                        break

            final_frame_indices = sorted(list(selected_set))
        else:
            final_frame_indices = unique_indices

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        pil_frames = []
        for f_idx in final_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                if max(h, w) > max_dim:
                    scale = float(max_dim) / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if Image is not None:
                    pil_frames.append(Image.fromarray(rgb))
                else:
                    pil_frames.append(rgb)

        cap.release()
        return pil_frames

    @staticmethod
    def _extract_status_from_text(text: str) -> str:
        """
        Parses text for status using word-boundary regex matching.
        Guarantees that negative words ('unsupported', 'not supported')
        do not accidentally match 'supported'.
        """
        # 1. Check for REJECTED / UNSUPPORTED / NOT SUPPORTED first
        if re.search(r'\b(?:rejected|not\s+supported|unsupported|disproven|false|negative|denied)\b', text, re.IGNORECASE):
            return "REJECTED"
        # 2. Check for UNCERTAIN / INCONCLUSIVE
        elif re.search(r'\b(?:uncertain|inconclusive|unclear|ambiguous|partially\s+supported|doubtful)\b', text, re.IGNORECASE):
            return "UNCERTAIN"
        # 3. Check for SUPPORTED (with \b word boundary)
        elif re.search(r'\b(?:supported|confirmed|verified|positive|observed|detected)\b', text, re.IGNORECASE):
            return "SUPPORTED"
        return "UNCERTAIN"

    def _parse_cand_status(self, response_text: str, class_name: str) -> Tuple[str, str]:
        """Parses Qwen-VL response text to assign SUPPORTED, REJECTED, or UNCERTAIN per candidate."""
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        class_parts = [p.strip().lower() for p in class_name.split("/")]

        for line in lines:
            line_lower = line.lower()
            if (class_name.lower() in line_lower) or any(p in line_lower for p in class_parts if len(p) > 3):
                status = self._extract_status_from_text(line)
                return status, line

        if re.search(r'\b(?:none|no\s+activity|no\s+action)\b', response_text, re.IGNORECASE):
            return "REJECTED", "VLM indicated no activity is visibly supported."

        return "UNCERTAIN", "Class was not explicitly referenced in VLM output; preserved as uncertain."

    @staticmethod
    def _is_strong_xclip_candidate(cand: ActivityInterval) -> bool:
        """Determines if a candidate has exceptionally strong X-CLIP evidence."""
        return (
            cand.max_similarity >= 0.28 or
            cand.relative_margin >= 0.03 or
            cand.ranking_score >= 0.40
        )

    def _get_candidate_peak_timestamp(
        self,
        cand: ActivityInterval,
        timelines: Optional[Dict[str, Dict[str, np.ndarray]]]
    ) -> float:
        """Finds peak activation timestamp for candidate from X-CLIP timelines."""
        if timelines and cand.activity_class in timelines:
            ts = timelines[cand.activity_class]["timestamps"]
            scores = timelines[cand.activity_class]["raw_scores"]
            mask = (ts >= cand.start_time) & (ts <= cand.end_time)
            if np.any(mask):
                sub_ts = ts[mask]
                sub_scores = scores[mask]
                return float(sub_ts[np.argmax(sub_scores)])
        return float((cand.start_time + cand.end_time) / 2.0)

    def _reexamine_disagreement(
        self,
        cand: ActivityInterval,
        video_metadata: VideoMetadata,
        peak_timestamp: float,
        config: PipelineConfig
    ) -> Tuple[str, str]:
        """
        Re-examines a strong X-CLIP candidate using targeted high-resolution frames
        centered tightly around the peak window.
        """
        try:
            sub_start = max(0.0, peak_timestamp - 1.0)
            sub_end = min(video_metadata.duration_sec, peak_timestamp + 1.0)

            targeted_frames = self._extract_clip_frames(
                video_path=video_metadata.video_path,
                clip_start=sub_start,
                clip_end=sub_end,
                fps=video_metadata.fps,
                total_frames=video_metadata.total_frames,
                target_fps=config.vlm_fps,
                max_frames=8,
                max_dim=config.vlm_max_img_dim,
                peak_timestamps=[peak_timestamp]
            )

            if not targeted_frames:
                return "UNCERTAIN", "Targeted frame extraction failed; preserved as uncertain."

            reexamine_prompt = (
                f"You are an objective video action verifier re-examining a specific candidate event.\n"
                f"Candidate Activity: '{cand.activity_class}' at timestamps {sub_start:.1f}s to {sub_end:.1f}s.\n"
                f"A strong localized feature detection was observed at {peak_timestamp:.1f}s.\n"
                f"Carefully examine these targeted sequential frames.\n"
                f"Question: Are visible bodily movements, physical interactions, or phenomena corresponding to "
                f"'{cand.activity_class}' observable?\n"
                f"Rules:\n"
                f"- Do NOT infer intent or mental states; verify physical actions only.\n"
                f"- Output exactly one line:\n"
                f"Status: <SUPPORTED, REJECTED, or UNCERTAIN> | Evidence: <brief description of visible actions>"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in targeted_frames],
                        {"type": "text", "text": reexamine_prompt}
                    ]
                }
            ]

            try:
                chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[chat_prompt],
                    images=targeted_frames,
                    padding=True,
                    return_tensors="pt"
                )
            except Exception:
                inputs = self.processor(
                    text=[f"<|im_start|>user\n{reexamine_prompt}<|im_end|>\n<|im_start|>assistant\n"],
                    images=targeted_frames,
                    padding=True,
                    return_tensors="pt"
                )

            target_device = self.model.device if hasattr(self.model, "device") else "cuda:0"
            inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            logger.info(f"Targeted re-examination for '{cand.activity_class}' response: {response}")

            re_status = self._extract_status_from_text(response)
            if re_status == "SUPPORTED":
                return "SUPPORTED", f"Verified on targeted re-examination: {response}"
            elif re_status == "UNCERTAIN":
                return "UNCERTAIN", f"Targeted re-examination uncertain: {response}"
            else:
                # Disagreement remains: Preserve candidate with UNCERTAIN status due to strong X-CLIP detection
                return "UNCERTAIN", (
                    f"VLM re-examination indicated rejection ({response}), but preserved as UNCERTAIN "
                    f"due to strong X-CLIP response (max_sim={cand.max_similarity:.3f}, margin={cand.relative_margin:.3f})"
                )
        except Exception as e:
            logger.warning(f"Error during targeted re-examination for '{cand.activity_class}': {e}")
            return "UNCERTAIN", f"Re-examination exception: {e}; preserved as UNCERTAIN."


# ==============================================================================
# 12. Localized Boundary Refinement
# ==============================================================================
class BoundaryRefiner:
    """
    Refines start and end boundaries using fine-grained temporal windows evaluated
    ONLY within the local search range [start - 1.0, end + 1.0].
    """
    @staticmethod
    def refine_candidate_boundaries(
        candidates: List[ActivityInterval],
        video_metadata: VideoMetadata,
        inference_engine: Optional[XCLIPInferenceEngine],
        config: PipelineConfig
    ) -> List[ActivityInterval]:
        if not candidates or not config.enable_refinement or inference_engine is None:
            return candidates

        for cand in candidates:
            search_start = max(0.0, cand.start_time - 1.0)
            search_end = min(video_metadata.duration_sec, cand.end_time + 1.0)

            if (search_end - search_start) < config.refinement_window_duration:
                continue

            # Generate fine-grained sub-windows ONLY in this candidate's local search range
            sub_windows: List[TemporalWindow] = []
            curr = search_start
            win_id = 0
            while curr + config.refinement_window_duration <= search_end:
                sub_windows.append(TemporalWindow(
                    window_id=win_id,
                    start_time=round(curr, 3),
                    end_time=round(curr + config.refinement_window_duration, 3),
                    center_time=round(curr + config.refinement_window_duration / 2.0, 3),
                    start_frame=int(round(curr * video_metadata.fps)),
                    end_frame=min(video_metadata.total_frames - 1, int(round((curr + config.refinement_window_duration) * video_metadata.fps))),
                    frame_indices=np.linspace(
                        int(round(curr * video_metadata.fps)),
                        min(video_metadata.total_frames - 1, int(round((curr + config.refinement_window_duration) * video_metadata.fps))),
                        config.refinement_frames,
                        dtype=int
                    ).tolist()
                ))
                curr += config.refinement_stride
                win_id += 1

            if not sub_windows:
                continue

            try:
                sub_clips = CachedFrameExtractor.extract_window_frames_block(video_metadata.video_path, sub_windows)
                sim_matrix = inference_engine.infer_windows(sub_clips)
                c_idx = inference_engine.classes.index(cand.activity_class)
                sub_scores = sim_matrix[:, c_idx]

                # Compute relative margins on sub-windows
                sub_margins_matrix = compute_relative_margins(sim_matrix)
                sub_margins = sub_margins_matrix[:, c_idx]

                thresh = config.class_thresholds.get(cand.activity_class, config.default_threshold) * 0.9
                rel_thresh = config.rel_margin_threshold

                # Dual qualification condition: similarity AND relative margin
                active_idx = np.where((sub_scores >= thresh) & (sub_margins >= rel_thresh))[0]

                if len(active_idx) > 0:
                    cand.start_time = round(float(sub_windows[active_idx[0]].start_time), 2)
                    cand.end_time = round(float(sub_windows[active_idx[-1]].end_time), 2)
                    cand.refined = True

                    # Recalculate all interval metrics
                    cand.mean_similarity = float(np.mean(sub_scores[active_idx]))
                    cand.max_similarity = float(np.max(sub_scores[active_idx]))
                    cand.relative_margin = float(np.mean(sub_margins[active_idx]))
                    cand.max_relative_margin = float(np.max(sub_margins[active_idx]))

                    # Recalculate ranking score and confidence
                    logoic_sigmoid = 1.0 / (1.0 + math.exp(-cand.logoic_score)) if config.enable_logoic else 0.5
                    rel_margin_contrib = float(np.clip(cand.relative_margin * 5.0, -0.5, 0.5))

                    cand.ranking_score = float(
                        0.35 * cand.mean_similarity +
                        0.25 * cand.max_similarity +
                        0.20 * rel_margin_contrib +
                        0.20 * logoic_sigmoid
                    )

                    abs_component = 0.5 * cand.max_similarity + 0.5 * cand.mean_similarity
                    margin_mod = max(0.0, cand.relative_margin + 0.05) * 2.0
                    raw_confidence = 0.40 * abs_component + 0.35 * margin_mod + 0.25 * logoic_sigmoid
                    cand.confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Local refinement fallback for '{cand.activity_class}' due to: {e}")

        return candidates


# ==============================================================================
# 13. Confidence Score Generation (Uncalibrated Ranking Score)
# ==============================================================================
class ConfidenceEstimator:
    """
    Computes final detection confidence score.
    Documented explicitly as an uncalibrated ranking score rather than a probability.
    Synthesizes peak similarity, mean similarity, relative class margin, and LogOIC contrast.
    """
    @staticmethod
    def compute_confidence(candidates: List[ActivityInterval], config: PipelineConfig) -> List[ActivityInterval]:
        for cand in candidates:
            # Absolute similarity component
            abs_component = 0.5 * cand.max_similarity + 0.5 * cand.mean_similarity

            # Relative margin component (reward standing out over competing classes)
            margin_mod = max(0.0, cand.relative_margin + 0.05) * 2.0

            # LogOIC contrast component
            logoic_mod = 1.0 / (1.0 + math.exp(-cand.logoic_score)) if config.enable_logoic else 0.5

            raw_confidence = 0.40 * abs_component + 0.35 * margin_mod + 0.25 * logoic_mod
            cand.confidence = float(np.clip(raw_confidence, 0.0, 1.0))

        return candidates


# ==============================================================================
# Structured Output Generation
# ==============================================================================
class OutputFormatter:
    """Formats pipeline results into the assignment standard JSON format."""
    @staticmethod
    def create_json(
        video_name: str,
        activities: List[ActivityInterval],
        metadata: Optional[VideoMetadata] = None,
        config: Optional[PipelineConfig] = None
    ) -> Dict[str, Any]:
        result = {
            "video": video_name,
            "activities": [act.to_dict() for act in activities]
        }

        if config and config.detailed_output and metadata:
            result["_metadata"] = {
                "duration_sec": metadata.duration_sec,
                "fps": metadata.fps,
                "resolution": metadata.resolution,
                "total_frames": metadata.total_frames,
                "total_windows": metadata.num_windows,
                "total_detections": len(activities)
            }

        return result


# ==============================================================================
# Master Pipeline Orchestrator
# ==============================================================================
class ZeroShotActivityPipeline:
    """Master orchestrator integrating all 15 revised pipeline components."""
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config if config is not None else PipelineConfig()
        logger.info(
            f"Initializing Zero-Shot Activity Pipeline (aggregation='{self.config.prompt_aggregation}', "
            f"smoothing_sigma={self.config.smoothing_sigma})..."
        )
        self.inference_engine = XCLIPInferenceEngine(self.config)
        self.vlm_verifier = None
        if self.config.enable_vlm_verification:
            self.vlm_verifier = QwenVLVerifier(self.config)

    def process_video(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        start_time = time.time()
        video_path = Path(video_path)
        logger.info(f"=== Starting Processing: {video_path.name} ===")

        # 1. Video Input and Metadata Extraction
        metadata = VideoMetadataExtractor.extract(video_path, self.config)

        # 2. Temporal Video Chunking
        windows = TemporalChunker.generate_windows(
            metadata=metadata,
            window_duration=self.config.window_duration,
            window_stride=self.config.window_stride,
            frames_per_window=self.config.frames_per_window
        )

        # 15. Scalable Long-Video Processing: Block-based temporal chunking
        block_size = self.config.temporal_block_windows
        num_windows = len(windows)
        sim_matrices: List[np.ndarray] = []

        logger.info(f"Processing {num_windows} windows in blocks of {block_size} to conserve memory...")
        for b_start in range(0, num_windows, block_size):
            b_end = min(b_start + block_size, num_windows)
            block_windows = windows[b_start:b_end]

            # Extract frames for this block
            block_clips = CachedFrameExtractor.extract_window_frames_block(str(video_path), block_windows)
            # Infer with X-CLIP
            block_sim = self.inference_engine.infer_windows(block_clips)
            sim_matrices.append(block_sim)

            # Free block memory
            del block_clips
            gc.collect()

        similarity_matrix = np.concatenate(sim_matrices, axis=0) if sim_matrices else np.empty((0, len(self.inference_engine.classes)))

        # 3. Introduce Relative Class Evidence: R_c(t) = S_c(t) - max_{j != c} S_j(t)
        relative_margins = compute_relative_margins(similarity_matrix)

        # 4. Improved Temporal Smoothing: Gaussian smoothing (sigma=0.5) on both similarities & margins
        timelines = TemporalScoreAggregator.process_timelines(
            similarity_matrix=similarity_matrix,
            relative_margins=relative_margins,
            windows=windows,
            classes=self.inference_engine.classes,
            config=self.config
        )

        # 5 & 6. Candidate Activity Interval Generation (using both relative and absolute evidence)
        candidates = CandidateIntervalGenerator.generate_candidates(timelines, windows, self.config)

        # 8. Distance-Weighted LogOIC Ranking & Candidate Ranking Score
        # Computes LogOIC ranking score (or fallback ranking score when LogOIC is disabled)
        candidates = LogOICTemporalLocalizer.compute_logoic_and_ranking(
            candidates, timelines, windows, self.config
        )

        # 9. Class-Wise NMS (restricted strictly to the same class using ranking_score)
        candidates = ClassWiseNMS.apply_nms(candidates, self.config.nms_iou_threshold)

        # 10. Cross-Class Overlap Grouping (form event proposals with multi-class hypotheses)
        proposals = CrossClassGrouper.group_cross_class_candidates(candidates, self.config)

        # 11. Real Qwen-VL Verification Stage
        if self.config.enable_vlm_verification:
            if self.vlm_verifier is None:
                self.vlm_verifier = QwenVLVerifier(self.config)
            proposals = self.vlm_verifier.verify_proposals(proposals, metadata, timelines, self.config)

            # Filter candidates: retain SUPPORTED hypotheses at full strength,
            # and retain UNCERTAIN hypotheses with adjusted confidence.
            # Discard only confirmed false-alarm REJECTED hypotheses.
            verified_candidates = []
            for prop in proposals:
                for cand in prop.candidates:
                    if cand.vlm_decision == "SUPPORTED":
                        verified_candidates.append(cand)
                    elif cand.vlm_decision == "UNCERTAIN":
                        cand.confidence = round(cand.confidence * 0.85, 4)
                        verified_candidates.append(cand)
                        logger.info(
                            f"Retained UNCERTAIN candidate '{cand.activity_class}' at "
                            f"[{cand.start_time:.1f}s - {cand.end_time:.1f}s] with adjusted confidence {cand.confidence}. "
                            f"Evidence: {cand.vlm_evidence}"
                        )
                    elif cand.vlm_decision == "REJECTED":
                        logger.info(
                            f"VLM rejected false-alarm candidate '{cand.activity_class}' at "
                            f"[{cand.start_time:.1f}s - {cand.end_time:.1f}s]. Reason: {cand.vlm_evidence}"
                        )
                    else:
                        verified_candidates.append(cand)
            candidates = verified_candidates

        # 12. Localized Boundary Refinement
        if self.config.enable_refinement:
            candidates = BoundaryRefiner.refine_candidate_boundaries(
                candidates, metadata, self.inference_engine, self.config
            )

        # 13. Confidence Score Generation (uncalibrated ranking score)
        activities = ConfidenceEstimator.compute_confidence(candidates, self.config)

        # 14. Export Timelines and Candidates if requested
        if self.config.export_timeline:
            self._export_timeline_data(video_path.stem, timelines, activities, proposals)

        # Structured Output Generation
        output_json = OutputFormatter.create_json(
            video_name=metadata.video_name,
            activities=activities,
            metadata=metadata,
            config=self.config
        )

        total_elapsed = time.time() - start_time
        logger.info(
            f"=== Finished {video_path.name}: Detected {len(activities)} activities in {total_elapsed:.2f}s ==="
        )
        return output_json

    def _export_timeline_data(
        self,
        video_stem: str,
        timelines: Dict[str, Dict[str, np.ndarray]],
        activities: List[ActivityInterval],
        proposals: List[EventProposal]
    ):
        """Exports raw/smoothed similarities, relative margins, candidates, and proposals."""
        export_dir = Path("timelines_export")
        export_dir.mkdir(exist_ok=True)
        csv_file = export_dir / f"{video_stem}_timelines.csv"

        classes = list(timelines.keys())
        first_class = classes[0]
        timestamps = timelines[first_class]["timestamps"]

        # CSV Header
        header = ["timestamp"]
        for c in classes:
            header.extend([f"{c}_raw", f"{c}_smoothed", f"{c}_rel_margin_raw", f"{c}_rel_margin_smoothed"])
        rows = [",".join(header)]

        for i, t in enumerate(timestamps):
            row = [f"{t:.2f}"]
            for c in classes:
                row.append(f"{timelines[c]['raw_scores'][i]:.4f}")
                row.append(f"{timelines[c]['smoothed_scores'][i]:.4f}")
                row.append(f"{timelines[c]['raw_relative_margins'][i]:.4f}")
                row.append(f"{timelines[c]['smoothed_relative_margins'][i]:.4f}")
            rows.append(",".join(row))

        csv_file.write_text("\n".join(rows), encoding="utf-8")
        logger.info(f"Exported activation timeline CSV to '{csv_file}'.")

        # Export Candidate & Event Proposal Details
        diag_file = export_dir / f"{video_stem}_proposals.json"
        diag_data = {
            "proposals": [
                {
                    "proposal_id": p.proposal_id,
                    "start_time": p.start_time,
                    "end_time": p.end_time,
                    "hypotheses": [
                        {
                            "class": c.activity_class,
                            "ranking_score": round(c.ranking_score, 4),
                            "confidence": round(c.confidence, 4),
                            "mean_sim": round(c.mean_similarity, 4),
                            "max_sim": round(c.max_similarity, 4),
                            "rel_margin": round(c.relative_margin, 4),
                            "logoic_score": round(c.logoic_score, 4),
                            "vlm_decision": c.vlm_decision,
                            "vlm_verified": c.vlm_verified,
                            "vlm_evidence": c.vlm_evidence
                        } for c in p.candidates
                    ]
                } for p in proposals
            ]
        }
        with open(diag_file, "w", encoding="utf-8") as f:
            json.dump(diag_data, f, indent=2)
        logger.info(f"Exported event proposals diagnostic JSON to '{diag_file}'.")


# ==============================================================================
# Hardcoded Execution Paths & Main Runner (Single Video)
# ==============================================================================
INPUT_VIDEO_PATH = Path("selected_benchmark_data/videos/Fighting_Assault_Fighting003_x264.mp4")
LOCAL_TEST_PATH = Path("cerebralzip_test_data/videos/Fighting_Assault_Fighting003_x264.mp4")
LOCAL_SUITE_PATH = Path("cerebralzip_benchmark_suite/videos/Fighting_Assault_Fighting003_x264.mp4")
KAGGLE_VIDEO_PATH = Path("/kaggle/input/selected-benchmark-data/videos/Fighting_Assault_Fighting003_x264.mp4")
KAGGLE_FALLBACK_PATH = Path("/kaggle/input/datasets/neo110/cerebralzip-assignment1-videos/videos/Fire_Smoke_Arson022_x264.mp4")
OUTPUT_JSON_PATH = Path("activity_results.json")


def main():
    """
    Main entry point executing the updated Zero-Shot Activity Detection Pipeline
    for a single hardcoded video without command-line arguments.
    """
    if INPUT_VIDEO_PATH.is_file():
        target_video = INPUT_VIDEO_PATH
    elif LOCAL_TEST_PATH.is_file():
        target_video = LOCAL_TEST_PATH
    elif LOCAL_SUITE_PATH.is_file():
        target_video = LOCAL_SUITE_PATH
    elif KAGGLE_VIDEO_PATH.is_file():
        target_video = KAGGLE_VIDEO_PATH
    elif KAGGLE_FALLBACK_PATH.is_file():
        target_video = KAGGLE_FALLBACK_PATH
    else:
        target_video = None
        for pattern in [
            "cerebralzip_test_data/videos/*.mp4",
            "selected_benchmark_data/videos/*.mp4",
            "cerebralzip_benchmark_suite/videos/*.mp4",
            "cerebralzip_benchmark_videos/videos/*.mp4",
            "videos/*.mp4",
            "*.mp4",
            "**/*.mp4"
        ]:
            matches = list(Path(".").glob(pattern))
            if matches:
                target_video = matches[0]
                break

        if target_video is None or not target_video.is_file():
            logger.error(f"Target video file not found at '{INPUT_VIDEO_PATH}' or '{KAGGLE_VIDEO_PATH}'.")
            sys.exit(1)

    logger.info(f"Target Video: '{target_video.resolve()}'")

    # Initialize configuration with revised parameters:
    # - prompt_aggregation='mean'
    # - smoothing_sigma=0.5
    # - enable_logoic=True
    # - enable_cross_class_grouping=True
    # - enable_refinement=True
    # - enable_vlm_verification=False (optional plug-in)
    config = PipelineConfig(
        model_name="microsoft/xclip-base-patch32",
        window_duration=4.0,           # 4-second temporal window
        window_stride=1.0,             # 1-second stride
        frames_per_window=8,           # 8 uniformly sampled frames
        batch_size=16,                 # Optimized for GPU batching (2x T4 on Kaggle)
        use_fp16=True,                 # FP16 mixed precision
        prompt_aggregation="mean",     # Changed default from 'max' to 'mean'
        smoothing_type="gaussian",     # 1D Gaussian temporal smoothing
        smoothing_sigma=0.5,           # Sigma 0.5 to prevent over-smoothing
        enable_logoic=True,            # Distance-weighted LogOIC localization
        enable_cross_class_grouping=True, # Multi-hypothesis event proposals
        enable_vlm_verification=True,  # Real Qwen2-VL verification stage
        vlm_model_name="Qwen/Qwen2-VL-2B-Instruct", # Ultra-fast 2B vision-language model
        vlm_fps=4.0,                   # Sample frames at 4 FPS from candidate interval
        vlm_context_sec=1.0,           # ~1s context before and after
        vlm_max_frames=16,             # Max frames to protect GPU memory
        vlm_max_img_dim=768,           # 768px max dimension
        enable_refinement=True,        # Localized boundary refinement
        export_timeline=True,          # Export CSV timelines & diagnostic JSON
        detailed_output=True           # Include detailed metadata
    )

    # Initialize and execute pipeline
    pipeline = ZeroShotActivityPipeline(config)
    result = pipeline.process_video(target_video)

    # Save structured output JSON
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Processing complete! Results saved to '{OUTPUT_JSON_PATH.resolve()}'.")
    print("\n" + "=" * 60)
    print(f"PIPELINE RUN COMPLETE — Saved: {OUTPUT_JSON_PATH.name}")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()