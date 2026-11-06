#!/usr/bin/env python3
"""
================================================================================
Centralized Global Configuration
================================================================================
Defines all classes, hyperparameters, detection thresholds, hardware settings,
and filesystem paths. All pipeline modules and evaluation scripts must import
their configurations and paths from this central module.
================================================================================
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

# Optimize CUDA memory allocation on NVIDIA GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Root workspace directory
WORKSPACE_ROOT = Path(__file__).resolve().parent

# ==============================================================================
# 1. Activity Class Taxonomy
# ==============================================================================
ALL_CANONICAL_CLASSES: List[str] = [
    "Fighting/Physical Assault",
    "Falling",
    "Fire/Smoke",
    "Intrusion/Trespassing",
    "Loitering",
    "Vandalism",
    "Theft/Chain Snatching",
    "Crowd Formation/Unusual Gathering",
    "Running in a Restricted Zone",
    "Abandoned Object"
]

EXCLUDED_BENCHMARK_CLASSES: Set[str] = {
    "Loitering",
    "Crowd Formation/Unusual Gathering",
    "Running in a Restricted Zone"
}

KNOWN_CAUSAL_CO_OCCURRENCES: Set[Tuple[str, str]] = {
    ("Fire/Smoke", "Intrusion/Trespassing"),
    ("Intrusion/Trespassing", "Fire/Smoke"),
    ("Fire/Smoke", "Vandalism"),
    ("Vandalism", "Fire/Smoke"),
    ("Fire/Smoke", "Loitering"),
    ("Loitering", "Fire/Smoke"),
    ("Theft/Chain Snatching", "Intrusion/Trespassing"),
    ("Intrusion/Trespassing", "Theft/Chain Snatching"),
    ("Theft/Chain Snatching", "Loitering"),
    ("Loitering", "Theft/Chain Snatching"),
    ("Theft/Chain Snatching", "Running in a Restricted Zone"),
    ("Running in a Restricted Zone", "Theft/Chain Snatching"),
    ("Vandalism", "Intrusion/Trespassing"),
    ("Intrusion/Trespassing", "Vandalism"),
    ("Fighting/Physical Assault", "Crowd Formation/Unusual Gathering"),
    ("Crowd Formation/Unusual Gathering", "Fighting/Physical Assault"),
    ("Fighting/Physical Assault", "Running in a Restricted Zone"),
    ("Running in a Restricted Zone", "Fighting/Physical Assault")
}

CLASS_NORMALIZATION_MAP: Dict[str, str] = {
    "fighting": "Fighting/Physical Assault",
    "physical assault": "Fighting/Physical Assault",
    "assault": "Fighting/Physical Assault",
    "falling": "Falling",
    "fall": "Falling",
    "fire": "Fire/Smoke",
    "smoke": "Fire/Smoke",
    "arson": "Fire/Smoke",
    "intrusion": "Intrusion/Trespassing",
    "trespassing": "Intrusion/Trespassing",
    "burglary": "Intrusion/Trespassing",
    "loitering": "Loitering",
    "vandalism": "Vandalism",
    "theft": "Theft/Chain Snatching",
    "chain snatching": "Theft/Chain Snatching",
    "crowd": "Crowd Formation/Unusual Gathering",
    "running": "Running in a Restricted Zone",
    "abandoned object": "Abandoned Object"
}


def normalize_class_name(raw: str) -> str:
    """Maps raw label strings to canonical class names."""
    if not raw:
        return "Unknown"
    cleaned = raw.strip().lower()
    for pattern, canonical in CLASS_NORMALIZATION_MAP.items():
        if pattern in cleaned:
            return canonical
    for canonical in ALL_CANONICAL_CLASSES:
        if canonical.lower() in cleaned:
            return canonical
    return raw.strip()


def parse_timestamp(v: Any, default_val: float = 0.0) -> float:
    """Parses timestamps from float, int, MM:SS, or None."""
    if v is None:
        return default_val
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        if ":" in v:
            parts = v.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        try:
            return float(v)
        except ValueError:
            return default_val
    return default_val


# ==============================================================================
# 2. Pipeline Master Hyperparameters & Thresholds
# ==============================================================================
@dataclass
class PipelineConfig:
    # Model backbone
    model_name: str = "microsoft/xclip-base-patch32"
    window_duration: float = 4.0
    window_stride: float = 1.0
    frames_per_window: int = 8
    batch_size: int = 16
    use_fp16: bool = True
    temporal_block_windows: int = 120

    # Prompt aggregation & temporal smoothing
    prompt_aggregation: str = "mean"
    smoothing_sigma: float = 0.5

    # Thresholds
    default_threshold: float = 0.20

    class_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "Fighting/Physical Assault": 0.15,
        "Falling": 0.10,
        "Fire/Smoke": 0.12,
        "Intrusion/Trespassing": 0.22,
        "Loitering": 0.20,
        "Vandalism": 0.15,
        "Theft/Chain Snatching": 0.16,
        "Crowd Formation/Unusual Gathering": 0.20,
        "Running in a Restricted Zone": 0.20,
        "Abandoned Object": 0.15
    })
    rel_margin_threshold: float = -0.015
    min_consecutive_windows: int = 2
    max_merge_gap: float = 2.0
    min_duration: float = 1.5
    max_duration: float = 300.0

    # LogOIC Contrast
    enable_logoic: bool = True
    logoic_context_ratio: float = 0.25
    logoic_tau: float = 2.0
    logoic_alpha: float = 0.3

    # NMS & Cross-Class Grouping
    nms_iou_threshold: float = 0.3
    enable_cross_class_grouping: bool = True
    cross_class_iou_threshold: float = 0.3
    cross_class_overlap_ratio: float = 0.5

    # VLM Verification
    enable_vlm_verification: bool = True
    vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    vlm_fps: float = 4.0
    vlm_context_sec: float = 1.0
    vlm_max_frames: int = 16
    vlm_max_img_dim: int = 768


# ==============================================================================
# 3. Path Configuration (Anchored to 'results/' and Dataset Candidates)
# ==============================================================================
def discover_dataset_dir() -> Path:
    candidates = [
        Path("/kaggle/input/datasets/neo110/cerebralzip-assignment1-videos/cerebralzip_test_data/cerebralzip_test_data"),
        WORKSPACE_ROOT / "cerebralzip_test_data",
        WORKSPACE_ROOT / "selected_benchmark_data",
        Path("/kaggle/input/selected-benchmark-data"),
        Path("/kaggle/input/cerebralzip-test-data"),
        WORKSPACE_ROOT
    ]
    for c in candidates:
        if (c / "labels").is_dir() or (c / "videos").is_dir():
            return c
    return WORKSPACE_ROOT / "cerebralzip_test_data"


DATASET_DIR: Path = discover_dataset_dir()
VIDEOS_DIR: Path = DATASET_DIR / "videos"
LABELS_DIR: Path = DATASET_DIR / "labels"

# Centralized results directory
RESULTS_DIR: Path = WORKSPACE_ROOT / "results"
TIMELINES_DIR: Path = RESULTS_DIR / "timelines_export"
PREDICTIONS_JSON: Path = RESULTS_DIR / "benchmark_predictions.json"
OUTPUT_METRICS_JSON: Path = RESULTS_DIR / "quantitative_metrics.json"
OUTPUT_CONFUSION_CSV: Path = RESULTS_DIR / "confusion_matrix.csv"
OUTPUT_REPORT_MD: Path = RESULTS_DIR / "qualitative_analysis_report.md"
OUTPUT_QUALITATIVE_JSON: Path = RESULTS_DIR / "qualitative_analysis.json"

# Ensure results directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMELINES_DIR.mkdir(parents=True, exist_ok=True)


def get_default_config() -> PipelineConfig:
    """Returns a hardware-aware PipelineConfig."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    return PipelineConfig(
        use_fp16=has_cuda,
        batch_size=16 if has_cuda else 4,
        enable_vlm_verification=has_cuda
    )

