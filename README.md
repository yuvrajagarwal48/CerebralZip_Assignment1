# Zero-Shot Temporal Action Localization in Surveillance Videos

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

A modular, zero-shot temporal action localization (TAL) and anomaly detection system for surveillance and CCTV footage. The framework combines video-language contrastive representations (**X-CLIP**) with grounded physical bodily action verification (**Qwen2-VL**), relative class margin estimation, Log-Outer-Inner Contrast (LogOIC), and calibrated dynamic confidence scoring.

---

## Architecture Overview

```text
Input Video (MP4/AVI)
         │
         ▼
[Stage 1: Temporal Windowing]
   • Sliding 4.0s windows at 1.0s stride (8 frames/window)
   • Single-pass cached frame extraction in blocks of 120 windows
         │
         ▼
[Stage 2: X-CLIP Video-Language Scoring]
   • Pre-encoded multi-prompt embeddings across 10 canonical activity classes
   • Cosine similarity dot-product matrix S_c(t)
         │
         ▼
[Stage 3: Relative Class Margin & Gaussian Smoothing]
   • Margin calculation: R_c(t) = S_c(t) - max_{j != c} S_j(t)
   • 1D Gaussian temporal smoothing (sigma = 0.5)
         │
         ▼
[Stage 4: LogOIC Contrast & Dynamic Range Calibration]
   • Inner/Outer context contrast: log((S_in + eps) / (S_out + eps)) + alpha * S_max
   • Dynamic normalization:
       Norm_Abs    = clip(((0.5*S_max + 0.5*S_in) - 0.16) / 0.16, 0.0, 1.0)
       Norm_Margin = clip((R_c + 0.01) / 0.04, 0.0, 1.0)
       Norm_LogOIC = sigmoid(LogOIC)
       Conf        = 0.45 * Norm_Abs + 0.35 * Norm_Margin + 0.20 * Norm_LogOIC
         │
         ▼
[Stage 5: Class-Wise NMS & Multi-Hypothesis Grouping]
   • Greedy 1D temporal NMS (IoU threshold = 0.3)
   • Cross-class proposal clustering for overlapping multi-class events
         │
         ▼
[Stage 6: Qwen2-VL Grounded Action Verification]
   • 4 FPS sampling from candidate interval (+/- 1.0s context)
   • Bodily physical movement screening (SUPPORTED / UNCERTAIN / REJECTED)
   • UNCERTAIN attenuated by 0.85x; REJECTED false alarms pruned
         │
         ▼
Outputs: Bounded Intervals, Calibrated Confidence, Timeline CSVs, Video Clips
```

---

## Canonical Activity Taxonomy

The pipeline detects and localizes 10 standard surveillance activity classes:
1. `Fighting/Physical Assault`
2. `Falling`
3. `Fire/Smoke`
4. `Intrusion/Trespassing`
5. `Loitering`
6. `Vandalism`
7. `Theft/Chain Snatching`
8. `Crowd Formation/Unusual Gathering`
9. `Running in a Restricted Zone`
10. `Abandoned Object`

---

## Directory Structure

```text
.
├── config.py                     # Centralized taxonomy, thresholds, and paths
├── prompts.py                    # Multi-prompt physical bodily action descriptions
├── run_pipeline.py               # Master CLI entry point (runs benchmark & evaluations)
├── setup.py                      # Environment initialization and model pre-caching
├── streamlit_app.py              # Interactive web demo (upload, visualize, cut clips)
├── zero_shot_activity_pipeline.py# Original monolithic baseline pipeline (preserved)
├── requirements.txt              # Project dependencies
│
├── pipeline/                     # Modular pipeline package
│   ├── __init__.py               # Exports predict_video, ZeroShotActivityPipeline, etc.
│   ├── runner.py                 # Master pipeline runner & single-video CLI
│   ├── inference.py              # XCLIPInferenceEngine with pre-encoded prompts
│   ├── metadata.py               # VideoMetadata, ActivityInterval, EventProposal
│   ├── chunking.py               # TemporalWindow and single-pass block extractor
│   ├── scoring.py                # Relative margins, Gaussian smoothing, intervals
│   ├── logoic.py                 # FreeZAD-inspired LogOIC temporal contrast
│   ├── calibration.py            # Multi-Modal Dynamic Range Confidence Calibration
│   ├── nms.py                    # 1D temporal IoU & class-wise NMS
│   ├── grouping.py               # Multi-hypothesis cross-class event clustering
│   └── vlm.py                    # Qwen2-VL vision-language verification
│
├── scripts/                      # Evaluation and batch runners
│   ├── run_benchmark_pipeline.py # Batch inference across all benchmark videos
│   ├── evaluate_quantitative.py  # Target-anchored metrics & dual-stream ablation
│   ├── evaluate_qualitative.py   # Qualitative error analysis & ASCII ribbons
│   ├── run_pipeline.py           # Script orchestrator (Phases 1-3)
│   └── setup.py                  # Package installer and HuggingFace cache utility
│
├── results/                      # Centralized output directory
│   ├── benchmark_predictions.json# Consolidated predictions across videos
│   ├── quantitative_metrics.json # Full quantitative evaluation payload
│   ├── confusion_matrix.csv      # 10x10 + background confusion matrix
│   ├── qualitative_analysis_report.md # Markdown diagnostic report with ASCII ribbons
│   ├── qualitative_analysis.json # Structured qualitative failure/success cases
│   └── timelines_export/         # Per-video activation CSVs and proposal audits
│
└── cerebralzip_test_data/        # Benchmark dataset (videos/ and labels/)
```

---

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Pre-cache Models & Verify Environment
Run the automated setup utility to verify CUDA acceleration and pre-download backbone weights:
```bash
python setup.py
```
*Note: If CUDA is detected, models run in FP16 with DataParallel; on CPU, the pipeline operates in FP32 with CPU-safe inference.*

---

## How to Run

### 1. Process a Single Video

#### In Python Code:
```python
from pipeline import predict_video

# Run pipeline on any video file
result = predict_video("cerebralzip_test_data/videos/Falling_fall-01-cam0-rgb.mp4")

print(f"Video: {result['video']} (Duration: {result['duration']}s)")
for act in result["activities"]:
    print(f"-> {act['class']:<25} [{act['start_time']:>5.1f}s - {act['end_time']:>5.1f}s] | Conf: {act['confidence']:.4f}")
```

#### Via Command Line:
```bash
# Pass any custom video path
python pipeline/runner.py cerebralzip_test_data/videos/Falling_fall-01-cam0-rgb.mp4

# Or run with no arguments to process the first discovered benchmark sample
python pipeline/runner.py
```

---

### 2. Interactive Web Application (Streamlit)

Launch the interactive UI to upload videos, inspect the 6-stage decision process, view tabular predictions and activation curves, and automatically cut/download video clips of detected intervals:

```bash
streamlit run streamlit_app.py
```

---

### 3. Full Benchmark & Evaluation Suite

Run the end-to-end benchmark suite (batch inference $\rightarrow$ quantitative evaluation $\rightarrow$ qualitative ASCII diagnostics):

```bash
python run_pipeline.py
```

Or run individual evaluation phases separately:
```bash
# Phase 1: Batch inference across all benchmark videos
python scripts/run_benchmark_pipeline.py

# Phase 2: Target-anchored quantitative evaluation & dual-stream ablation
python scripts/evaluate_quantitative.py

# Phase 3: Qualitative error analysis with ASCII timeline ribbons
python scripts/evaluate_qualitative.py
```

---

## Evaluation Metrics & Reporting

The evaluation scripts report:
1. **Target-Anchored Accuracy & Precision**: Evaluates primary incident localization while distinguishing genuine false alarms from precursor/co-occurring actions (e.g., vandalism or trespassing preceding an arson incident).
2. **Dual-Stream Ablation (Pure X-CLIP vs. Full Pipeline [+VLM])**: Demonstrates the impact of Qwen2-VL verification on eliminating background false alarms.
3. **Temporal Localization**: Strict tIoU@0.5, semantic vicinity hits ($\text{center offset} \le 5.0\text{s}$), and start/end boundary Mean Absolute Error (MAE).
4. **Per-Class Breakdown**: Precision, Recall, and F1 across each active benchmark class.
5. **ASCII Ribbon Timelines**: Visual representations comparing ground-truth intervals with model predictions directly in markdown reports.

All generated reports and metrics are saved to the [`results/`](results/) folder.

