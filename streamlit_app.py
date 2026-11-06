#!/usr/bin/env python3
"""
streamlit_app.py: Interactive Zero-Shot Temporal Action Localization Web Demo.
Accepts video uploads or benchmark sample selection, visualizes the 7-stage
pipeline decision workflow, presents calibrated prediction intervals, and
automatically extracts and plays video sub-clips of each detected activity.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import pandas as pd
import streamlit as st

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ALL_CANONICAL_CLASSES,
    VIDEOS_DIR,
    TIMELINES_DIR,
    get_default_config,
    PipelineConfig
)
from pipeline import predict_video, VideoMetadata

# Configure Page
st.set_page_config(
    page_title="Zero-Shot Activity Localization",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .step-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .badge-supported {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-uncertain {
        background-color: #FEF08A;
        color: #713F12;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def extract_video_clip(
    src_video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str
) -> bool:
    """Extracts a temporal segment between start_sec and end_sec into an MP4 file."""
    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(total_frames - 1, int(round(end_sec * fps)))

    if start_frame >= end_frame:
        cap.release()
        return False

    # Use mp4v or avc1 codec for wide browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur_frame = start_frame
    while cap.isOpened() and cur_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cur_frame += 1

    cap.release()
    out.release()
    return True


# Sidebar Configuration & Specifications
with st.sidebar:
    st.image("https://img.icons8.com/color/96/cctv.png", width=64)
    st.title("TAL Architecture")
    st.markdown("**Dual-Stream Zero-Shot Temporal Action Localization**")

    st.markdown("---")
    st.subheader("Model Specifications")
    st.markdown(r"""
    - **Vision Backbone**: `microsoft/xclip-base-patch32`
    - **VLM Verifier**: `Qwen/Qwen2-VL-2B-Instruct`
    - **Window Duration**: 4.0 seconds (8 frames)
    - **Window Stride**: 1.0 second
    - **Temporal Smoothing**: Gaussian ($\sigma=0.5$)
    - **Contrast Metric**: LogOIC with Context Ratio 0.25
    - **Confidence Calibration**: Multi-Modal Dynamic Range
    """)

    st.markdown("---")
    st.subheader("10 Canonical Activities")
    for cls_name in ALL_CANONICAL_CLASSES:
        st.markdown(f"• `{cls_name}`")


# Header
st.markdown('<div class="main-title">🎯 Zero-Shot Temporal Action Localization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated video surveillance activity detection, temporal interval bounding, and vision-language verification.</div>', unsafe_allow_html=True)

# Input Mode Tabs: Upload vs. Benchmark Dataset
tab_upload, tab_benchmark = st.tabs(["📤 Upload Surveillance Video", "📂 Select from Benchmark Suite"])

video_to_process: Optional[str] = None
video_display_name: str = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a video file (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload surveillance or CCTV footage for automated zero-shot activity detection."
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_to_process = tfile.name
        video_display_name = uploaded_file.name

with tab_benchmark:
    if VIDEOS_DIR.is_dir():
        benchmark_files = sorted(list(VIDEOS_DIR.glob("*.mp4")))
        if benchmark_files:
            selected_vid = st.selectbox(
                "Choose a benchmark video:",
                options=[vf.name for vf in benchmark_files]
            )
            if selected_vid and video_to_process is None:
                video_to_process = str(VIDEOS_DIR / selected_vid)
                video_display_name = selected_vid
        else:
            st.info("No benchmark videos found in 'cerebralzip_test_data/videos/'.")
    else:
        st.info("Benchmark videos folder not detected.")

# Display Video and Pipeline Preview
if video_to_process is not None:
    st.markdown("---")
    col_vid, col_steps = st.columns([1.2, 1.0])

    with col_vid:
        st.subheader("Input Video Preview")
        st.video(video_to_process)

        # Video Metadata Preview
        try:
            meta = VideoMetadata.extract(video_to_process)
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Duration", f"{meta.duration_sec:.1f}s")
            m_col2.metric("FPS", f"{meta.fps:.1f}")
            m_col3.metric("Resolution", f"{meta.resolution[0]}x{meta.resolution[1]}")
        except Exception:
            pass

    with col_steps:
        st.subheader("Pipeline Decision Workflow")
        st.markdown(r"""
        <div class="step-card">
            <strong>Stage 1: Temporal Windowing</strong><br>
            Extracts sliding 4.0s clips at 1.0s stride with 8-frame uniform sampling.
        </div>
        <div class="step-card">
            <strong>Stage 2: X-CLIP Action Embedding</strong><br>
            Computes zero-shot dot-product similarity against pre-encoded text descriptions.
        </div>
        <div class="step-card">
            <strong>Stage 3: Relative Class Evidence</strong><br>
            Derives competitive margin \(R_c(t) = S_c(t) - \max_{j \neq c} S_j(t)\) with Gaussian temporal smoothing.
        </div>
        <div class="step-card">
            <strong>Stage 4: LogOIC Contrast & Calibration</strong><br>
            Calculates inner/outer context ratio and multi-modal dynamic calibrated confidence.
        </div>
        <div class="step-card">
            <strong>Stage 5: Class-Wise NMS & Grouping</strong><br>
            Suppresses overlapping duplicates and clusters cross-class hypotheses.
        </div>
        <div class="step-card">
            <strong>Stage 6: Qwen2-VL Grounded Verification</strong><br>
            Screens physical bodily action evidence, pruning false alarms.
        </div>
        """, unsafe_allow_html=True)

    # Execution Action
    st.markdown("---")
    if st.button("🚀 Run Temporal Action Localization Pipeline", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing Pipeline & Pre-encoding Text Descriptions...")

        status_box = st.status("Running Zero-Shot Detection Pipeline...", expanded=True)
        try:
            start_time = time.time()
            status_box.write("⚙️ Stage 1 & 2: Chunking video and extracting X-CLIP action embeddings...")
            progress_bar.progress(25, text="Extracting X-CLIP Action Embeddings...")

            status_box.write("📊 Stage 3 & 4: Calculating relative margins, Gaussian smoothing, and LogOIC contrast...")
            progress_bar.progress(50, text="Evaluating LogOIC & Calibrating Dynamic Confidence...")

            status_box.write("🧠 Stage 5 & 6: Applying Class-Wise NMS and executing Qwen-VL action verification...")
            progress_bar.progress(75, text="Running Qwen2-VL Grounded Verification...")

            # Call Master Pipeline
            result = predict_video(video_to_process)
            elapsed = time.time() - start_time

            progress_bar.progress(100, text=f"Processing Complete in {elapsed:.2f}s!")
            status_box.update(label=f"✅ Video processing complete ({elapsed:.2f}s)", state="complete", expanded=False)

            # Store in session state for persistence
            st.session_state["last_result"] = result
            st.session_state["last_video"] = video_to_process
            st.session_state["last_elapsed"] = elapsed

        except Exception as e:
            status_box.update(label="❌ Pipeline execution error", state="error")
            st.error(f"Error during pipeline execution: {e}")

# Display Results
if "last_result" in st.session_state and st.session_state.get("last_video") == video_to_process:
    res = st.session_state["last_result"]
    elapsed = st.session_state.get("last_elapsed", 0.0)
    activities = res.get("activities", [])

    st.markdown("---")
    st.header("📊 Detection & Localization Results")

    # Metrics Summary Cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Activities Detected", len(activities))
    max_conf = max([a.get("confidence", 0.0) for a in activities]) if activities else 0.0
    col_m2.metric("Max Calibrated Confidence", f"{max_conf:.3f}")
    col_m3.metric("Video Duration", f"{res.get('duration', 0.0):.1f}s")
    col_m4.metric("Pipeline Runtime", f"{elapsed:.2f}s")

    if not activities:
        st.info("No abnormal activities exceeded the calibrated detection thresholds for this video clip.")
    else:
        # Structured Results Table
        table_rows = []
        for a in activities:
            table_rows.append({
                "Activity Class": a.get("class", "Unknown"),
                "Start Time (s)": f"{a.get('start_time', 0.0):.1f}",
                "End Time (s)": f"{a.get('end_time', 0.0):.1f}",
                "Duration (s)": f"{(a.get('end_time', 0.0) - a.get('start_time', 0.0)):.1f}",
                "Confidence": f"{a.get('confidence', 0.0):.4f}",
                "VLM Status": a.get("vlm_decision", "N/A"),
                "VLM Bodily Action Evidence": a.get("vlm_evidence", "N/A")
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # Plot Activation Timeline Curves if available
        vid_stem = Path(video_to_process).stem
        timeline_csv = TIMELINES_DIR / f"{vid_stem}_timelines.csv"
        if timeline_csv.is_file():
            st.subheader("📈 Temporal Activation & Margin Curves")
            try:
                df_time = pd.read_csv(timeline_csv)
                smooth_cols = [c for c in df_time.columns if c.endswith("_smoothed")]
                if smooth_cols:
                    chart_df = df_time[["timestamp"] + smooth_cols].rename(
                        columns={c: c.replace("_smoothed", "") for c in smooth_cols}
                    ).set_index("timestamp")
                    st.line_chart(chart_df)
            except Exception as e:
                st.caption(f"Could not render activation curves: {e}")

        # Clipped Video Segment Extraction & Playback
        st.markdown("---")
        st.subheader("🎬 Clipped Video Segments of Detected Activities")
        st.markdown("Individual video segments extracted according to localized temporal boundaries:")

        clip_cols = st.columns(min(3, max(1, len(activities))))
        temp_clips_dir = Path(tempfile.gettempdir()) / "tal_extracted_clips"
        temp_clips_dir.mkdir(parents=True, exist_ok=True)

        for idx, act in enumerate(activities):
            col_target = clip_cols[idx % len(clip_cols)]
            with col_target:
                s_t = act.get("start_time", 0.0)
                e_t = act.get("end_time", 0.0)
                cls_title = act.get("class", "Activity")
                conf_val = act.get("confidence", 0.0)
                vlm_dec = act.get("vlm_decision", "SUPPORTED")

                st.markdown(f"**Event #{idx + 1}: {cls_title}**")
                clip_path = str(temp_clips_dir / f"clip_{vid_stem}_{idx}_{int(s_t)}_{int(e_t)}.mp4")

                # Extract sub-clip
                extracted = extract_video_clip(
                    src_video_path=video_to_process,
                    start_sec=s_t,
                    end_sec=e_t,
                    output_path=clip_path
                )

                if extracted and os.path.exists(clip_path):
                    st.video(clip_path)
                    st.caption(f"⏱️ Interval: **[{s_t:.1f}s - {e_t:.1f}s]** | Conf: **{conf_val:.3f}**")
                    if vlm_dec == "SUPPORTED":
                        st.markdown('<span class="badge-supported">✓ VLM SUPPORTED</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge-uncertain">~ VLM {vlm_dec}</span>', unsafe_allow_html=True)
                    if act.get("vlm_evidence"):
                        st.caption(f"📝 *{act.get('vlm_evidence')}*")

                    # Download button
                    with open(clip_path, "rb") as f_clip:
                        st.download_button(
                            label=f"⬇️ Download Event #{idx + 1} Clip",
                            data=f_clip,
                            file_name=f"event_{idx + 1}_{cls_title.replace('/', '_')}_{int(s_t)}s_{int(e_t)}s.mp4",
                            mime="video/mp4",
                            key=f"dl_clip_{idx}"
                        )
                else:
                    st.warning(f"Could not extract sub-clip for interval [{s_t:.1f}s - {e_t:.1f}s].")
