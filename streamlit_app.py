#!/usr/bin/env python3
"""
streamlit_app.py: Interactive Zero-Shot Temporal Action Localization Web Demo.
Accepts video uploads or local path selection, visualizes the 6-stage
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
    st.title("🎯 TAL Architecture")
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


# Page Title
st.title("🎯 Zero-Shot Temporal Action Localization")
st.caption("Automated video surveillance activity detection, temporal interval bounding, and vision-language verification.")

# Video Selection: NO video preloaded by default
st.markdown("### 1. Select or Upload Video")
input_mode = st.radio(
    "Choose video input method:",
    options=["Upload Video File", "Select from Benchmark Dataset", "Enter Video File Path"],
    horizontal=True
)

video_to_process: Optional[str] = None
video_display_name: str = ""

if input_mode == "Upload Video File":
    uploaded_file = st.file_uploader(
        "Upload a surveillance video (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload CCTV or surveillance footage to analyze."
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_to_process = tfile.name
        video_display_name = uploaded_file.name

elif input_mode == "Select from Benchmark Dataset":
    if VIDEOS_DIR.is_dir():
        benchmark_files = sorted(list(VIDEOS_DIR.glob("*.mp4")))
        file_options = ["-- None (Select a video) --"] + [vf.name for vf in benchmark_files]
        selected_vid = st.selectbox("Choose a benchmark video:", options=file_options, index=0)
        if selected_vid != "-- None (Select a video) --":
            video_to_process = str(VIDEOS_DIR / selected_vid)
            video_display_name = selected_vid
    else:
        st.info(f"Benchmark videos folder '{VIDEOS_DIR}' not found.")

elif input_mode == "Enter Video File Path":
    entered_path = st.text_input(
        "Path to video file:",
        placeholder="e.g. cerebralzip_test_data/videos/Fighting_Assault_Fighting003_x264.mp4"
    )
    if entered_path.strip():
        p = Path(entered_path.strip())
        if p.is_file():
            video_to_process = str(p)
            video_display_name = p.name
        else:
            st.error(f"File not found: '{entered_path}'")

# If no video has been chosen, show guide and stop
if not video_to_process:
    st.info("👆 Please upload a video file or select a video from the options above to begin.")
else:
    st.markdown("---")
    st.markdown("### 2. Video Preview & Metadata")
    col_vid, col_steps = st.columns([1.2, 1.0])

    with col_vid:
        st.video(video_to_process)
        try:
            meta = VideoMetadata.extract(video_to_process)
            m1, m2, m3 = st.columns(3)
            m1.metric("Duration", f"{meta.duration_sec:.1f}s")
            m2.metric("FPS", f"{meta.fps:.1f}")
            m3.metric("Resolution", f"{meta.resolution[0]}x{meta.resolution[1]}")
        except Exception as e:
            st.caption(f"Could not read metadata: {e}")

    with col_steps:
        st.markdown("#### Pipeline Decision Workflow")
        with st.expander("Stage 1: Temporal Windowing", expanded=True):
            st.write("Extracts sliding 4.0s clips at 1.0s stride with 8-frame uniform sampling.")
        with st.expander("Stage 2: X-CLIP Action Embedding", expanded=False):
            st.write("Computes zero-shot dot-product similarity against pre-encoded text descriptions.")
        with st.expander("Stage 3: Relative Class Evidence", expanded=False):
            st.write(r"Derives competitive margin $R_c(t) = S_c(t) - \max_{j \neq c} S_j(t)$ with Gaussian smoothing ($\sigma=0.5$).")
        with st.expander("Stage 4: LogOIC Contrast & Calibration", expanded=False):
            st.write("Calculates inner/outer context ratio and multi-modal dynamic calibrated confidence.")
        with st.expander("Stage 5: Class-Wise NMS & Grouping", expanded=False):
            st.write("Suppresses overlapping duplicates and clusters cross-class hypotheses.")
        with st.expander("Stage 6: Qwen2-VL Grounded Verification", expanded=False):
            st.write("Screens physical bodily action evidence, pruning false alarms.")

    # Execution Action
    st.markdown("---")
    st.markdown("### 3. Pipeline Execution")
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

            # Run prediction
            result = predict_video(video_to_process)
            elapsed = time.time() - start_time

            progress_bar.progress(100, text=f"Processing Complete in {elapsed:.2f}s!")
            status_box.update(label=f"✅ Video processing complete ({elapsed:.2f}s)", state="complete", expanded=False)

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
    st.markdown("### 4. Detection & Localization Results")

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

        # Plot Activation Timeline Curves if CSV available
        vid_stem = Path(video_to_process).stem
        timeline_csv = TIMELINES_DIR / f"{vid_stem}_timelines.csv"
        if timeline_csv.is_file():
            st.markdown("#### Temporal Activation & Margin Curves")
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
        st.markdown("### 5. Clipped Video Segments of Detected Activities")
        st.caption("Individual video segments extracted according to localized temporal boundaries:")

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

                extracted = extract_video_clip(
                    src_video_path=video_to_process,
                    start_sec=s_t,
                    end_sec=e_t,
                    output_path=clip_path
                )

                if extracted and os.path.exists(clip_path):
                    st.video(clip_path)
                    st.caption(f"⏱️ Interval: **[{s_t:.1f}s - {e_t:.1f}s]** | Conf: **{conf_val:.3f}** | VLM: **{vlm_dec}**")
                    if act.get("vlm_evidence"):
                        st.caption(f"📝 *{act.get('vlm_evidence')}*")

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
