#!/usr/bin/env python3
"""
streamlit_app.py: Robust Zero-Shot Temporal Action Localization Web Demo.
Engineered for reliable execution across local environments, Kaggle, and reverse proxies (localtunnel/ngrok/cloudflared).

Key Features:
- No video preloaded on startup.
- Green highlighted cards for SUPPORTED activities; Orange highlighted cards for UNCERTAIN activities.
- Browser-native H.264 video clipping with FFmpeg and OpenCV fallback.
- 100% resilient UI: Zero reliance on dynamically imported JS chunks (st.metric, st.radio, st.dataframe eliminated).
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
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

# Page Configuration
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
    """
    Extracts a temporal sub-clip between start_sec and end_sec.
    Prioritizes FFmpeg with browser-native H.264 (yuv420p + faststart + zero timestamps)
    for guaranteed HTML5 browser playback, with automatic OpenCV fallback.
    """
    duration = max(0.5, end_sec - start_sec)
    ffmpeg_bin = shutil.which("ffmpeg") or ("/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else None)

    # 1. Primary: FFmpeg (available by default on Kaggle, Colab, and standard Linux)
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", f"{start_sec:.2f}",
            "-i", str(src_video_path),
            "-t", f"{duration:.2f}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            "-an",
            str(output_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
            if res.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 100:
                return True
        except Exception:
            pass

        # Secondary FFmpeg attempt: Stream copy
        cmd_copy = [
            ffmpeg_bin,
            "-y",
            "-ss", f"{start_sec:.2f}",
            "-i", str(src_video_path),
            "-t", f"{duration:.2f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path)
        ]
        try:
            res_c = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if res_c.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 100:
                return True
        except Exception:
            pass

    # 2. Secondary fallback: OpenCV frame-by-frame extraction
    cap = cv2.VideoCapture(str(src_video_path))
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

    out = None
    for fourcc_str in ['avc1', 'H264', 'mp4v']:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if out.isOpened():
                break
        except Exception:
            continue

    if out is None or not out.isOpened():
        cap.release()
        return False

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
    return os.path.isfile(output_path) and os.path.getsize(output_path) > 100


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
st.markdown("### 1. Select Video Source")
input_mode = st.selectbox(
    "Choose video input method:",
    options=[
        "-- Select an input method --",
        "Enter Video File Path (Recommended for Kaggle)",
        "Select from Benchmark Dataset",
        "Upload Video File"
    ],
    index=0
)

video_to_process: Optional[str] = None
video_display_name: str = ""

if input_mode == "Enter Video File Path (Recommended for Kaggle)":
    entered_path = st.text_input(
        "Enter full path to video file on disk:",
        placeholder="e.g. cerebralzip_test_data/videos/Fighting_Assault_Fighting003_x264.mp4 or /kaggle/input/..."
    )
    if entered_path.strip():
        p = Path(entered_path.strip())
        if p.is_file():
            video_to_process = str(p)
            video_display_name = p.name
        else:
            st.error(f"Video file not found at: '{entered_path}'")

elif input_mode == "Select from Benchmark Dataset":
    if VIDEOS_DIR.is_dir():
        benchmark_files = sorted(list(VIDEOS_DIR.glob("*.mp4")))
        file_options = ["-- Select a benchmark video --"] + [vf.name for vf in benchmark_files]
        selected_vid = st.selectbox("Choose benchmark video:", options=file_options, index=0)
        if selected_vid != "-- Select a benchmark video --":
            video_to_process = str(VIDEOS_DIR / selected_vid)
            video_display_name = selected_vid
    else:
        st.info(f"Benchmark videos directory '{VIDEOS_DIR}' not found.")

elif input_mode == "Upload Video File":
    uploaded_file = st.file_uploader(
        "Upload a video file (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload surveillance footage to analyze."
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_to_process = tfile.name
        video_display_name = uploaded_file.name

# If no video is selected, show initial prompt and stop
if not video_to_process:
    st.info("👆 Please select an input method and provide a video above to begin.")
else:
    st.markdown("---")
    st.markdown("### 2. Video Preview & Metadata")
    col_vid, col_steps = st.columns([1.2, 1.0])

    with col_vid:
        st.markdown(f"**Loaded Video:** `{video_display_name}`")
        try:
            with open(video_to_process, "rb") as f_v:
                vid_bytes = f_v.read()
            st.video(vid_bytes, format="video/mp4", start_time=0)
        except Exception:
            st.video(video_to_process, format="video/mp4", start_time=0)

        try:
            meta = VideoMetadata.extract(video_to_process)
            m1, m2, m3 = st.columns(3)
            m1.info(f"⏱️ **Duration**\n\n### {meta.duration_sec:.1f}s")
            m2.info(f"🎞️ **FPS**\n\n### {meta.fps:.1f}")
            m3.info(f"📐 **Resolution**\n\n### {meta.resolution[0]}x{meta.resolution[1]}")
        except Exception as e:
            st.caption(f"Could not read metadata: {e}")

    with col_steps:
        st.markdown("#### Pipeline Decision Workflow")
        st.info("**Stage 1: Temporal Windowing** — Sliding 4.0s clips at 1.0s stride (8 frames/window).")
        st.info("**Stage 2: X-CLIP Action Embedding** — Zero-shot similarity against pre-encoded prompts.")
        st.info(r"**Stage 3: Relative Class Evidence** — Competitive margin $R_c(t) = S_c(t) - \max_{j \neq c} S_j(t)$ with Gaussian smoothing ($\sigma=0.5$).")
        st.info("**Stage 4: LogOIC Contrast & Calibration** — Inner/outer contrast with dynamic range calibration.")
        st.info("**Stage 5: Class-Wise NMS & Grouping** — Suppresses duplicates and clusters multi-hypothesis proposals.")
        st.info("**Stage 6: Qwen2-VL Grounded Verification** — Bodily action screening: :green[**SUPPORTED**] retained at full strength, :orange[**UNCERTAIN**] attenuated (0.85x), REJECTED pruned.")

    # Execution Action
    st.markdown("---")
    st.markdown("### 3. Pipeline Execution")
    if st.button("🚀 Run Temporal Action Localization Pipeline", type="primary", use_container_width=True):
        status_box = st.info("⚙️ Running Zero-Shot Temporal Action Localization Pipeline (X-CLIP + Qwen2-VL)...")

        try:
            start_time = time.time()
            result = predict_video(video_to_process)
            elapsed = time.time() - start_time

            st.session_state["last_result"] = result
            st.session_state["last_video"] = video_to_process
            st.session_state["last_elapsed"] = elapsed

            st.success(f"✅ Video processing complete in {elapsed:.2f}s!")

        except Exception as e:
            st.error(f"❌ Error during pipeline execution: {e}")

# Display Results
if "last_result" in st.session_state and st.session_state.get("last_video") == video_to_process:
    res = st.session_state["last_result"]
    elapsed = st.session_state.get("last_elapsed", 0.0)
    activities = res.get("activities", [])

    st.markdown("---")
    st.markdown("### 4. Detection & Localization Results")

    # Metrics Summary Cards using core st.info (zero chunk dependencies)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.info(f"🎯 **Activities Detected**\n\n### {len(activities)}")
    max_conf = max([a.get("confidence", 0.0) for a in activities]) if activities else 0.0
    col_m2.info(f"⭐ **Max Confidence**\n\n### {max_conf:.3f}")
    col_m3.info(f"⏱️ **Video Duration**\n\n### {res.get('duration', 0.0):.1f}s")
    col_m4.info(f"⚡ **Pipeline Runtime**\n\n### {elapsed:.2f}s")

    if not activities:
        st.info("No abnormal activities exceeded the calibrated detection thresholds for this video clip.")
    else:
        # Structured Results Table using st.table (reliable plain HTML rendering, zero JS chunk dependency)
        table_rows = []
        for idx, a in enumerate(activities, 1):
            vlm_dec = a.get("vlm_decision", "N/A")
            status_label = "🟢 SUPPORTED" if vlm_dec == "SUPPORTED" else ("🟠 UNCERTAIN" if vlm_dec == "UNCERTAIN" else vlm_dec)
            table_rows.append({
                "#": idx,
                "Activity Class": a.get("class", "Unknown"),
                "Start (s)": f"{a.get('start_time', 0.0):.1f}",
                "End (s)": f"{a.get('end_time', 0.0):.1f}",
                "Duration (s)": f"{(a.get('end_time', 0.0) - a.get('start_time', 0.0)):.1f}",
                "Calibrated Conf": f"{a.get('confidence', 0.0):.4f}",
                "VLM Status": status_label,
                "VLM Evidence": a.get("vlm_evidence", "N/A")
            })
        st.table(pd.DataFrame(table_rows))

        # Clipped Video Segment Extraction & Playback
        st.markdown("---")
        st.markdown("### 5. Clipped Video Segments of Detected Activities")
        st.caption("Color-coded event cards: :green[**🟢 GREEN = SUPPORTED**] (Grounded bodily action confirmed) | :orange[**🟠 ORANGE = UNCERTAIN**] (Action inferred/ambiguous)")

        vid_stem = Path(video_to_process).stem
        temp_clips_dir = Path(tempfile.gettempdir()) / "tal_extracted_clips"
        temp_clips_dir.mkdir(parents=True, exist_ok=True)

        # Render 2 clips per row for spacious, clear video playback
        clip_cols = st.columns(min(2, max(1, len(activities))))

        for idx, act in enumerate(activities):
            col_target = clip_cols[idx % len(clip_cols)]
            with col_target:
                s_t = act.get("start_time", 0.0)
                e_t = act.get("end_time", 0.0)
                dur_t = max(0.1, e_t - s_t)
                cls_title = act.get("class", "Activity")
                conf_val = act.get("confidence", 0.0)
                vlm_dec = act.get("vlm_decision", "SUPPORTED")
                vlm_ev = act.get("vlm_evidence", "")
                is_supported = (vlm_dec == "SUPPORTED")

                # Show SUPPORTED in prominent GREEN box, UNCERTAIN in prominent ORANGE box
                if is_supported:
                    st.success(
                        f"**🟢 Event #{idx + 1}: {cls_title} [SUPPORTED]**\n\n"
                        f"⏱️ **Interval:** `[{s_t:.1f}s - {e_t:.1f}s]` (Duration: `{dur_t:.1f}s`) | "
                        f"**Calibrated Conf:** `{conf_val:.3f}`  \n"
                        f"📝 **VLM Evidence:** *{vlm_ev if vlm_ev else 'Bodily action verified'}*"
                    )
                else:
                    st.warning(
                        f"**🟠 Event #{idx + 1}: {cls_title} [UNCERTAIN]**\n\n"
                        f"⏱️ **Interval:** `[{s_t:.1f}s - {e_t:.1f}s]` (Duration: `{dur_t:.1f}s`) | "
                        f"**Calibrated Conf:** `{conf_val:.3f}` (0.85x adjusted)  \n"
                        f"📝 **VLM Evidence:** *{vlm_ev if vlm_ev else 'Bodily action uncertain'}*"
                    )

                clip_path = str(temp_clips_dir / f"clip_{vid_stem}_{idx}_{int(s_t)}_{int(e_t)}.mp4")

                extracted = extract_video_clip(
                    src_video_path=video_to_process,
                    start_sec=s_t,
                    end_sec=e_t,
                    output_path=clip_path
                )

                if extracted and os.path.isfile(clip_path) and os.path.getsize(clip_path) > 100:
                    try:
                        with open(clip_path, "rb") as f_c:
                            c_bytes = f_c.read()
                        st.video(c_bytes, format="video/mp4", start_time=0)
                    except Exception:
                        st.video(clip_path, format="video/mp4", start_time=0)
                else:
                    st.warning(f"Could not extract sub-clip for interval [{s_t:.1f}s - {e_t:.1f}s].")

        # Optional Temporal Activation Curves inside expander
        timeline_csv = TIMELINES_DIR / f"{vid_stem}_timelines.csv"
        if timeline_csv.is_file():
            with st.expander("📈 View Temporal Activation Data", expanded=False):
                try:
                    df_time = pd.read_csv(timeline_csv)
                    st.table(df_time.head(15))
                except Exception as e:
                    st.caption(f"Could not render activation data: {e}")
