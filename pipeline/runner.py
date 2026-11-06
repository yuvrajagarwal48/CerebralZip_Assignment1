#!/usr/bin/env python3
"""
pipeline/runner.py: Master Zero-Shot Temporal Action Localization Pipeline Runner.
Coordinates metadata extraction, temporal windowing, X-CLIP inference,
relative margin calculation, Gaussian temporal smoothing, candidate interval proposal,
class-wise NMS, multi-hypothesis cross-class grouping, Qwen2-VL verification,
and diagnostic timeline export.
"""

import gc
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PipelineConfig, get_default_config, TIMELINES_DIR

try:
    from .metadata import VideoMetadata, ActivityInterval, EventProposal
    from .chunking import TemporalWindow, TemporalChunker
    from .inference import XCLIPInferenceEngine
    from .scoring import compute_relative_margins, smooth_timelines, generate_candidate_intervals
    from .nms import apply_class_wise_nms
    from .grouping import group_cross_class_candidates
    from .vlm import QwenVLVerifier
except ImportError:
    from pipeline.metadata import VideoMetadata, ActivityInterval, EventProposal
    from pipeline.chunking import TemporalWindow, TemporalChunker
    from pipeline.inference import XCLIPInferenceEngine
    from pipeline.scoring import compute_relative_margins, smooth_timelines, generate_candidate_intervals
    from pipeline.nms import apply_class_wise_nms
    from pipeline.grouping import group_cross_class_candidates
    from pipeline.vlm import QwenVLVerifier

logger = logging.getLogger("ZeroShotPipeline")


class ZeroShotActivityPipeline:
    """End-to-end zero-shot temporal activity localization pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_default_config()
        logger.info(f"Initializing ZeroShotActivityPipeline with backbone: {self.config.model_name}")
        self.inference_engine = XCLIPInferenceEngine(self.config)
        self.vlm_verifier: Optional[QwenVLVerifier] = None
        if self.config.enable_vlm_verification:
            try:
                self.vlm_verifier = QwenVLVerifier(self.config)
            except Exception as e:
                logger.warning(f"Could not load VLM verifier ({e}). Proceeding without VLM verification.")
                self.config.enable_vlm_verification = False

    def predict_video(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Executes full zero-shot temporal action localization pipeline on a single video.

        Args:
            video_path: Path to the input video file (.mp4, .avi, etc.)

        Returns:
            Dict containing video name, duration, and list of detected activity intervals.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found at: '{video_path}'")

        logger.info(f"=== Starting Processing: {video_path.name} ===")

        # 1. Video Metadata Extraction
        metadata = VideoMetadata.extract(video_path)
        logger.info(
            f"Extracted Metadata: '{metadata.video_name}' | Duration: {metadata.duration_sec:.2f}s | "
            f"FPS: {metadata.fps:.2f} | Total Frames: {metadata.total_frames}"
        )

        # 2. Temporal Window Generation
        windows = TemporalChunker.generate_windows(
            metadata=metadata,
            window_duration=self.config.window_duration,
            window_stride=self.config.window_stride,
            frames_per_window=self.config.frames_per_window
        )
        logger.info(f"Generated {len(windows)} temporal windows (duration={self.config.window_duration}s, stride={self.config.window_stride}s).")

        # 3. Scalable Long-Video Processing: Block-based window feature extraction & inference
        block_size = self.config.temporal_block_windows
        num_windows = len(windows)
        sim_matrices: List[np.ndarray] = []

        logger.info(f"Extracting clips and scoring windows in blocks of {block_size}...")
        for b_start in range(0, num_windows, block_size):
            b_end = min(b_start + block_size, num_windows)
            block_windows = windows[b_start:b_end]

            # Extract window clips with frame caching
            block_clips = TemporalChunker.extract_block_clips(str(video_path), block_windows)
            block_sim = self.inference_engine.infer_windows(block_clips)
            sim_matrices.append(block_sim)

            del block_clips
            gc.collect()

        similarity_matrix = np.concatenate(sim_matrices, axis=0) if sim_matrices else np.empty((0, len(self.inference_engine.classes)))

        # 4. Relative Class Evidence: R_c(t) = S_c(t) - max_{j != c} S_j(t)
        relative_margins = compute_relative_margins(similarity_matrix)

        # 5. Gaussian Temporal Smoothing
        smoothed_sim, smoothed_margin = smooth_timelines(
            similarity_matrix=similarity_matrix,
            relative_margins=relative_margins,
            sigma=self.config.smoothing_sigma
        )
        timestamps = np.array([w.center_time for w in windows])

        # 6. Candidate Interval Generation with LogOIC & Calibrated Confidence
        candidates = generate_candidate_intervals(
            classes=self.inference_engine.classes,
            windows=windows,
            smoothed_sim=smoothed_sim,
            smoothed_margin=smoothed_margin,
            config=self.config
        )
        logger.info(f"Generated {len(candidates)} candidate activity intervals across all classes.")

        # 7. Class-Wise NMS
        candidates = apply_class_wise_nms(candidates, self.config.nms_iou_threshold)
        logger.info(f"Retained {len(candidates)} candidates after class-wise NMS (IoU threshold={self.config.nms_iou_threshold}).")

        # 8. Cross-Class Overlap Grouping into Event Proposals
        proposals = group_cross_class_candidates(
            candidates=candidates,
            iou_threshold=self.config.cross_class_iou_threshold,
            overlap_ratio=self.config.cross_class_overlap_ratio,
            enable_grouping=self.config.enable_cross_class_grouping
        )
        logger.info(f"Formed {len(proposals)} multi-hypothesis event proposals.")

        # 9. Qwen2-VL Vision-Language Verification Stage
        if self.config.enable_vlm_verification and self.vlm_verifier is not None:
            logger.info(f"Running Qwen-VL verification on {len(proposals)} event proposals...")
            self.vlm_verifier.verify(
                video_path=str(video_path),
                proposals=proposals,
                fps=metadata.fps,
                total_frames=metadata.total_frames,
                duration=metadata.duration_sec
            )

        # 10. Filter Hypotheses: Keep SUPPORTED, Keep UNCERTAIN (0.85x confidence), Prune REJECTED
        final_activities: List[ActivityInterval] = []
        for prop in proposals:
            for cand in prop.candidates:
                if cand.vlm_decision == "SUPPORTED":
                    final_activities.append(cand)
                elif cand.vlm_decision == "UNCERTAIN":
                    cand.confidence = round(cand.confidence * 0.85, 4)
                    final_activities.append(cand)
                    logger.info(
                        f"Retained UNCERTAIN candidate '{cand.activity_class}' at "
                        f"[{cand.start_time:.1f}s - {cand.end_time:.1f}s] with adjusted confidence {cand.confidence}."
                    )
                elif cand.vlm_decision == "REJECTED":
                    logger.info(
                        f"VLM rejected false-alarm candidate '{cand.activity_class}' at "
                        f"[{cand.start_time:.1f}s - {cand.end_time:.1f}s]. Reason: {cand.vlm_evidence}"
                    )
                else:
                    final_activities.append(cand)

        # 11. Export Diagnostic Timelines and Proposals
        self._export_diagnostic_data(
            video_stem=video_path.stem,
            classes=self.inference_engine.classes,
            timestamps=timestamps,
            similarity_matrix=similarity_matrix,
            smoothed_sim=smoothed_sim,
            relative_margins=relative_margins,
            smoothed_margin=smoothed_margin,
            proposals=proposals
        )

        # 12. Structure Output Payload
        output_payload = {
            "video": metadata.video_name,
            "duration": round(metadata.duration_sec, 2),
            "activities": [a.to_dict() for a in final_activities]
        }

        logger.info(f"=== Completed Processing {video_path.name}: {len(final_activities)} detected activity intervals ===")
        return output_payload

    def _export_diagnostic_data(
        self,
        video_stem: str,
        classes: List[str],
        timestamps: np.ndarray,
        similarity_matrix: np.ndarray,
        smoothed_sim: np.ndarray,
        relative_margins: np.ndarray,
        smoothed_margin: np.ndarray,
        proposals: List[EventProposal]
    ):
        """Exports activation CSV timelines and proposal audit JSON to results/timelines_export/."""
        TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Activation Timelines CSV
        csv_file = TIMELINES_DIR / f"{video_stem}_timelines.csv"
        csv_rows = ["timestamp," + ",".join(f"{c}_raw,{c}_smoothed,{c}_rel_raw,{c}_rel_smoothed" for c in classes)]
        for i, t in enumerate(timestamps):
            vals = [f"{t:.2f}"]
            for c in range(len(classes)):
                vals.extend([
                    f"{similarity_matrix[i, c]:.4f}",
                    f"{smoothed_sim[i, c]:.4f}",
                    f"{relative_margins[i, c]:.4f}",
                    f"{smoothed_margin[i, c]:.4f}"
                ])
            csv_rows.append(",".join(vals))
        csv_file.write_text("\n".join(csv_rows), encoding="utf-8")
        logger.info(f"Exported activation timeline CSV to '{csv_file}'.")

        # 2. Event Proposals Diagnostic JSON
        proposals_file = TIMELINES_DIR / f"{video_stem}_proposals.json"
        diag_data = {
            "proposals": [
                {
                    "proposal_id": p.proposal_id,
                    "start_time": p.start_time,
                    "end_time": p.end_time,
                    "hypotheses": [
                        {
                            "class": c.activity_class,
                            "confidence": round(c.confidence, 4),
                            "ranking_score": round(c.ranking_score, 4),
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
        with open(proposals_file, "w", encoding="utf-8") as f:
            json.dump(diag_data, f, indent=2)
        logger.info(f"Exported event proposals diagnostic JSON to '{proposals_file}'.")


# Pipeline Singleton Instance Cache
_GLOBAL_PIPELINE: Optional[ZeroShotActivityPipeline] = None


def predict_video(
    video_path: Union[str, Path],
    config: Optional[PipelineConfig] = None
) -> Dict[str, Any]:
    """
    Top-level pipeline interface function. Accepts a single video path and returns predictions.
    Re-uses the pipeline backbone to avoid re-loading weights between runs.
    """
    global _GLOBAL_PIPELINE
    if config is not None:
        pipeline = ZeroShotActivityPipeline(config)
    else:
        if _GLOBAL_PIPELINE is None:
            _GLOBAL_PIPELINE = ZeroShotActivityPipeline()
        pipeline = _GLOBAL_PIPELINE

    return pipeline.predict_video(video_path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    if len(sys.argv) > 1:
        target_video = Path(sys.argv[1])
    else:
        # Fallback to first discovered video in dataset
        video_candidates = [
            Path("cerebralzip_test_data/videos/Fighting_Assault_Fighting003_x264.mp4"),
            Path("cerebralzip_test_data/videos/Falling_fall-01-cam0-rgb.mp4")
        ]
        target_video = next((c for c in video_candidates if c.is_file()), None)
        if target_video is None:
            matches = list(Path("cerebralzip_test_data/videos").glob("*.mp4"))
            target_video = matches[0] if matches else None

    if target_video is None or not target_video.is_file():
        logger.error("No valid video file provided or discovered.")
        sys.exit(1)

    logger.info(f"Executing pipeline on single video: '{target_video.resolve()}'")
    predictions = predict_video(target_video)

    print("\n" + "=" * 80)
    print(f"PREDICTION RESULTS FOR: {predictions.get('video')}")
    print(f"Video Duration: {predictions.get('duration')}s")
    print(f"Detected Activities: {len(predictions.get('activities', []))}")
    print("=" * 80)
    for idx, act in enumerate(predictions.get("activities", []), 1):
        print(f"[{idx}] {act['class']:<30} | {act['start_time']:>6.1f}s - {act['end_time']:>6.1f}s "
              f"(dur: {act['end_time'] - act['start_time']:>4.1f}s) | Conf: {act['confidence']:.4f} | VLM: {act.get('vlm_decision', 'N/A')}")
        if act.get("vlm_evidence"):
            print(f"    Evidence: {act['vlm_evidence']}")
    print("=" * 80 + "\n")


