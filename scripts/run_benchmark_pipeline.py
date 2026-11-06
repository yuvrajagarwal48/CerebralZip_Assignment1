#!/usr/bin/env python3
"""
scripts/run_benchmark_pipeline.py
Batch inference runner: Discovers all benchmark videos, executes
the 10-class zero-shot temporal action localization pipeline, and exports timelines,
proposal audits, and consolidated predictions into results/.

Config and Prompts are strictly imported from central config.py and prompts.py.
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path so config, prompts, and pipeline are directly importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    VIDEOS_DIR,
    PREDICTIONS_JSON,
    TIMELINES_DIR,
    RESULTS_DIR,
    get_default_config
)
from pipeline import ZeroShotActivityPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BenchmarkRunner")


def main():
    logger.info("================================================================================")
    logger.info("        ZERO-SHOT TEMPORAL ACTION LOCALIZATION BENCHMARK RUNNER")
    logger.info("================================================================================")
    logger.info(f"Videos Directory:     {VIDEOS_DIR}")
    logger.info(f"Timelines Directory:  {TIMELINES_DIR}")
    logger.info(f"Predictions Output:   {PREDICTIONS_JSON}")

    if not VIDEOS_DIR.is_dir():
        logger.error(f"Videos directory '{VIDEOS_DIR}' not found. Please verify dataset location.")
        sys.exit(1)

    video_files = sorted(list(VIDEOS_DIR.glob("*.mp4")))
    if not video_files:
        logger.error(f"No .mp4 video files found in '{VIDEOS_DIR}'.")
        sys.exit(1)

    logger.info(f"Discovered {len(video_files)} benchmark video(s).")

    # Ensure results directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize master pipeline once with hardware-aware config
    config = get_default_config()
    pipeline = ZeroShotActivityPipeline(config=config)

    predictions = {}
    # If partial predictions already exist, allow resuming
    if PREDICTIONS_JSON.is_file():
        try:
            with open(PREDICTIONS_JSON, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            logger.info(f"Loaded existing predictions for {len(predictions)} video(s).")
        except Exception:
            predictions = {}

    for idx, vf in enumerate(video_files, 1):
        logger.info(f"[{idx}/{len(video_files)}] Processing '{vf.name}'...")
        try:
            res = pipeline.predict_video(vf)
            predictions[vf.name] = res["activities"]

            # Save checkpoint after each processed video
            with open(PREDICTIONS_JSON, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"  -> Predicted {len(res['activities'])} activity intervals. Checkpoint saved.")
        except Exception as e:
            logger.error(f"Error processing video '{vf.name}': {e}", exc_info=True)
            predictions[vf.name] = []

    logger.info("================================================================================")
    logger.info(f"Benchmark run complete. Saved predictions for {len(predictions)} video(s) to '{PREDICTIONS_JSON}'.")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()

