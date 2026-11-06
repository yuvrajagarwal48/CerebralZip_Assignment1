#!/usr/bin/env python3
"""
================================================================================
Master Pipeline Orchestrator
================================================================================
Executes the complete Zero-Shot Temporal Action Localization benchmark suite:
  1. scripts/run_benchmark_pipeline.py  (Runs pipeline on all videos, outputs results/benchmark_predictions.json & timelines)
  2. scripts/evaluate_quantitative.py   (Target-anchored quantitative evaluation & ablation, outputs metrics JSON & confusion CSV)
  3. scripts/evaluate_qualitative.py    (Deep qualitative error analysis, outputs ASCII timeline markdown & diagnostics JSON)
================================================================================
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.run_benchmark_pipeline as run_benchmark
import scripts.evaluate_quantitative as evaluate_quant
import scripts.evaluate_qualitative as evaluate_qual

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] MasterPipeline: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MasterPipeline")


def run_full_pipeline():
    start_time = time.time()
    logger.info("================================================================================")
    logger.info("     MASTER PIPELINE: ZERO-SHOT TAL BENCHMARK EXECUTION & EVALUATION")
    logger.info("================================================================================")

    # Phase 1: Video Inference & Candidate Generation
    logger.info("\n>>> Phase 1: Video Inference & Multi-Hypothesis Proposal Generation <<<")
    run_benchmark.main()

    # Phase 2: Target-Anchored Quantitative Evaluation & Ablation
    logger.info("\n>>> Phase 2: Target-Anchored Quantitative Evaluation & Ablation <<<")
    evaluate_quant.main()

    # Phase 3: Deep Qualitative Analysis & Report Generation
    logger.info("\n>>> Phase 3: Qualitative Diagnostic Ribbon Analysis <<<")
    evaluate_qual.main()

    total_elapsed = time.time() - start_time
    logger.info("\n================================================================================")
    logger.info(f"ALL PHASES COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s")
    logger.info("All outputs are consolidated inside the 'results/' directory:")
    logger.info("  * results/benchmark_predictions.json")
    logger.info("  * results/quantitative_metrics.json")
    logger.info("  * results/confusion_matrix.csv")
    logger.info("  * results/qualitative_analysis_report.md")
    logger.info("  * results/qualitative_analysis.json")
    logger.info("  * results/timelines_export/")
    logger.info("================================================================================")


def main():
    run_full_pipeline()


if __name__ == "__main__":
    main()
