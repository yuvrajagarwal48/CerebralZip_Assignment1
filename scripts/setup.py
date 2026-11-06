#!/usr/bin/env python3
"""
================================================================================
Setup & Model Pre-caching Script
================================================================================
Objective:
1. Installs all required packages from 'requirements.txt'.
2. Inspects and logs hardware acceleration environment (CUDA / GPU / CPU).
3. Pre-downloads and caches HuggingFace models:
   - Primary Zero-Shot Action Detector: 'microsoft/xclip-base-patch32'
   - VLM Verification Stage: 'Qwen/Qwen2-VL-2B-Instruct'
4. Runs a verification check to ensure all pipelines are operational.
================================================================================
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Setup: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Setup")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
REQUIREMENTS_FILE = WORKSPACE_DIR / "requirements.txt"


def install_requirements():
    """Installs dependencies from requirements.txt."""
    if not REQUIREMENTS_FILE.is_file():
        logger.warning(f"'{REQUIREMENTS_FILE}' not found. Skipping pip install.")
        return

    logger.info(f"Installing dependencies from '{REQUIREMENTS_FILE.resolve()}'...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
        res = subprocess.run(cmd, check=True)
        logger.info("Successfully verified / installed requirements.txt.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        sys.exit(1)


def check_hardware_environment():
    """Detects CUDA GPUs, PyTorch version, and memory allocation."""
    logger.info("--- Inspecting Hardware & Deep Learning Environment ---")
    try:
        import torch
        logger.info(f"PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"CUDA Available: True ({num_gpus} GPU(s) detected)")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                logger.info(f"  * GPU {i}: {props.name} | Total VRAM: {vram_gb:.2f} GB")
        else:
            logger.info("CUDA Available: False (Running in CPU mode)")
    except ImportError:
        logger.error("PyTorch is not installed.")
        sys.exit(1)


def download_xclip_model(model_name: str = "microsoft/xclip-base-patch32"):
    """Pre-downloads and caches X-CLIP model and processor."""
    logger.info(f"\n--- Pre-downloading & Caching X-CLIP Model ('{model_name}') ---")
    try:
        import torch
        from transformers import AutoProcessor, AutoModel

        logger.info(f"Downloading AutoProcessor for '{model_name}'...")
        processor = AutoProcessor.from_pretrained(model_name)

        logger.info(f"Downloading AutoModel for '{model_name}'...")
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Sanity check: test tokenization
        test_text = ["A person falling onto the ground.", "A person punching another person."]
        inputs = processor(text=test_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        # Robustly extract tensor shape whether get_text_features returns a Tensor or ModelOutput
        if isinstance(text_features, torch.Tensor):
            feat_shape = text_features.shape
        elif hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
            feat_shape = text_features.pooler_output.shape
        elif hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
            feat_shape = text_features.text_embeds.shape
        elif isinstance(text_features, (tuple, list)) and len(text_features) > 0:
            feat_shape = text_features[0].shape
        else:
            feat_shape = "(extracted)"

        logger.info(f"X-CLIP pre-caching verified successfully! Text features shape: {feat_shape}")

    except Exception as e:
        logger.error(f"Error caching X-CLIP model '{model_name}': {e}", exc_info=True)
        sys.exit(1)


def download_qwen_vlm_model(model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    """Pre-downloads and caches Qwen2-VL model and processor."""
    logger.info(f"\n--- Pre-downloading & Caching Qwen2-VL Model ('{model_name}') ---")
    try:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2VLForConditionalGeneration
            model_cls = Qwen2VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq

        logger.info(f"Downloading AutoProcessor for '{model_name}'...")
        processor = AutoProcessor.from_pretrained(model_name)

        logger.info(f"Downloading model weights for '{model_name}'...")
        dtype = torch.float16 if (torch.cuda.is_available()) else torch.float32
        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        model.eval()
        logger.info(f"Qwen2-VL pre-caching verified successfully ({model_name})!")

    except Exception as e:
        logger.warning(
            f"Note: Qwen2-VL pre-caching encountered an issue: {e}. "
            "The primary pipeline can still run zero-shot X-CLIP detection with enable_vlm_verification=False."
        )


def main():
    """Master setup sequence."""
    logger.info("================================================================================")
    logger.info("STARTING ZERO-SHOT TAL ENVIRONMENT SETUP")
    logger.info("================================================================================")

    # 1. Install pip requirements
    install_requirements()

    # 2. Check hardware environment
    check_hardware_environment()

    # 3. Pre-download X-CLIP
    download_xclip_model()

    # 4. Pre-download Qwen2-VL
    download_qwen_vlm_model()

    logger.info("\n================================================================================")
    logger.info("SETUP COMPLETE: Environment is fully initialized and models are cached.")
    logger.info("You can now run: python run_pipeline.py")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()

