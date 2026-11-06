#!/usr/bin/env python3
"""
Root Pipeline Runner.
Executes the master zero-shot activity detection and benchmark evaluation pipeline.
"""
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from run_pipeline import main

if __name__ == "__main__":
    main()

