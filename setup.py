#!/usr/bin/env python3
"""
Root setup entry point.
Delegates to scripts/setup.py to install requirements and pre-download models.
"""
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from setup import main

if __name__ == "__main__":
    main()

