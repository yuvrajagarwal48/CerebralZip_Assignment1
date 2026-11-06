#!/usr/bin/env python3
"""
scripts/config.py: Re-exports centralized global configuration from config.py.
"""
import sys
from pathlib import Path

# Ensure root workspace is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import *
from prompts import ACTIVITY_PROMPTS
