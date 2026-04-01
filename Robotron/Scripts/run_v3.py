#!/usr/bin/env python3
"""Launch script for Robotron AI v3.

Run from the Robotron/Scripts directory:
    python3 run_v3.py
"""
import sys
import os

# Ensure the Scripts directory is on the path so 'v3' is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3.main import main
main()
