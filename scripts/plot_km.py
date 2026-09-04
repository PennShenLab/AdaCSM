#!/usr/bin/env python3
"""Compatibility entrypoint for KM plotting under scripts/."""

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "plot_km.py"), run_name="__main__")
