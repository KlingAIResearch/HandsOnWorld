#!/usr/bin/env python3
"""Estimator subprocess used by prepare_videos.py (also supports separate environments)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handsonworld.data.video_preprocessing import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
