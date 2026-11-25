#!/usr/bin/env python3
"""Export a completed Self-Forcing DCP student and adapter into a portable bundle."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handsonworld.distillation.export import export_student  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--control-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    print(export_student(args.checkpoint, args.control_checkpoint, args.output, dtype=args.dtype))


if __name__ == "__main__":
    main()
