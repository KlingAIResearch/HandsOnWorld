#!/usr/bin/env python3
"""Verify the shipped backend before installing it in the active environment."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def verify_backend(root):
    manifest = json.loads((root / "backend_manifest.json").read_text())
    errors = []
    for relative, info in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing backend file: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            errors.append(f"Modified backend file: {relative}")
    listed = set(manifest["files"])
    for path in root.rglob("*.py"):
        if path.relative_to(root).as_posix() not in listed:
            errors.append(f"Untracked backend source: {path.relative_to(root)}")
    if errors:
        raise ValueError("Backend integrity check failed:\n" + "\n".join(errors))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "third_party" / "DiffSynth-Studio"
    manifest = verify_backend(root)
    print(f"Verified {len(manifest['files'])} backend files; base {manifest['upstream_commit']}")
    if not args.verify_only:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(root)], check=True)


if __name__ == "__main__":
    main()
