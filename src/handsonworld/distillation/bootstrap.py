"""Locate the pinned backends without depending on adjacent repositories."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def repository_root():
    return Path(__file__).resolve().parents[3]


def backend_paths(config):
    root = repository_root()
    return [root / "src", Path(config["backend_root"]), root / "third_party/FastGen"]


def configure_backends(config):
    paths = backend_paths(config)
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ["HAND_MESH_FACES_PATH"] = str(config["hand_mesh_faces_path"])
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
    return paths


def worker_environment(config):
    paths = backend_paths(config)
    existing = os.environ.get("PYTHONPATH", "")
    return dict(os.environ,
                PYTHONPATH=os.pathsep.join([*(str(p) for p in paths), *([existing] if existing else [])]),
                HAND_MESH_FACES_PATH=str(config["hand_mesh_faces_path"]))
