"""Prepare user-captioned videos for HandsOnWorld training.

Estimator execution lives in ``scripts/run_video_preprocessing.py``.
This module owns the stable boundary around that runner: input normalization,
artifact validation, processing reports, and the final training manifest.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .manifest import write_manifest


BACKEND_FIELDS = (
    "input_video",
    "video",
    "prompt",
    "cam_pose",
    "hand_pose",
    "camera_intrinsics",
    "width",
    "height",
    "num_frames",
    "status",
    "message",
)
REPORT_FIELDS = ("input_row", *BACKEND_FIELDS)


@dataclass(frozen=True)
class VideoInput:
    input_row: int
    video: Path
    prompt: str


@dataclass(frozen=True)
class CollectionSummary:
    successes: int
    rejected: int
    errors: int


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_video_inputs(path: str | Path, *, require_files: bool = True) -> list[VideoInput]:
    """Load ``video,prompt`` rows and resolve video paths relative to the CSV."""
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"video", "prompt"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"input CSV is missing columns: {sorted(missing)}")
        result = []
        seen: set[Path] = set()
        for line, row in enumerate(reader, start=2):
            raw_video = (row.get("video") or "").strip()
            prompt = row.get("prompt") or ""
            if not raw_video:
                raise ValueError(f"missing video at input line {line}")
            if not prompt.strip():
                raise ValueError(f"missing prompt at input line {line}")
            video = _resolve(csv_path.parent, raw_video)
            if require_files and not video.is_file():
                raise FileNotFoundError(f"video at input line {line} does not exist: {video}")
            if video in seen:
                raise ValueError(f"duplicate video at input line {line}: {video}")
            seen.add(video)
            result.append(VideoInput(line, video, prompt))
    if not result:
        raise ValueError("input CSV contains no video rows")
    return result


def write_normalized_input(path: str | Path, rows: Iterable[VideoInput]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video", "prompt"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"video": str(row.video), "prompt": row.prompt})


def build_preprocessing_command(
    *,
    python: str,
    runner: Path,
    input_csv: Path,
    output_dir: Path,
    hawor_root: Path | None,
    sampling: str,
    min_pass_ratio: float,
    smpl_model_path: Path,
    vposer_path: Path,
    device: str,
    fit_iterations: int = 500,
    fit_learning_rate: float = 0.05,
    fill_mode: str = "interpolation",
    estimator: str = "hawor",
    estimator_config: Path | None = None,
) -> list[str]:
    """Build the shared estimation/filtering runner command.

    The paper setting fixes shared shape parameters per chunk and uses a strict
    ``pass_ratio > min_pass_ratio`` decision in the public filter.
    """
    if sampling not in {"chunk_first", "all_frames"}:
        raise ValueError("sampling must be 'chunk_first' or 'all_frames'")
    if not math.isfinite(min_pass_ratio) or not 0 <= min_pass_ratio <= 1:
        raise ValueError("min_pass_ratio must be finite and between 0 and 1")
    if fit_iterations < 0:
        raise ValueError("fit_iterations must be non-negative")
    if not math.isfinite(fit_learning_rate) or fit_learning_rate <= 0:
        raise ValueError("fit_learning_rate must be finite and positive")
    if fill_mode not in {"interpolation", "network"}:
        raise ValueError("fill_mode must be 'interpolation' or 'network'")
    if estimator == "hawor" and hawor_root is None and estimator_config is None:
        raise ValueError("--hawor-root is required for the HaWoR estimator")
    command = [
        python,
        str(runner),
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--sampling",
        sampling,
        "--min-pass-ratio",
        str(min_pass_ratio),
        "--beta-mode",
        "chunk_shared",
        "--smpl-model-path",
        str(smpl_model_path),
        "--vposer-path",
        str(vposer_path),
        "--device",
        device,
        "--fit-iterations",
        str(fit_iterations),
        "--fit-learning-rate",
        str(fit_learning_rate),
        "--fill-mode",
        fill_mode,
        "--estimator", estimator,
    ]
    if hawor_root is not None:
        command.extend(["--hawor-root", str(hawor_root)])
    if estimator_config is not None:
        command.extend(["--estimator-config", str(estimator_config)])
    return command


def build_hawor_command(**kwargs) -> list[str]:
    """Compatibility alias for the default HaWoR estimator."""
    return build_preprocessing_command(**kwargs)


def _load_backend_rows(path: Path) -> dict[Path, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(BACKEND_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"backend manifest is missing columns: {sorted(missing)}")
        result: dict[Path, dict[str, str]] = {}
        for line, row in enumerate(reader, start=2):
            raw_video = (row.get("input_video") or "").strip()
            if not raw_video:
                raise ValueError(f"backend manifest has an empty input_video at line {line}")
            video = _resolve(path.parent, raw_video)
            if video in result:
                raise ValueError(f"backend manifest contains duplicate video: {video}")
            result[video] = dict(row)
    return result


def _artifact(path: Path, value: str, field: str) -> Path:
    if not value.strip():
        raise ValueError(f"missing {field}")
    result = _resolve(path.parent, value.strip())
    if not result.is_file():
        raise FileNotFoundError(f"missing {field} artifact: {result}")
    return result


def _positive_int(value: str, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def probe_video(path: Path) -> tuple[int, int, int]:
    """Return decoded ``(frames, width, height)`` using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate aligned output videos")
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, subprocess.CalledProcessError) as error:
        raise ValueError(f"could not decode aligned output video: {path}") from error
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"video must contain exactly one video stream: {path}")
    stream = streams[0]
    raw_frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_frames in (None, "N/A"):
        raise ValueError(f"ffprobe could not determine decoded frame count: {path}")
    return (
        _positive_int(str(raw_frames), "decoded video frames"),
        _positive_int(str(stream.get("width", "")), "decoded video width"),
        _positive_int(str(stream.get("height", "")), "decoded video height"),
    )


def validate_backend_row(manifest: Path, row: dict[str, str]) -> dict[str, object]:
    """Validate one runner result and return canonical absolute artifact paths."""
    cam_path = _artifact(manifest, row.get("cam_pose", ""), "cam_pose")
    hand_path = _artifact(manifest, row.get("hand_pose", ""), "hand_pose")
    video_path = _artifact(manifest, row.get("video", ""), "video")
    intrinsics_path = _artifact(
        manifest, row.get("camera_intrinsics", ""), "camera_intrinsics"
    )
    cam = np.load(cam_path, mmap_mode="r", allow_pickle=False)
    hand = np.load(hand_path, mmap_mode="r", allow_pickle=False)
    intrinsics = np.load(intrinsics_path, mmap_mode="r", allow_pickle=False)
    if cam.ndim != 3 or cam.shape[1:] != (4, 4):
        raise ValueError(f"cam_pose must have shape (T, 4, 4), got {cam.shape}")
    if hand.ndim != 3 or hand.shape[1:] != (1556, 3):
        raise ValueError(f"hand_pose must have shape (T, 1556, 3), got {hand.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"camera_intrinsics must have shape (3, 3), got {intrinsics.shape}")
    num_frames = _positive_int(row.get("num_frames", ""), "num_frames")
    if len(cam) != num_frames or len(hand) != num_frames:
        raise ValueError(
            "artifact length mismatch: "
            f"num_frames={num_frames}, cam_pose={len(cam)}, hand_pose={len(hand)}"
        )
    # Share the same contract for live estimators and reused manifests. Import
    # lazily because the processor also uses probe_video from this module.
    from .estimators import Reconstruction, validate_intrinsics, validate_reconstruction

    validate_reconstruction(Reconstruction(cam, hand), num_frames)
    validate_intrinsics(intrinsics)
    width = _positive_int(row.get("width", ""), "width")
    height = _positive_int(row.get("height", ""), "height")
    video_frames, video_width, video_height = probe_video(video_path)
    if (video_frames, video_width, video_height) != (num_frames, width, height):
        raise ValueError(
            "aligned video metadata mismatch: "
            f"expected frames/width/height={(num_frames, width, height)}, "
            f"decoded={(video_frames, video_width, video_height)}"
        )
    return {
        "video": str(video_path),
        "cam_pose": str(cam_path),
        "hand_pose": str(hand_path),
        "camera_intrinsics": str(intrinsics_path),
        "width": width,
        "height": height,
        "num_frames": num_frames,
    }


def _write_report(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REPORT_FIELDS})


def collect_backend_results(
    inputs: Sequence[VideoInput],
    backend_manifest: str | Path,
    *,
    train_csv: str | Path,
    report_csv: str | Path,
) -> CollectionSummary:
    """Collect validated successes and record every failure explicitly."""
    manifest = Path(backend_manifest).expanduser().resolve()
    backend = _load_backend_rows(manifest)
    training_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    rejected = 0
    errors = 0
    for item in inputs:
        raw = backend.pop(item.video, None)
        report: dict[str, object] = {
            "input_row": item.input_row,
            "input_video": str(item.video),
            "prompt": item.prompt,
        }
        if raw is None:
            report.update(status="missing", message="runner produced no result for this input")
            report_rows.append(report)
            errors += 1
            continue
        status = (raw.get("status") or "").strip().lower()
        if status != "success":
            report.update(status=status or "failed", message=raw.get("message", ""))
            report_rows.append(report)
            if status in {"filtered", "rejected"}:
                rejected += 1
            else:
                errors += 1
            continue
        try:
            artifacts = validate_backend_row(manifest, raw)
        except (FileNotFoundError, OSError, ValueError) as error:
            report.update(status="failed", message=str(error))
            report_rows.append(report)
            errors += 1
            continue
        report.update(artifacts, status="success", message=raw.get("message", ""))
        report_rows.append(report)
        training_rows.append({
            "video": artifacts["video"],
            "prompt": item.prompt,
            "cam_pose": artifacts["cam_pose"],
            "hand_pose": artifacts["hand_pose"],
            "camera_intrinsics": artifacts["camera_intrinsics"],
            "width": artifacts["width"],
            "height": artifacts["height"],
            "num_frames": artifacts["num_frames"],
            "allow_random_start_frame": 0,
            "source": "custom",
            "status": "success",
        })
    if backend:
        extras = ", ".join(str(path) for path in sorted(backend, key=str)[:3])
        raise ValueError(f"backend manifest contains videos absent from the input CSV: {extras}")
    write_manifest(Path(train_csv), training_rows)
    _write_report(Path(report_csv), report_rows)
    return CollectionSummary(len(training_rows), rejected, errors)


def run_hawor(command: Sequence[str]) -> int:
    return subprocess.run(list(command), check=False).returncode


def execute_and_collect(
    inputs: Sequence[VideoInput],
    backend_manifest: Path,
    *,
    command: Sequence[str] | None,
    train_csv: Path,
    report_csv: Path,
) -> tuple[int, CollectionSummary]:
    """Run the backend, retaining a partial manifest when the backend exits nonzero."""
    previous = (
        (backend_manifest.stat().st_mtime_ns, backend_manifest.stat().st_size)
        if command is not None and backend_manifest.is_file()
        else None
    )
    backend_status = run_hawor(command) if command is not None else 0
    if backend_status != 0:
        current = (
            (backend_manifest.stat().st_mtime_ns, backend_manifest.stat().st_size)
            if backend_manifest.is_file()
            else None
        )
        if current is None or current == previous:
            raise subprocess.CalledProcessError(backend_status, list(command or ()))
    summary = collect_backend_results(
        inputs, backend_manifest, train_csv=train_csv, report_csv=report_csv
    )
    return backend_status, summary


def default_runner(repo_root: Path) -> Path:
    return repo_root / "scripts" / "run_video_preprocessing.py"


def default_python() -> str:
    return sys.executable
