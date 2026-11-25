"""Write and validate training CSVs produced by dataset preprocessing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


FIELDNAMES = (
    "video", "prompt", "poses", "pose_video", "cam_pose", "hand_pose",
    "camera_intrinsics", "width", "height", "num_frames",
    "allow_random_start_frame", "source", "status", "fps", "frame_stride",
    "skip_initial_frames", "subject", "sequence", "split", "message",
)


def write_manifest(output: Path, rows: Iterable[dict[str, Any]]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
            count += 1
    return count


def validate_manifest(path: Path) -> dict[str, int]:
    counts = {"egovid": 0, "arctic": 0}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDNAMES) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            source = row["source"]
            if source not in {"egovid", "arctic", "custom"}:
                raise ValueError(f"Unknown source {source!r} at line {line}")
            expected_random = "1" if source == "arctic" else "0"
            if row["allow_random_start_frame"] != expected_random:
                raise ValueError(
                    f"{source} allow_random_start_frame must be {expected_random} at line {line}"
                )
            for key in ("video", "cam_pose", "hand_pose"):
                if not row[key]:
                    raise ValueError(f"Missing {key} at line {line}")
            counts[source] = counts.get(source, 0) + 1
    return counts

