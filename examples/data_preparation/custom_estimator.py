"""Template for connecting a hand estimator to ``prepare_videos.py``.

Replace the two ``NotImplementedError`` methods with calls to your estimator.
HandsOnWorld handles protagonist filtering, overlap selection, output validation,
and training CSV export. No intermediate NPZ conversion is required.

Usage
-----
Copy this adapter into an importable module, for example ``my_estimator.py``.
Install it in the environment used by --python, or put its directory on
PYTHONPATH. Then run from the HandsOnWorld repository:

    python scripts/prepare_videos.py --input-csv clips.csv --output-dir data/custom
        --estimator my_estimator:CustomEstimator --estimator-config estimator.json
        --device cuda:0

The command above is wrapped for readability; enter it on one line. The optional
estimator.json is a JSON object of constructor arguments, for example
{"checkpoint": "/path/to/model"}. ``device`` comes from --device, not the JSON;
``min_num_frames`` is also reserved for the shared pipeline.

Built-in HaWoR adapter
----------------------
``--estimator hawor --hawor-root /path/to/HaWoR`` uses the adapter shipped with
HandsOnWorld to call an unmodified external HaWoR installation. Install HaWoR
and its model assets following its upstream instructions. You do not need to
implement this example or add any files to HaWoR to use that backend. Use the
contract below only when connecting a different estimator.

Estimate output
---------------
Process the video's native spatial size. Model-internal resizing is fine, but
map its results back before returning. Do not crop or resize to the training
resolution here: training handles that resize and the corresponding intrinsics.
If you resample time, return the processed video's path and align ALL frame
indices and reconstruction outputs to its decoded frames. The default pipeline
requires at least 81 frames.

``EstimatedClip`` contains the video path (prefer absolute), positive integer
num_frames/width/height, its intrinsic matrix, and all candidate hand chunks
BEFORE protagonist or overlap filtering. ``intrinsics`` is finite (3, 3), has
positive fx/fy, zero skew, and bottom row [0, 0, 1]. Optional ``context`` can hold
camera estimates or other backend state for reconstruction.

For each ``HandCandidate``, let N be the number of observations in that chunk:

* chunk_id: nonempty ID including person/track identity; (chunk_id, hand_side)
  must be unique within the clip. hand_side is "left" or "right".
* frames: sorted, unique integer array (N,), with 0 <= index < num_frames.
* wrist_cam / middle_mcp_cam: (N, 3) wrist and middle-finger MCP positions in
  METRES, in OpenCV camera coordinates (+x right, +y down, +z forward).
* confidence: finite (N,) values in [0, 1].
* valid: optional (N,) bool mask. Valid observations must have finite anchors.
  If omitted, validity is inferred from finite anchors. Invalid observations
  count as failures in the filter's sampled-frame denominator.
* projected_area: finite nonnegative scalar, the mean projected hand area in
  pixels, used to rank overlapping chunks on the same side.
* payload: optional private track data needed for reconstruction.

The default filter checks the first frame per chunk. --sampling all_frames
checks all observations and retains the whole chunk only if its pass ratio is
strictly greater than --min-pass-ratio (default 0.9). Shared body fitting uses
one shape per chunk. Same-side overlap selection runs AFTER this filter.

Reconstruction output
---------------------
``reconstruct`` receives only retained candidates. Honor this selection; do not
restore rejected tracks. Use their payloads and clip.context as needed, and
return ``Reconstruction`` with finite arrays aligned to all T=clip.num_frames:

* camera: (T, 4, 4) world-to-camera transforms, with translation in metres,
  proper orthonormal rotations and homogeneous bottom row [0, 0, 0, 1].
* hand_vertices: (T, 1556, 3) world-space vertices in metres, in the SAME world
  frame as camera. Use MANO topology/order: 778 left vertices followed by 778
  right vertices. Fill an entirely absent hand with zeros. Fill temporal gaps
  using the retained tracks rather than reintroducing rejected candidates.

Camera SLAM may run here (as in HaWoR), or its results may be reused from
clip.context. Do not change the clip's video, dimensions, frame count or
intrinsics in reconstruction. HandsOnWorld writes the resulting control arrays,
filter_decisions.json, processing_report.csv and train.csv; the adapter does not
need to implement those exports or handle captions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from handsonworld.data.estimators import (
    EstimatedClip,
    HandCandidate,
    HandEstimator,
    Reconstruction,
)


class CustomEstimator(HandEstimator):
    """Example adapter; rename this class for your estimator package."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        checkpoint: str | None = None,
        **options: Any,
    ) -> None:
        # ``device`` comes from --device. The remaining values can come from
        # the JSON object passed to --estimator-config.
        self.device = device
        self.checkpoint = checkpoint
        self.options = options

    def _estimate_with_external_model(self, video: Path, work_dir: Path) -> dict[str, Any]:
        """Run detection/tracking and return the fields consumed below."""
        raise NotImplementedError("connect your detector and hand estimator here")

    def _reconstruct_with_external_model(
        self,
        clip: EstimatedClip,
        selected_backend_tracks: list[Any],
    ) -> dict[str, np.ndarray]:
        """Reconstruct camera and dense hands for the selected tracks only."""
        raise NotImplementedError("connect camera and dense-hand reconstruction here")

    def estimate(self, video: Path, work_dir: Path) -> EstimatedClip:
        result = self._estimate_with_external_model(video, work_dir)

        # Map every candidate track before filtering. Four details matter:
        # 1. frame indices refer to ``result["video"]`` and its native size;
        # 2. wrist/MCP anchors are OpenCV camera-space XYZ in metres;
        # 3. chunk_id includes person/track identity and is unique per side;
        # 4. payload keeps the private track object needed by reconstruction.
        candidates = [
            HandCandidate(
                chunk_id=str(track["chunk_id"]),
                hand_side=str(track["hand_side"]),
                frames=np.asarray(track["frames"], dtype=np.int64),
                wrist_cam=np.asarray(track["wrist_cam"], dtype=np.float32),
                middle_mcp_cam=np.asarray(track["middle_mcp_cam"], dtype=np.float32),
                confidence=np.asarray(track["confidence"], dtype=np.float32),
                projected_area=float(track["projected_area"]),
                valid=(
                    None
                    if track.get("valid") is None
                    else np.asarray(track["valid"], dtype=bool)
                ),
                payload=track,
            )
            for track in result["tracks"]
        ]
        return EstimatedClip(
            video=Path(result.get("video", video)),
            num_frames=int(result["num_frames"]),
            width=int(result["width"]),
            height=int(result["height"]),
            intrinsics=np.asarray(result["intrinsics"], dtype=np.float32),
            candidates=candidates,
            context=result.get("context"),
        )

    def reconstruct(
        self,
        clip: EstimatedClip,
        selected: list[HandCandidate],
    ) -> Reconstruction:
        # HandsOnWorld calls this only after protagonist and overlap filtering.
        # Pass exactly these payloads to the backend; do not restore rejected tracks.
        result = self._reconstruct_with_external_model(
            clip,
            [candidate.payload for candidate in selected],
        )
        return Reconstruction(
            camera=np.asarray(result["camera"], dtype=np.float32),
            hand_vertices=np.asarray(result["hand_vertices"], dtype=np.float32),
        )
