"""HaWoR export adapters without importing HaWoR itself."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..schema import HandObservations


WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9


def _broadcast_strings(value: str | np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(size, str(array.item()))
    if array.shape != (size,):
        raise ValueError(f"{name} must be a scalar or have shape ({size},)")
    return array.astype(str)


def from_camera_joints(
    joints_cam: np.ndarray,
    *,
    clip_id: str | np.ndarray,
    chunk_id: str | np.ndarray,
    frame_index: np.ndarray,
    hand_side: str | np.ndarray,
    valid: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    projected_area: np.ndarray | None = None,
) -> HandObservations:
    """Map HaWoR/MANO camera joints 0 and 9 to the public schema."""
    joints = np.asarray(joints_cam, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1] <= MIDDLE_MCP_INDEX or joints.shape[2] != 3:
        raise ValueError("joints_cam must have shape (N, J, 3) with J >= 10")
    size = len(joints)
    frames = np.asarray(frame_index)
    if frames.shape != (size,):
        raise ValueError(f"frame_index must have shape ({size},)")
    if valid is None:
        valid = np.isfinite(joints[:, (WRIST_INDEX, MIDDLE_MCP_INDEX)]).all(axis=(1, 2))
    if confidence is None:
        confidence = np.ones(size, dtype=np.float32)
    return HandObservations(
        clip_id=_broadcast_strings(clip_id, size, "clip_id"),
        chunk_id=_broadcast_strings(chunk_id, size, "chunk_id"),
        frame_index=frames,
        hand_side=_broadcast_strings(hand_side, size, "hand_side"),
        wrist_cam=joints[:, WRIST_INDEX],
        middle_mcp_cam=joints[:, MIDDLE_MCP_INDEX],
        valid=valid,
        confidence=confidence,
        projected_area=projected_area,
    )


def load_camera_joints_npz(path: str | Path) -> HandObservations:
    """Load a pre-filter candidate export containing ``joints_cam``.

    Required fields are ``joints_cam``, ``coordinate_frame``, ``units``,
    ``clip_id``, ``chunk_id``, ``frame_index``, and ``hand_side``. Optional
    fields are ``valid``, ``confidence``, and ``projected_area``.
    """
    with np.load(path, allow_pickle=False) as data:
        required = (
            "joints_cam", "coordinate_frame", "units", "clip_id", "chunk_id",
            "frame_index", "hand_side",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"missing HaWoR candidate fields: {', '.join(missing)}")
        coordinate_frame = str(np.asarray(data["coordinate_frame"]).item())
        units = str(np.asarray(data["units"]).item())
        if coordinate_frame != "camera" or units != "meters":
            raise ValueError(
                "HaWoR candidates must explicitly declare coordinate_frame='camera' "
                "and units='meters'"
            )
        return from_camera_joints(
            data["joints_cam"],
            clip_id=data["clip_id"],
            chunk_id=data["chunk_id"],
            frame_index=data["frame_index"],
            hand_side=data["hand_side"],
            valid=data["valid"] if "valid" in data else None,
            confidence=data["confidence"] if "confidence" in data else None,
            projected_area=data["projected_area"] if "projected_area" in data else None,
        )


def load_cam_space_json(
    path: str | Path,
    *,
    mano_model_path: str | Path,
    clip_id: str,
    chunk_id: str,
    hand_side: str,
    coordinate_frame: str,
    units: str,
    frame_index: np.ndarray | None = None,
    device: str = "cpu",
) -> HandObservations:
    """Convert one HaWoR ``cam_space`` JSON using a user-provided MANO model.

    HaWoR's normal ``cam_space`` output is written after candidate filtering.
    It must not be treated as a complete candidate set when re-running the
    protagonist filter.
    """
    model_path = Path(mano_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MANO model path does not exist: {model_path}")
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right'")
    if coordinate_frame != "camera" or units != "meters":
        raise ValueError(
            "cam_space conversion requires explicit coordinate_frame='camera' and units='meters'"
        )
    try:
        import torch
        import smplx
    except ImportError as error:  # pragma: no cover
        raise ImportError("cam_space JSON conversion requires torch, smplx, and MANO assets") from error

    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    required = ("init_root_orient", "init_hand_pose", "init_trans", "init_betas")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"missing cam_space fields: {', '.join(missing)}")
    root = torch.as_tensor(values["init_root_orient"], dtype=torch.float32, device=device)
    pose = torch.as_tensor(values["init_hand_pose"], dtype=torch.float32, device=device)
    transl = torch.as_tensor(values["init_trans"], dtype=torch.float32, device=device)
    betas = torch.as_tensor(values["init_betas"], dtype=torch.float32, device=device)
    if root.ndim == 4 and root.shape[0] == 1:
        root = root[0]
    if pose.ndim == 5 and pose.shape[0] == 1:
        pose = pose[0]
    if transl.ndim == 3 and transl.shape[0] == 1:
        transl = transl[0]
    if betas.ndim == 3 and betas.shape[0] == 1:
        betas = betas[0]
    if root.ndim != 3 or root.shape[1:] != (3, 3):
        raise ValueError("init_root_orient must contain rotation matrices with shape (T, 3, 3)")
    if pose.ndim != 4 or pose.shape[1:] != (15, 3, 3):
        raise ValueError("init_hand_pose must have shape (T, 15, 3, 3)")
    size = len(root)
    if transl.shape != (size, 3) or betas.shape != (size, 10):
        raise ValueError("init_trans/init_betas length must match the pose sequence")
    if frame_index is None:
        frame_index = np.arange(size, dtype=np.int64)

    mano = smplx.create(
        model_path=str(model_path),
        model_type="mano",
        is_rhand=hand_side == "right",
        use_pca=False,
    ).to(device)
    with torch.no_grad():
        output = mano(
            global_orient=root,
            hand_pose=pose,
            transl=transl,
            betas=betas,
            pose2rot=False,
            return_verts=False,
        )
    return from_camera_joints(
        output.joints.detach().cpu().numpy(),
        clip_id=clip_id,
        chunk_id=chunk_id,
        frame_index=np.asarray(frame_index),
        hand_side=hand_side,
    )
