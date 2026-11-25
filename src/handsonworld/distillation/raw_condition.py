"""Render raw pose/camera arrays into the shared 48-channel control tensor."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


class RawConditioner:
    def __init__(
        self,
        backend_root: str | Path,
        height: int,
        width: int,
        representation: str,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if representation not in {"normal_plucker", "reflection_plucker"}:
            raise ValueError(f"Unsupported control representation: {representation}")
        backend = Path(backend_root).expanduser().resolve()
        if not (backend / "diffsynth").is_dir():
            raise ValueError(f"backend_root does not contain diffsynth: {backend}")
        if str(backend) not in sys.path:
            sys.path.insert(0, str(backend))
        from diffsynth.pipelines.wan_video import (
            WanVideoUnit_HandPosePluckerControlAdapter,
            WanVideoUnit_Pose2Plucker,
            WanVideoUnit_RebasePoseToClipStart,
        )

        self.rebase = WanVideoUnit_RebasePoseToClipStart()
        self.plucker = WanVideoUnit_Pose2Plucker()
        self.hand = WanVideoUnit_HandPosePluckerControlAdapter()
        self.height, self.width = int(height), int(width)
        self.representation = representation
        self.device, self.dtype = torch.device(device), dtype

    @torch.no_grad()
    def __call__(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cam = data["raw_cam_pose"]
        hand = data["raw_hand_pose"]
        intrinsics = data["raw_camera_intrinsics"]
        original_heights = data.get("raw_original_height")
        original_widths = data.get("raw_original_width")
        if cam.ndim == 3:
            cam, hand, intrinsics = cam.unsqueeze(0), hand.unsqueeze(0), intrinsics.unsqueeze(0)
        if cam.ndim != 4 or hand.ndim != 4 or intrinsics.ndim != 3:
            raise ValueError("Raw controls must be unbatched or batched camera/hand/intrinsics tensors")
        pipe = SimpleNamespace(
            device=self.device,
            torch_dtype=self.dtype,
            dit=SimpleNamespace(control_adapter=True, in_dim_control_adapter=48),
        )
        controls, masks = [], []
        for index in range(cam.shape[0]):
            original_height = (
                int(original_heights.reshape(-1)[index].item()) if original_heights is not None else self.height
            )
            original_width = (
                int(original_widths.reshape(-1)[index].item()) if original_widths is not None else self.width
            )
            rebased = self.rebase.process(pipe, poses=None, cam_pose=cam[index], hand_pose=hand[index])
            rays = self.plucker.process(
                pipe, cam_pose=rebased["cam_pose"], camera_intrinsics=intrinsics[index],
                height=self.height, width=self.width,
                original_height=original_height, original_width=original_width,
            )
            rendered = self.hand.process(
                pipe, poses=None, hand_pose=rebased["hand_pose"], cam_pose=rebased["cam_pose"],
                camera_intrinsics=intrinsics[index], plucker_rays=rays["plucker_rays"],
                control_camera_valid_mask=rays["control_camera_valid_mask"],
                original_height=original_height, original_width=original_width,
                hand_pose_render_mode=self.representation, hand_pose_attr_space="world", y=None,
            )
            controls.append(rendered["control_camera_latents_input"])
            masks.append(rays["control_camera_valid_mask"].reshape(1))
        return torch.cat(controls), torch.cat(masks)


RawAdapterConditioner = RawConditioner
