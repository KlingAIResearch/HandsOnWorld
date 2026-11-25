"""Optional SMPL body fitting backend used by the paper filter.

Importing this module does not import PyTorch, SMPL-X, or VPoser. Those
dependencies are loaded only when a fitter is constructed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


DEFAULT_WEIGHTS = {
    "data_weight": 10.0,
    "body_pose_prior": 5e-3,
    "shape_prior": 3e-2,
    "wrist_prior": 0.1,
    "angle_prior": 0.0,
    "gaze_weight": 1.0,
}


def _torch_dependencies():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover
        raise ImportError("SMPL body fitting requires the optional dependency torch") from error
    return torch, functional


def _optional_dependencies():
    torch, functional = _torch_dependencies()
    try:
        import smplx
        from human_body_prior.tools.model_loader import load_vposer
    except ImportError as error:  # pragma: no cover - exact missing package varies
        raise ImportError(
            "SMPL body fitting requires the optional fitting dependencies: "
            "torch, smplx, and human-body-prior"
        ) from error
    return torch, functional, smplx, load_vposer


class SMPLBodyFitter:
    """Fit a VPoser-parameterized SMPL body to camera-space hand anchors.

    Despite the dependency package being named ``smplx``, this backend creates
    ``model_type='smpl'`` to match the research implementation.
    """

    left_eye_vertex = 2808
    right_eye_vertex = 6270
    left_wrist_joint = 20
    left_middle_mcp_joint = 22
    right_wrist_joint = 21
    right_middle_mcp_joint = 23
    latent_dim = 32

    def __init__(
        self,
        *,
        smpl_model_path: str | Path,
        vposer_path: str | Path,
        gender: str = "neutral",
        device: str = "cpu",
    ) -> None:
        smpl_path = Path(smpl_model_path)
        poser_path = Path(vposer_path)
        if not smpl_path.exists():
            raise FileNotFoundError(f"SMPL model path does not exist: {smpl_path}")
        if not poser_path.exists():
            raise FileNotFoundError(f"VPoser path does not exist: {poser_path}")
        torch, functional, smplx, load_vposer = _optional_dependencies()
        self.torch = torch
        self.functional = functional
        self.device = torch.device(device)
        self.body_model = smplx.create(
            model_path=str(smpl_path),
            model_type="smpl",
            gender=gender,
            use_face_contour=False,
            num_betas=10,
        ).to(self.device)
        self.vposer, _ = load_vposer(str(poser_path), vp_model="snapshot")
        self.vposer = self.vposer.to(self.device)
        self.vposer.eval()
        for model in (self.body_model, self.vposer):
            for parameter in model.parameters():
                parameter.requires_grad = False
        self.faces_tensor = torch.as_tensor(
            self.body_model.faces, dtype=torch.long, device=self.device
        )

    @classmethod
    def from_components(cls, body_model, vposer, faces, *, device: str = "cpu"):
        """Construct from lightweight components for tests or custom loaders."""
        torch, functional = _torch_dependencies()
        instance = cls.__new__(cls)
        instance.torch = torch
        instance.functional = functional
        instance.device = torch.device(device)
        instance.body_model = body_model
        instance.vposer = vposer
        instance.faces_tensor = torch.as_tensor(faces, dtype=torch.long, device=instance.device)
        return instance

    def create_initial_params(self, batch_size: int, beta_mode: str = "chunk_shared"):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if beta_mode not in {"chunk_shared", "per_frame"}:
            raise ValueError("beta_mode must be 'chunk_shared' or 'per_frame'")
        torch = self.torch
        orient = torch.zeros(batch_size, 3, device=self.device)
        orient[:, -1] = torch.pi
        beta_rows = 1 if beta_mode == "chunk_shared" else batch_size
        return {
            "body_pose_latent": torch.zeros(
                batch_size, self.latent_dim, device=self.device, requires_grad=True
            ),
            "wrist_pose": torch.zeros(batch_size, 6, device=self.device, requires_grad=True),
            "global_orient": orient.detach().clone().requires_grad_(True),
            "transl": torch.zeros(batch_size, 3, device=self.device, requires_grad=True),
            "betas": torch.zeros(beta_rows, 10, device=self.device, requires_grad=True),
        }

    @staticmethod
    def _validate_anchors(anchors: object) -> np.ndarray:
        result = np.asarray(anchors, dtype=np.float32)
        if result.ndim != 3 or result.shape[1:] != (2, 3):
            raise ValueError(f"anchors must have shape (T, 2, 3), got {result.shape}")
        if len(result) == 0:
            raise ValueError("anchors must contain at least one frame")
        if not np.isfinite(result).all():
            raise ValueError("anchors must contain only finite values")
        return result

    def _normalize_hand_sides(self, hand_side, batch_size: int):
        torch = self.torch
        if isinstance(hand_side, torch.Tensor):
            values = hand_side.to(device=self.device)
            if values.ndim == 0:
                values = values.expand(batch_size)
            if tuple(values.shape) != (batch_size,):
                raise ValueError(f"hand_side must be scalar or have shape ({batch_size},)")
            if not torch.all((values == 0) | (values == 1)):
                raise ValueError("hand_side tensor values must be 0 or 1")
            return values.to(dtype=torch.long)
        if isinstance(hand_side, str):
            if hand_side not in {"left", "right"}:
                raise ValueError("hand_side must be 'left' or 'right'")
            values = torch.full(
                (batch_size,), hand_side == "right", dtype=torch.long, device=self.device
            )
        else:
            array = np.asarray(hand_side)
            if array.ndim == 0:
                array = np.repeat(array, batch_size)
            if array.shape != (batch_size,):
                raise ValueError(f"hand_side must be scalar or have shape ({batch_size},)")
            normalized = []
            for value in array.tolist():
                if value in ("right", True, 1):
                    normalized.append(1)
                elif value in ("left", False, 0):
                    normalized.append(0)
                else:
                    raise ValueError("hand_side values must be left/right or boolean")
            values = torch.tensor(normalized, dtype=torch.long, device=self.device)
        return values

    def _expand_betas(self, params, batch_size: int):
        betas = params["betas"]
        if betas.ndim != 2 or betas.shape[1] != 10:
            raise ValueError("betas must have shape (1, 10) or (T, 10)")
        if betas.shape[0] == 1:
            return betas.expand(batch_size, -1)
        if betas.shape[0] == batch_size:
            return betas
        raise ValueError(f"betas has {betas.shape[0]} rows for {batch_size} frames")

    def forward_body(self, params):
        body_pose = self.vposer.decode(params["body_pose_latent"], output_type="aa")
        if body_pose.ndim == 4:
            body_pose = body_pose.reshape(body_pose.shape[0], -1)
        elif body_pose.ndim == 3:
            body_pose = body_pose.squeeze(1)
        if body_pose.ndim != 2 or body_pose.shape[1] != 63:
            raise ValueError(f"VPoser must decode to 63 axis-angle values, got {body_pose.shape}")
        # The research path appends the two SMPL wrist joints as zero rotations.
        body_pose = self.torch.cat((body_pose, self.torch.zeros_like(body_pose[:, :6])), dim=-1)
        batch_size = params["body_pose_latent"].shape[0]
        return self.body_model(
            global_orient=params["global_orient"],
            body_pose=body_pose,
            transl=params["transl"],
            betas=self._expand_betas(params, batch_size),
            return_full_pose=True,
        )

    def _eye_normals(self, vertices):
        if not hasattr(self, "_left_eye_face"):
            faces = self.faces_tensor.detach().cpu().numpy()
            left = np.flatnonzero(np.any(faces == self.left_eye_vertex, axis=1))
            right = np.flatnonzero(np.any(faces == self.right_eye_vertex, axis=1))
            if not len(left) or not len(right):
                raise ValueError("SMPL mesh does not contain the expected eye vertices")
            self._left_eye_face = int(left[0])
            self._right_eye_face = int(right[0])
        left_triangle = vertices[:, self.faces_tensor[self._left_eye_face]]
        right_triangle = vertices[:, self.faces_tensor[self._right_eye_face]]
        normalize = self.functional.normalize
        left_normal = normalize(
            self.torch.cross(
                left_triangle[:, 1] - left_triangle[:, 0],
                left_triangle[:, 2] - left_triangle[:, 0],
                dim=-1,
            ),
            dim=-1,
        )
        right_normal = normalize(
            self.torch.cross(
                right_triangle[:, 1] - right_triangle[:, 0],
                right_triangle[:, 2] - right_triangle[:, 0],
                dim=-1,
            ),
            dim=-1,
        )
        return left_normal, right_normal

    def _body_points(self, output, hand_sides):
        vertices, joints = output.vertices, output.joints
        eye = (vertices[:, self.left_eye_vertex] + vertices[:, self.right_eye_vertex]) / 2
        wrist_index = self.right_wrist_joint * hand_sides + self.left_wrist_joint * (1 - hand_sides)
        middle_index = (
            self.right_middle_mcp_joint * hand_sides
            + self.left_middle_mcp_joint * (1 - hand_sides)
        )
        batch = self.torch.arange(len(joints), device=joints.device)
        return eye, joints[batch, wrist_index], joints[batch, middle_index]

    def compute_loss(
        self,
        params,
        anchors,
        hand_side="right",
        *,
        weights: Mapping[str, float] | None = None,
    ):
        torch = self.torch
        functional = self.functional
        anchors = torch.as_tensor(anchors, dtype=torch.float32, device=self.device)
        if anchors.ndim != 3 or tuple(anchors.shape[1:]) != (2, 3) or len(anchors) == 0:
            raise ValueError("anchors must have shape (T, 2, 3) with T > 0")
        if not torch.isfinite(anchors).all():
            raise ValueError("anchors must contain only finite values")
        resolved = dict(DEFAULT_WEIGHTS)
        if weights:
            resolved.update(weights)
        if resolved["angle_prior"] != 0:
            raise ValueError(
                "angle_prior is unavailable in the standalone release; set its weight to 0"
            )
        hand_sides = self._normalize_hand_sides(hand_side, len(anchors))
        output = self.forward_body(params)
        eye, wrist, middle = self._body_points(output, hand_sides)
        target_eye = torch.zeros_like(eye)
        loss_eye = functional.mse_loss(eye, target_eye, reduction="sum")
        loss_wrist = functional.mse_loss(
            torch.stack((wrist, middle), dim=1), anchors, reduction="sum"
        )
        left_normal, right_normal = self._eye_normals(output.vertices)
        target_gaze = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand_as(eye)
        target_right = torch.tensor((1.0, 0.0, 0.0), device=self.device).expand_as(eye)
        eye_axis = functional.normalize(
            output.vertices[:, self.right_eye_vertex]
            - output.vertices[:, self.left_eye_vertex],
            dim=1,
        )
        loss_right = torch.sum(1.0 - torch.sum(eye_axis * target_right, dim=1))
        gaze = functional.normalize((left_normal + right_normal) / 2.0, dim=1)
        loss_gaze = torch.sum(1.0 - torch.sum(gaze * target_gaze, dim=1))
        loss_data = loss_eye + loss_wrist + resolved["gaze_weight"] * (loss_gaze + loss_right)
        pose_prior = params["body_pose_latent"].pow(2).sum()
        wrist_prior = params["wrist_pose"].pow(2).sum()
        # This L2 prior is a generic squared norm. Expanding first preserves the
        # original per-frame sum and accumulates gradients into a shared beta.
        shape_prior = self._expand_betas(params, len(anchors)).pow(2).sum()
        total = (
            resolved["data_weight"] * loss_data
            + resolved["body_pose_prior"] * pose_prior
            + resolved["wrist_prior"] * wrist_prior
            + resolved["shape_prior"] * shape_prior
        )
        report = {
            "total": float(total.detach()),
            "data": float(loss_data.detach()),
            "eye": float(loss_eye.detach()),
            "wrist": float(loss_wrist.detach()),
            "gaze": float(loss_gaze.detach()),
            "right": float(loss_right.detach()),
            "body_pose_prior": float(pose_prior.detach()),
            "shape_prior": float(shape_prior.detach()),
            "angle_prior": 0.0,
        }
        return total, report

    def fit_sequence(
        self,
        anchors,
        hand_side="right",
        *,
        n_iter: int = 500,
        lr: float = 0.05,
        temporal_smooth_weight: float = 0.0,
        beta_mode: str = "chunk_shared",
        verbose: bool = False,
    ):
        anchors_array = self._validate_anchors(anchors)
        hand_sides = self._normalize_hand_sides(hand_side, len(anchors_array))
        params = self.create_initial_params(len(anchors_array), beta_mode=beta_mode)
        trainable = [value for value in params.values() if value.is_leaf and value.requires_grad]
        optimizer = self.torch.optim.Adam(trainable, lr=lr)
        history = []
        for step in range(n_iter):
            optimizer.zero_grad()
            loss, report = self.compute_loss(params, anchors_array, hand_sides)
            if len(anchors_array) > 1:
                smooth = sum(
                    self.torch.sum((params[name][1:] - params[name][:-1]) ** 2)
                    for name in ("body_pose_latent", "global_orient", "transl")
                )
                loss = loss + temporal_smooth_weight * smooth
                report["smooth"] = float(smooth.detach())
                report["total"] = float(loss.detach())
            loss.backward()
            optimizer.step()
            history.append(report)
            if verbose and (step % 100 == 0 or step == n_iter - 1):
                print(f"fit step {step}: total={report['total']:.6f}")
        return params, history

    def check_first_person(self, params, anchors, hand_side="right"):
        torch = self.torch
        anchors_array = self._validate_anchors(anchors)
        anchors_tensor = torch.as_tensor(anchors_array, device=self.device)
        hand_sides = self._normalize_hand_sides(hand_side, len(anchors_array))
        with torch.no_grad():
            output = self.forward_body(params)
            eye, wrist, middle = self._body_points(output, hand_sides)
            eye_distance = torch.linalg.vector_norm(eye, dim=1)
            wrist_distance = torch.linalg.vector_norm(wrist - anchors_tensor[:, 0], dim=1)
            middle_distance = torch.linalg.vector_norm(middle - anchors_tensor[:, 1], dim=1)
            target_direction = self.functional.normalize(
                anchors_tensor[:, 1] - anchors_tensor[:, 0], dim=-1
            )
            fitted_direction = self.functional.normalize(middle - wrist, dim=-1)
            angle = torch.acos(
                torch.sum(target_direction * fitted_direction, dim=1).clamp(-0.99999, 0.99999)
            ) / torch.pi * 180
            passed = (
                (eye_distance < 0.1)
                & (wrist_distance < 0.1)
                & (middle_distance < 0.1)
                & (angle < 45)
            )
            confidence = (1.0 - eye_distance / 0.2).clamp(0, 1) * (
                1.0 - wrist_distance / 0.2
            ).clamp(0, 1)
        debug = {
            "eye_distance": eye_distance.cpu().numpy(),
            "wrist_distance": wrist_distance.cpu().numpy(),
            "middle_mcp_distance": middle_distance.cpu().numpy(),
            "direction_angle_degrees": angle.cpu().numpy(),
        }
        return passed.cpu().numpy(), confidence.cpu().numpy(), debug

    def make_checker(
        self,
        *,
        n_iter: int = 500,
        lr: float = 0.05,
        beta_mode: str = "chunk_shared",
        verbose: bool = False,
    ):
        """Return a checker compatible with :func:`filter_chunks`."""
        def checker(anchors: np.ndarray, hand_side: str):
            params, history = self.fit_sequence(
                anchors,
                hand_side,
                n_iter=n_iter,
                lr=lr,
                temporal_smooth_weight=0.0,
                beta_mode=beta_mode,
                verbose=verbose,
            )
            passed, confidence, debug = self.check_first_person(params, anchors, hand_side)
            debug["loss_history"] = history
            return passed, confidence, debug

        return checker
