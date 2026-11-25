# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from ..models.wan_video_dit import sinusoidal_embedding_1d


class JointOccPluckerFeatureBuilder:
    JOINT_MAP_21 = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    MANO_FINGERTIP_IDXS = [744, 320, 443, 554, 671]

    def __init__(self, joint_regressor_path=None):
        self.joint_regressor_path = joint_regressor_path or self._find_default_joint_regressor_path()
        self.joint_regressor_np = None
        self.fingertip_idxs_np = np.asarray(self.MANO_FINGERTIP_IDXS, dtype=np.int64)
        self.joint_map_np = np.asarray(self.JOINT_MAP_21, dtype=np.int64)

    def _find_default_joint_regressor_path(self):
        candidates = [
            os.environ.get("HAND_JOINT_REGRESSOR_PATH"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def _get_joint_regressor(self, device):
        if self.joint_regressor_np is None:
            if self.joint_regressor_path is None:
                raise FileNotFoundError(
                    "Could not find MANO joint regressor. Set HAND_JOINT_REGRESSOR_PATH to your resource file."
                )
            if self.joint_regressor_path.endswith(".npz"):
                data = np.load(self.joint_regressor_path)
                self.joint_regressor_np = data["J_regressor"].astype(np.float32)
                if "fingertip_idxs" in data:
                    self.fingertip_idxs_np = data["fingertip_idxs"].astype(np.int64)
                if "joint_map" in data:
                    self.joint_map_np = data["joint_map"].astype(np.int64)
            elif self.joint_regressor_path.endswith(".npy"):
                self.joint_regressor_np = np.load(self.joint_regressor_path).astype(np.float32)
            elif self.joint_regressor_path.endswith(".pkl"):
                with open(self.joint_regressor_path, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
                regressor = data["J_regressor"]
                if hasattr(regressor, "toarray"):
                    regressor = regressor.toarray()
                self.joint_regressor_np = np.asarray(regressor, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported HAND_JOINT_REGRESSOR_PATH: {self.joint_regressor_path}")

        regressor = torch.from_numpy(self.joint_regressor_np).to(device=device, dtype=torch.float32)
        fingertip_idxs = torch.from_numpy(self.fingertip_idxs_np).to(device=device, dtype=torch.long)
        joint_map = torch.from_numpy(self.joint_map_np).to(device=device, dtype=torch.long)
        return regressor, fingertip_idxs, joint_map

    def vertices_to_mano21_joints(self, hand_pose, device):
        if hand_pose.shape[2] < 1556:
            raise ValueError(
                "joint_occ_plucker expects two MANO hands as 1556 vertices, "
                f"got {hand_pose.shape[2]} vertices."
            )
        regressor, fingertip_idxs, joint_map = self._get_joint_regressor(device)

        def convert_one_hand(vertices):
            joints_16 = torch.einsum("jv,btvc->btjc", regressor, vertices)
            fingertips = vertices.index_select(dim=2, index=fingertip_idxs)
            joints_21 = torch.cat([joints_16, fingertips], dim=2)
            return joints_21.index_select(dim=2, index=joint_map)

        left_joints = convert_one_hand(hand_pose[:, :, :778])
        right_joints = convert_one_hand(hand_pose[:, :, 778:1556])
        return torch.cat([left_joints, right_joints], dim=2)

    @staticmethod
    def world_joints_to_camera(joints, cam_pose):
        num_frames = min(joints.shape[1], cam_pose.shape[1])
        joints = joints[:, :num_frames]
        cam_pose = cam_pose[:, :num_frames]
        ones = torch.ones((*joints.shape[:-1], 1), dtype=joints.dtype, device=joints.device)
        joints_h = torch.cat([joints, ones], dim=-1)
        return torch.einsum("btij,btnj->btni", cam_pose, joints_h)[..., :3]

    @staticmethod
    def project_joints_to_latent_grid(joints_cam, K_full, downsample_rate=8):
        b, t, n, _ = joints_cam.shape
        K = K_full.clone()
        K[:2, :] = K[:2, :] / downsample_rate
        joints_flat = joints_cam.reshape(b * t, n, 3)
        projected = torch.bmm(K.unsqueeze(0).expand(b * t, -1, -1), joints_flat.transpose(1, 2))
        z_metric = torch.clamp(projected[:, 2:3], min=0.2)
        uv = projected[:, :2] / z_metric
        disparity = (1.0 / z_metric) / 5.0
        return uv.transpose(1, 2).view(b, t, n, 2), disparity.transpose(1, 2).view(b, t, n, 1)

    @staticmethod
    def downsample_wan_style(tensor, ft=4):
        if tensor.shape[1] == 1:
            return tensor
        t0 = tensor[:, 0:1]
        t_rest = tensor[:, 1:]
        t_segments = t_rest.shape[1] // ft
        if t_segments <= 0:
            return t0
        t_rest = t_rest[:, :t_segments * ft]
        t_rest = t_rest.view(t_rest.shape[0], t_segments, ft, *t_rest.shape[2:]).mean(dim=2)
        return torch.cat([t0, t_rest], dim=1)

    @staticmethod
    def sinusoidal_encoding_3d(coords, dim=32):
        b, t, n, _ = coords.shape
        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=coords.device, dtype=coords.dtype)
            * -(torch.log(torch.tensor(10000.0, device=coords.device, dtype=coords.dtype)) / max(half_dim - 1, 1))
        )
        args = coords.reshape(-1, 1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding.view(b, t, n, 3, dim).mean(dim=3)

    @staticmethod
    def generate_joint_heatmaps(uv_coords, grid_height, grid_width, sigma=2.0):
        b, t, n, _ = uv_coords.shape
        y_range = torch.arange(grid_height, device=uv_coords.device, dtype=uv_coords.dtype)
        x_range = torch.arange(grid_width, device=uv_coords.device, dtype=uv_coords.dtype)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
        dist_sq = (
            (grid_x.view(1, 1, 1, grid_height, grid_width) - uv_coords[..., 0].view(b, t, n, 1, 1)) ** 2
            + (grid_y.view(1, 1, 1, grid_height, grid_width) - uv_coords[..., 1].view(b, t, n, 1, 1)) ** 2
        )
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    @staticmethod
    def get_occlusion_mask(uv_coords, disparity, dist_threshold=1.0):
        diff = uv_coords.unsqueeze(3) - uv_coords.unsqueeze(2)
        dists = torch.norm(diff, dim=-1)
        is_overlapping = dists < dist_threshold
        disp_i = disparity.unsqueeze(3)
        disp_j = disparity.unsqueeze(2)
        i_is_behind_j = (disp_i < (disp_j - 1e-3)).squeeze(-1)
        return (~torch.logical_and(is_overlapping, i_is_behind_j).any(dim=3)).float().unsqueeze(-1)

    @staticmethod
    def sift_z0_features(z0, heatmaps_0, vis_mask_0):
        weights = heatmaps_0 / (heatmaps_0.sum(dim=(-2, -1), keepdim=True) + 1e-6)
        pooled_feats = torch.einsum("bnhw,bchw->bnc", weights, z0)
        return pooled_feats * vis_mask_0

    @staticmethod
    def propagate_z0_depth_aware(heatmaps_latent, pooled_feats, disparity_latent, scale=5.0):
        logits = torch.log(heatmaps_latent + 1e-6) + disparity_latent.unsqueeze(-1) * scale
        attn_weights = F.softmax(logits, dim=2)
        splatted = torch.einsum("btnhw,bnc->btchw", attn_weights, pooled_feats)
        opacity = heatmaps_latent.sum(dim=2, keepdim=True)
        return splatted * opacity

    @staticmethod
    def extract_reference_vae_latent(y, batch_size, grid_height, grid_width, device):
        if y is None:
            raise ValueError(
                "joint_occ_plucker requires Wan I2V y latents from WanVideoUnit_ImageEmbedderVAE. "
                "Pass input_image during inference, and include input_image in --extra_inputs during training."
            )
        if y.ndim != 5 or y.shape[1] < 20:
            raise ValueError(
                "joint_occ_plucker expects y with shape [B, >=20, T_lat, H/8, W/8] "
                "where channels 0:4 are mask and 4:20 are first-frame VAE latents; "
                f"got {tuple(y.shape)}."
            )
        z0 = y[:, 4:20, 0].to(device=device, dtype=torch.float32)
        if z0.shape[0] != batch_size:
            if z0.shape[0] == 1:
                z0 = z0.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(f"y batch {z0.shape[0]} does not match hand_pose batch {batch_size}.")
        if z0.shape[-2:] != (grid_height, grid_width):
            z0 = F.interpolate(z0.float(), size=(grid_height, grid_width), mode="bilinear", align_corners=False)
        return z0

    def build_joint_occ_latents(self, hand_pose, cam_pose, K, z0, grid_height, grid_width, device):
        joints_world = self.vertices_to_mano21_joints(hand_pose, device)
        joints_cam = self.world_joints_to_camera(joints_world, cam_pose)
        uv_coords, disparity = self.project_joints_to_latent_grid(joints_cam, K, downsample_rate=8)
        combined_coords = torch.cat([uv_coords, disparity], dim=-1)

        coord_feat = self.sinusoidal_encoding_3d(combined_coords, dim=32)
        joint_ids = torch.arange(joints_cam.shape[2], device=device, dtype=coord_feat.dtype)
        id_feat = sinusoidal_embedding_1d(32, joint_ids).view(1, 1, joints_cam.shape[2], 32)
        id_feat = id_feat.expand(joints_cam.shape[0], joints_cam.shape[1], -1, -1)
        joint_feats = torch.cat([coord_feat, id_feat], dim=-1)

        heatmaps_full = self.generate_joint_heatmaps(uv_coords, grid_height, grid_width)
        vis_mask_full = self.get_occlusion_mask(uv_coords, disparity)
        pooled_feats = self.sift_z0_features(z0, heatmaps_full[:, 0], vis_mask_full[:, 0])
        struct_map_full = torch.einsum("btnhw,btnd->btdhw", heatmaps_full, joint_feats)

        heatmaps_latent = self.downsample_wan_style(heatmaps_full, ft=4)
        struct_map_latent = self.downsample_wan_style(struct_map_full, ft=4)
        disparity_latent = self.downsample_wan_style(disparity, ft=4)
        occ_latent = self.propagate_z0_depth_aware(
            heatmaps_latent,
            pooled_feats,
            disparity_latent,
            scale=5.0,
        )
        return torch.cat(
            [
                struct_map_latent.permute(0, 2, 1, 3, 4).contiguous(),
                occ_latent.permute(0, 2, 1, 3, 4).contiguous(),
            ],
            dim=1,
        )


def resize_packed_plucker_to_latent_grid(packed_plucker, grid_height, grid_width):
    b, c, t, h, w = packed_plucker.shape
    plucker = packed_plucker.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    plucker = F.interpolate(plucker, size=(grid_height, grid_width), mode="bilinear", align_corners=False)
    return plucker.view(b, t, c, grid_height, grid_width).permute(0, 2, 1, 3, 4).contiguous()
