# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wan_video_dit import sinusoidal_embedding_1d
from .wan_video_vae import CausalConv3d

class CausalConv1d(nn.Conv1d):
    """
    Causal 1d convolution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # For 1D: padding format is (left, right)
        # We want causal padding: pad on the left (past), no padding on the right (future)
        self._padding = (2 * self.padding[0], 0)
        self.padding = (0,)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[0] > 0:
            cache_x = cache_x.to(x.device)
            # Concatenate along the time dimension (dim=2 for shape [B, C, T])
            x = torch.cat([cache_x, x], dim=2)
            # Reduce left padding by the cache size
            padding[0] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)

class WanPoseControllerModel(torch.nn.Module):
    def __init__(self, pose_dim=48 * 6, dim=1536):
        super().__init__()
        self.pose_dim = pose_dim
        self.freq_num = 12
        self.g_e = nn.Sequential(
            CausalConv1d(pose_dim * self.freq_num, dim, 3, 2, 1),
            nn.SiLU(),
            CausalConv1d(dim, dim, 3, 2, 1),
        )
        self.g_m = nn.Sequential(
            CausalConv1d(pose_dim * self.freq_num, dim, 3, 2, 1),
            nn.SiLU(),
            CausalConv1d(dim, dim * 6, 3, 2, 1),
        )
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

    def forward(self, poses, e_t):
        B, T, _ = poses.shape
        poses = sinusoidal_embedding_1d(self.freq_num, poses.reshape(-1)).reshape(B, T, -1) # [B, T, pose_dim * freq_num]
        # print(poses.shape)
        
        poses = poses.transpose(-1, -2) # [B, T, pose_dim]
        e_p, m_p = self.g_e(poses).transpose(-1, -2), self.g_m(poses).transpose(-1, -2) # [B, T//4, dim]
        # print(e_p.shape, e_t.shape, m_p.shape)
        emb = self.linear(nn.functional.silu(e_p + e_t)) + m_p
        return emb

    def init(self):
        state_dict = self.g_e[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.g_e[-1].load_state_dict(state_dict)

        state_dict = self.g_m[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.g_m[-1].load_state_dict(state_dict)

        state_dict = self.linear[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.linear[-1].load_state_dict(state_dict)


class WanHandMotionTokenEncoder(torch.nn.Module):
    """Temporal pose encoder for patch-token addition, not AdaLN/t_mod control."""

    def __init__(self, pose_dim=297, dim=1536, separate_hands=False):
        super().__init__()
        self.pose_dim = pose_dim
        self.dim = dim
        self.separate_hands = separate_hands
        encoder_pose_dim = 16 * 9 if separate_hands else pose_dim
        self.encoder = nn.Sequential(
            CausalConv1d(encoder_pose_dim, dim, 3, 2, 1),
            nn.SiLU(),
            CausalConv1d(dim, dim, 3, 2, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.init()

    def forward(self, poses):
        # poses: [B, T, pose_dim]
        if self.separate_hands:
            if poses.shape[-1] < 2 * 16 * 9:
                raise ValueError("separate_hands expects poses last dim >= 288, got %d." % poses.shape[-1])
            b, t, _ = poses.shape
            left_pose = poses[..., :16 * 9]
            right_pose = poses[..., 16 * 9:2 * 16 * 9]
            poses = torch.stack([left_pose, right_pose], dim=1).reshape(b * 2, t, 16 * 9)
            tokens = self.encoder(poses.transpose(-1, -2)).transpose(-1, -2)
            tokens = self.proj(tokens)
            return tokens.reshape(b, 2, tokens.shape[1], self.dim).permute(0, 2, 1, 3)

        # output: [B, ceil(T/4), dim]
        tokens = self.encoder(poses.transpose(-1, -2)).transpose(-1, -2)
        return self.proj(tokens)

    def init(self):
        state_dict = self.encoder[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.encoder[-1].load_state_dict(state_dict)

        state_dict = self.proj[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.proj[-1].load_state_dict(state_dict)


class WanPoseLatentsControllerModel(torch.nn.Module):
    def __init__(self, pose_dim=48, dim=1536):
        super().__init__()
        self.pose_dim = pose_dim
        self.g_e = nn.Conv3d(
            pose_dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.g_m = nn.Conv3d(
            pose_dim, dim * 6, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

    def forward(self, poses, e_t):
        # poses [B x 48 x T x H x W]
        
        # poses = poses.transpose(1, 2) # [B, 3, T, H, W]
        e_p, m_p = self.g_e(poses).permute(0, 2, 3, 4, 1).flatten(1, 3), self.g_m(poses).permute(0, 2, 3, 4, 1).flatten(1, 3) # [B, T//4xHxW, dim]
        emb = self.linear(nn.functional.silu(e_p + e_t)) + m_p
        return emb

    def init(self):
        state_dict = self.g_e.state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.g_e.load_state_dict(state_dict)

        state_dict = self.g_m.state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.g_m.load_state_dict(state_dict)

        state_dict = self.linear[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.linear[-1].load_state_dict(state_dict)

class WanPoseVideoControllerFeatModel(nn.Module):
    def __init__(self, in_channels=6+3, dim=1536):
        super().__init__()

        # [B, 6, T, H, W] -> [B, 64, T/2, H/4, W/4]
        self.conv1 = nn.Sequential(
            CausalConv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )
        
        # 中间特征提取：T 下采样 2x，空间下采样 4x
        # -> [B, 256, T/2, H/16, W/16]
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=(1, 7, 7), stride=(1, 4, 4), padding=(0, 2, 2)),
            nn.SiLU()
        )
        
        # 最终映射：T 下采样 2x，空间下采样 2x，通道数映射到目标维度
        # -> [B, embed_dim, T/4, H/32, W/32]
        # self.g_e = CausalConv3d(dim // 2, dim, kernel_size=(3, 5, 5), stride=(2, 4, 4), padding=1)
        # self.g_m = CausalConv3d(dim // 2, dim * 6, kernel_size=(3, 5, 5), stride=(2, 4, 4), padding=1)

        self.g_e = CausalConv3d(256, 256, kernel_size=3, stride=2, padding=1)
        self.g_m = CausalConv3d(256, dim, kernel_size=3, stride=2, padding=1)

        self.linear = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, dim),
        )

    def forward(self, poses):
        """
        rays: [B, 6/9, T, H, W]
        """
        x = self.conv1(poses)
        x = self.conv2(x)

        e_p, m_p = self.g_e(x).permute(0, 2, 3, 4, 1).flatten(1, 3), self.g_m(x).permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, T//4xHxW, dim]
        emb = self.linear(nn.functional.silu(e_p)) + m_p

        return emb


class WanPoseVideoControllerModel(nn.Module):
    def __init__(self, in_channels=6+3, dim=1536, spatial_downsample_factor=32):
        super().__init__()
        if spatial_downsample_factor not in (16, 32):
            raise ValueError("spatial_downsample_factor must be 16 or 32")
        final_stride = (2, 2, 2) if spatial_downsample_factor == 32 else (2, 1, 1)

        # [B, 6, T, H, W] -> [B, 64, T/2, H/4, W/4]
        self.conv1 = nn.Sequential(
            CausalConv3d(in_channels, dim // 16, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 2, 2)),
            nn.SiLU()
        )
        
        # 中间特征提取：T 下采样 2x，空间下采样 4x
        # -> [B, 256, T/2, H/16, W/16]
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 16, dim // 4, kernel_size=(1, 7, 7), stride=(1, 4, 4), padding=(0, 2, 2)),
            nn.SiLU()
        )
        
        # 最终映射：T 下采样 2x，空间下采样 1x/2x，通道数映射到目标维度
        # -> [B, embed_dim, T/4, H/16 or H/32, W/16 or W/32]
        # self.g_e = CausalConv3d(dim // 2, dim, kernel_size=(3, 5, 5), stride=(2, 4, 4), padding=1)
        # self.g_m = CausalConv3d(dim // 2, dim * 6, kernel_size=(3, 5, 5), stride=(2, 4, 4), padding=1)

        self.g_e = CausalConv3d(dim // 4, dim, kernel_size=3, stride=final_stride, padding=1)
        self.g_m = CausalConv3d(dim // 4, dim * 6, kernel_size=3, stride=final_stride, padding=1)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

    def forward(self, poses, e_t=None):
        """
        rays: [B, 6/9, T, H, W]
        """
        x = self.conv1(poses)
        x = self.conv2(x)

        e_p, m_p = self.g_e(x).permute(0, 2, 3, 4, 1).flatten(1, 3), self.g_m(x).permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, T//4xHxW, dim]
        emb = self.linear(nn.functional.silu(e_p + e_t)) + m_p

        return emb
    
    def init(self):
        state_dict = self.g_e.state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.g_e.load_state_dict(state_dict)

        state_dict = self.g_m.state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.g_m.load_state_dict(state_dict)

        state_dict = self.linear[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.linear[-1].load_state_dict(state_dict)


class WanPoseVideoControllerModel16(WanPoseVideoControllerModel):
    def __init__(self, in_channels=6+3, dim=5120):
        super().__init__(
            in_channels=in_channels,
            dim=dim,
            spatial_downsample_factor=16,
        )
