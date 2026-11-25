# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except (ImportError, RuntimeError):
    BlockMask = ()
    create_block_mask = flex_attention = None
    FLEX_ATTENTION_AVAILABLE = False

_compiled_flex_attention = None


def _run_compiled_flex_attention(query, key, value, block_mask):
    """Lazily compile the production CUDA kernel without affecting SFT imports."""
    global _compiled_flex_attention
    if os.environ.get("HANDSONWORLD_FLEX_COMPILE", "1") == "0":
        raise RuntimeError(
            "CUDA block-causal training requires compiled FlexAttention; "
            "unset HANDSONWORLD_FLEX_COMPILE=0"
        )
    if _compiled_flex_attention is None:
        try:
            import torch._dynamo as _dynamo

            _dynamo.config.optimize_ddp = False
        except Exception:
            pass
        _compiled_flex_attention = torch.compile(
            flex_attention,
            dynamic=False,
            # Match the training runtime: PyTorch 2.6's default backward
            # candidate can be filtered out on Ampere for 128-wide heads.
            mode=os.environ.get("TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"),
        )
    return _compiled_flex_attention(
        query, key, value, block_mask=block_mask
    )
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def layer_norm_float(module: nn.LayerNorm, x: torch.Tensor):
    """Apply LayerNorm in FP32 even when affine parameters are stored in BF16."""
    weight = module.weight.float() if module.weight is not None else None
    bias = module.bias.float() if module.bias is not None else None
    return F.layer_norm(x.float(), module.normalized_shape, weight, bias, module.eps)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attention_mask=None):
        if attention_mask is None:
            x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
            return x
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        if BlockMask and isinstance(attention_mask, BlockMask):
            original_length = q.shape[2]
            padded_length = math.ceil(original_length / 128) * 128
            if padded_length != original_length:
                padding = (0, 0, 0, padded_length - original_length)
                q, k, v = (F.pad(tensor, padding) for tensor in (q, k, v))
            x = _run_compiled_flex_attention(
                q.contiguous(), k.contiguous(), v.contiguous(), attention_mask
            )
            x = x[:, :, :original_length]
        else:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, attention_mask=None, kv_cache=None, store_kv=False):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if kv_cache is not None and kv_cache.get("k") is not None:
            cached_k = kv_cache["k"]
            cached_v = kv_cache["v"]
            k_full = torch.cat((cached_k, k), dim=1)
            v_full = torch.cat((cached_v, v), dim=1)
        else:
            k_full, v_full = k, v
        if store_kv and kv_cache is not None:
            kv_cache["k"] = k_full.detach()
            kv_cache["v"] = v_full.detach()
        x = self.attn(q, k_full, v_full, attention_mask=attention_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, kv_cache=None, store_kv=False):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        if (
            kv_cache is not None
            and kv_cache.get("k") is not None
            and not torch.is_grad_enabled()
        ):
            k, v = kv_cache["k"], kv_cache["v"]
        else:
            k = self.norm_k(self.k(ctx))
            v = self.v(ctx)
            if store_kv and kv_cache is not None:
                kv_cache["k"], kv_cache["v"] = k.detach(), v.detach()
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self, x, context, t_mod, freqs, attention_mask=None,
        self_kv_cache=None, cross_kv_cache=None, store_kv=False,
    ):
        causal_execution = (
            attention_mask is not None or self_kv_cache is not None
            or cross_kv_cache is not None or store_kv
        )
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        modulation = self.modulation.to(device=t_mod.device)
        if causal_execution:
            modulation = modulation.float()
            t_mod = t_mod.float()
        else:
            modulation = modulation.to(dtype=t_mod.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        if not causal_execution:
            input_x = modulate(self.norm1(x), shift_msa, scale_msa)
            x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
            x = x + self.cross_attn(self.norm3(x), context)
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            return self.gate(x, gate_mlp, self.ffn(input_x))

        input_x = modulate(layer_norm_float(self.norm1, x), shift_msa, scale_msa).to(x.dtype)
        self_output = self.self_attn(
            input_x, freqs, attention_mask=attention_mask,
            kv_cache=self_kv_cache, store_kv=store_kv,
        )
        x = self.gate(x.float(), gate_msa, self_output.float()).to(input_x.dtype)
        x = x + self.cross_attn(
            layer_norm_float(self.norm3, x).to(x.dtype), context,
            kv_cache=cross_kv_cache, store_kv=store_kv,
        )
        input_x = modulate(layer_norm_float(self.norm2, x), shift_mlp, scale_mlp).to(x.dtype)
        x = self.gate(x.float(), gate_mlp, self.ffn(input_x).float()).to(input_x.dtype)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        downscale_factor_control_adapter: int = 8,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.in_dim_control_adapter = in_dim_control_adapter if add_control_adapter else None

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
                downscale_factor=downscale_factor_control_adapter,
            )
        else:
            self.control_adapter = None

    def patchify(
        self,
        x: torch.Tensor,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        control_camera_valid_mask: Optional[torch.Tensor] = None,
        hand_motion_tokens: Optional[torch.Tensor] = None,
        hand_motion_masks_input: Optional[torch.Tensor] = None,
    ):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            # import ipdb; ipdb.set_trace()
            y_camera = self.control_adapter(control_camera_latents_input) * control_camera_valid_mask
            # if control_camera_valid_mask is not None:
            #     y_camera = y_camera * control_camera_valid_mask.to(device=y_camera.device, dtype=y_camera.dtype)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        if hand_motion_tokens is not None:
            hand_motion_tokens = hand_motion_tokens.to(device=x.device, dtype=x.dtype)
            if hand_motion_masks_input is not None:
                hand_motion_masks_input = hand_motion_masks_input.to(device=x.device, dtype=x.dtype)
                b, c, f, h, w = hand_motion_masks_input.shape
                if c != 8:
                    raise ValueError("hand_motion_masks_input expects 8 packed channels, got %d." % c)
                if f != x.shape[2]:
                    raise ValueError(
                        "hand_motion_masks_input frame dim %d does not match patch tokens frame dim %d."
                        % (f, x.shape[2])
                    )
                hand_masks = hand_motion_masks_input.reshape(b, 2, 4, f, h, w).amax(dim=2)
                hand_masks = hand_masks.permute(0, 2, 1, 3, 4).reshape(b * f, 2, h, w)
                hand_masks = F.interpolate(hand_masks, size=x.shape[-2:], mode="area")
                hand_masks = hand_masks.reshape(b, f, 2, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
                hand_masks = hand_masks.clamp(0.0, 1.0)
                if hand_motion_tokens.ndim == 4:
                    if hand_motion_tokens.shape[2] != 2:
                        raise ValueError("4D hand_motion_tokens expects side dim 2, got %d." % hand_motion_tokens.shape[2])
                    token_field = (
                        hand_motion_tokens.permute(0, 3, 1, 2).unsqueeze(-1).unsqueeze(-1)
                        * hand_masks.permute(0, 2, 1, 3, 4).unsqueeze(1)
                    ).sum(dim=3)
                    x = x + token_field * control_camera_valid_mask
                else:
                    hand_mask = hand_masks.amax(dim=1, keepdim=True)
                    x = x + hand_motion_tokens.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1) * hand_mask * control_camera_valid_mask
            else:
                if hand_motion_tokens.ndim == 4:
                    raise ValueError("side hand_motion_tokens require hand_motion_masks_input.")
                x = x + hand_motion_tokens.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1) * control_camera_valid_mask
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    @staticmethod
    def block_causal_mask(
        num_frames: int,
        frame_seqlen: int,
        chunk_size: int = 4,
        fixed_first_frame: bool = True,
        device=None,
    ):
        """Dense SDPA mask matching FastGen's temporal block-causal contract."""
        if chunk_size < 1 or num_frames < 1 or frame_seqlen < 1:
            raise ValueError("num_frames, frame_seqlen and chunk_size must be positive")
        frame_ids = torch.arange(num_frames, device=device)
        if fixed_first_frame:
            block_ids = torch.where(frame_ids == 0, 0, 1 + (frame_ids - 1) // chunk_size)
        else:
            block_ids = frame_ids // chunk_size
        token_blocks = block_ids.repeat_interleave(frame_seqlen)
        return token_blocks[:, None] >= token_blocks[None, :]

    def flex_block_causal_mask(
        self,
        num_frames: int,
        frame_seqlen: int,
        chunk_size: int = 4,
        fixed_first_frame: bool = True,
        device=None,
    ):
        """Sparse production mask; dense masks are reserved for CPU parity tests."""
        if not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError("Full-video CUDA causal training requires torch FlexAttention")
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128
        ends = torch.zeros(padded_length, dtype=torch.long, device=device)
        start = 0
        if fixed_first_frame:
            ends[:frame_seqlen] = frame_seqlen
            start = 1
        while start < num_frames:
            end = min(start + chunk_size, num_frames)
            ends[start * frame_seqlen:end * frame_seqlen] = end * frame_seqlen
            start = end

        def mask_mod(batch, head, query_index, key_index):
            del batch, head
            return (key_index < ends[query_index]) | (query_index == key_index)

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            device=device,
            # Compiled construction avoids materializing the full token mask
            # for production CUDA sequences. CPU parity tests stay eager.
            _compile=torch.device(device).type == "cuda",
        )

    def _causal_freqs(self, f, h, w, start_frame, device):
        if start_frame + f > self.freqs[0].shape[0]:
            raise ValueError(f"RoPE frame range [{start_frame},{start_frame + f}) exceeds table")
        return torch.cat([
            self.freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(device)

    def forward_causal(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        control_hidden_states: Optional[torch.Tensor] = None,
        chunk_size: int = 4,
        fixed_first_frame: bool = True,
        start_frame: int = 0,
        self_kv_caches=None,
        cross_kv_caches=None,
        store_kv: bool = False,
        full_causal: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        """Causal execution using this model's canonical parameters.

        Full-video CUDA training uses a sparse FlexAttention block mask; CPU
        tests use a dense SDPA oracle. Autoregressive calls pass per-block
        caches and attend to their detached history.
        """
        if use_gradient_checkpointing and store_kv:
            raise ValueError("KV mutation is incompatible with checkpoint recomputation")
        batch, _, frames, _, _ = x.shape
        if timestep.ndim == 0:
            timestep = timestep.expand(batch)
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(batch, frames)
        if timestep.shape != (batch, frames):
            raise ValueError(f"timestep must have shape [B] or [B,F], got {tuple(timestep.shape)}")
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1)).to(x.dtype)
        ).view(batch, frames, self.dim)
        t_mod = self.time_projection(t).unflatten(-1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            if y is None or clip_feature is None:
                raise ValueError("Wan2.1 I2V requires y and clip_feature")
            x = torch.cat([x, y], dim=1)
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)

        x = self.patch_embedding(x)
        if control_hidden_states is not None:
            if control_hidden_states.shape != x.shape:
                raise ValueError(
                    f"control feature shape {tuple(control_hidden_states.shape)} != patch shape {tuple(x.shape)}"
                )
            x = x + control_hidden_states.to(x)
        f, h, w = x.shape[2:]
        if f != frames:
            raise ValueError("temporal patching other than one is unsupported by causal execution")
        frame_seqlen = h * w
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        token_t_mod = t_mod.repeat_interleave(frame_seqlen, dim=1)
        token_t = t.repeat_interleave(frame_seqlen, dim=1)
        freqs = self._causal_freqs(f, h, w, start_frame, x.device)
        attention_mask = None
        if full_causal:
            if x.is_cuda:
                mask_key = (f, frame_seqlen, chunk_size, fixed_first_frame, x.device)
                if getattr(self, "_flex_mask_key", None) != mask_key:
                    self._flex_mask = self.flex_block_causal_mask(
                        f, frame_seqlen, chunk_size, fixed_first_frame, x.device
                    )
                    self._flex_mask_key = mask_key
                attention_mask = self._flex_mask
            else:
                attention_mask = self.block_causal_mask(
                    f, frame_seqlen, chunk_size, fixed_first_frame, x.device
                )

        if self_kv_caches is None:
            self_kv_caches = [None] * len(self.blocks)
        if cross_kv_caches is None:
            cross_kv_caches = [None] * len(self.blocks)
        if len(self_kv_caches) != len(self.blocks) or len(cross_kv_caches) != len(self.blocks):
            raise ValueError("one self/cross KV cache is required per transformer block")

        # Streaming gradient replay runs chunks in reverse after the rollout
        # has populated the complete cache. A chunk may only see the prefix
        # before its absolute start frame, matching FastGen's cache index
        # contract. Normal forward rollout already has exactly this length.
        prefix_tokens = start_frame * frame_seqlen
        visible_self_caches = []
        for cache in self_kv_caches:
            if cache is None or cache.get("k") is None or cache["k"].shape[1] <= prefix_tokens:
                visible_self_caches.append(cache)
                continue
            visible_self_caches.append({
                **cache,
                "k": cache["k"][:, :prefix_tokens],
                "v": cache["v"][:, :prefix_tokens],
            })

        # Non-reentrant checkpointing recomputes blocks during backward. Keep
        # each call pinned to the cache history visible during its forward;
        # later AR chunks replace dict entries and must not change recompute
        # tensor shapes. Tensor values are immutable, so shallow snapshots are
        # sufficient and do not duplicate the large KV storage.
        block_self_caches = visible_self_caches
        block_cross_caches = cross_kv_caches
        if self.training and use_gradient_checkpointing:
            block_self_caches = [dict(cache) if cache is not None else None for cache in visible_self_caches]
            block_cross_caches = [dict(cache) if cache is not None else None for cache in cross_kv_caches]

        def run_block(block, hidden, index):
            return block(
                hidden, context, token_t_mod, freqs,
                attention_mask=attention_mask,
                self_kv_cache=block_self_caches[index],
                cross_kv_cache=block_cross_caches[index],
                store_kv=store_kv,
            )

        for index, block in enumerate(self.blocks):
            if self.training and use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    lambda hidden, module=block, i=index: run_block(module, hidden, i),
                    x, use_reentrant=False,
                )
            else:
                x = run_block(block, x, index)
        x = self.head(x, token_t)
        return self.unpatchify(x, (f, h, w))

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
