"""Wan I2V wrappers for FastGen losses using the canonical DiffSynth model.

The wrappers intentionally contain no second transformer implementation. Both
bidirectional teachers and causal students own a ``diffsynth.WanModel`` under
``transformer`` so checkpoint names and SFT parameters stay identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

from fastgen.networks.network import CausalFastGenNetwork, FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES

from .specs import WAN_MODEL_SPECS, get_model_spec

try:
    from diffsynth.models.wan_video_dit import WanModel, precompute_freqs_cis_3d
except ImportError as exc:  # pragma: no cover - gives CLI users a useful error
    raise ImportError(
        "DiffSynth backend is not importable; add HandsOnWorld/third_party/"
        "DiffSynth-Studio to PYTHONPATH before importing handsonworld.models.wan"
    ) from exc


Prediction = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def _canonical_architecture(config: Optional[dict], spec: dict) -> dict:
    if config is None:
        return dict(spec["architecture"])
    config = dict(config)
    aliases = {
        "in_channels": "in_dim",
        "out_channels": "out_dim",
        "num_attention_heads": "num_heads",
    }
    for source, target in aliases.items():
        if source in config and target not in config:
            config[target] = config[source]
    if "dim" not in config and "attention_head_dim" in config and "num_heads" in config:
        config["dim"] = config["attention_head_dim"] * config["num_heads"]
    allowed = set(spec["architecture"])
    canonical = {key: value for key, value in config.items() if key in allowed}
    missing = sorted(allowed - set(canonical))
    if missing:
        for key in missing:
            if key in spec["architecture"]:
                canonical[key] = spec["architecture"][key]
    canonical["patch_size"] = tuple(canonical["patch_size"])
    return canonical


class WanI2V(FastGenNetwork):
    """Bidirectional Wan teacher with the FastGen network contract."""

    MODEL_ID_VER_2_1_I2V_14B_480P = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    MODEL_ID_VER_2_1_I2V_14B_720P = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    MODEL_ID_VER_2_2_TI2V_5B_720P = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    def __init__(
        self,
        model_id_or_local_path: str = MODEL_ID_VER_2_2_TI2V_5B_720P,
        model_type: Optional[str] = None,
        transformer_config: Optional[dict] = None,
        load_pretrained: bool = True,
        disable_grad_ckpt: bool = False,
        enable_logvar_linear: bool = True,
        use_fsdp_checkpoint: bool = True,
        use_wan_official_sinusoidal: bool = True,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        **model_kwargs,
    ) -> None:
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)
        self.spec = get_model_spec(model_type or model_id_or_local_path)
        self.model_type = self.spec["model_type"]
        self.model_id = self.spec["diffusers_model_id"]
        self.model_id_or_local_path = str(model_id_or_local_path)
        self.inner_dim = self.spec["dim"]
        self.latent_channels = self.spec["latent_channels"]
        self.concat_mask = self.spec["concat_mask"]
        self.use_image_encoder = self.spec["use_image_encoder"]
        self.expand_timesteps = self.spec["expand_timesteps"]
        self.disable_grad_ckpt = bool(disable_grad_ckpt)
        self.use_fsdp_checkpoint = bool(use_fsdp_checkpoint)
        self._use_wan_official_sinusoidal = bool(use_wan_official_sinusoidal)

        architecture = _canonical_architecture(transformer_config, self.spec)
        self.transformer = WanModel(**architecture)
        if enable_logvar_linear:
            self.transformer.logvar_linear = torch.nn.Linear(self.inner_dim, 1)
        if load_pretrained:
            self._load_local_pretrained(model_id_or_local_path)

    def _load_local_pretrained(self, source: str) -> None:
        path = Path(source)
        if not path.exists():
            raise ValueError(
                "Canonical Wan loading requires a local DiffSynth/official checkpoint. "
                "Use load_pretrained=False and stream_wan_official_local_state for sharded initialization."
            )
        if path.is_dir():
            # Large official shards must be streamed after FSDP materialization.
            if (path / "diffusion_pytorch_model.safetensors.index.json").is_file():
                raise ValueError(
                    "Sharded checkpoint directories must be loaded with "
                    "stream_wan_official_local_state after sharding"
                )
            candidates = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.pth"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one model file under {path}, found {len(candidates)}")
            path = candidates[0]
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        for prefix in ("state_dict", "model", "generator"):
            if isinstance(state, dict) and isinstance(state.get(prefix), dict):
                state = state[prefix]
                break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint object in {path}")
        canonical = {}
        for key, value in state.items():
            for prefix in ("pipe.dit.", "dit.", "transformer.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            canonical[key] = value
        result = self.transformer.load_state_dict(canonical, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint/model mismatch: missing={result.missing_keys[:8]}, "
                f"unexpected={result.unexpected_keys[:8]}"
            )

    def reset_nonpersistent_state(self) -> None:
        """Recreate RoPE tables after meta-device model materialization."""
        head_dim = self.transformer.dim // self.spec["num_heads"]
        device = self.transformer.patch_embedding.weight.device
        self.transformer.freqs = tuple(x.to(device) for x in precompute_freqs_cis_3d(head_dim))

    def fully_shard(self, **kwargs):
        from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

        for block in self.transformer.blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)
        # The wrappers intentionally call the causal entry point directly.
        # Register it so FSDP2 still unshards/reshards the transformer root.
        register_fsdp_forward_method(self.transformer, "forward_causal")
        return self

    def preserve_conditioning(self, x: torch.Tensor, condition: Optional[Dict]) -> torch.Tensor:
        if self.concat_mask or not condition or "first_frame_cond" not in condition:
            return x
        x = x.clone()
        x[:, :, :1] = condition["first_frame_cond"][:, :, :1]
        return x

    def _replace_first_frame(self, first_frame: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.clone()
        latents[:, :, :1] = first_frame[:, :, :1]
        return latents

    def _i2v_inputs(
        self, x_t: torch.Tensor, first_frame: torch.Tensor, cur_start_frame: int = 0
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.concat_mask:
            if cur_start_frame == 0:
                x_t = self._replace_first_frame(first_frame, x_t)
            return x_t, None
        batch, _, frames, height, width = x_t.shape
        first_frame = first_frame[:, :, cur_start_frame:cur_start_frame + frames]
        if first_frame.shape[2] != frames:
            raise ValueError("Wan2.1 first_frame_cond must cover the requested latent frames")
        mask = x_t.new_zeros(batch, 4, frames, height, width)
        if cur_start_frame == 0:
            mask[:, :, 0] = 1
        return x_t, torch.cat((mask, first_frame), dim=1)

    @staticmethod
    def _scaled_timestep(t: torch.Tensor) -> torch.Tensor:
        return t * 1000.0

    def _convert_prediction(
        self, x_t: torch.Tensor, output: torch.Tensor, t: torch.Tensor, target: str
    ) -> torch.Tensor:
        if t.ndim == 2:
            t = t[:, None, :, None, None]
        return self.noise_scheduler.convert_model_output(
            x_t, output, t, src_pred_type=self.net_pred_type, target_pred_type=target
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        skip_layers: Optional[List[int]] = None,
        **fwd_kwargs,
    ) -> Prediction:
        del r, skip_layers
        if return_features_early or feature_indices:
            raise NotImplementedError("Feature extraction is not used by the released distillation methods")
        if not isinstance(condition, dict):
            raise TypeError("condition must be a dict")
        for key in ("text_embeds", "first_frame_cond"):
            if key not in condition:
                raise ValueError(f"condition must contain {key!r}")
        target = fwd_pred_type or self.net_pred_type
        if target not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported prediction type {target!r}")
        context = condition["text_embeds"]
        if isinstance(context, list):
            context = torch.stack(context)
        model_x, y = self._i2v_inputs(x_t, condition["first_frame_cond"])
        model_t = t
        if model_t.ndim == 1:
            model_t = model_t[:, None].expand(-1, x_t.shape[2])
        control = condition.get("control_hidden_states")
        if control is not None:
            control = control.detach()
        output = self.transformer.forward_causal(
            model_x,
            self._scaled_timestep(model_t),
            context,
            clip_feature=condition.get("encoder_hidden_states_image"),
            y=y,
            control_hidden_states=control,
            full_causal=False,
            use_gradient_checkpointing=self.training and not self.disable_grad_ckpt,
        )
        output = self._convert_prediction(x_t, output, t, target)
        if not self.concat_mask:
            output = self._replace_first_frame(condition["first_frame_cond"], output)
        if return_logvar:
            if not hasattr(self.transformer, "logvar_linear"):
                raise ValueError("return_logvar=True requires enable_logvar_linear=True")
            scaled = self._scaled_timestep(t)
            if scaled.ndim > 1:
                scaled = scaled.mean(dim=1)
            emb = self.transformer.time_embedding(
                __import__("diffsynth.models.wan_video_dit", fromlist=["sinusoidal_embedding_1d"])
                .sinusoidal_embedding_1d(self.transformer.freq_dim, scaled)
                .to(x_t.dtype)
            )
            return output, self.transformer.logvar_linear(emb)
        return output


class CausalWanI2V(CausalFastGenNetwork, WanI2V):
    """Block-causal Wan with full-training and append-only KV-cache paths."""

    def __init__(
        self,
        *args,
        chunk_size: int = 4,
        total_num_frames: int = 21,
        fixed_first_frame: bool = True,
        delete_cache_on_clear: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            **kwargs,
        )
        self.fixed_first_frame = bool(fixed_first_frame)
        self._delete_cache_on_clear = bool(delete_cache_on_clear)
        self._self_kv: dict[str, list[dict]] = {}
        self._cross_kv: dict[str, list[dict]] = {}

    def _tag_caches(self, tag: str) -> tuple[list[dict], list[dict]]:
        if tag not in self._self_kv:
            self._self_kv[tag] = [{} for _ in self.transformer.blocks]
            self._cross_kv[tag] = [{} for _ in self.transformer.blocks]
        return self._self_kv[tag], self._cross_kv[tag]

    def clear_caches(self, *, reuse_storage: bool = False) -> None:
        del reuse_storage
        self._self_kv.clear()
        self._cross_kv.clear()
        if self._delete_cache_on_clear and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _per_frame_t(t: torch.Tensor, batch: int, frames: int, fixed_prefix: bool) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(batch)
        if t.ndim == 1:
            t = t[:, None].expand(batch, frames)
        if t.shape != (batch, frames):
            raise ValueError(f"Expected timestep [B] or [B,F], got {tuple(t.shape)}")
        if fixed_prefix:
            t = t.clone()
            t[:, 0] = 0
        return t

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        cache_tag: str = "pos",
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = False,
        **fwd_kwargs,
    ) -> Prediction:
        del r
        if return_features_early or feature_indices:
            raise NotImplementedError("Feature extraction is not used by the released distillation methods")
        if not isinstance(condition, dict):
            raise TypeError("condition must be a dict")
        target = fwd_pred_type or self.net_pred_type
        if target not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported prediction type {target!r}")
        first_frame = condition["first_frame_cond"]
        fixed_prefix = self.fixed_first_frame and cur_start_frame == 0
        model_input = x_t
        if fixed_prefix:
            model_input = self._replace_first_frame(first_frame, x_t)
        model_x, y = self._i2v_inputs(model_input, first_frame, cur_start_frame)
        per_frame_t = self._per_frame_t(t, x_t.shape[0], x_t.shape[2], fixed_prefix)
        context = condition["text_embeds"]
        if isinstance(context, list):
            context = torch.stack(context)
        control = condition.get("control_hidden_states")
        if control is not None and control.shape[2] != x_t.shape[2]:
            control = control[:, :, cur_start_frame:cur_start_frame + x_t.shape[2]]
        if control is not None:
            control = control.detach()
        self_caches = cross_caches = None
        if is_ar or store_kv:
            self_caches, cross_caches = self._tag_caches(cache_tag)
        output = self.transformer.forward_causal(
            model_x,
            self._scaled_timestep(per_frame_t),
            context,
            clip_feature=condition.get("encoder_hidden_states_image"),
            y=y,
            control_hidden_states=control,
            chunk_size=self.chunk_size,
            fixed_first_frame=self.fixed_first_frame,
            start_frame=cur_start_frame,
            self_kv_caches=self_caches,
            cross_kv_caches=cross_caches,
            store_kv=store_kv,
            full_causal=not is_ar,
            use_gradient_checkpointing=(
                self.training
                and not store_kv
                and not self.disable_grad_ckpt
                and self.use_fsdp_checkpoint
            ),
        )
        output = self._convert_prediction(x_t, output, t, target)
        if fixed_prefix:
            output = output.clone()
            output[:, :, :1] = first_frame[:, :, :1] if target == "x0" else 0
        if return_logvar:
            if not hasattr(self.transformer, "logvar_linear"):
                raise ValueError("return_logvar=True requires enable_logvar_linear=True")
            mean_t = self._scaled_timestep(per_frame_t).mean(dim=1)
            from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

            emb = self.transformer.time_embedding(
                sinusoidal_embedding_1d(self.transformer.freq_dim, mean_t).to(x_t.dtype)
            )
            return output, self.transformer.logvar_linear(emb)
        return output

    @staticmethod
    def _chunk_ranges(num_frames: int, chunk_size: int, fixed_first_frame: bool):
        start = 0
        if fixed_first_frame:
            yield 0, 1
            start = 1
        while start < num_frames:
            end = min(start + chunk_size, num_frames)
            yield start, end
            start = end

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 1.0,
        sample_steps: int = 4,
        num_steps: Optional[int] = None,
        t_list: Optional[torch.Tensor | list[float]] = None,
        shift: float = 1.0,
        context_noise: float = 0.0,
        return_trajectory: bool = False,
        **kwargs,
    ):
        """Autoregressive rectified-flow sampler used by ODE export and inference."""
        del context_noise, kwargs
        if condition is None:
            raise ValueError("condition is required")
        if num_steps is not None:
            sample_steps = num_steps
        if t_list is None:
            timesteps = torch.linspace(1, 0, sample_steps + 1, device=noise.device, dtype=torch.float32)
            if shift != 1:
                timesteps = timesteps * shift / (1 + (shift - 1) * timesteps)
        else:
            timesteps = torch.as_tensor(t_list, device=noise.device, dtype=torch.float32)
            if timesteps.ndim != 1 or timesteps.numel() < 2:
                raise ValueError("t_list must contain at least two scalar timesteps")
        scale = 1.0 if guidance_scale is None else float(guidance_scale)
        if scale != 1 and neg_condition is None:
            raise ValueError("neg_condition is required when guidance_scale != 1")
        x = noise.clone()
        if self.fixed_first_frame:
            x[:, :, :1] = condition["first_frame_cond"][:, :, :1]
        trajectory = [x.clone()] if return_trajectory else None
        self.clear_caches()
        ranges = list(self._chunk_ranges(x.shape[2], self.chunk_size, self.fixed_first_frame))
        if self.fixed_first_frame:
            prefix_t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            for tag, cond in (("pos", condition), ("neg", neg_condition)):
                if cond is not None and (tag == "pos" or scale != 1):
                    self(x[:, :, :1], prefix_t, cond, cache_tag=tag, store_kv=True, is_ar=True)
            ranges = ranges[1:]
        for start, end in ranges:
            chunk = x[:, :, start:end]
            for current, next_t in zip(timesteps[:-1], timesteps[1:]):
                t = current.expand(x.shape[0]).to(x.dtype)
                positive = self(
                    chunk, t, condition, cache_tag="pos", cur_start_frame=start,
                    store_kv=False, is_ar=True,
                )
                if scale == 1:
                    flow = positive
                else:
                    negative = self(
                        chunk, t, neg_condition, cache_tag="neg", cur_start_frame=start,
                        store_kv=False, is_ar=True,
                    )
                    flow = negative + scale * (positive - negative)
                chunk = chunk + (next_t - current).to(chunk) * flow
                if trajectory is not None:
                    state = x.clone()
                    state[:, :, start:end] = chunk
                    trajectory.append(state)
            x[:, :, start:end] = chunk
            zero_t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            for tag, cond in (("pos", condition), ("neg", neg_condition)):
                if cond is not None and (tag == "pos" or scale != 1):
                    self(
                        chunk, zero_t, cond, cache_tag=tag, cur_start_frame=start,
                        store_kv=True, is_ar=True,
                    )
        self.clear_caches()
        if return_trajectory:
            return x, torch.stack(trajectory, dim=1)
        return x


__all__ = ["CausalWanI2V", "WanI2V", "WAN_MODEL_SPECS", "get_model_spec"]
