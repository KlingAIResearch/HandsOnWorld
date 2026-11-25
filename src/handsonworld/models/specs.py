"""Canonical architecture metadata for the supported self-forcing backbones."""

from __future__ import annotations

from copy import deepcopy


DISTILLATION_MODEL_TYPES = (
    "wan2.1-i2v-14b-480p",
    "wan2.2-ti2v-5b",
)

_COMMON = {
    "patch_size": (1, 2, 2),
    "temporal_downscale": 4,
    "text_dim": 4096,
    "freq_dim": 256,
    "eps": 1e-6,
    "control_channels": 48,
}

WAN_MODEL_SPECS = {
    "wan2.1-i2v-14b-480p": {
        "model_type": "wan2.1-i2v-14b-480p",
        "model_id": "Wan-AI/Wan2.1-I2V-14B-480P",
        "diffusers_model_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "dim": 5120,
        "ffn_dim": 13824,
        "num_heads": 40,
        "num_layers": 40,
        "latent_channels": 16,
        "in_dim": 36,
        "out_dim": 16,
        "spatial_downscale": 8,
        "has_image_input": True,
        "has_image_pos_emb": False,
        "concat_mask": True,
        "use_image_encoder": True,
        "image_context_tokens": 257,
        "image_context_dim": 1280,
        "expand_timesteps": False,
        "seperated_timestep": False,
        "require_vae_embedding": True,
        "require_clip_embedding": True,
        "fuse_vae_embedding_in_latents": False,
        **_COMMON,
    },
    "wan2.2-ti2v-5b": {
        "model_type": "wan2.2-ti2v-5b",
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "diffusers_model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "dim": 3072,
        "ffn_dim": 14336,
        "num_heads": 24,
        "num_layers": 30,
        "latent_channels": 48,
        "in_dim": 48,
        "out_dim": 48,
        "spatial_downscale": 16,
        "has_image_input": False,
        "has_image_pos_emb": False,
        "concat_mask": False,
        "use_image_encoder": False,
        "image_context_tokens": None,
        "image_context_dim": None,
        "expand_timesteps": True,
        "seperated_timestep": True,
        "require_vae_embedding": False,
        "require_clip_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        **_COMMON,
    },
}

_ALIASES = {
    "Wan-AI/Wan2.1-I2V-14B-480P": "wan2.1-i2v-14b-480p",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": "wan2.1-i2v-14b-480p",
    "Wan-AI/Wan2.2-TI2V-5B": "wan2.2-ti2v-5b",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "wan2.2-ti2v-5b",
}


def get_model_spec(model_type: str) -> dict:
    """Return an isolated model spec for a supported type, ID, or local path."""
    key = _ALIASES.get(model_type, model_type)
    if key not in WAN_MODEL_SPECS:
        lowered = str(model_type).lower()
        matches = [name for alias, name in _ALIASES.items() if alias.lower() in lowered]
        if not matches:
            matches = [name for name in WAN_MODEL_SPECS if name in lowered]
        if len(set(matches)) == 1:
            key = matches[0]
    if key not in WAN_MODEL_SPECS:
        raise ValueError(
            f"Unsupported self-forcing model {model_type!r}; expected one of {DISTILLATION_MODEL_TYPES}"
        )
    spec = deepcopy(WAN_MODEL_SPECS[key])
    spec["architecture"] = {
        name: spec[name]
        for name in (
            "dim", "in_dim", "ffn_dim", "out_dim", "text_dim", "freq_dim", "eps",
            "patch_size", "num_heads", "num_layers", "has_image_input", "has_image_pos_emb",
            "seperated_timestep", "require_vae_embedding", "require_clip_embedding",
            "fuse_vae_embedding_in_latents",
        )
    }
    return spec
