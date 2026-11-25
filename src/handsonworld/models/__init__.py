"""Shared model definitions used by SFT and self-forcing."""

from .specs import DISTILLATION_MODEL_TYPES, WAN_MODEL_SPECS, get_model_spec

__all__ = ["DISTILLATION_MODEL_TYPES", "WAN_MODEL_SPECS", "get_model_spec"]
