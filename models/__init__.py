from __future__ import annotations

from torch import nn

from .attention_unet import AttentionUNet
from .proposed_model import LightweightTransformerAttentionUNet
from .transformer_unet import TransformerUNet
from .unet import ConfigurableUNet


def build_model(config: dict) -> nn.Module:
    cfg = dict(config)
    name = str(cfg.pop("name")).lower()
    if name == "unet":
        return ConfigurableUNet(use_attention=False, use_transformer=False, **cfg)
    if name == "attention_unet":
        return AttentionUNet(**cfg)
    if name == "transformer_unet":
        return TransformerUNet(**cfg)
    if name in {"proposed", "lightweight_transformer_attention_unet"}:
        return LightweightTransformerAttentionUNet(**cfg)
    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "build_model",
    "ConfigurableUNet",
    "AttentionUNet",
    "TransformerUNet",
    "LightweightTransformerAttentionUNet",
]
