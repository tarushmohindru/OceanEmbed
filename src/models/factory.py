"""Model construction and CNN parameter-count matching."""

from __future__ import annotations

from torch import nn

from src.common.config import Config
from src.models.cnn_unet import CNNUNet
from src.models.vit_cnn import ViTCNN


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _vit(cfg: Config, cin: int, cout: int, hw: tuple[int, int]) -> ViTCNN:
    m = cfg["model"]
    return ViTCNN(cin, cout, hw, patch=m["patch"], dim=m["dim"], depth=m["depth"],
                  heads=m["heads"], mlp_ratio=m["mlp_ratio"], dropout=m["dropout"],
                  stem_base_channels=m["stem_base_channels"],
                  use_skip_stem=m["use_skip_stem"])  # fmt: skip


def match_cnn_base_channels(
    cfg: Config, in_channels: int, out_channels: int, grid_hw_padded: tuple[int, int],
    candidates=(32, 40, 48, 56, 64, 72, 80),
) -> int:  # fmt: skip
    """Base channels whose CNNUNet parameter count is closest to the ViTCNN's (within 10%)."""
    target = count_parameters(_vit(cfg, in_channels, out_channels, grid_hw_padded))
    best = min(candidates, key=lambda b: abs(count_parameters(CNNUNet(in_channels, out_channels, b))
                                             - target))  # fmt: skip
    n = count_parameters(CNNUNet(in_channels, out_channels, best))
    if abs(n - target) / target > 0.10:
        raise ValueError(f"no CNN base within 10% of ViT-CNN ({target} params); best {best}: {n}")
    return best


def build_model(
    cfg: Config, model_name: str, in_channels: int, out_channels: int,
    grid_hw_padded: tuple[int, int],
) -> nn.Module:  # fmt: skip
    if model_name == "vit_cnn":
        return _vit(cfg, in_channels, out_channels, grid_hw_padded)
    if model_name == "cnn_unet":
        base = cfg["cnn"]["base_channels"]
        if base == "auto":
            base = match_cnn_base_channels(cfg, in_channels, out_channels, grid_hw_padded)
        return CNNUNet(in_channels, out_channels, int(base))
    raise ValueError(f"unknown model {model_name!r}")
