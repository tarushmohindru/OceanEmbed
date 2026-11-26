"""Masked, depth-weighted MSE plus a horizontal-gradient term. Tensors are (B, D, Hp, Wp)."""

from __future__ import annotations

import torch

from src.common.config import Config
from src.common.normalization import NormStats


def depth_weights(cfg: Config, norm_stats: NormStats) -> torch.Tensor:
    """(D,) weights with mean 1 for ``train.depth_weights`` = uniform | thermocline | variance."""
    mode = cfg["train"]["depth_weights"]
    z = torch.tensor(cfg["depths_m"], dtype=torch.float32)
    if mode == "uniform":
        w = torch.ones_like(z)
    elif mode == "thermocline":
        w = torch.where((z >= 50) & (z <= 200), float(cfg["train"]["thermocline_weight"]), 1.0)
    elif mode == "variance":
        w = torch.as_tensor(norm_stats.out_std, dtype=torch.float32) ** 2
    else:
        raise ValueError(f"unknown depth_weights {mode!r}")
    return w / w.mean()


def masked_mse_per_depth(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(D,) sum over (B, H, W) of m * (pred - y)^2 / max(sum m, 1)."""
    mf = m.to(pred.dtype)
    se = (mf * (pred - y) ** 2).sum(dim=(0, 2, 3))
    return se / mf.sum(dim=(0, 2, 3)).clamp_min(1.0)


def gradient_loss_per_depth(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(D,) mean absolute error of forward differences along lon and lat, over valid pairs.

    Grid spacing differences with latitude (cos 5-23 deg = 0.996-0.92) are ignored.
    """
    total = 0.0
    for dim in (-1, -2):
        n = pred.shape[dim]

        def hi(t: torch.Tensor, dim=dim, n=n) -> torch.Tensor:
            return t.narrow(dim, 1, n - 1)

        def lo(t: torch.Tensor, dim=dim, n=n) -> torch.Tensor:
            return t.narrow(dim, 0, n - 1)

        mm = (hi(m) & lo(m)).to(pred.dtype)
        diff = (hi(pred) - lo(pred)) - (hi(y) - lo(y))
        total = total + (mm * diff.abs()).sum(dim=(0, 2, 3)) / mm.sum(dim=(0, 2, 3)).clamp_min(1.0)
    return total


def total_loss(
    pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor, w: torch.Tensor, lam: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = sum_d w_d mse_d + lam * sum_d grad_d (computed in float32)."""
    pred, y = pred.float(), y.float()
    mse = masked_mse_per_depth(pred, y, m)
    grad = gradient_loss_per_depth(pred, y, m) if lam else torch.zeros_like(mse)
    loss = (w.to(mse.device) * mse).sum() + lam * grad.sum()
    return loss, {"loss": float(loss.detach()), "mse": float(mse.detach().mean()),
                  "grad": float(grad.detach().mean())}  # fmt: skip
