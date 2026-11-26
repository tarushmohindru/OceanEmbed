"""Streaming per-depth RMSE in degC (same definition as src/evaluation/metrics.py)."""

from __future__ import annotations

import numpy as np
import torch


class StreamingDepthRMSE:
    def __init__(self, out_std: torch.Tensor):
        self.out_std = torch.as_tensor(out_std, dtype=torch.float64)
        self.se = torch.zeros_like(self.out_std)
        self.n = torch.zeros_like(self.out_std)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> None:
        mf = m.to(torch.float64)
        err = (pred.double() - y.double()) ** 2 * mf
        self.se += err.sum(dim=(0, 2, 3)).cpu()
        self.n += mf.sum(dim=(0, 2, 3)).cpu()

    def compute(self) -> dict:
        rmse = torch.sqrt(self.se / self.n.clamp_min(1.0)) * self.out_std
        rmse = torch.where(self.n > 0, rmse, torch.full_like(rmse, float("nan")))
        r = rmse.numpy()
        return {"rmse_degC": r, "mean_rmse_degC": float(np.nanmean(r))}
