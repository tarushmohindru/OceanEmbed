"""Per-step linear warmup followed by cosine decay to ``min_lr``."""

from __future__ import annotations

import math

import torch


def warmup_cosine(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warm = max(1, warmup_epochs * steps_per_epoch)
    total = max(warm + 1, total_epochs * steps_per_epoch)
    floor = min_lr / base_lr

    def factor(step: int) -> float:
        if step < warm:
            return step / warm
        prog = min(1.0, (step - warm) / max(1, total - 1 - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
