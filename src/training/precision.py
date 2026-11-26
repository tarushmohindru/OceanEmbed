"""Mixed-precision helpers (bf16 where supported, else fp16 + GradScaler; off on CPU)."""

from __future__ import annotations

import contextlib

import torch

from src.common.config import Config


def resolve_amp(cfg: Config, device: torch.device) -> torch.dtype | None:
    mode = cfg["train"]["amp"]
    if mode == "off" or device.type != "cuda":
        return None
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but not supported on this GPU")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_ctx(dtype: torch.dtype | None, device: torch.device):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


class _NoScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


def make_scaler(dtype: torch.dtype | None):
    if dtype == torch.float16:
        return torch.amp.GradScaler("cuda")
    return _NoScaler()
