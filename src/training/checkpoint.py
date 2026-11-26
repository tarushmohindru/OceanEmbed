"""Atomic checkpoint save/load including RNG states."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from src.common.gitinfo import git_hash
from src.common.io import atomic_write_path

_REPO = Path(__file__).resolve().parents[2]


def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, epoch: int,
                    best_val: float, cfg_raw: dict, extra: dict) -> None:  # fmt: skip
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "cfg": cfg_raw,
        "git_hash": git_hash(_REPO),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),  # noqa: NPY002
        "python_rng": random.getstate(),
        **extra,
    }
    with atomic_write_path(Path(path)) as tmp:
        torch.save(state, tmp)


def load_checkpoint(path: Path, map_location) -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])  # noqa: NPY002
    random.setstate(state["python_rng"])
