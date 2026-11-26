"""Normalization statistics and (de)normalization. The only implementation in the project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.config import Config


@dataclass(frozen=True, eq=False)
class NormStats:
    """Per-channel input and per-depth target mean/std, in config order."""

    in_mean: np.ndarray  # (C,)
    in_std: np.ndarray  # (C,)
    out_mean: np.ndarray  # (D,)
    out_std: np.ndarray  # (D,)
    years_used: list[int]


def load_norm_stats(path: Path, cfg: Config) -> NormStats:
    """Read ``norm_stats.json``; fail if channel or depth keys differ from the config."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    inputs = list(cfg["inputs"])
    depth_keys = [str(int(z)) for z in cfg["depths_m"]]
    if set(d["inputs"]) != set(inputs):
        raise ValueError(f"norm_stats inputs {sorted(d['inputs'])} != config {sorted(inputs)}")
    if set(d["targets"]) != set(depth_keys):
        raise ValueError(f"norm_stats depths {sorted(d['targets'])} != config {depth_keys}")
    f32 = np.float32
    return NormStats(
        in_mean=np.array([d["inputs"][c]["mean"] for c in inputs], dtype=f32),
        in_std=np.array([d["inputs"][c]["std"] for c in inputs], dtype=f32),
        out_mean=np.array([d["targets"][k]["mean"] for k in depth_keys], dtype=f32),
        out_std=np.array([d["targets"][k]["std"] for k in depth_keys], dtype=f32),
        years_used=[int(y) for y in d["years_used"]],
    )


def _bcast(vec: np.ndarray, x, axis: int):
    """Reshape ``vec`` to broadcast along ``axis`` of ``x`` (numpy array or torch tensor)."""
    axis = axis % x.ndim
    if vec.shape[0] != x.shape[axis]:
        raise ValueError(f"axis {axis} has size {x.shape[axis]}, stats have {vec.shape[0]}")
    shape = [1] * x.ndim
    shape[axis] = vec.shape[0]
    if type(x).__module__.split(".")[0] == "torch":
        import torch

        return torch.as_tensor(vec, dtype=x.dtype, device=x.device).reshape(shape)
    return vec.reshape(shape).astype(x.dtype, copy=False)


def normalize_x(x, s: NormStats, channel_axis: int):
    """``(x - mean) / std`` along the channel axis."""
    return (x - _bcast(s.in_mean, x, channel_axis)) / _bcast(s.in_std, x, channel_axis)


def normalize_y(y, s: NormStats, depth_axis: int):
    """``(y - mean) / std`` along the depth axis."""
    return (y - _bcast(s.out_mean, y, depth_axis)) / _bcast(s.out_std, y, depth_axis)


def denormalize_y(y, s: NormStats, depth_axis: int):
    """``y * std + mean`` along the depth axis (back to degrees C)."""
    return y * _bcast(s.out_std, y, depth_axis) + _bcast(s.out_mean, y, depth_axis)
