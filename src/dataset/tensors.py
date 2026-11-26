"""Padding, masks, optional extra channels and anomaly base -> torch tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.common.config import Config
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.dataset.arrays import SplitArrays, load_split_numpy

log = get_logger(__name__)


@dataclass
class SplitTensors:
    split: str
    x: torch.Tensor  # (N, C_in, Hp, Wp)
    y: torch.Tensor  # (N, D, Hp, Wp) normalized absolute target, 0 where invalid
    m: torch.Tensor  # (N, D, Hp, Wp) bool
    base: torch.Tensor  # (N, D, Hp, Wp) zeros (absolute) or normalized climatology (anomaly)
    t_index: torch.Tensor  # (N,)
    H: int
    W: int
    channel_names: list[str]


def pad_multiple_for(cfg: Config, model_name: str) -> int:
    if model_name == "vit_cnn":
        return int(cfg["model"]["patch"])
    if model_name == "cnn_unet":
        from src.models.cnn_unet import CNNUNet

        return 2**CNNUNet.N_DOWN
    raise ValueError(f"unknown model {model_name!r}")


def pad_to_multiple(a: np.ndarray, multiple: int, value: float = 0.0) -> np.ndarray:
    """Pad the last two axes at the end (north rows, east columns) to the next multiple."""
    h, w = a.shape[-2:]
    hp, wp = -(-h // multiple) * multiple, -(-w // multiple) * multiple
    if (hp, wp) == (h, w):
        return a
    pad = [(0, 0)] * (a.ndim - 2) + [(0, hp - h), (0, wp - w)]
    return np.pad(a, pad, constant_values=value)


def extra_channels(cfg: Config, arrays: SplitArrays) -> tuple[np.ndarray, list[str]]:
    """Optional planes per ``extra_inputs``: (N, k, H, W) and their names."""
    n, _, h, w = arrays.X.shape
    planes, names = [], []
    ex = cfg.get("extra_inputs", {})
    if ex.get("land_mask"):
        planes.append(np.broadcast_to(arrays.ocean.astype("f4"), (n, 1, h, w)))
        names.append("ocean_mask")
    if ex.get("doy_sincos"):
        ang = 2 * np.pi * arrays.doy / 366.0
        for f, nm in ((np.sin, "doy_sin"), (np.cos, "doy_cos")):
            planes.append(np.broadcast_to(f(ang).astype("f4")[:, None, None, None], (n, 1, h, w)))
            names.append(nm)
    if ex.get("latlon"):
        lat = np.linspace(-1, 1, h, dtype="f4")[:, None]
        lon = np.linspace(-1, 1, w, dtype="f4")[None, :]
        planes.append(np.broadcast_to(np.broadcast_to(lat, (h, w)), (n, 1, h, w)))
        planes.append(np.broadcast_to(np.broadcast_to(lon, (h, w)), (n, 1, h, w)))
        names += ["lat", "lon"]
    if not planes:
        return np.zeros((n, 0, h, w), "f4"), []
    return np.concatenate(planes, axis=1).astype("f4"), names


def tensors_from_arrays(
    cfg: Config, arrays: SplitArrays, model_name: str, device: torch.device, smoke: bool = False
) -> SplitTensors:
    if smoke:
        keep = slice(0, 31)
        for f in ("t_index", "doy", "X", "Y", "clim_Y_norm"):
            setattr(arrays, f, getattr(arrays, f)[keep])
        arrays.dates = arrays.dates[keep]
        log.info("smoke mode: %s limited to %d samples", arrays.split, len(arrays.t_index))
    names = list(cfg["inputs"])
    drop = set(cfg["train"].get("drop_inputs", []))
    keep_c = [i for i, nm in enumerate(names) if nm not in drop]
    x = arrays.X[:, keep_c]
    extra, extra_names = extra_channels(cfg, arrays)
    x = np.concatenate([x, extra], axis=1)
    channel_names = [names[i] for i in keep_c] + extra_names
    log.info("%s channels: %s", arrays.split, channel_names)
    m = arrays.depth_mask[None] & np.isfinite(arrays.Y)
    y = np.nan_to_num(arrays.Y, nan=0.0)
    anomaly = cfg["train"]["target_mode"] == "anomaly"
    mult = pad_multiple_for(cfg, model_name)
    x, y = pad_to_multiple(x, mult), pad_to_multiple(y, mult)
    m = pad_to_multiple(m, mult, False)
    n, d, hp, wp = y.shape
    if anomaly:
        base_t = torch.from_numpy(pad_to_multiple(np.nan_to_num(arrays.clim_Y_norm, nan=0.0), mult))
    else:
        base_t = torch.zeros(1, 1, 1, 1).expand(n, d, hp, wp)
    assert not np.isnan(x).any() and not np.isnan(y).any()
    h, w = arrays.X.shape[-2:]
    assert not m[..., h:, :].any() and not m[..., :, w:].any(), "mask True on padding"
    resident = bool(cfg["train"].get("gpu_resident", True))
    target = device if resident else torch.device("cpu")

    def to(a):
        t = torch.as_tensor(a)
        if target.type == "cpu" and torch.cuda.is_available() and not resident:
            return t.pin_memory()
        return t.to(target)

    return SplitTensors(
        split=arrays.split,
        x=to(np.ascontiguousarray(x, dtype="f4")),
        y=to(np.ascontiguousarray(y, dtype="f4")),
        m=to(np.ascontiguousarray(m)),
        base=base_t.to(target) if anomaly else base_t.to(target),
        t_index=torch.as_tensor(arrays.t_index, dtype=torch.int64),
        H=h,
        W=w,
        channel_names=channel_names,
    )


def build_split_tensors(
    cfg: Config, paths: Paths, split: str, model_name: str, device: torch.device,
    smoke: bool = False,
) -> SplitTensors:  # fmt: skip
    arrays = load_split_numpy(cfg, paths, split, load_degc=False)
    return tensors_from_arrays(cfg, arrays, model_name, device, smoke)
