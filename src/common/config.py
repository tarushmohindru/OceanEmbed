"""Load, override and validate ``config.yaml``."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common.logging_utils import get_logger

log = get_logger(__name__)

_REQUIRED = [
    "seed", "paths", "region", "grid_res", "dates", "splits", "depths_m", "inputs",
    "channels", "products", "preprocess", "model", "train", "eval",
]  # fmt: skip
_NEW_KEY_OK_UNDER = {"extra_inputs", "train"}
_MISSING: Any = ...


class ConfigError(ValueError):
    """Raised for an invalid configuration; the message lists every problem."""


@dataclass(frozen=True, eq=False)
class Config:
    """Resolved configuration. ``raw`` is nested dicts and is read-only by convention."""

    raw: dict
    path: Path

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Return ``cfg.get("train.lr")``; ``KeyError`` if absent and no default given."""
        node: Any = self.raw
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default is _MISSING:
                raise KeyError(f"config key not found: {dotted}")
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def hash(self) -> str:
        """SHA-256 of the resolved YAML (sorted keys)."""
        text = yaml.safe_dump(self.raw, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def dump(self, path: Path) -> None:
        """Write the resolved YAML to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.raw, sort_keys=False), encoding="utf-8")


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Read ``path``, apply ``dotted.key=value`` overrides, validate, return a ``Config``."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    raw = apply_overrides(raw, overrides or [])
    validate(raw)
    return Config(raw=raw, path=path)


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Return a deep copy of ``raw`` with each ``dotted.key=value`` override applied.

    ``value`` is parsed with ``yaml.safe_load``. New top-level keys are an error; new nested keys
    are allowed only below ``extra_inputs`` and ``train``.
    """
    out = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like dotted.key=value, got {item!r}")
        key, text = item.split("=", 1)
        key = key.strip()
        value = yaml.safe_load(text)
        parts = key.split(".")
        node = out
        for i, part in enumerate(parts[:-1]):
            if not isinstance(node, dict):
                raise ConfigError(f"override {key!r}: {'.'.join(parts[:i])} is not a mapping")
            if part not in node:
                if i == 0 or parts[0] not in _NEW_KEY_OK_UNDER:
                    raise ConfigError(f"override {key!r}: unknown key {part!r}")
                node[part] = {}
            node = node[part]
        last = parts[-1]
        if not isinstance(node, dict):
            raise ConfigError(f"override {key!r}: parent is not a mapping")
        if last not in node and (len(parts) == 1 or parts[0] not in _NEW_KEY_OK_UNDER):
            raise ConfigError(f"override {key!r}: creating a new key is not allowed here")
        node[last] = value
    return out


def _is_num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _check_region(raw: dict, errors: list[str]) -> None:
    r = raw["region"]
    res = raw["grid_res"]
    if not r["lat_min"] < r["lat_max"]:
        errors.append("region: lat_min must be < lat_max")
    if not r["lon_min"] < r["lon_max"]:
        errors.append("region: lon_min must be < lon_max")
    for _name, lo, hi in (("lat", "lat_min", "lat_max"), ("lon", "lon_min", "lon_max")):
        n = (r[hi] - r[lo]) / res
        if abs(n - round(n)) > 1e-9:
            errors.append(f"region: ({hi}-{lo})/grid_res = {n} is not an integer")


def _check_dates_splits(raw: dict, errors: list[str]) -> None:
    start = pd.Timestamp(raw["dates"]["start"])
    end = pd.Timestamp(raw["dates"]["end"])
    if start > end:
        errors.append("dates: start must be <= end")
    prev_hi = None
    for name in ("train", "val", "test"):
        lo, hi = raw["splits"][name]
        if lo > hi:
            errors.append(f"splits.{name}: first year > last year")
        if lo < start.year or hi > end.year:
            errors.append(f"splits.{name}: years {lo}-{hi} outside dates range")
        if prev_hi is not None and lo <= prev_hi:
            errors.append(f"splits.{name}: must start after the previous split ends ({prev_hi})")
        prev_hi = hi
    if raw["splits"].get("gap_days", 0) < 0:
        errors.append("splits.gap_days must be >= 0")


def _check_depths(raw: dict, errors: list[str]) -> None:
    depths = raw["depths_m"]
    if any(b <= a for a, b in zip(depths, depths[1:], strict=False)):
        errors.append("depths_m must be strictly increasing")
    if depths[0] != 0:
        errors.append("depths_m: first value must be 0")
    if len(depths) != 15:
        log.warning("depths_m has %d entries (expected 15)", len(depths))
    seen: dict[Any, str] = {}
    for band, members in raw.get("depth_bands_m", {}).items():
        for d in members:
            if d not in depths:
                errors.append(f"depth_bands_m.{band}: depth {d} not in depths_m")
            if d in seen:
                errors.append(f"depth_bands_m: depth {d} is in both {seen[d]!r} and {band!r}")
            seen[d] = band


def _check_inputs(raw: dict, errors: list[str]) -> None:
    if len(set(raw["inputs"])) != len(raw["inputs"]):
        errors.append("inputs contains duplicates")
    for name in raw["inputs"]:
        ch = raw["channels"].get(name)
        if ch is None:
            errors.append(f"inputs: {name!r} has no entry in channels")
        elif ch.get("product") not in raw["products"]:
            errors.append(f"channels.{name}: product {ch.get('product')!r} not in products")


def _check_model_train(raw: dict, errors: list[str]) -> None:
    m, t = raw["model"], raw["train"]
    p = m["patch"]
    if not (isinstance(p, int) and p >= 2 and p & (p - 1) == 0):
        errors.append(f"model.patch must be a power of two >= 2, got {p!r}")
    if m["dim"] % m["heads"] != 0:
        errors.append("model.dim must be divisible by model.heads")
    for k in ("lr", "wd", "grad_lambda"):
        if not _is_num(t.get(k)):
            errors.append(f"train.{k} must be a number, got {t.get(k)!r} (write 3.0e-4, not 3e-4)")
    if t["depth_weights"] not in {"uniform", "thermocline", "variance"}:
        errors.append("train.depth_weights must be uniform|thermocline|variance")
    if t["target_mode"] not in {"absolute", "anomaly"}:
        errors.append("train.target_mode must be absolute|anomaly")
    if t["amp"] not in {"auto", "bf16", "fp16", "off"}:
        errors.append("train.amp must be auto|bf16|fp16|off")
    for name in t.get("drop_inputs", []):
        if name not in raw["inputs"]:
            errors.append(f"train.drop_inputs: {name!r} is not in inputs")
    if t["patience"] > t["epochs"]:
        errors.append("train.patience must be <= train.epochs")


def _check_eval(raw: dict, errors: list[str]) -> None:
    months = sorted(m for ms in raw["eval"]["seasons"].values() for m in ms)
    if months != list(range(1, 13)):
        errors.append("eval.seasons months must cover 1-12 exactly once")


def validate(raw: dict) -> None:
    """Check ``raw``; raise ``ConfigError`` listing ALL problems found."""
    errors = [f"missing top-level key: {k}" for k in _REQUIRED if k not in raw]
    if errors:
        raise ConfigError("\n".join(errors))
    checks = (
        _check_region, _check_dates_splits, _check_depths, _check_inputs, _check_model_train,
        _check_eval,
    )  # fmt: skip
    for check in checks:
        try:
            check(raw, errors)
        except (KeyError, TypeError, ValueError, IndexError) as e:
            errors.append(f"{check.__name__[7:]}: malformed section ({type(e).__name__}: {e})")
    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))
