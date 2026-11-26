"""Shared fixtures. All tests are deterministic and write only inside tmp_path."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.common.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The repo's config.yaml (validated on load)."""
    return load_config(REPO_ROOT / "config.yaml")


@pytest.fixture
def tiny_cfg(cfg, tmp_path) -> Config:
    """``cfg`` with data and results roots inside ``tmp_path``."""
    raw = copy.deepcopy(cfg.raw)
    raw["paths"]["data_root"] = str(tmp_path / "data")
    raw["paths"]["results_root"] = str(tmp_path / "results")
    return Config(raw=raw, path=cfg.path)


@pytest.fixture(scope="session")
def synthetic_processed(tmp_path_factory, cfg) -> Config:
    """Config pointing at a synthetic ``data/processed`` (3 years, 24x32 grid, 15 depths)."""
    from src.dataset.synthetic import make_synthetic_processed  # lazy: built in a later stage

    return make_synthetic_processed(tmp_path_factory.mktemp("synthetic"), cfg)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
