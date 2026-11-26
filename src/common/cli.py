"""Argument-parsing helpers shared by every entry point."""

from __future__ import annotations

import argparse

from src.common.config import Config, load_config
from src.common.logging_utils import configure_logging


def base_parser(description: str) -> argparse.ArgumentParser:
    """Parser with ``--config``, repeatable ``--set key=value``, ``--force``, ``--log-level``."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.lr=1.0e-4 (repeatable)",
    )  # fmt: skip
    p.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ...")
    return p


def parse_and_load(parser: argparse.ArgumentParser) -> tuple[argparse.Namespace, Config]:
    """Parse ``sys.argv``, configure logging, and load the (overridden) config."""
    args = parser.parse_args()
    configure_logging(args.log_level)
    cfg = load_config(args.config, args.set)
    return args, cfg
