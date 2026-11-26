"""Logger factory with UTC timestamps."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_TAG = "_oceanembed_handler"


class _UTCFormatter(logging.Formatter):
    converter = time.gmtime


def get_logger(name: str) -> logging.Logger:
    """Return the logger called ``name`` (configuration is done by ``configure_logging``)."""
    return logging.getLogger(name)


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger once; calling again replaces our handlers.

    Args:
        level: logging level name, e.g. ``"INFO"``.
        log_file: optional file that also receives the log lines.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()
    formatter = _UTCFormatter(_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)
    root.setLevel(level.upper())
