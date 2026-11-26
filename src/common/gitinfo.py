"""Git provenance helpers. Never raise."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
        )
    except (OSError, ValueError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip()


def git_hash(repo_root: Path) -> str:
    """Return the HEAD commit hash, with ``-dirty`` appended if the tree has changes.

    Returns ``"nogit"`` if ``repo_root`` is not a repository, has no commit, or git is missing.
    """
    head = _git(Path(repo_root), "rev-parse", "HEAD")
    if not head:
        return "nogit"
    status = _git(Path(repo_root), "status", "--porcelain")
    return head + ("-dirty" if status else "")
