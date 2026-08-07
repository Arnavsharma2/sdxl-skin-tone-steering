"""Helpers for deterministic runs and machine-readable provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without silently forcing algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for_index(index: int, seeds: list[int]) -> int:
    """Extend a finite seed list without repeating generated samples."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    cycle, offset = divmod(index, len(seeds))
    return seeds[offset] + cycle * 10_000


def stable_fingerprint(values: Mapping[str, Any], length: int = 12) -> str:
    """Return a stable short hash suitable for cache and run identifiers."""
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_provenance(project_root: str | Path = ".") -> dict[str, Any]:
    """Collect the minimum provenance needed to interpret an experiment."""
    root = Path(project_root).resolve()
    git_status = _git_value(root, "status", "--porcelain")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _version(name)
            for name in (
                "torch",
                "diffusers",
                "transformers",
                "numpy",
                "scikit-image",
                "facenet-pytorch",
                "mediapipe",
                "lpips",
            )
        },
        "git": {
            "commit": _git_value(root, "rev-parse", "HEAD"),
            "branch": _git_value(root, "branch", "--show-current"),
            "dirty": bool(git_status) if git_status is not None else None,
        },
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }
