#!/usr/bin/env python3
"""Recover individual panels from the historical Matplotlib strip.

This is an explicit provenance conversion, not regeneration. Metrics computed
from these crops remain a one-seed legacy re-analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return list(zip(changes[::2], changes[1::2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("strip", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    args = parser.parse_args()

    image = Image.open(args.strip).convert("RGB")
    array = np.asarray(image)
    nonwhite = np.any(array < 245, axis=2)
    row_candidates = nonwhite.mean(axis=1) > 0.55
    row_runs = runs(row_candidates)
    if not row_runs:
        raise SystemExit("Could not locate the image panel row")
    y1, y2 = max(row_runs, key=lambda item: item[1] - item[0])
    column_candidates = nonwhite[y1:y2].mean(axis=0) > 0.80
    panels = [item for item in runs(column_candidates) if item[1] - item[0] > 100]
    if len(panels) != len(args.alphas):
        raise SystemExit(f"Found {len(panels)} panels but received {len(args.alphas)} alphas")

    counterfactuals = args.output / "counterfactuals"
    counterfactuals.mkdir(parents=True, exist_ok=True)
    generated = []
    common_width = min(x2 - x1 for x1, x2 in panels)
    common_height = y2 - y1
    for alpha, (x1, x2) in zip(args.alphas, panels):
        panel = image.crop((x1, y1, x2, y2))
        if panel.width != common_width:
            left = (panel.width - common_width) // 2
            panel = panel.crop((left, 0, left + common_width, common_height))
        path = counterfactuals / f"alpha_{alpha:+.1f}.png"
        panel.save(path)
        generated.append(str(path))
        if alpha == 0:
            panel.save(args.output / "base_image.png")
    if not (args.output / "base_image.png").exists():
        raise SystemExit("The alpha list must include 0 to establish the crop-matched base")

    manifest = {
        "schema_version": "1.0",
        "source": str(args.strip),
        "source_sha256": hashlib.sha256(args.strip.read_bytes()).hexdigest(),
        "conversion": "threshold-detected Matplotlib panel crops",
        "crop_y": [int(y1), int(y2)],
        "crop_x": [[int(x1), int(x2)] for x1, x2 in panels],
        "standardized_panel_size": [int(common_width), int(common_height)],
        "alphas": args.alphas,
        "generated": generated,
        "limitation": "Derived from a composite PNG; use only as legacy pilot re-analysis.",
    }
    (args.output / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
