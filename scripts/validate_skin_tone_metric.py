#!/usr/bin/env python3
"""Validate the preregistered skin-tone proxy on held-out paired images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.metrics.skin_tone import SkinToneMetrics
from src.study.config import StudyConfigError, load_study_config
from src.study.measurement_validation import apply_rgb_gain, validation_summary


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_held_out_pairs(config) -> list[dict]:
    root = config.path.parent.parent
    manifest_path = resolve(root, config.data["training_manifest"])
    if not manifest_path.exists():
        raise StudyConfigError(f"Training manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    start = int(config.direction["train_pairs"])
    stop = start + int(config.direction["held_out_pairs"])
    pairs = manifest.get("pairs", [])[start:stop]
    if len(pairs) != stop - start:
        raise StudyConfigError(f"Expected {stop - start} held-out pairs, found {len(pairs)}")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/measurement_validation")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    root = config.path.parent.parent
    metric = SkinToneMetrics()
    pair_rows = []
    perturbation_rows = []
    perturbations = config.raw["measurement_validation"]["perturbations"]

    for item in load_held_out_pairs(config):
        images = {
            group: Image.open(resolve(root, item[group]["path"])).convert("RGB")
            for group in ("light", "dark")
        }
        measures = {group: metric.measure(image) for group, image in images.items()}
        complete = all(value is not None for value in measures.values())
        gap = (
            measures["light"].ita_degrees - measures["dark"].ita_degrees
            if complete
            else None
        )
        pair_rows.append(
            {
                "pair_id": item["pair_id"],
                "seed": item["seed"],
                "light_detected": measures["light"] is not None,
                "dark_detected": measures["dark"] is not None,
                "pair_complete": complete,
                "light_ita": measures["light"].ita_degrees if measures["light"] else None,
                "dark_ita": measures["dark"].ita_degrees if measures["dark"] else None,
                "ita_gap": gap,
                "ordered_light_above_dark": bool(gap is not None and gap > 0),
            }
        )
        for group, image in images.items():
            baseline = measures[group]
            if baseline is None:
                continue
            for name, spec in perturbations.items():
                changed = apply_rgb_gain(image, spec["rgb_gain"])
                shifted = metric.measure(changed, face_bbox=baseline.face_bbox)
                perturbation_rows.append(
                    {
                        "pair_id": item["pair_id"],
                        "seed": item["seed"],
                        "group": group,
                        "perturbation": name,
                        "baseline_ita": baseline.ita_degrees,
                        "perturbed_ita": shifted.ita_degrees if shifted else None,
                        "abs_ita_shift": (
                            abs(shifted.ita_degrees - baseline.ita_degrees)
                            if shifted
                            else None
                        ),
                    }
                )

    pair_frame = pd.DataFrame(pair_rows)
    perturbation_frame = pd.DataFrame(perturbation_rows)
    report = validation_summary(
        pair_frame,
        perturbation_frame,
        config.raw["measurement_validation"]["gates"],
    )
    report.update(
        {
            "schema_version": "1.0",
            "study_id": config.study_id,
            "config_fingerprint": config.fingerprint,
            "metric": config.raw["measurement_validation"]["metric"],
            "split": "held_out_direction_pairs",
        }
    )

    args.output.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(args.output / "paired_measurements.csv", index=False)
    perturbation_frame.to_csv(args.output / "illumination_sensitivity.csv", index=False)
    report_path = args.output / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))
    print(f"validation_report_sha256={digest}")
    if not report["passed"]:
        raise SystemExit("Measurement validation failed one or more preregistered gates")


if __name__ == "__main__":
    main()
