#!/usr/bin/env python3
"""Plot parent-versus-replication LPIPS effects and matched-target coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.study.replication import DIRECTION_LABELS, replication_family

METHOD_LABELS = {
    "prompt_only": "Prompt-only",
    "stepwise_unmasked": "Unmasked",
    "stepwise_masked": "Masked",
}
STUDY_STYLES = {
    "Parent": ("#4477AA", "o"),
    "Replication": ("#AA3377", "s"),
}


def load_tables(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = replication_family(pd.read_csv(analysis_dir / "paired_contrasts.csv"))
    coverage = pd.read_csv(analysis_dir / "matched_summary.csv")
    coverage = coverage[
        (coverage["target_change"] == 5.0)
        & coverage["method"].isin(METHOD_LABELS)
        & coverage["direction"].isin(DIRECTION_LABELS)
    ].copy()
    if len(coverage) != 6:
        raise ValueError(f"Expected six matched-coverage rows, found {len(coverage)}")
    return contrasts, coverage


def plot_replication(
    parent_dir: Path,
    replication_dir: Path,
    output_dir: Path,
) -> None:
    parent_effects, parent_coverage = load_tables(parent_dir)
    replication_effects, replication_coverage = load_tables(replication_dir)
    studies = {
        "Parent": (parent_effects, parent_coverage),
        "Replication": (replication_effects, replication_coverage),
    }

    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 8})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    effect_axis = axes[0]
    base_positions = {"lighter": 1.0, "darker": 0.0}
    offsets = {"Parent": 0.10, "Replication": -0.10}
    for study, (effects, _) in studies.items():
        color, marker = STUDY_STYLES[study]
        positions = np.array(
            [base_positions[label] + offsets[study] for label in effects["direction_label"]]
        )
        means = effects["mean_reference_advantage"].to_numpy(dtype=float)
        lower = means - effects["ci_low"].to_numpy(dtype=float)
        upper = effects["ci_high"].to_numpy(dtype=float) - means
        effect_axis.errorbar(
            means,
            positions,
            xerr=np.vstack([lower, upper]),
            fmt=marker,
            color=color,
            capsize=3,
            linewidth=1.4,
            label=study,
        )
    effect_axis.axvline(0, color="#444444", linewidth=0.8)
    effect_axis.set_yticks([0, 1], ["Darker", "Lighter"])
    effect_axis.set_xlabel("Masked LPIPS advantage")
    effect_axis.set_title("(a) Locked replication family", loc="left")
    effect_axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    effect_axis.legend(frameon=False, loc="lower right")

    coverage_axis = axes[1]
    methods = list(METHOD_LABELS)
    group_centers = np.arange(len(methods), dtype=float)
    width = 0.18
    combinations = [
        ("Parent", -1, -1.5 * width, ""),
        ("Replication", -1, -0.5 * width, "//"),
        ("Parent", 1, 0.5 * width, ""),
        ("Replication", 1, 1.5 * width, "//"),
    ]
    direction_colors = {-1: "#88CCEE", 1: "#CC6677"}
    for study, direction, offset, hatch in combinations:
        _, coverage = studies[study]
        values = []
        for method in methods:
            row = coverage[
                (coverage["method"] == method) & (coverage["direction"] == direction)
            ]
            values.append(float(row.iloc[0]["coverage_rate"]))
        coverage_axis.bar(
            group_centers + offset,
            values,
            width=width,
            color=direction_colors[direction],
            edgecolor="#333333",
            linewidth=0.5,
            hatch=hatch,
        )
    coverage_axis.set_xticks(
        group_centers, [METHOD_LABELS[method] for method in methods], rotation=15
    )
    coverage_axis.set_ylim(0, 1.03)
    coverage_axis.set_ylabel("Matched-target coverage")
    coverage_axis.set_title("(b) Coverage by campaign", loc="left")
    coverage_axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    coverage_axis.text(
        0.02,
        0.98,
        "Blue = lighter, red = darker; hatch = replication",
        transform=coverage_axis.transAxes,
        va="top",
        fontsize=7,
    )

    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"replication_comparison.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_analysis", type=Path)
    parser.add_argument("replication_analysis", type=Path)
    parser.add_argument("--output", type=Path, default=Path("paper/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_replication(args.parent_analysis, args.replication_analysis, args.output)


if __name__ == "__main__":
    main()
