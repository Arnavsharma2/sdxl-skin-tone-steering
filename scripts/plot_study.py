#!/usr/bin/env python3
"""Render publication-ready figures from the frozen study analysis tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")
plt.rcParams.update({"font.size": 7})


METHOD_LABELS = {
    "prompt_only": "Prompt-only",
    "posthoc_latent": "Post-hoc latent",
    "stepwise_unmasked": "Stepwise, unmasked",
    "stepwise_masked": "Stepwise, masked",
}
METHOD_COLORS = {
    "prompt_only": "#4477AA",
    "posthoc_latent": "#EE6677",
    "stepwise_unmasked": "#228833",
    "stepwise_masked": "#AA3377",
}
OUTCOME_LABELS = {
    "face_similarity": "Face similarity",
    "lpips": "LPIPS",
    "background_ssim": "Background SSIM",
    "total_pose_diff": "Pose difference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis",
        type=Path,
        help="Directory produced by scripts/analyze_study.py",
    )
    parser.add_argument("--output", type=Path, default=Path("paper/figures"))
    return parser.parse_args()


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_dose_response(descriptive: pd.DataFrame, output: Path) -> None:
    dose = descriptive[descriptive["metric"] == "skin_tone_change"].copy()
    methods = [method for method in METHOD_LABELS if method in set(dose["method"])]
    if not methods:
        raise ValueError("descriptive.csv has no skin_tone_change rows")
    fig, axes = plt.subplots(1, len(methods), figsize=(7.2, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods, strict=True):
        group = dose[dose["method"] == method].sort_values("alpha")
        means = group["mean"].to_numpy(dtype=float)
        lows = group["ci_low"].to_numpy(dtype=float)
        highs = group["ci_high"].to_numpy(dtype=float)
        errors = np.vstack([means - lows, highs - means])
        ax.errorbar(
            group["alpha"],
            means,
            yerr=errors,
            marker="o",
            color=METHOD_COLORS[method],
            capsize=3,
            linewidth=1.6,
        )
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.axhline(5, color="#999999", linewidth=0.8, linestyle="--")
        ax.axhline(-5, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_title(METHOD_LABELS[method], fontsize=8)
        ax.set_xlabel(r"Steering value $\alpha$")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Skin-tone change (ITA degrees)\npositive = darker")
    fig.suptitle("Confirmatory dose-response curves (mean and 95% seed bootstrap CI)", fontsize=9)
    fig.tight_layout()
    _save(fig, output, "confirmatory_dose_response")


def plot_matched_contrasts(contrasts: pd.DataFrame, output: Path) -> None:
    if contrasts.empty:
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No matched-change contrast had sufficient complete pairs.",
            ha="center",
            va="center",
        )
        _save(fig, output, "confirmatory_matched_contrasts")
        return
    outcomes = [outcome for outcome in OUTCOME_LABELS if outcome in set(contrasts["outcome"])]
    if not outcomes:
        raise ValueError("paired_contrasts.csv has no recognized preservation outcomes")
    fig, axes = plt.subplots(1, len(outcomes), figsize=(7.2, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    comparators = [
        method
        for method in ("prompt_only", "stepwise_unmasked")
        if method in set(contrasts["comparator"])
    ]
    row_keys = [(method, direction) for method in comparators for direction in (-1, 1)]
    y_positions = np.arange(len(row_keys))[::-1]
    row_labels = [
        f"{METHOD_LABELS[method]} - {'lighter' if direction < 0 else 'darker'}"
        for method, direction in row_keys
    ]
    for ax, outcome in zip(axes, outcomes, strict=True):
        subset = contrasts[contrasts["outcome"] == outcome]
        for y, (method, direction) in zip(y_positions, row_keys, strict=True):
            row = subset[
                (subset["comparator"] == method) & (subset["direction"] == direction)
            ]
            if row.empty:
                continue
            point = row.iloc[0]
            mean = float(point["mean_reference_advantage"])
            low = float(point["ci_low"])
            high = float(point["ci_high"])
            ax.errorbar(
                mean,
                y,
                xerr=np.array([[mean - low], [high - mean]]),
                marker="o",
                color=METHOD_COLORS[method],
                capsize=3,
                linewidth=1.6,
            )
        ax.axvline(0, color="#555555", linewidth=0.9)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=8)
        ax.set_xlabel("Advantage")
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_yticks(y_positions)
    axes[0].set_yticks(y_positions, row_labels)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Preservation at matched +/-5 ITA change (mean paired advantage, 95% CI)", fontsize=9)
    fig.tight_layout()
    _save(fig, output, "confirmatory_matched_contrasts")


def plot_completion(missingness: pd.DataFrame, output: Path) -> None:
    summary = (
        missingness.groupby("method", as_index=False)[["attempted", "generated", "evaluation_complete"]]
        .sum()
        .assign(
            generation_rate=lambda frame: frame["generated"] / frame["attempted"],
            evaluation_rate=lambda frame: frame["evaluation_complete"] / frame["attempted"],
        )
    )
    methods = [method for method in METHOD_LABELS if method in set(summary["method"])]
    summary = summary.set_index("method").loc[methods]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.bar(x - width / 2, summary["generation_rate"], width, label="Generated", color="#66CCEE")
    ax.bar(x + width / 2, summary["evaluation_rate"], width, label="Metric-complete", color="#228833")
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Condition completion rate")
    ax.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=15, ha="right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, output, "confirmatory_completion")


def main() -> None:
    args = parse_args()
    descriptive = pd.read_csv(args.analysis / "descriptive.csv")
    missingness = pd.read_csv(args.analysis / "missingness.csv")
    try:
        contrasts = pd.read_csv(args.analysis / "paired_contrasts.csv")
    except pd.errors.EmptyDataError:
        contrasts = pd.DataFrame()
    plot_dose_response(descriptive, args.output)
    plot_matched_contrasts(contrasts, args.output)
    plot_completion(missingness, args.output)
    print(f"Wrote publication figures to {args.output}")


if __name__ == "__main__":
    main()
