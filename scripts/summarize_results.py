#!/usr/bin/env python3
"""Aggregate run metadata and compute seed-level bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

METRICS = (
    "face_similarity",
    "landmark_rmse",
    "lpips",
    "background_ssim",
    "overall_ssim",
    "total_pose_diff",
    "overall_score",
    "skin_tone_change",
    "skin_delta_ita",
    "skin_delta_e",
)


def load_runs(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(root.rglob("metadata.json")):
        with path.open(encoding="utf-8") as handle:
            run = json.load(handle)
        for result in run.get("results", []):
            rows.append(
                {
                    "run": str(path.parent),
                    "seed": run.get("seed"),
                    "method": result.get("method", run.get("method", "stepwise_masked")),
                    "alpha": result.get("alpha"),
                    "evaluation_complete": result.get("evaluation_complete", False),
                    "counterfactual_success": result.get("counterfactual_success", False),
                    **{metric: result.get(metric) for metric in METRICS},
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return tuple(np.quantile(means, [tail, 1.0 - tail]).tolist())


def summarize(df: pd.DataFrame, resamples: int = 10_000) -> pd.DataFrame:
    records: list[dict] = []
    if df.empty:
        return pd.DataFrame()
    for (method, alpha), group in df.groupby(["method", "alpha"], dropna=False):
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            values = values[np.isfinite(values)]
            low, high = bootstrap_mean_ci(values, resamples=resamples)
            records.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Directory containing run metadata.json files")
    parser.add_argument("--output", type=Path, default=Path("experiments/summary"))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    long_df = load_runs(args.input)
    if long_df.empty:
        raise SystemExit(f"No result rows found under {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarize(long_df, resamples=args.resamples)
    long_df.to_csv(args.output / "results_long.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)

    audit = {
        "metadata_files": int(long_df["run"].nunique()),
        "rows": int(len(long_df)),
        "complete_rows": int(long_df["evaluation_complete"].fillna(False).sum()),
        "successful_rows": int(long_df["counterfactual_success"].fillna(False).sum()),
        "bootstrap_resamples": args.resamples,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Wrote {args.output / 'results_long.csv'}")
    print(f"Wrote {args.output / 'summary.csv'}")
    print(f"Complete evaluations: {audit['complete_rows']}/{audit['rows']}")


if __name__ == "__main__":
    main()
