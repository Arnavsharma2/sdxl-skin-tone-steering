#!/usr/bin/env python3
"""Build fail-closed confirmatory statistical and audit outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.analysis.statistics import analyze_comparisons, seed_cluster_bootstrap_ci
from src.metrics.protocol import protocol_record
from src.utils.config import ExperimentConfig

ANALYSIS_OUTPUT_SCHEMA_VERSION = "1.0"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def analysis_code_provenance() -> dict[str, Any]:
    project_root = Path(__file__).parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"analysis_code_commit": commit, "analysis_code_dirty": dirty}


def _success_value(result: dict) -> tuple[object, str]:
    """Read the current success field or its legacy compatibility alias."""
    if "counterfactual_success" in result:
        return result["counterfactual_success"], "counterfactual_success"
    if "is_disentangled" in result:
        return result["is_disentangled"], "legacy_is_disentangled"
    return None, "unreported"


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    """Accept flat rows and the documented nested evaluation compatibility form."""
    flattened = dict(result)
    for container in ("metrics", "evaluation"):
        nested = result.get(container)
        if isinstance(nested, dict):
            for key, value in nested.items():
                flattened.setdefault(key, value)
    return flattened


def _explicit_reasons(result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    raw = result.get("failure_reasons", [])
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            raw = decoded if isinstance(decoded, list) else [raw]
        except json.JSONDecodeError:
            raw = [raw]
    if isinstance(raw, (list, tuple)):
        reasons.extend(str(reason) for reason in raw if reason)
    failure = result.get("failure")
    if isinstance(failure, dict):
        stage = str(failure.get("stage", "unknown"))
        exception_type = str(failure.get("exception_type", "unknown"))
        reasons.append(f"failure:{stage}:{exception_type}")
    missing = result.get("missing_required_metrics")
    if isinstance(missing, (list, tuple)):
        reasons.extend(f"metric_missing:{metric}" for metric in missing)
    return reasons


def _normalise_planned_result(
    planned: dict[str, Any],
    result: dict[str, Any] | None,
    config: ExperimentConfig,
    *,
    integrity_reasons: Iterable[str] = (),
) -> dict[str, Any]:
    source = _flatten_result(result or {})
    row = {**planned, **source}
    reasons = [*_explicit_reasons(source), *integrity_reasons]
    status = row.get("status")
    if result is None:
        reasons.append("result_row_missing")
    if status != "complete":
        if status in {"failed", "setup_failed", "partial_failed"}:
            reasons.append(f"generation_failure:status:{status}")
        else:
            reasons.append(f"generation_unavailable:status:{status or 'missing'}")
    if status == "complete":
        if not row.get("image_sha256"):
            reasons.append("file_integrity_failure:missing_image_sha256")
        if not row.get("base_image_sha256"):
            reasons.append("file_integrity_failure:missing_base_image_sha256")

    for metric in config.evaluation.required_metrics:
        value = row.get(metric)
        if metric == "target_direction_correct":
            missing = value is None
        else:
            try:
                missing = not np.isfinite(float(value))
            except (TypeError, ValueError):
                missing = True
        if missing:
            reasons.append(f"metric_missing:{metric}")

    if source.get("face_detection_failed") is True:
        reasons.append("face_detection_failure")
    reasons = sorted(set(reasons))
    generation_valid = status == "complete" and not any(
        reason.startswith(("generation_failure:", "generation_unavailable:")) for reason in reasons
    )
    integrity_valid = not any(reason.startswith("file_integrity_failure:") for reason in reasons)
    required_metric_valid = not any(reason.startswith("metric_missing:") for reason in reasons)
    reported_complete = source.get("evaluation_complete") is True
    row.update(
        {
            "failure_reasons": json.dumps(reasons, separators=(",", ":")),
            "failure_reason_count": len(reasons),
            "face_detection_failure": any("face_detection_failure" in reason for reason in reasons),
            "generation_failure": any(
                reason.startswith("generation_failure:") for reason in reasons
            ),
            "file_integrity_failure": any(
                reason.startswith("file_integrity_failure:") for reason in reasons
            ),
            "analysis_row_valid": bool(
                generation_valid and integrity_valid and required_metric_valid and reported_complete
            ),
        }
    )
    success, success_source = _success_value(source)
    row["counterfactual_success"] = success
    row["counterfactual_success_source"] = success_source
    for metric in METRICS:
        row.setdefault(metric, None)
    row.setdefault("evaluation_complete", False)
    return row


def _manifest_row_key(row: dict[str, Any]) -> tuple[int, str, float]:
    return int(row["seed"]), str(row["method"]), float(row["alpha"])


def _verified_file(
    path: Path, expected_sha256: object, cache: dict[Path, tuple[str | None, str | None]]
) -> tuple[str | None, str | None]:
    if path in cache:
        status, actual = cache[path]
        if status is not None:
            return status, actual
        return (
            None if isinstance(expected_sha256, str) and actual == expected_sha256 else "mismatch",
            actual,
        )
    if not path.is_file():
        cache[path] = ("missing", None)
        return cache[path]
    try:
        actual = sha256_file(path)
    except OSError:
        cache[path] = ("unreadable", None)
        return cache[path]
    cache[path] = (None, actual)
    return (
        None if isinstance(expected_sha256, str) and actual == expected_sha256 else "mismatch",
        actual,
    )


def _file_integrity_reasons(
    root: Path,
    planned: dict[str, Any],
    result: dict[str, Any] | None,
    cache: dict[Path, tuple[str | None, str | None]],
) -> list[str]:
    source = _flatten_result(result or {})
    if source.get("status") != "complete":
        return []
    reasons: list[str] = []
    root_resolved = root.resolve()
    output_path = source.get("output_path", planned.get("output_path"))
    if not isinstance(output_path, str) or not output_path:
        reasons.append("file_integrity_failure:missing_output_path")
    else:
        candidate = (root / output_path).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            reasons.append("file_integrity_failure:output_path_escape")
        else:
            status, _ = _verified_file(candidate, source.get("image_sha256"), cache)
            if status:
                reasons.append(f"file_integrity_failure:image_{status}")

    base_path = (root / "seeds" / str(planned["seed"]) / "base.png").resolve()
    status, _ = _verified_file(base_path, source.get("base_image_sha256"), cache)
    if status:
        reasons.append(f"file_integrity_failure:base_image_{status}")
    return reasons


def load_confirmatory_results(
    root: Path, config_path: Path
) -> tuple[pd.DataFrame, ExperimentConfig, dict[str, Any]]:
    """Load the frozen matrix and merge result metadata without dropping cells."""
    config = ExperimentConfig.from_yaml(config_path)
    if not config.analysis.comparisons:
        raise ValueError("The selected config does not declare confirmatory comparisons")
    manifest_path = root / "study_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing confirmatory manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(config_path)
    embedded_config_hash = manifest.get("config", {}).get("sha256")
    if embedded_config_hash != config_sha256:
        raise ValueError("Manifest config hash does not match the analysis config")
    current_protocol = protocol_record()
    embedded_protocol_hash = manifest.get("metric_protocol", {}).get("sha256")
    if embedded_protocol_hash != current_protocol["sha256"]:
        raise ValueError("Manifest metric protocol hash does not match the frozen protocol")
    if manifest.get("study_id") != config.study_id:
        raise ValueError("Manifest study_id does not match the analysis config")

    planned_rows = manifest.get("rows")
    if not isinstance(planned_rows, list):
        raise ValueError("Manifest rows must be a list")
    planned_keys = [_manifest_row_key(row) for row in planned_rows]
    expected_keys = [
        (seed, method, float(alpha))
        for seed in config.evaluation.seeds
        for method in config.evaluation.methods
        for alpha in config.evaluation.alphas
    ]
    if len(planned_keys) != len(set(planned_keys)):
        raise ValueError("Manifest contains duplicate seed/method/alpha cells")
    if set(planned_keys) != set(expected_keys):
        raise ValueError("Manifest rows do not equal the frozen evaluation matrix")

    metadata_results: dict[tuple[int, str, float], dict[str, Any] | None] = {}
    duplicate_keys: set[tuple[int, str, float]] = set()
    metadata_files = 0
    for seed in config.evaluation.seeds:
        metadata_path = root / "seeds" / str(seed) / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata_files += 1
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("seed") != seed:
            raise ValueError(f"Seed metadata mismatch: {metadata_path}")
        if metadata.get("config_sha256") != config_sha256:
            raise ValueError(f"Seed metadata config hash mismatch: {metadata_path}")
        for result in metadata.get("results", []):
            key = _manifest_row_key(result)
            if key in metadata_results:
                duplicate_keys.add(key)
            else:
                metadata_results[key] = result

    source_commit = manifest.get("provenance", {}).get("git", {}).get("commit")
    if not source_commit:
        raise ValueError("Manifest does not record the source commit")
    settings = {
        "analysis": asdict(config.analysis),
        "bootstrap": asdict(config.evaluation.bootstrap),
        "multiplicity_correction": config.evaluation.multiplicity_correction,
    }
    analysis_settings_sha256 = stable_hash(settings)
    code_provenance = analysis_code_provenance()
    records: list[dict[str, Any]] = []
    file_cache: dict[Path, tuple[str | None, str | None]] = {}
    for planned in sorted(planned_rows, key=_manifest_row_key):
        key = _manifest_row_key(planned)
        result = None if key in duplicate_keys else metadata_results.get(key)
        row = _normalise_planned_result(
            planned,
            result,
            config,
            integrity_reasons=_file_integrity_reasons(root, planned, result, file_cache),
        )
        if key in duplicate_keys:
            reasons = json.loads(row["failure_reasons"])
            reasons.append("duplicate_result_row")
            row["failure_reasons"] = json.dumps(sorted(set(reasons)), separators=(",", ":"))
            row["failure_reason_count"] = len(set(reasons))
            row["analysis_row_valid"] = False
        row.update(
            {
                "config_sha256": config_sha256,
                "protocol_id": config.evaluation.protocol_id,
                "protocol_sha256": current_protocol["sha256"],
                "source_commit": source_commit,
                **code_provenance,
                "analysis_version": config.analysis.version,
                "analysis_rng_seed": config.analysis.rng_seed,
                "analysis_settings_sha256": analysis_settings_sha256,
            }
        )
        records.append(row)
    provenance = {
        "schema_version": ANALYSIS_OUTPUT_SCHEMA_VERSION,
        "study_id": config.study_id,
        "run_id": manifest.get("run_id"),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "protocol_id": config.evaluation.protocol_id,
        "protocol_sha256": current_protocol["sha256"],
        "source_commit": source_commit,
        **code_provenance,
        "analysis_version": config.analysis.version,
        "analysis_rng_seed": config.analysis.rng_seed,
        "analysis_settings": settings,
        "analysis_settings_sha256": analysis_settings_sha256,
        "metadata_files": metadata_files,
    }
    return pd.DataFrame(records), config, provenance


def _json_reasons(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def failure_counts(long_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in long_df.itertuples(index=False):
        for reason in _json_reasons(row.failure_reasons):
            records.append({"method": row.method, "alpha": row.alpha, "failure_reason": reason})
    if not records:
        return pd.DataFrame(columns=["method", "alpha", "failure_reason", "count"])
    return (
        pd.DataFrame(records)
        .groupby(["method", "alpha", "failure_reason"], sort=True, dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )


def aggregate_metrics(
    long_df: pd.DataFrame, config: ExperimentConfig, provenance: dict[str, Any]
) -> pd.DataFrame:
    """Build explicitly exploratory method/alpha summaries with missingness."""
    records: list[dict[str, Any]] = []
    for method in config.evaluation.methods:
        for alpha in config.evaluation.alphas:
            group = long_df.loc[long_df["method"].eq(method) & long_df["alpha"].eq(alpha)]
            for metric in METRICS:
                numeric = pd.to_numeric(group[metric], errors="coerce")
                finite_mask = np.isfinite(numeric.to_numpy(dtype=float))
                finite = numeric.loc[finite_mask]
                low = high = float("nan")
                if len(finite) >= 2:
                    seed_values = pd.DataFrame(
                        {
                            "seed": group.loc[finite_mask, "seed"].astype(int),
                            "effect_a_minus_b": finite.astype(float),
                        }
                    )
                    low, high = seed_cluster_bootstrap_ci(
                        seed_values,
                        resamples=config.evaluation.bootstrap.resamples,
                        confidence=config.evaluation.bootstrap.confidence_level,
                        rng_seed=config.analysis.rng_seed,
                        label=f"exploratory:{method}:{alpha:g}:{metric}",
                    )
                records.append(
                    {
                        "summary_role": "exploratory_descriptive",
                        "method": method,
                        "alpha": alpha,
                        "metric": metric,
                        "expected_seed_count": len(config.evaluation.seeds),
                        "finite_seed_count": int(len(finite)),
                        "missing_seed_count": int(len(group) - len(finite)),
                        "mean": float(finite.mean()) if len(finite) else None,
                        "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
                        "median": float(finite.median()) if len(finite) else None,
                        "ci95_low": low if np.isfinite(low) else None,
                        "ci95_high": high if np.isfinite(high) else None,
                        "evaluation_incomplete_rows": int(
                            (~group["evaluation_complete"].fillna(False).astype(bool)).sum()
                        ),
                        "face_detection_failure_rows": int(
                            group["face_detection_failure"].astype(bool).sum()
                        ),
                        "generation_failure_rows": int(
                            group["generation_failure"].astype(bool).sum()
                        ),
                        "file_integrity_failure_rows": int(
                            group["file_integrity_failure"].astype(bool).sum()
                        ),
                        **{
                            key: provenance[key]
                            for key in (
                                "config_sha256",
                                "protocol_sha256",
                                "source_commit",
                                "analysis_code_commit",
                                "analysis_code_dirty",
                                "analysis_version",
                                "analysis_rng_seed",
                                "analysis_settings_sha256",
                            )
                        },
                    }
                )
    return pd.DataFrame(records)


def _attach_provenance(frame: pd.DataFrame, provenance: dict[str, Any]) -> pd.DataFrame:
    attached = frame.copy()
    for key in (
        "config_sha256",
        "protocol_sha256",
        "source_commit",
        "analysis_code_commit",
        "analysis_code_dirty",
        "analysis_version",
        "analysis_rng_seed",
        "analysis_settings_sha256",
    ):
        attached[key] = provenance[key]
    return attached


def run_analysis(root: Path, output: Path, config_path: Path) -> dict[str, Any]:
    long_df, config, provenance = load_confirmatory_results(root, config_path)
    seed_contrasts, contrasts = analyze_comparisons(long_df, config)
    aggregates = aggregate_metrics(long_df, config, provenance)
    failures = _attach_provenance(failure_counts(long_df), provenance)
    seed_contrasts = _attach_provenance(seed_contrasts, provenance)
    contrasts = _attach_provenance(contrasts, provenance)

    output.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output / "results_long.csv", index=False)
    aggregates.to_csv(output / "aggregate_metrics.csv", index=False)
    seed_contrasts.to_csv(output / "seed_matched_contrasts.csv", index=False)
    contrasts.to_csv(output / "confirmatory_contrasts.csv", index=False)
    failures.to_csv(output / "failure_counts.csv", index=False)
    invalid = contrasts.loc[~contrasts["confirmatory_computable"].astype(bool)]
    audit = {
        **provenance,
        "expected_rows": config.evaluation.matrix.expected_rows,
        "observed_long_rows": int(len(long_df)),
        "valid_analysis_rows": int(long_df["analysis_row_valid"].astype(bool).sum()),
        "incomplete_evaluation_rows": int(
            (~long_df["evaluation_complete"].fillna(False).astype(bool)).sum()
        ),
        "face_detection_failure_rows": int(long_df["face_detection_failure"].astype(bool).sum()),
        "generation_failure_rows": int(long_df["generation_failure"].astype(bool).sum()),
        "file_integrity_failure_rows": int(long_df["file_integrity_failure"].astype(bool).sum()),
        "bootstrap_resamples": config.evaluation.bootstrap.resamples,
        "bootstrap_cluster_unit": config.evaluation.bootstrap.cluster_unit,
        "confidence_level": config.evaluation.bootstrap.confidence_level,
        "holm_family": [
            comparison.id
            for comparison in config.analysis.comparisons
            if comparison.role == "secondary"
        ],
        "confirmatory_analysis_computable": bool(invalid.empty),
        "not_computable_estimates": invalid[
            ["comparison_id", "not_computable_reason", "included_seed_count"]
        ].to_dict(orient="records"),
        "outputs": {
            "results_long": "results_long.csv",
            "aggregate_metrics": "aggregate_metrics.csv",
            "seed_matched_contrasts": "seed_matched_contrasts.csv",
            "confirmatory_contrasts": "confirmatory_contrasts.csv",
            "failure_counts": "failure_counts.csv",
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit


# Backward-compatible helpers remain available for the historical pilot. They
# are descriptive only and are not used by the confirmatory CLI above.
def load_runs(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(root.rglob("metadata.json")):
        with path.open(encoding="utf-8") as handle:
            run = json.load(handle)
        for result in run.get("results", []):
            success, success_source = _success_value(result)
            rows.append(
                {
                    "run": str(path.parent),
                    "seed": result.get("seed", run.get("seed")),
                    "method": result.get("method", run.get("method", "stepwise_masked")),
                    "alpha": result.get("alpha"),
                    "evaluation_complete": result.get("evaluation_complete", False),
                    "counterfactual_success": success,
                    "counterfactual_success_source": success_source,
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
        return float("nan"), float("nan")
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
            values = pd.to_numeric(group[metric], errors="coerce")
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


def build_audit(long_df: pd.DataFrame, *, resamples: int) -> dict:
    """Historical descriptive audit retained for legacy callers."""
    success = long_df["counterfactual_success"].astype("boolean")
    return {
        "metadata_files": int(long_df["run"].nunique()),
        "rows": int(len(long_df)),
        "complete_rows": int(long_df["evaluation_complete"].fillna(False).sum()),
        "successful_rows": int(success.fillna(False).sum()),
        "unsuccessful_rows": int(success.eq(False).fillna(False).sum()),
        "unreported_success_rows": int(success.isna().sum()),
        "legacy_success_rows": int(
            long_df["counterfactual_success_source"].eq("legacy_is_disentangled").sum()
        ),
        "bootstrap_resamples": resamples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Confirmatory run containing study_manifest.json")
    parser.add_argument("--output", type=Path, default=Path("experiments/summary"))
    parser.add_argument("--config", type=Path, default=Path("configs/full_study.yaml"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 after writing audit outputs if any confirmatory estimate is unavailable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = run_analysis(args.input, args.output, args.config)
    print(f"Wrote audited analysis outputs to {args.output}")
    print(
        f"Valid analysis rows: {audit['valid_analysis_rows']}/{audit['expected_rows']}; "
        f"confirmatory computable: {audit['confirmatory_analysis_computable']}"
    )
    if args.strict and not audit["confirmatory_analysis_computable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
