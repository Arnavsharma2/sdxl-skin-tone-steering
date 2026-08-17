#!/usr/bin/env python3
"""Verify that manuscript numbers agree with retained analysis artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPERS = (ROOT / "paper/main.tex", ROOT / "paper/wacv/main.tex")


def normalize(value: str) -> str:
    """Collapse TeX source whitespace without changing content."""

    return re.sub(r"\s+", " ", value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


@dataclass(frozen=True)
class Claim:
    label: str
    snippet: str
    evidence: dict[str, object]


def build_claims(root: Path = ROOT) -> tuple[list[Claim], list[Path]]:
    parent_dir = root / "experiments/analysis/confirmatory_cuda"
    replication_dir = root / "experiments/analysis/replication_cuda"
    assessment_dir = root / "experiments/analysis/replication_assessment"
    robustness_dir = root / "experiments/analysis/robustness"
    stability_dir = root / "experiments/analysis/direction_stability_cuda"
    validation_parent_path = (
        root / "experiments/measurement_validation_cuda/validation_report.json"
    )
    validation_replication_path = (
        root
        / "experiments/measurement_validation_replication_cuda/validation_report.json"
    )
    source_paths = [
        validation_parent_path,
        validation_replication_path,
        parent_dir / "analysis_audit.json",
        parent_dir / "monotonicity.csv",
        parent_dir / "matched_summary.csv",
        parent_dir / "paired_contrasts.csv",
        replication_dir / "analysis_audit.json",
        replication_dir / "matched_summary.csv",
        assessment_dir / "replication_audit.json",
        assessment_dir / "replication_family.csv",
        robustness_dir / "tolerance_contrasts.csv",
        stability_dir / "direction_stability_summary.csv",
    ]

    validation_parent = json.loads(validation_parent_path.read_text())
    validation_replication = json.loads(validation_replication_path.read_text())
    parent_audit = json.loads((parent_dir / "analysis_audit.json").read_text())
    replication_audit = json.loads((replication_dir / "analysis_audit.json").read_text())
    decision_audit = json.loads((assessment_dir / "replication_audit.json").read_text())
    parent_mono = pd.read_csv(parent_dir / "monotonicity.csv")
    parent_coverage = pd.read_csv(parent_dir / "matched_summary.csv")
    parent_contrasts = pd.read_csv(parent_dir / "paired_contrasts.csv")
    replication_coverage = pd.read_csv(replication_dir / "matched_summary.csv")
    replication_family = pd.read_csv(assessment_dir / "replication_family.csv")
    robustness = pd.read_csv(robustness_dir / "tolerance_contrasts.csv")
    stability = pd.read_csv(stability_dir / "direction_stability_summary.csv")

    claims: list[Claim] = []

    claims.append(
        Claim(
            "parent measurement validation",
            (
                "ordered the light-descriptor image above the dark-descriptor image in "
                f"{round(validation_parent['pair_order_accuracy'] * 32)} of 32 pairs "
                f"({validation_parent['pair_order_accuracy'] * 100:.1f}\\%), and produced "
                "a median within-pair separation of "
                f"{validation_parent['median_pair_gap_ita']:.2f} ITA degrees"
            ),
            validation_parent,
        )
    )
    claims.append(
        Claim(
            "parent perturbation validation",
            (
                "the median absolute ITA shift was "
                f"{validation_parent['median_abs_ita_shift']:.2f} degrees and the 95th "
                f"percentile was {validation_parent['p95_abs_ita_shift']:.2f} degrees"
            ),
            validation_parent,
        )
    )
    claims.append(
        Claim(
            "parent completion",
            (
                f"produced all {parent_audit['input_rows']} prespecified condition rows "
                f"with {parent_audit['input_rows']} unique keys"
            ),
            parent_audit,
        )
    )

    mono = {
        row.method: row for row in parent_mono.itertuples(index=False)
    }
    claims.append(
        Claim(
            "parent monotonicity",
            (
                "correlations between $\\alpha$ and skin-tone change were "
                f"{mono['prompt_only'].mean_spearman:.2f} for prompt-only, "
                f"{mono['stepwise_unmasked'].mean_spearman:.2f} for unmasked stepwise, "
                f"{mono['stepwise_masked'].mean_spearman:.2f} for masked stepwise, and "
                f"{mono['posthoc_latent'].mean_spearman:.2f} for post-hoc addition"
            ),
            {name: float(row.mean_spearman) for name, row in mono.items()},
        )
    )
    claims.append(
        Claim(
            "parent post-hoc seed dependence",
            (
                f"The post-hoc median was {mono['posthoc_latent'].median_spearman:.2f} "
                "but its mean 95\\% interval was "
                f"$[{mono['posthoc_latent'].ci_low:.2f},{mono['posthoc_latent'].ci_high:.2f}]$"
            ),
            {
                "median": float(mono["posthoc_latent"].median_spearman),
                "ci_low": float(mono["posthoc_latent"].ci_low),
                "ci_high": float(mono["posthoc_latent"].ci_high),
            },
        )
    )

    def coverage(method: str, direction: int, frame: pd.DataFrame) -> pd.Series:
        return one_row(frame, method=method, direction=direction)

    parent_coverage_rows = {
        (method, direction): coverage(method, direction, parent_coverage)
        for method in ("prompt_only", "stepwise_unmasked", "stepwise_masked")
        for direction in (-1, 1)
    }
    claims.append(
        Claim(
            "parent matched coverage table",
            " \\\\ ".join(
                [
                    "Prompt-only & 7 (23\\%) & 18 (60\\%)",
                    "Stepwise, unmasked & 13 (43\\%) & 15 (50\\%)",
                    "Stepwise, masked & 19 (63\\%) & 26 (87\\%)",
                ]
            ),
            {
                f"{method}:{direction}": int(row.complete_seeds)
                for (method, direction), row in parent_coverage_rows.items()
            },
        )
    )
    masked_lighter = parent_coverage_rows[("stepwise_masked", -1)]
    masked_darker = parent_coverage_rows[("stepwise_masked", 1)]
    claims.append(
        Claim(
            "parent achieved target change",
            (
                "mean achieved changes were "
                f"{masked_lighter.mean_achieved_change:.2f} ITA lighter and "
                f"{masked_darker.mean_achieved_change:.2f} ITA darker"
            ),
            {
                "lighter": float(masked_lighter.mean_achieved_change),
                "darker": float(masked_darker.mean_achieved_change),
            },
        )
    )

    def contrast(
        frame: pd.DataFrame, comparator: str, direction: int, outcome: str
    ) -> pd.Series:
        return one_row(
            frame,
            reference_method="stepwise_masked",
            comparator=comparator,
            direction=direction,
            outcome=outcome,
        )

    parent_lpips_lighter = contrast(
        parent_contrasts, "stepwise_unmasked", -1, "lpips"
    )
    parent_lpips_darker = contrast(
        parent_contrasts, "stepwise_unmasked", 1, "lpips"
    )
    claims.append(
        Claim(
            "parent masked-versus-unmasked LPIPS",
            (
                "paired LPIPS advantage was "
                f"{parent_lpips_lighter.mean_reference_advantage:.4f} "
                f"($[{parent_lpips_lighter.ci_low:.4f},{parent_lpips_lighter.ci_high:.4f}]$, "
                f"$p_{{\\mathrm{{Holm}}}}={parent_lpips_lighter.p_holm:.3f}$) for lighter "
                f"edits and {parent_lpips_darker.mean_reference_advantage:.4f} "
                f"($[{parent_lpips_darker.ci_low:.4f},{parent_lpips_darker.ci_high:.4f}]$, "
                f"$p_{{\\mathrm{{Holm}}}}={parent_lpips_darker.p_holm:.3f}$) for darker edits"
            ).replace("=0.", "=."),
            {
                "lighter": parent_lpips_lighter.to_dict(),
                "darker": parent_lpips_darker.to_dict(),
            },
        )
    )

    robust_pair_ranges = {}
    for tolerance in (1.0, 2.0):
        rows = robustness[robustness["tolerance_ita"] == tolerance]
        robust_pair_ranges[str(int(tolerance))] = [
            int(rows.n_pairs.min()),
            int(rows.n_pairs.max()),
        ]
    claims.append(
        Claim(
            "robustness shared sample ranges",
            (
                f"$n={robust_pair_ranges['1'][0]}$--{robust_pair_ranges['1'][1]} at one "
                f"degree and $n={robust_pair_ranges['2'][0]}$--{robust_pair_ranges['2'][1]} "
                "at two degrees"
            ),
            robust_pair_ranges,
        )
    )

    stability_rows = {
        (int(row.pair_count), row.metric): row
        for row in stability.itertuples(index=False)
    }
    table_rows = []
    for pair_count in (8, 16, 32):
        raw = stability_rows[(pair_count, "raw_cosine")]
        masked = stability_rows[(pair_count, "masked_cosine")]
        table_rows.append(
            f"{pair_count} & {raw.median:.2f} [{raw.q025:.2f}, {raw.q975:.2f}] & "
            f"{masked.median:.2f} [{masked.q025:.2f}, {masked.q975:.2f}]"
        )
    claims.append(
        Claim(
            "direction stability table",
            " \\\\ ".join(table_rows),
            {
                f"{pair_count}:{metric}": {
                    "median": float(row.median),
                    "q025": float(row.q025),
                    "q975": float(row.q975),
                }
                for (pair_count, metric), row in stability_rows.items()
                if metric in {"raw_cosine", "masked_cosine"}
            },
        )
    )

    claims.append(
        Claim(
            "replication measurement validation",
            (
                f"detection was {validation_replication['detection_rate']:.3f}, pair "
                f"ordering {validation_replication['pair_order_accuracy']:.3f}, the median "
                f"pair gap {validation_replication['median_pair_gap_ita']:.2f} ITA degrees, "
                "and median/95th-percentile perturbation shifts "
                f"{validation_replication['median_abs_ita_shift']:.2f}/"
                f"{validation_replication['p95_abs_ita_shift']:.2f} degrees"
            ),
            validation_replication,
        )
    )
    claims.append(
        Claim(
            "replication completion",
            (
                f"completed all {replication_audit['input_rows']} frozen conditions over "
                f"{replication_audit['unique_seeds']} new seeds"
            ),
            replication_audit,
        )
    )

    replication_coverage_rows = {
        (method, direction): coverage(method, direction, replication_coverage)
        for method in ("prompt_only", "stepwise_unmasked", "stepwise_masked")
        for direction in (-1, 1)
    }
    claims.append(
        Claim(
            "replication matched coverage",
            (
                "Masked target coverage was "
                f"{int(replication_coverage_rows[('stepwise_masked', -1)].complete_seeds)}/30 "
                f"lighter and {int(replication_coverage_rows[('stepwise_masked', 1)].complete_seeds)}/30 "
                "darker, versus "
                f"{int(replication_coverage_rows[('stepwise_unmasked', -1)].complete_seeds)}/30 "
                "in each direction for unmasked steering and "
                f"{int(replication_coverage_rows[('prompt_only', -1)].complete_seeds)}/30 lighter "
                f"and {int(replication_coverage_rows[('prompt_only', 1)].complete_seeds)}/30 "
                "darker for prompt-only"
            ),
            {
                f"{method}:{direction}": int(row.complete_seeds)
                for (method, direction), row in replication_coverage_rows.items()
            },
        )
    )

    family_rows = {
        row.direction_label: row for row in replication_family.itertuples(index=False)
    }
    family_table_rows = []
    for label in ("lighter", "darker"):
        row = family_rows[label]
        adjusted_p = f"{row.p_holm_replication_family:.4f}".lstrip("0")
        family_table_rows.append(
            f"{label.title()} & {row.n_pairs} & {row.mean_reference_advantage:.4f} & "
            f"[{row.ci_low:.4f}, {row.ci_high:.4f}] & "
            + adjusted_p
        )
    claims.append(
        Claim(
            "locked replication table",
            " \\\\ ".join(family_table_rows),
            {label: row._asdict() for label, row in family_rows.items()},
        )
    )
    claims.append(
        Claim(
            "locked replication decision",
            (
                f"\\emph{{inconclusive coverage}}: only {family_rows['lighter'].n_pairs} "
                f"lighter and {family_rows['darker'].n_pairs} darker seeds were shared "
                "between matched methods, below the prespecified minimum of "
                f"{decision_audit['minimum_shared_matched_pairs_per_direction']}"
            ),
            {
                "decision": decision_audit["decision"],
                "minimum_pairs": decision_audit[
                    "minimum_shared_matched_pairs_per_direction"
                ],
            },
        )
    )
    agreement = decision_audit["direction_agreement"]
    claims.append(
        Claim(
            "cross-campaign direction agreement",
            (
                f"raw directions had cosine {agreement['raw_cosine']:.3f}; applying the "
                "frozen spatial mask raised cross-campaign cosine to "
                f"{agreement['masked_cosine']:.3f}. Their norm ratio was "
                f"{agreement['norm_ratio_replication_to_parent']:.2f}"
            ),
            agreement,
        )
    )
    return claims, source_paths


def verify(root: Path = ROOT) -> dict[str, object]:
    claims, source_paths = build_claims(root)
    paper_paths = (root / "paper/main.tex", root / "paper/wacv/main.tex")
    papers = {path: normalize(path.read_text()) for path in paper_paths}
    failures = []
    records = []
    for claim in claims:
        snippet = normalize(claim.snippet)
        missing = [str(path.relative_to(root)) for path, text in papers.items() if snippet not in text]
        records.append(
            {
                "label": claim.label,
                "evidence": claim.evidence,
                "present_in_all_papers": not missing,
            }
        )
        if missing:
            failures.append({"label": claim.label, "missing_from": missing, "snippet": snippet})

    result = {
        "schema_version": "1.0",
        "papers": {
            str(path.relative_to(root)): sha256_file(path) for path in paper_paths
        },
        "evidence_files": {
            str(path.relative_to(root)): sha256_file(path) for path in source_paths
        },
        "claim_count": len(records),
        "claims": records,
        "failures": failures,
        "passed": not failures,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/analysis/manuscript_claim_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("claim_count", "failures", "passed")}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
