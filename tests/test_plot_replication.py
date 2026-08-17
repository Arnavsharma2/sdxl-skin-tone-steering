from pathlib import Path

import pandas as pd

from scripts.plot_replication import plot_replication


def _write_analysis(path: Path, scale: float) -> None:
    path.mkdir()
    effects = []
    for direction, mean in [(-1, 0.02 * scale), (1, 0.08 * scale)]:
        effects.append(
            {
                "reference_method": "stepwise_masked",
                "comparator": "stepwise_unmasked",
                "direction": direction,
                "target_change": 5.0,
                "outcome": "lpips",
                "n_pairs": 15,
                "mean_reference_advantage": mean,
                "median_reference_advantage": mean,
                "ci_low": mean - 0.01,
                "ci_high": mean + 0.01,
                "p_value": 0.01,
                "p_holm": 0.02,
            }
        )
    pd.DataFrame(effects).to_csv(path / "paired_contrasts.csv", index=False)
    coverage = []
    for method, rate in [
        ("prompt_only", 0.3),
        ("stepwise_unmasked", 0.5),
        ("stepwise_masked", 0.8),
    ]:
        for direction in (-1, 1):
            coverage.append(
                {
                    "method": method,
                    "direction": direction,
                    "target_change": 5.0,
                    "coverage_rate": rate,
                }
            )
    pd.DataFrame(coverage).to_csv(path / "matched_summary.csv", index=False)


def test_plot_replication_writes_vector_and_raster_outputs(tmp_path):
    parent = tmp_path / "parent"
    replication = tmp_path / "replication"
    output = tmp_path / "figures"
    _write_analysis(parent, 1.0)
    _write_analysis(replication, 0.8)

    plot_replication(parent, replication, output)

    assert (output / "replication_comparison.pdf").stat().st_size > 1000
    assert (output / "replication_comparison.png").stat().st_size > 1000
