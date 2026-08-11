from types import SimpleNamespace

from PIL import Image

from scripts import evaluate_sweep


class FakeToneMetrics:
    """Return a deterministic target value without loading metric models."""

    def measure(self, _image):
        return SimpleNamespace(relative_lstar=0.0)


class FakeEvaluator:
    """Minimal evaluator used to exercise sweep aggregation only."""

    def __init__(self, device):
        self.device = device
        self.skin_tone_metrics = FakeToneMetrics()

    def metric_provenance(self):
        return {"fixture": True}


def test_alpha_zero_only_sweep_writes_invalid_summary_instead_of_crashing(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    counterfactuals = input_dir / "counterfactuals"
    counterfactuals.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(input_dir / "base_image.png")
    Image.new("RGB", (8, 8)).save(counterfactuals / "alpha_0.png")
    monkeypatch.setattr(evaluate_sweep, "CounterfactualEvaluator", FakeEvaluator)

    summary = evaluate_sweep.evaluate(input_dir, tmp_path / "output", "cpu")

    assert summary["pair_count"] == 0
    assert summary["evaluation_completion_rate"] == 0.0
    assert summary["counterfactual_success_rate"] == 0.0
    assert not summary["target_monotonicity_valid"]
    assert summary["target_monotonicity_failure"] == "missing_or_unexpected_alpha"
    assert (tmp_path / "output" / "pair_metrics.csv").is_file()
    assert (tmp_path / "output" / "summary.json").is_file()


def test_strict_validation_rejects_complete_but_nonmonotonic_sweep():
    summary = {
        "evaluation_completion_rate": 1.0,
        "target_monotonicity_valid": True,
        "strictly_monotonic_target_response": False,
    }

    assert evaluate_sweep.strict_validation_failed(summary)
