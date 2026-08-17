import pandas as pd

from scripts.analyze_replication import (
    assess_replication as frozen_assess_replication,
)
from scripts.analyze_replication import (
    replication_family as frozen_replication_family,
)
from src.study.replication import assess_replication, replication_family


def _contrasts(estimates=(0.02, 0.08), pairs=(15, 16), lows=(0.01, 0.04)):
    return pd.DataFrame(
        {
            "reference_method": ["stepwise_masked", "stepwise_masked"],
            "comparator": ["stepwise_unmasked", "stepwise_unmasked"],
            "direction": [-1, 1],
            "target_change": [5.0, 5.0],
            "outcome": ["lpips", "lpips"],
            "n_pairs": list(pairs),
            "mean_reference_advantage": list(estimates),
            "ci_low": list(lows),
            "ci_high": [0.04, 0.12],
            "p_value": [0.01, 0.02],
        }
    )


def test_replication_family_applies_two_test_holm_rule():
    family = replication_family(_contrasts())
    assert family["direction_label"].tolist() == ["lighter", "darker"]
    assert family["p_holm_replication_family"].tolist() == [0.02, 0.02]
    assert assess_replication(family) == "strict_replication"


def test_replication_decision_prioritizes_coverage_and_sign():
    sparse = replication_family(_contrasts(pairs=(11, 16)))
    assert assess_replication(sparse) == "inconclusive_coverage"

    adverse = replication_family(_contrasts(estimates=(-0.01, 0.08), lows=(-0.03, 0.04)))
    assert assess_replication(adverse) == "failure_to_replicate"

    uncertain = replication_family(_contrasts(lows=(-0.01, 0.04)))
    assert assess_replication(uncertain) == "directional_replication"


def test_plotting_helper_matches_frozen_decision_implementation():
    contrasts = _contrasts()
    frozen = frozen_replication_family(contrasts)
    helper = replication_family(contrasts)
    pd.testing.assert_frame_equal(frozen, helper)
    assert frozen_assess_replication(frozen) == assess_replication(helper)
