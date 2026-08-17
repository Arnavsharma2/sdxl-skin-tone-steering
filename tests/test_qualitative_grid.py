from __future__ import annotations

import pandas as pd

from scripts.make_qualitative_grid import METHODS, select_seed


def test_select_seed_uses_complete_coverage_then_median_base_ita() -> None:
    rows = []
    for seed, ita in ((1, 10.0), (2, 20.0), (3, 30.0)):
        for method in METHODS:
            for direction in (-1, 1):
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "direction": direction,
                        "target_change": 5.0,
                        "match_complete": True,
                        "original_ita": ita,
                    }
                )

    seed, selected, audit = select_seed(pd.DataFrame(rows))

    assert seed == 2
    assert len(selected) == 6
    assert audit["maximum_complete_rows"] == 6
