from src.utils.reproducibility import stable_fingerprint


def test_fingerprint_is_order_independent_and_sensitive():
    first = stable_fingerprint({"seed": 1, "prompt": "portrait"})
    reordered = stable_fingerprint({"prompt": "portrait", "seed": 1})
    changed = stable_fingerprint({"seed": 2, "prompt": "portrait"})
    assert first == reordered
    assert first != changed
    assert len(first) == 12
