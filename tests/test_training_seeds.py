from src.utils.reproducibility import seed_for_index


def test_seed_extension_does_not_repeat_after_first_cycle():
    seeds = [42, 137]
    generated = [seed_for_index(index, seeds) for index in range(6)]
    assert generated == [42, 137, 10042, 10137, 20042, 20137]
    assert len(generated) == len(set(generated))
