import pytest

torch = pytest.importorskip("torch")

from src.latent.vector_discovery import (  # noqa: E402
    RaceVectorExtractor,
    SkinToneDirectionExtractor,
)


def test_compatibility_alias_points_to_narrower_api():
    assert RaceVectorExtractor is SkinToneDirectionExtractor


def test_paired_difference_and_mask_are_deterministic():
    extractor = SkinToneDirectionExtractor(device="cpu")
    light = [torch.zeros(1, 4, 4), torch.ones(1, 4, 4)]
    dark = [tensor + 2 for tensor in light]
    mask = torch.ones(4, 4)

    direction = extractor.extract_from_pairs(light, dark, spatial_mask=mask)

    assert torch.equal(direction, torch.full((1, 4, 4), 2.0))


def test_paired_difference_rejects_invalid_samples():
    extractor = SkinToneDirectionExtractor(device="cpu")
    with pytest.raises(ValueError, match="at least one"):
        extractor.extract_from_pairs([], [])
    with pytest.raises(ValueError, match="same number"):
        extractor.extract_from_pairs([torch.zeros(1, 2, 2)], [torch.zeros(1, 2, 2)] * 2)
    with pytest.raises(ValueError, match="same shape"):
        extractor.extract_from_pairs(
            [torch.zeros(1, 2, 2)],
            [torch.zeros(1, 3, 3)],
        )


def test_center_mask_is_bounded_and_center_weighted():
    mask = SkinToneDirectionExtractor(device="cpu").create_center_mask(
        9,
        9,
        center_weight=1.0,
        edge_weight=0.3,
        radius=1.0,
    )
    assert float(mask.min()) >= 0.3
    assert float(mask.max()) <= 1.0
    assert mask[4, 4] > mask[0, 0]
