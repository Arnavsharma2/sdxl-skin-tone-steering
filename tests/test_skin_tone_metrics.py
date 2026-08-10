import numpy as np

from src.metrics.skin_tone_metrics import SkinToneMetrics


def portrait(skin_rgb, background_rgb=(210, 210, 210)):
    image = np.full((100, 100, 3), background_rgb, dtype=np.uint8)
    image[25:75, 25:75] = skin_rgb
    skin_mask = np.zeros((100, 100), dtype=bool)
    skin_mask[25:75, 25:75] = True
    reference_mask = np.zeros((100, 100), dtype=bool)
    reference_mask[:40, :20] = True
    reference_mask[:40, -20:] = True
    return image, skin_mask, reference_mask


def test_relative_lstar_tracks_skin_change_with_fixed_background():
    metrics = SkinToneMetrics(min_skin_pixels=100, min_reference_pixels=100)
    light, skin_mask, reference_mask = portrait((190, 135, 105))
    dark, _, _ = portrait((125, 78, 55))

    first = metrics.measure(light, skin_mask=skin_mask, reference_mask=reference_mask)
    second = metrics.measure(dark, skin_mask=skin_mask, reference_mask=reference_mask)

    assert first is not None and second is not None
    assert second.relative_lstar < first.relative_lstar
    assert second.ita_degrees < first.ita_degrees


def test_large_reference_shift_invalidates_target_change(monkeypatch):
    metrics = SkinToneMetrics(
        min_skin_pixels=100,
        min_reference_pixels=100,
        max_reference_shift=5.0,
    )
    original, skin_mask, reference_mask = portrait((190, 135, 105), (220, 220, 220))
    shifted, _, _ = portrait((160, 105, 78), (140, 140, 140))
    masks = iter([skin_mask, skin_mask])
    references = iter([reference_mask, reference_mask])
    monkeypatch.setattr(metrics, "create_skin_mask", lambda _: next(masks))
    monkeypatch.setattr(metrics, "create_reference_mask", lambda _: next(references))

    comparison = metrics.compare(original, shifted)
    assert not comparison["illumination_stable"]
    assert comparison["skin_tone_change"] is None
