import numpy as np
import pytest

from src.metrics.skin_tone_metrics import SkinToneMetrics


class NoFaceBackend:
    def detect(self, _image):
        return None


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


def test_moderate_global_exposure_is_first_order_corrected(monkeypatch):
    metrics = SkinToneMetrics(min_skin_pixels=100, min_reference_pixels=100)
    original, skin_mask, reference_mask = portrait((160, 110, 80), (200, 200, 200))
    exposed = np.clip(original.astype(int) + 10, 0, 255).astype(np.uint8)
    masks = iter([skin_mask, skin_mask])
    references = iter([reference_mask, reference_mask])
    monkeypatch.setattr(metrics, "create_skin_mask", lambda _: next(masks))
    monkeypatch.setattr(metrics, "create_reference_mask", lambda _: next(references))

    comparison = metrics.compare(original, exposed)

    assert comparison["illumination_stable"]
    assert abs(comparison["skin_tone_change"]) < 1.0
    assert comparison["reference_lstar_shift"] > 3.0


def test_local_skin_lighting_remains_a_known_metric_sensitivity(monkeypatch):
    metrics = SkinToneMetrics(min_skin_pixels=100, min_reference_pixels=100)
    original, skin_mask, reference_mask = portrait((160, 110, 80), (200, 200, 200))
    locally_lit = original.copy()
    locally_lit[skin_mask] = np.clip(
        locally_lit[skin_mask].astype(int) + 20, 0, 255
    ).astype(np.uint8)
    masks = iter([skin_mask, skin_mask])
    references = iter([reference_mask, reference_mask])
    monkeypatch.setattr(metrics, "create_skin_mask", lambda _: next(masks))
    monkeypatch.setattr(metrics, "create_reference_mask", lambda _: next(references))

    comparison = metrics.compare(original, locally_lit)

    assert comparison["illumination_stable"]
    assert comparison["reference_lstar_shift"] == 0.0
    assert comparison["skin_tone_change"] > 5.0


def test_face_detection_failure_is_an_invalid_target_outcome():
    metrics = SkinToneMetrics(
        min_skin_pixels=100,
        min_reference_pixels=100,
        landmark_backend=NoFaceBackend(),
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    assert metrics.create_skin_mask(image) is None
    assert metrics.measure(image) is None
    comparison = metrics.compare(image, image)
    assert comparison["skin_tone_change"] is None
    assert not comparison["illumination_stable"]


def test_incorrect_skin_mask_shape_fails_closed():
    metrics = SkinToneMetrics(min_skin_pixels=100, min_reference_pixels=100)
    image, _, reference_mask = portrait((190, 135, 105))
    incorrect_skin_mask = np.ones((100, 101), dtype=bool)

    measurement = metrics.measure(
        image,
        skin_mask=incorrect_skin_mask,
        reference_mask=reference_mask,
    )

    assert measurement is None


def test_incorrect_reference_mask_shape_fails_closed():
    metrics = SkinToneMetrics(min_skin_pixels=100, min_reference_pixels=100)
    image, skin_mask, _ = portrait((190, 135, 105))
    incorrect_reference_mask = np.ones((100, 101), dtype=bool)

    measurement = metrics.measure(
        image,
        skin_mask=skin_mask,
        reference_mask=incorrect_reference_mask,
    )

    assert measurement is None


def test_small_nonbinary_and_overlapping_masks_fail_closed():
    metrics = SkinToneMetrics(min_skin_pixels=100, min_reference_pixels=100)
    image, skin_mask, reference_mask = portrait((190, 135, 105))
    too_small = np.zeros_like(skin_mask)
    too_small[:5, :5] = True
    assert metrics.measure(image, skin_mask=too_small, reference_mask=reference_mask) is None

    nonbinary = skin_mask.astype(np.uint8) * 2
    assert metrics.measure(image, skin_mask=nonbinary, reference_mask=reference_mask) is None

    overlapping_reference = reference_mask.copy()
    overlapping_reference[25:30, 25:30] = True
    assert (
        metrics.measure(
            image,
            skin_mask=skin_mask,
            reference_mask=overlapping_reference,
        )
        is None
    )


def test_colored_reference_fails_neutrality_check():
    metrics = SkinToneMetrics(min_skin_pixels=100, min_reference_pixels=100)
    image, skin_mask, reference_mask = portrait(
        (190, 135, 105), background_rgb=(255, 0, 0)
    )

    assert metrics.measure(image, skin_mask=skin_mask, reference_mask=reference_mask) is None


def test_non_uint8_image_is_rejected_before_measurement():
    metrics = SkinToneMetrics(min_skin_pixels=100, min_reference_pixels=100)

    with pytest.raises(ValueError, match="uint8 RGB"):
        metrics.measure(np.zeros((100, 100, 3), dtype=np.float32))
