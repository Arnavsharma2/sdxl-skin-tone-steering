import numpy as np
from PIL import Image, ImageEnhance

from src.metrics.skin_tone import SkinToneMetrics, individual_typology_angle, srgb_to_cielab


def solid_face(rgb):
    return Image.fromarray(np.full((160, 160, 3), rgb, dtype=np.uint8))


def studio_face(rgb):
    image = np.full((256, 256, 3), 235, dtype=np.uint8)
    image[48:208, 48:208] = rgb
    return Image.fromarray(image)


def test_srgb_reference_white_is_close_to_cielab_white():
    lab = srgb_to_cielab(np.array([[255, 255, 255]], dtype=np.uint8))[0]
    assert np.allclose(lab, [100.0, 0.0, 0.0], atol=0.02)


def test_ita_uses_standard_lstar_bstar_definition():
    assert np.isclose(individual_typology_angle(60.0, 10.0), 45.0)


def test_darker_cheek_colour_has_positive_signed_change():
    metric = SkinToneMetrics(min_pixels=100)
    bbox = (10, 10, 140, 140)
    result = metric.compare(
        solid_face((190, 135, 105)),
        solid_face((105, 70, 55)),
        original_bbox=bbox,
        edited_bbox=bbox,
    )
    assert result["skin_tone_complete"]
    assert result["skin_tone_change"] > 0
    assert result["lightness_change"] < 0


def test_detection_failure_is_missing_not_a_center_fallback(monkeypatch):
    metric = SkinToneMetrics()
    monkeypatch.setattr(metric, "_detect_largest_face", lambda _image: None)
    result = metric.compare(solid_face((180, 130, 100)), solid_face((170, 120, 90)))
    assert not result["skin_tone_complete"]
    assert result["skin_tone_change"] is None
    assert result["skin_tone_failure"] == "original_face_or_skin_region_missing"


def test_white_reference_reduces_global_exposure_sensitivity():
    metric = SkinToneMetrics(min_pixels=100)
    image = studio_face((150, 100, 75))
    darker_exposure = ImageEnhance.Brightness(image).enhance(0.9)
    bbox = (48, 48, 160, 160)
    baseline = metric.measure(image, face_bbox=bbox)
    shifted = metric.measure(darker_exposure, face_bbox=bbox)
    assert baseline.normalization_applied
    assert shifted.normalization_applied
    assert abs(baseline.ita_degrees - shifted.ita_degrees) < 1.0
