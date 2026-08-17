"""Evaluation metrics with lazy optional-dependency imports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CounterfactualEvaluator",
    "DisentanglementMetrics",
    "EvaluationResult",
    "EvaluationThresholds",
    "IdentityPreservationMetrics",
    "SkinToneMeasurement",
    "SkinToneMetrics",
    "StructuralPreservationMetrics",
    "individual_typology_angle",
    "srgb_to_cielab",
]

_EXPORTS = {
    "CounterfactualEvaluator": (".evaluator", "CounterfactualEvaluator"),
    "EvaluationResult": (".evaluator", "EvaluationResult"),
    "EvaluationThresholds": (".evaluator", "EvaluationThresholds"),
    "DisentanglementMetrics": (".disentanglement_metrics", "DisentanglementMetrics"),
    "IdentityPreservationMetrics": (".identity_metrics", "IdentityPreservationMetrics"),
    "StructuralPreservationMetrics": (
        ".structural_metrics",
        "StructuralPreservationMetrics",
    ),
    "SkinToneMeasurement": (".skin_tone", "SkinToneMeasurement"),
    "SkinToneMetrics": (".skin_tone", "SkinToneMetrics"),
    "individual_typology_angle": (".skin_tone", "individual_typology_angle"),
    "srgb_to_cielab": (".skin_tone", "srgb_to_cielab"),
}


def __getattr__(name: str):
    """Load a metric only when requested, preserving optional dependencies."""

    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
