"""Validated configuration contract for pilot and confirmatory studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.utils.reproducibility import seed_for_index, stable_fingerprint

SUPPORTED_METHODS = (
    "prompt_only",
    "posthoc_latent",
    "stepwise_unmasked",
    "stepwise_masked",
)


class StudyConfigError(ValueError):
    """Raised when a study manifest is incomplete or internally inconsistent."""


def _require(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise StudyConfigError(f"Missing required field {section}.{key}")
    return mapping[key]


@dataclass(frozen=True)
class StudyConfig:
    """Thin validated view over an immutable YAML study manifest."""

    path: Path
    raw: Mapping[str, Any]

    @property
    def study_id(self) -> str:
        return str(self.raw["study_id"])

    @property
    def status(self) -> str:
        return str(self.raw["status"])

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.raw, length=16)

    @property
    def model(self) -> Mapping[str, Any]:
        return self.raw["model"]

    @property
    def direction(self) -> Mapping[str, Any]:
        return self.raw["direction"]

    @property
    def evaluation(self) -> Mapping[str, Any]:
        return self.raw["evaluation"]

    @property
    def data(self) -> Mapping[str, Any]:
        return self.raw["data"]

    @property
    def prompts(self) -> Mapping[str, Any]:
        return self.raw["prompts"]

    @property
    def analysis(self) -> Mapping[str, Any]:
        return self.raw["analysis"]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.evaluation["seeds"])

    @property
    def alphas(self) -> tuple[float, ...]:
        if "method_alphas" not in self.evaluation:
            return tuple(float(value) for value in self.evaluation["alphas"])
        return tuple(
            sorted(
                {
                    alpha
                    for method in self.methods
                    for alpha in self.alphas_for(method)
                }
            )
        )

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.evaluation["methods"])

    @property
    def matched_change_methods(self) -> tuple[str, ...]:
        """Methods eligible for the prespecified matched-change contrasts."""

        values = self.analysis.get("matched_change_methods", self.methods)
        return tuple(str(value) for value in values)

    def alphas_for(self, method: str) -> tuple[float, ...]:
        """Return the prespecified alpha grid for one method."""

        grids = self.evaluation.get("method_alphas")
        values = grids[method] if grids is not None else self.evaluation["alphas"]
        return tuple(float(value) for value in values)

    def prompt_for(self, method: str, alpha: float) -> str:
        """Return the prespecified prompt for one method/alpha condition."""

        base = str(self.prompts["base"])
        if method != "prompt_only" or abs(alpha) < 1e-12:
            return base
        levels = self.prompts["prompt_only_levels"]
        candidates = (str(alpha), f"{alpha:g}", f"{alpha:.2f}")
        descriptor = next((levels[key] for key in candidates if key in levels), None)
        if descriptor is None:
            raise StudyConfigError(f"No prompt-only descriptor for alpha={alpha:g}")
        return str(self.prompts["attribute_template"]).format(
            skin_tone=str(descriptor)
        )

    def assert_confirmatory_ready(self) -> None:
        """Reject a run that has not frozen all confirmatory design choices."""

        if self.status != "preregistered":
            raise StudyConfigError(
                "Confirmatory execution requires status: preregistered; "
                f"current status is {self.status!r}"
            )
        targets = self.analysis.get("matched_change_targets", [])
        if not targets:
            raise StudyConfigError(
                "analysis.matched_change_targets must be frozen after calibration"
            )
        revision = self.model.get("revision")
        if not revision or str(revision).lower() in {"main", "latest", "null", "none"}:
            raise StudyConfigError("model.revision must be an immutable revision")
        validation_hash = self.raw["measurement_validation"].get("report_sha256")
        if not isinstance(validation_hash, str) or len(validation_hash) != 64:
            raise StudyConfigError(
                "measurement_validation.report_sha256 must freeze a passing report"
            )
        manifest_hash = self.data.get("training_manifest_sha256")
        if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
            raise StudyConfigError(
                "data.training_manifest_sha256 must freeze the paired-data manifest"
            )


def _validate(raw: Mapping[str, Any]) -> None:
    for key in (
        "schema_version",
        "study_id",
        "status",
        "attribute",
        "model",
        "direction",
        "data",
        "prompts",
        "evaluation",
        "analysis",
        "measurement_validation",
        "reporting",
    ):
        _require(raw, key, "root")

    if raw["attribute"] != "visual_skin_tone":
        raise StudyConfigError("attribute must be visual_skin_tone")
    if raw["status"] not in {"pilot", "planned", "calibration", "preregistered"}:
        raise StudyConfigError("status must be pilot, planned, calibration, or preregistered")

    model = raw["model"]
    for key in (
        "id",
        "revision",
        "scheduler",
        "inference_steps",
        "guidance_scale",
        "height",
        "width",
    ):
        _require(model, key, "model")
    if int(model["inference_steps"]) < 1:
        raise StudyConfigError("model.inference_steps must be positive")
    if int(model["height"]) % 8 or int(model["width"]) % 8:
        raise StudyConfigError("model.height and model.width must be divisible by 8")

    direction = raw["direction"]
    for key in ("estimator", "train_pairs", "held_out_pairs", "spatial_mask"):
        _require(direction, key, "direction")
    if direction["estimator"] != "paired_mean_difference":
        raise StudyConfigError("Only paired_mean_difference is confirmatory")
    if int(direction["train_pairs"]) < 2 or int(direction["held_out_pairs"]) < 1:
        raise StudyConfigError("Direction splits require >=2 train and >=1 held-out pairs")

    evaluation = raw["evaluation"]
    for key in ("seeds", "methods", "required_metrics", "bootstrap"):
        _require(evaluation, key, "evaluation")
    seeds = [int(value) for value in evaluation["seeds"]]
    methods = [str(value) for value in evaluation["methods"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise StudyConfigError("evaluation.seeds must be non-empty and unique")
    data_seed_schedule = raw["data"].get("seed_schedule")
    if data_seed_schedule is not None:
        data_seeds = [int(value) for value in data_seed_schedule]
        if not data_seeds or len(data_seeds) != len(set(data_seeds)):
            raise StudyConfigError("data.seed_schedule must be non-empty and unique")
        pair_count = int(direction["train_pairs"]) + int(direction["held_out_pairs"])
        expanded_data_seeds = {
            seed_for_index(index, data_seeds) for index in range(pair_count)
        }
        overlap = sorted(expanded_data_seeds.intersection(seeds))
        if overlap:
            raise StudyConfigError(
                "Direction-data and evaluation seeds must be disjoint; "
                f"overlap: {overlap}"
            )
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise StudyConfigError(f"Unsupported methods: {', '.join(unknown)}")
    if len(methods) != len(set(methods)):
        raise StudyConfigError("evaluation.methods must be unique")
    if set(methods) != set(SUPPORTED_METHODS):
        raise StudyConfigError("The confirmatory design requires all four supported methods")
    has_shared = "alphas" in evaluation
    has_method_grids = "method_alphas" in evaluation
    if has_shared == has_method_grids:
        raise StudyConfigError(
            "evaluation must define exactly one of alphas or method_alphas"
        )
    raw_grids = (
        {method: evaluation["alphas"] for method in methods}
        if has_shared
        else evaluation["method_alphas"]
    )
    if set(raw_grids) != set(methods):
        raise StudyConfigError("evaluation.method_alphas must cover every method exactly")
    alpha_grids = {
        method: [float(value) for value in raw_grids[method]] for method in methods
    }
    for method, alphas in alpha_grids.items():
        if not alphas or 0.0 not in alphas:
            raise StudyConfigError(
                f"Alpha grid for {method} must include zero"
            )
        if raw["status"] != "calibration" and (
            not any(alpha < 0 for alpha in alphas)
            or not any(alpha > 0 for alpha in alphas)
        ):
            raise StudyConfigError(
                f"Alpha grid for {method} must include both directions"
            )
        if len(alphas) != len(set(alphas)):
            raise StudyConfigError(f"Alpha grid for {method} must be unique")
    required = set(evaluation["required_metrics"])
    if "skin_tone_change" not in required or "face_similarity" not in required:
        raise StudyConfigError(
            "required_metrics must include skin_tone_change and face_similarity"
        )

    bootstrap = evaluation["bootstrap"]
    if int(_require(bootstrap, "resamples", "evaluation.bootstrap")) < 1000:
        raise StudyConfigError("bootstrap.resamples must be at least 1000")
    confidence = float(_require(bootstrap, "confidence_level", "evaluation.bootstrap"))
    if not 0.5 < confidence < 1.0:
        raise StudyConfigError("bootstrap.confidence_level must be between 0.5 and 1")

    prompts = raw["prompts"]
    for key in ("base", "negative", "attribute_template", "prompt_only_levels"):
        _require(prompts, key, "prompts")
    levels = prompts["prompt_only_levels"]
    for alpha in alpha_grids["prompt_only"]:
        if abs(alpha) < 1e-12:
            continue
        keys = (str(alpha), f"{alpha:g}", f"{alpha:.2f}")
        if not any(key in levels for key in keys):
            raise StudyConfigError(f"Missing prompt_only_levels entry for alpha={alpha:g}")

    analysis = raw["analysis"]
    if analysis.get("reference_method") not in methods:
        raise StudyConfigError("analysis.reference_method must name an evaluated method")
    if float(analysis.get("match_tolerance_ita", 0)) <= 0:
        raise StudyConfigError("analysis.match_tolerance_ita must be positive")
    matched_methods = [
        str(value) for value in analysis.get("matched_change_methods", methods)
    ]
    feasibility_methods = [
        str(value) for value in analysis.get("feasibility_only_methods", [])
    ]
    if not matched_methods or len(matched_methods) != len(set(matched_methods)):
        raise StudyConfigError(
            "analysis.matched_change_methods must be non-empty and unique"
        )
    if len(feasibility_methods) != len(set(feasibility_methods)):
        raise StudyConfigError("analysis.feasibility_only_methods must be unique")
    unknown_analysis_methods = sorted(
        (set(matched_methods) | set(feasibility_methods)) - set(methods)
    )
    if unknown_analysis_methods:
        raise StudyConfigError(
            "Analysis names unevaluated methods: "
            + ", ".join(unknown_analysis_methods)
        )
    if set(matched_methods) & set(feasibility_methods):
        raise StudyConfigError(
            "Matched-change and feasibility-only methods must be disjoint"
        )
    if set(matched_methods) | set(feasibility_methods) != set(methods):
        raise StudyConfigError(
            "Analysis method roles must cover every evaluated method"
        )
    if analysis.get("reference_method") not in matched_methods:
        raise StudyConfigError(
            "analysis.reference_method must be eligible for matched-change analysis"
        )

    validation = raw["measurement_validation"]
    gates = _require(validation, "gates", "measurement_validation")
    for name in (
        "min_detection_rate",
        "min_pair_order_accuracy",
        "min_median_pair_gap_ita",
        "max_median_abs_ita_shift",
        "max_p95_abs_ita_shift",
    ):
        if float(_require(gates, name, "measurement_validation.gates")) < 0:
            raise StudyConfigError(f"measurement_validation.gates.{name} must be nonnegative")
    perturbations = _require(validation, "perturbations", "measurement_validation")
    if not perturbations:
        raise StudyConfigError("measurement_validation.perturbations must be non-empty")


def load_study_config(path: str | Path) -> StudyConfig:
    """Load, validate, and return a study configuration."""

    resolved = Path(path).resolve()
    with resolved.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise StudyConfigError("Study configuration must be a YAML mapping")
    _validate(raw)
    return StudyConfig(path=resolved, raw=raw)
