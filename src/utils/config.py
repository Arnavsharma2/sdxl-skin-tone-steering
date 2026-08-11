"""
Configuration management for experiments.

This module provides configuration dataclasses and utilities.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import yaml

CANONICAL_REQUIRED_METRICS = frozenset(
    {
        "background_ssim",
        "face_similarity",
        "lpips",
        "skin_tone_change",
        "target_direction_correct",
        "total_pose_diff",
    }
)
FROZEN_EVALUATION_PROTOCOL_ID = "tmlr_evaluation_protocol_v1"


@dataclass
class ModelConfig:
    """Configuration for generative model."""

    type: str = "stable_diffusion"  # stable_diffusion or stylegan
    name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = "cuda"
    dtype: str = "float16"  # float16 or float32
    enable_xformers: bool = True
    enable_cpu_offload: bool = False
    scheduler: str = "dpm_solver_multistep_karras"
    inference_steps: int = 25
    guidance_scale: float = 7.5


@dataclass
class VectorConfig:
    """Configuration for latent-direction extraction."""

    method: str = "supervised"  # supervised, unsupervised, pca
    num_pairs: int = 100
    alpha_range: Tuple[float, float] = (-2.0, 2.0)
    normalize: bool = True

    # Optimization settings
    optimization_enabled: bool = True
    num_iterations: int = 100
    learning_rate: float = 0.01
    lambda_identity: float = 0.7
    lambda_attribute: float = 0.3


@dataclass
class ThresholdConfig:
    """Prespecified engineering gates used by the evaluator."""

    face_similarity: float = 0.85
    landmark_rmse: float = 5.0
    lpips: float = 0.3
    background_ssim: float = 0.75
    pose_angle_diff: float = 5.0
    min_abs_skin_tone_change: float = 2.0


@dataclass
class SpatialMaskConfig:
    """Spatial attenuation applied to a denoising-time direction."""

    type: str = "gaussian_center"
    radius: float = 1.0
    center_weight: float = 1.0
    edge_weight: float = 0.3


@dataclass
class DirectionConfig:
    """Prespecified direction-estimation settings."""

    estimator: str = "paired_mean_difference"
    train_pairs: int = 8
    held_out_pairs: int = 0
    training_seed_bases: list[int] = field(
        default_factory=lambda: [42, 137, 256, 512, 777, 1536, 2048, 3141]
    )
    held_out_seed_bases: list[int] = field(default_factory=list)
    seed_extension_stride: int = 10_000
    deterministic_vae_encoding: bool = True
    optimization: bool = False
    spatial_mask: SpatialMaskConfig = field(default_factory=SpatialMaskConfig)

    def training_seeds(self) -> list[int]:
        """Expand base seeds deterministically to the prespecified pair count."""
        return self._expand_seeds(self.training_seed_bases, self.train_pairs, "training")

    def held_out_seeds(self) -> list[int]:
        """Expand held-out pair seeds without sharing direction-training seeds."""
        if self.held_out_pairs == 0:
            if self.held_out_seed_bases:
                raise ValueError(
                    "direction.held_out_seed_bases must be empty when held_out_pairs is zero"
                )
            return []
        return self._expand_seeds(self.held_out_seed_bases, self.held_out_pairs, "held_out")

    def _expand_seeds(self, bases: list[int], count: int, label: str) -> list[int]:
        if not bases:
            raise ValueError(f"direction.{label}_seed_bases must not be empty")
        if count < 1:
            raise ValueError(f"direction.{label}_pairs must be positive")
        if self.seed_extension_stride < 1:
            raise ValueError("direction.seed_extension_stride must be positive")
        return [
            bases[index % len(bases)]
            + (index // len(bases)) * self.seed_extension_stride
            for index in range(count)
        ]


@dataclass
class SkinToneMetricConfig:
    """Configuration for the independently measured target response."""

    id: str = "relative_cheek_CIELAB_Lstar_v1"
    face_regions: str = "geometric_cheek_mask"
    illumination_reference: str = "neutral_portrait_border"
    max_reference_lstar_shift: float = 8.0
    minimum_directional_change: float = 2.0


@dataclass
class BootstrapConfig:
    """Prespecified uncertainty calculation."""

    resamples: int = 10_000
    confidence_level: float = 0.95
    cluster_unit: str = "seed"


@dataclass
class EvaluationMatrixConfig:
    """Declared shape of the complete evaluation grid."""

    pairing_unit: str = "seed"
    expected_seeds: int = 0
    expected_methods: int = 0
    expected_alphas: int = 0
    expected_rows: int = 0
    expected_nonzero_alpha_rows: int = 0


@dataclass
class EvaluationConfig:
    """Held-out generation and measurement protocol."""

    protocol_id: str = FROZEN_EVALUATION_PROTOCOL_ID
    seeds: list[int] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    required_metrics: list[str] = field(default_factory=list)
    skin_tone_metric: SkinToneMetricConfig = field(default_factory=SkinToneMetricConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    multiplicity_correction: str = "holm"
    matrix: EvaluationMatrixConfig = field(default_factory=EvaluationMatrixConfig)
    note: str = ""


@dataclass
class ReportingConfig:
    """Prespecified reporting policy."""

    primary_outcome: str = ""
    uncertainty: str = "seed_cluster_bootstrap_confidence_interval"
    report_failures_and_missingness: bool = True
    composite_score_is_primary: bool = False


@dataclass
class DataConfig:
    """Configuration for data handling."""

    input_dir: str = "data/raw"
    output_dir: str = "experiments/results"
    cache_latents: bool = True
    latent_cache_dir: str = "data/embeddings"
    image_size: Tuple[int, int] = (512, 512)


@dataclass
class LoggingConfig:
    """Configuration for experiment logging."""

    use_wandb: bool = False
    wandb_project: str = "disentangled-race-vector"
    wandb_entity: Optional[str] = None
    save_frequency: int = 10
    log_images: bool = True
    verbose: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Study protocol
    schema_version: str = "1.0"
    study_id: str = "default"
    status: str = "development"
    attribute: str = "visual_skin_tone"
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    # Experiment metadata
    experiment_name: str = "default"
    seed: int = 42
    description: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            ExperimentConfig instance
        """
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {path}")

        # Parse nested configs
        config = cls()

        if "model" in config_dict:
            model_dict = config_dict["model"].copy()
            if "id" in model_dict:
                if "name" in model_dict:
                    raise ValueError("model must not define both 'id' and legacy 'name'")
                model_dict["name"] = model_dict.pop("id")
            config.model = ModelConfig(**model_dict)

        if "vector" in config_dict:
            # Handle tuple conversion for alpha_range
            vector_dict = config_dict["vector"].copy()
            if "alpha_range" in vector_dict:
                vector_dict["alpha_range"] = tuple(vector_dict["alpha_range"])

            # Flatten optimization settings if present
            if "optimization" in vector_dict:
                opt_dict = vector_dict.pop("optimization")
                if "enabled" in opt_dict:
                    vector_dict["optimization_enabled"] = opt_dict["enabled"]
                if "num_iterations" in opt_dict:
                    vector_dict["num_iterations"] = opt_dict["num_iterations"]
                if "learning_rate" in opt_dict:
                    vector_dict["learning_rate"] = opt_dict["learning_rate"]
                if "lambda_identity" in opt_dict:
                    vector_dict["lambda_identity"] = opt_dict["lambda_identity"]
                if "lambda_attribute" in opt_dict:
                    vector_dict["lambda_attribute"] = opt_dict["lambda_attribute"]

            config.vector = VectorConfig(**vector_dict)

        if "thresholds" in config_dict:
            config.thresholds = ThresholdConfig(**config_dict["thresholds"])

        if "direction" in config_dict:
            direction_dict = config_dict["direction"].copy()
            if "spatial_mask" in direction_dict:
                direction_dict["spatial_mask"] = SpatialMaskConfig(
                    **direction_dict["spatial_mask"]
                )
            config.direction = DirectionConfig(**direction_dict)

        if "evaluation" in config_dict:
            evaluation_dict = config_dict["evaluation"].copy()
            if "skin_tone_metric" in evaluation_dict:
                evaluation_dict["skin_tone_metric"] = SkinToneMetricConfig(
                    **evaluation_dict["skin_tone_metric"]
                )
            if "bootstrap" in evaluation_dict:
                evaluation_dict["bootstrap"] = BootstrapConfig(
                    **evaluation_dict["bootstrap"]
                )
            if "matrix" in evaluation_dict:
                evaluation_dict["matrix"] = EvaluationMatrixConfig(
                    **evaluation_dict["matrix"]
                )
            config.evaluation = EvaluationConfig(**evaluation_dict)

        if "reporting" in config_dict:
            config.reporting = ReportingConfig(**config_dict["reporting"])

        if "data" in config_dict:
            data_dict = config_dict["data"].copy()
            if "image_size" in data_dict:
                img_size = data_dict["image_size"]
                if isinstance(img_size, int):
                    data_dict["image_size"] = (img_size, img_size)
                else:
                    data_dict["image_size"] = tuple(img_size)
            config.data = DataConfig(**data_dict)

        if "logging" in config_dict:
            config.logging = LoggingConfig(**config_dict["logging"])

        # Top-level fields
        if "experiment_name" in config_dict:
            config.experiment_name = config_dict["experiment_name"]
        if "seed" in config_dict:
            config.seed = config_dict["seed"]
        if "description" in config_dict:
            config.description = config_dict["description"]

        for field_name in ("schema_version", "study_id", "status", "attribute"):
            if field_name in config_dict:
                setattr(config, field_name, config_dict[field_name])

        if "study_id" in config_dict and "experiment_name" not in config_dict:
            config.experiment_name = config.study_id

        config.validate()
        return config

    def validate(self) -> None:
        """Reject internally inconsistent study protocols before any run starts."""
        evaluation = self.evaluation
        matrix = evaluation.matrix

        # Preserve compatibility with legacy/default training-only configs.
        if self.study_id == "default" and not any(
            (evaluation.seeds, evaluation.methods, evaluation.alphas)
        ):
            return

        if not evaluation.seeds:
            raise ValueError("evaluation.seeds must not be empty")
        if evaluation.protocol_id != FROZEN_EVALUATION_PROTOCOL_ID:
            raise ValueError(
                "evaluation.protocol_id must be "
                f"{FROZEN_EVALUATION_PROTOCOL_ID}"
            )
        from src.metrics.protocol import load_protocol

        protocol = load_protocol()
        if asdict(self.thresholds) != protocol["thresholds"]:
            raise ValueError("thresholds do not match the frozen evaluation protocol")
        if evaluation.required_metrics != protocol["required_pair_metrics"]:
            raise ValueError(
                "evaluation.required_metrics do not match the frozen evaluation protocol"
            )
        expected_alphas = protocol["metrics"]["monotonicity"]["expected_alphas"]
        if evaluation.alphas != expected_alphas:
            raise ValueError("evaluation.alphas do not match the frozen evaluation protocol")
        target_protocol = protocol["metrics"]["target_response"]
        expected_target_config = {
            "id": target_protocol["id"],
            "face_regions": target_protocol["face_regions_id"],
            "illumination_reference": target_protocol["illumination_reference_id"],
            "max_reference_lstar_shift": target_protocol[
                "pair_max_absolute_reference_lstar_shift"
            ],
            "minimum_directional_change": self.thresholds.min_abs_skin_tone_change,
        }
        if asdict(evaluation.skin_tone_metric) != expected_target_config:
            raise ValueError(
                "evaluation.skin_tone_metric does not match the frozen evaluation protocol"
            )
        if len(evaluation.seeds) != len(set(evaluation.seeds)):
            raise ValueError("evaluation.seeds must be unique")
        if not evaluation.methods:
            raise ValueError("evaluation.methods must not be empty")
        if len(evaluation.methods) != len(set(evaluation.methods)):
            raise ValueError("evaluation.methods must be unique")
        if not evaluation.alphas:
            raise ValueError("evaluation.alphas must not be empty")
        if len(evaluation.alphas) != len(set(evaluation.alphas)):
            raise ValueError("evaluation.alphas must be unique")

        training_seeds = self.direction.training_seeds()
        if len(training_seeds) != len(set(training_seeds)):
            raise ValueError("expanded direction-training seeds must be unique")
        held_out_seeds = self.direction.held_out_seeds()
        if len(held_out_seeds) != len(set(held_out_seeds)):
            raise ValueError("expanded held-out pair seeds must be unique")
        pair_overlap = set(training_seeds) & set(held_out_seeds)
        if pair_overlap:
            overlap = ", ".join(str(seed) for seed in sorted(pair_overlap))
            raise ValueError(f"training and held-out pair seeds overlap: {overlap}")
        training_overlap = set(training_seeds) & set(evaluation.seeds)
        if training_overlap:
            overlap = ", ".join(str(seed) for seed in sorted(training_overlap))
            raise ValueError(f"training and evaluation seeds overlap: {overlap}")
        held_out_overlap = set(held_out_seeds) & set(evaluation.seeds)
        if held_out_overlap:
            overlap = ", ".join(str(seed) for seed in sorted(held_out_overlap))
            raise ValueError(f"held-out pair and evaluation seeds overlap: {overlap}")

        required_metrics = set(evaluation.required_metrics)
        if required_metrics != CANONICAL_REQUIRED_METRICS:
            missing = sorted(CANONICAL_REQUIRED_METRICS - required_metrics)
            extra = sorted(required_metrics - CANONICAL_REQUIRED_METRICS)
            raise ValueError(
                f"evaluation.required_metrics mismatch; missing={missing}, extra={extra}"
            )

        if matrix.pairing_unit != "seed" or evaluation.bootstrap.cluster_unit != "seed":
            raise ValueError("evaluation matrix and bootstrap must both use seed as the unit")

        expected_rows = len(evaluation.seeds) * len(evaluation.methods) * len(evaluation.alphas)
        nonzero_alphas = sum(alpha != 0 for alpha in evaluation.alphas)
        expected_nonzero = len(evaluation.seeds) * len(evaluation.methods) * nonzero_alphas
        expected_matrix = {
            "expected_seeds": len(evaluation.seeds),
            "expected_methods": len(evaluation.methods),
            "expected_alphas": len(evaluation.alphas),
            "expected_rows": expected_rows,
            "expected_nonzero_alpha_rows": expected_nonzero,
        }
        for field_name, expected in expected_matrix.items():
            actual = getattr(matrix, field_name)
            if actual != expected:
                raise ValueError(
                    f"evaluation.matrix.{field_name} is {actual}; expected {expected}"
                )

        if (
            evaluation.skin_tone_metric.minimum_directional_change
            != self.thresholds.min_abs_skin_tone_change
        ):
            raise ValueError(
                "skin-tone minimum_directional_change must match "
                "thresholds.min_abs_skin_tone_change"
            )

    def to_yaml(self, path: str):
        """
        Save configuration to YAML file.

        Args:
            path: Path to save config
        """
        config_dict = self.to_dict()

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Saved config to {path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        model_dict = asdict(self.model)

        vector_dict = asdict(self.vector)
        # Convert tuple to list for YAML serialization
        if isinstance(vector_dict.get("alpha_range"), tuple):
            vector_dict["alpha_range"] = list(vector_dict["alpha_range"])

        thresholds_dict = asdict(self.thresholds)

        data_dict = asdict(self.data)
        # Convert tuple to list for YAML serialization
        if isinstance(data_dict.get("image_size"), tuple):
            data_dict["image_size"] = list(data_dict["image_size"])

        logging_dict = asdict(self.logging)

        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "status": self.status,
            "attribute": self.attribute,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "description": self.description,
            "model": model_dict,
            "vector": vector_dict,
            "thresholds": thresholds_dict,
            "direction": asdict(self.direction),
            "evaluation": asdict(self.evaluation),
            "reporting": asdict(self.reporting),
            "data": data_dict,
            "logging": logging_dict,
        }

    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = [
            f"ExperimentConfig(name='{self.experiment_name}')",
            f"  Model: {self.model.type} ({self.model.name})",
            f"  Vector Method: {self.vector.method}",
            f"  Alpha Range: {self.vector.alpha_range}",
            f"  Optimization: {'enabled' if self.vector.optimization_enabled else 'disabled'}",
            f"  Device: {self.model.device}",
            f"  Seed: {self.seed}",
        ]
        return "\n".join(lines)


def create_default_config(output_path: Optional[str] = None) -> ExperimentConfig:
    """
    Create default configuration.

    Args:
        output_path: Optional path to save config

    Returns:
        Default ExperimentConfig
    """
    config = ExperimentConfig(
        experiment_name="default",
        description="Default experiment configuration",
    )

    if output_path:
        config.to_yaml(output_path)

    return config
