"""
Configuration management for experiments.

This module provides configuration dataclasses and utilities.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for generative model."""

    type: str = "stable_diffusion"  # stable_diffusion or stylegan
    name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = "cuda"
    dtype: str = "float16"  # float16 or float32
    enable_xformers: bool = True
    enable_cpu_offload: bool = False


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
    """Evaluation thresholds for disentanglement."""

    face_similarity: float = 0.85
    landmark_rmse: float = 5.0
    lpips: float = 0.3
    background_ssim: float = 0.90
    pose_angle_diff: float = 5.0


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
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse nested configs
        config = cls()

        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])

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

        return config

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
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "description": self.description,
            "model": model_dict,
            "vector": vector_dict,
            "thresholds": thresholds_dict,
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
