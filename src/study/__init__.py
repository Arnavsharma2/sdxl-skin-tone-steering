"""Config-driven confirmatory study infrastructure."""

from .config import StudyConfig, StudyConfigError, load_study_config

__all__ = ["StudyConfig", "StudyConfigError", "load_study_config"]
