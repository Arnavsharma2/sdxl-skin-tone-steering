#!/usr/bin/env python3
"""Plan or execute the frozen matched confirmatory generation matrix.

Planning is the default and never loads SDXL. Expensive generation requires
the explicit ``--execute`` flag plus an exact model revision and direction
artifact. The runner writes a study manifest and one checkpointed metadata
file per evaluation seed so failures remain auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from src.latent.vector_discovery import SkinToneDirectionExtractor
from src.utils.config import ExperimentConfig
from src.utils.reproducibility import collect_provenance, seed_everything, stable_fingerprint

SUPPORTED_METHODS = (
    "prompt_only",
    "posthoc_latent",
    "stepwise_unmasked",
    "stepwise_masked",
)
RUNNER_SCHEMA_VERSION = "1.0"
PROMPT_POLICY_ID = "matched_portrait_prompt_with_directional_descriptor_v1"

BASE_PROMPT = (
    "professional headshot portrait, neutral expression, clean white studio background, "
    "soft diffused studio lighting, sharp focus on face, centered composition, "
    "natural skin tones, no glasses no jewelry no hat, facing camera directly, "
    "high quality photography, 85mm lens"
)
NEGATIVE_PROMPT = (
    "multiple people, accessories, sunglasses, jewelry, hat, cap, hood, blurry, "
    "low quality, cartoon, illustration, painting, watermark, text, extreme lighting, "
    "heavy shadows, overexposed, underexposed, cropped face"
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON so interrupted checkpoints are not truncated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def alpha_slug(alpha: float) -> str:
    """Return a stable filesystem representation of an alpha value."""
    rendered = format(float(alpha), "+.8g")
    return rendered.replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def prompt_for_alpha(alpha: float, alphas: Iterable[float]) -> str:
    """Change only the visible-skin-tone descriptor for the prompt baseline."""
    if alpha == 0:
        return BASE_PROMPT
    max_abs_alpha = max(abs(value) for value in alphas)
    strength = "subtly" if abs(alpha) < max_abs_alpha else "distinctly"
    direction = "darker" if alpha > 0 else "lighter"
    return f"{BASE_PROMPT}, {strength} {direction} visible skin tone"


@dataclass(frozen=True)
class PlannedRow:
    """One prespecified seed × method × alpha cell."""

    seed: int
    method: str
    alpha: float
    effective_prompt: str
    output_path: str

    @property
    def row_id(self) -> str:
        return f"seed_{self.seed}__{self.method}__alpha_{alpha_slug(self.alpha)}"

    def to_result(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "seed": self.seed,
            "method": self.method,
            "alpha": self.alpha,
            "status": "planned",
            "effective_prompt": self.effective_prompt,
            "output_path": self.output_path,
            "image_sha256": None,
            "base_image_sha256": None,
            "duration_seconds": None,
            "failure": None,
        }


def build_plan(config: ExperimentConfig) -> list[PlannedRow]:
    """Expand the validated configuration into the exact declared matrix."""
    unknown = sorted(set(config.evaluation.methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unsupported confirmatory methods: {', '.join(unknown)}")

    rows = []
    for seed in config.evaluation.seeds:
        for method in config.evaluation.methods:
            for alpha in config.evaluation.alphas:
                prompt = (
                    prompt_for_alpha(alpha, config.evaluation.alphas)
                    if method == "prompt_only"
                    else BASE_PROMPT
                )
                relative_path = (
                    Path("seeds")
                    / str(seed)
                    / "images"
                    / method
                    / f"alpha_{alpha_slug(alpha)}.png"
                )
                rows.append(
                    PlannedRow(
                        seed=seed,
                        method=method,
                        alpha=float(alpha),
                        effective_prompt=prompt,
                        output_path=relative_path.as_posix(),
                    )
                )

    if len(rows) != config.evaluation.matrix.expected_rows:
        raise ValueError(
            f"Expanded {len(rows)} rows; expected {config.evaluation.matrix.expected_rows}"
        )
    nonzero_rows = sum(row.alpha != 0 for row in rows)
    if nonzero_rows != config.evaluation.matrix.expected_nonzero_alpha_rows:
        raise ValueError(
            f"Expanded {nonzero_rows} nonzero-alpha rows; expected "
            f"{config.evaluation.matrix.expected_nonzero_alpha_rows}"
        )
    return rows


def _resize_direction(direction: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Move and resize a direction to match a generated latent."""
    resized = direction.to(device=latent.device, dtype=latent.dtype)
    if resized.dim() == 3:
        resized = resized.unsqueeze(0)
    if latent.dim() == 3:
        latent_shape = latent.unsqueeze(0).shape
    else:
        latent_shape = latent.shape
    if resized.shape[-2:] != latent_shape[-2:]:
        resized = F.interpolate(
            resized,
            size=latent_shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if resized.shape[1] != latent_shape[1]:
        raise ValueError(
            f"Direction channels {resized.shape[1]} do not match latent channels {latent_shape[1]}"
        )
    return resized.squeeze(0) if latent.dim() == 3 else resized


def _masked_direction(direction: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    """Apply the frozen spatial mask to an unmasked direction artifact."""
    mask_config = config.direction.spatial_mask
    extractor = SkinToneDirectionExtractor(device=str(direction.device))
    mask = extractor.create_center_mask(
        height=direction.shape[-2],
        width=direction.shape[-1],
        center_weight=mask_config.center_weight,
        edge_weight=mask_config.edge_weight,
        falloff=mask_config.type.removesuffix("_center"),
        radius=mask_config.radius,
    )
    while mask.dim() < direction.dim():
        mask = mask.unsqueeze(0)
    return direction * mask.to(device=direction.device, dtype=direction.dtype)


def load_direction(path: str | Path, device: str) -> torch.Tensor:
    """Load a finite 3-D or 4-D direction tensor from a release artifact."""
    artifact = torch.load(path, map_location=device, weights_only=True)
    if isinstance(artifact, dict):
        artifact = artifact.get("direction")
    if not isinstance(artifact, torch.Tensor):
        raise ValueError("Direction artifact must be a tensor or contain a 'direction' tensor")
    if artifact.dim() not in (3, 4):
        raise ValueError("Direction tensor must have three or four dimensions")
    if not torch.isfinite(artifact).all():
        raise ValueError("Direction tensor contains non-finite values")
    return artifact.to(device)


ModelFactory = Callable[[ExperimentConfig, str, str], Any]


class ConfirmatoryRunner:
    """Materialize the manifest and optionally execute its generation cells."""

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path,
        *,
        direction_path: str | Path | None = None,
        model_revision: str | None = None,
        device: str | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.project_root = Path(__file__).parents[1].resolve()
        self.config_path = Path(config_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.direction_path = Path(direction_path).resolve() if direction_path else None
        self.model_revision = model_revision
        self.device = device or self._default_device()
        self.config = ExperimentConfig.from_yaml(self.config_path)
        self.rows = build_plan(self.config)
        self.model_factory = model_factory or self._default_model_factory
        self.config_sha256 = sha256_file(self.config_path)
        self.provenance = collect_provenance(self.project_root)
        self.direction_metadata = self._direction_metadata()
        self.run_id = stable_fingerprint(
            {
                "study_id": self.config.study_id,
                "config_sha256": self.config_sha256,
                "model_id": self.config.model.name,
                "model_revision": self.model_revision,
                "direction_sha256": self.direction_metadata["sha256"],
                "code_commit": self.provenance["git"]["commit"],
            },
            length=20,
        )

    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _default_model_factory(
        config: ExperimentConfig,
        revision: str,
        device: str,
    ) -> Any:
        from src.models.stable_diffusion import StableDiffusionWrapper

        dtype = torch.float16 if device == "cuda" else torch.float32
        return StableDiffusionWrapper(
            device=device,
            dtype=dtype,
            model_id=config.model.name,
            revision=revision,
            enable_xformers=config.model.enable_xformers,
            enable_cpu_offload=config.model.enable_cpu_offload,
        )

    def _direction_metadata(self) -> dict[str, Any]:
        if self.direction_path is None:
            return {"path": None, "sha256": None, "size_bytes": None}
        if not self.direction_path.is_file():
            return {
                "path": str(self.direction_path),
                "sha256": None,
                "size_bytes": None,
            }
        return {
            "path": str(self.direction_path),
            "sha256": sha256_file(self.direction_path),
            "size_bytes": self.direction_path.stat().st_size,
        }

    def generation_settings(self) -> dict[str, Any]:
        width, height = self.config.data.image_size
        return {
            "base_prompt": BASE_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "prompt_policy_id": PROMPT_POLICY_ID,
            "scheduler": self.config.model.scheduler,
            "inference_steps": self.config.model.inference_steps,
            "guidance_scale": self.config.model.guidance_scale,
            "width": width,
            "height": height,
        }

    def _manifest(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "study_id": self.config.study_id,
            "run_id": self.run_id,
            "status": status,
            "created_at_utc": utc_now(),
            "terminology": {
                "estimated_attribute": "rendered visual skin tone",
                "explicitly_not_estimated": ["race", "ethnicity", "personal identity"],
            },
            "config": {
                "path": str(self.config_path),
                "sha256": self.config_sha256,
                "parsed": self.config.to_dict(),
            },
            "model": {
                "id": self.config.model.name,
                "requested_revision": self.model_revision,
                "resolved_revision": None,
            },
            "direction_artifact": deepcopy(self.direction_metadata),
            "generation": self.generation_settings(),
            "thresholds": asdict(self.config.thresholds),
            "metric_protocol": asdict(self.config.evaluation.skin_tone_metric),
            "matrix": {
                **asdict(self.config.evaluation.matrix),
                "methods": list(self.config.evaluation.methods),
                "seeds": list(self.config.evaluation.seeds),
                "alphas": list(self.config.evaluation.alphas),
            },
            "provenance": self.provenance,
            "execution": {
                "device": self.device,
                "argv": list(sys.argv),
                "started_at_utc": None,
                "finished_at_utc": None,
            },
            "summary": {
                "planned_rows": len(self.rows),
                "completed_rows": 0,
                "failed_rows": 0,
                "unattempted_rows": len(self.rows),
            },
            "failures": [],
            "rows": [row.to_result() for row in self.rows],
        }

    def _seed_metadata(
        self,
        seed: int,
        results: list[dict[str, Any]],
        *,
        status: str,
        resolved_revision: str | None = None,
        base_artifact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        failures = [result["failure"] for result in results if result["failure"]]
        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "study_id": self.config.study_id,
            "run_id": self.run_id,
            "status": status,
            "seed": seed,
            "study_manifest": "../../study_manifest.json",
            "config_sha256": self.config_sha256,
            "model_id": self.config.model.name,
            "model_revision": resolved_revision or self.model_revision,
            "direction_artifact": deepcopy(self.direction_metadata),
            "generation": self.generation_settings(),
            "thresholds": asdict(self.config.thresholds),
            "provenance": self.provenance,
            "base_artifact": base_artifact,
            "failures": failures,
            "results": results,
        }

    def _rows_for_seed(self, seed: int) -> list[PlannedRow]:
        return [row for row in self.rows if row.seed == seed]

    def write_plan(self) -> Path:
        """Write the complete plan without importing or loading the model."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest("planned")
        write_json(self.output_dir / "study_manifest.json", manifest)
        for seed in self.config.evaluation.seeds:
            results = [row.to_result() for row in self._rows_for_seed(seed)]
            metadata = self._seed_metadata(seed, results, status="planned")
            write_json(self.output_dir / "seeds" / str(seed) / "metadata.json", metadata)
        return self.output_dir / "study_manifest.json"

    def _validate_execution_inputs(self) -> None:
        if not self.model_revision:
            raise ValueError("--execute requires --model-revision")
        if self.direction_path is None or not self.direction_path.is_file():
            raise ValueError("--execute requires an existing --direction artifact")
        if self.provenance["git"]["dirty"]:
            raise ValueError("Refusing confirmatory execution from a dirty worktree")

    def _model_kwargs(self, prompt: str, seed: int) -> dict[str, Any]:
        generation = self.generation_settings()
        return {
            "prompt": prompt,
            "negative_prompt": generation["negative_prompt"],
            "seed": seed,
            "num_inference_steps": generation["inference_steps"],
            "guidance_scale": generation["guidance_scale"],
            "height": generation["height"],
            "width": generation["width"],
        }

    def _generate_row(
        self,
        model: Any,
        row: PlannedRow,
        base_image: Any,
        base_latent: torch.Tensor,
        direction: torch.Tensor,
        masked_direction: torch.Tensor,
    ) -> Any:
        if row.alpha == 0:
            return base_image.copy()
        if row.method == "prompt_only":
            image, _ = model.generate_from_prompt(
                **self._model_kwargs(row.effective_prompt, row.seed)
            )
            return image
        if row.method == "posthoc_latent":
            matched_direction = _resize_direction(direction, base_latent)
            return model.decode_latent(base_latent + row.alpha * matched_direction)
        if row.method == "stepwise_unmasked":
            image, _ = model.generate_steered(
                race_vector=direction,
                alpha=row.alpha,
                **self._model_kwargs(BASE_PROMPT, row.seed),
            )
            return image
        if row.method == "stepwise_masked":
            image, _ = model.generate_steered(
                race_vector=masked_direction,
                alpha=row.alpha,
                **self._model_kwargs(BASE_PROMPT, row.seed),
            )
            return image
        raise AssertionError(f"Unreachable method: {row.method}")

    @staticmethod
    def _failure(stage: str, error: BaseException, row_id: str | None = None) -> dict[str, Any]:
        return {
            "row_id": row_id,
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "recorded_at_utc": utc_now(),
        }

    def execute(self) -> Path:
        """Execute the declared matrix, checkpointing every attempted row."""
        self._validate_execution_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest("running")
        manifest["execution"]["started_at_utc"] = utc_now()
        manifest_path = self.output_dir / "study_manifest.json"
        write_json(manifest_path, manifest)

        try:
            direction = load_direction(self.direction_path, self.device)
            masked_direction = _masked_direction(direction, self.config)
            model = self.model_factory(self.config, self.model_revision, self.device)
            resolved_revision = getattr(model, "resolved_revision", None)
            manifest["model"]["resolved_revision"] = resolved_revision
            write_json(manifest_path, manifest)
        except Exception as error:
            failure = self._failure("setup", error)
            manifest["status"] = "setup_failed"
            manifest["failures"] = [failure]
            manifest["execution"]["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            raise

        row_lookup = {result["row_id"]: result for result in manifest["rows"]}
        try:
            for seed in self.config.evaluation.seeds:
                seed_everything(seed)
                planned_rows = self._rows_for_seed(seed)
                results = [deepcopy(row_lookup[row.row_id]) for row in planned_rows]
                result_lookup = {result["row_id"]: result for result in results}
                seed_dir = self.output_dir / "seeds" / str(seed)
                base_path = seed_dir / "base.png"
                base_artifact = {
                    "path": base_path.relative_to(self.output_dir).as_posix(),
                    "sha256": None,
                }

                try:
                    base_image, base_latent = model.generate_from_prompt(
                        **self._model_kwargs(BASE_PROMPT, seed)
                    )
                    if not isinstance(base_latent, torch.Tensor):
                        raise TypeError("Base generation did not return a latent tensor")
                    base_path.parent.mkdir(parents=True, exist_ok=True)
                    base_image.save(base_path)
                    base_artifact["sha256"] = sha256_file(base_path)
                except Exception as error:
                    for row in planned_rows:
                        result = result_lookup[row.row_id]
                        result["status"] = "failed"
                        result["failure"] = self._failure(
                            "base_generation", error, row.row_id
                        )
                        row_lookup[row.row_id] = deepcopy(result)
                    metadata = self._seed_metadata(
                        seed,
                        results,
                        status="failed",
                        resolved_revision=resolved_revision,
                        base_artifact=base_artifact,
                    )
                    write_json(seed_dir / "metadata.json", metadata)
                    continue

                for row in planned_rows:
                    result = result_lookup[row.row_id]
                    started = time.perf_counter()
                    try:
                        image = self._generate_row(
                            model,
                            row,
                            base_image,
                            base_latent,
                            direction,
                            masked_direction,
                        )
                        output_path = self.output_dir / row.output_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(output_path)
                        result.update(
                            {
                                "status": "complete",
                                "image_sha256": sha256_file(output_path),
                                "base_image_sha256": base_artifact["sha256"],
                                "failure": None,
                            }
                        )
                    except Exception as error:
                        result["status"] = "failed"
                        result["failure"] = self._failure("row_generation", error, row.row_id)
                    finally:
                        result["duration_seconds"] = time.perf_counter() - started
                        row_lookup[row.row_id] = deepcopy(result)
                        seed_status = (
                            "partial_failed"
                            if any(item["status"] == "failed" for item in results)
                            else "running"
                        )
                        metadata = self._seed_metadata(
                            seed,
                            results,
                            status=seed_status,
                            resolved_revision=resolved_revision,
                            base_artifact=base_artifact,
                        )
                        write_json(seed_dir / "metadata.json", metadata)

                seed_status = (
                    "partial_failed"
                    if any(item["status"] == "failed" for item in results)
                    else "complete"
                )
                metadata = self._seed_metadata(
                    seed,
                    results,
                    status=seed_status,
                    resolved_revision=resolved_revision,
                    base_artifact=base_artifact,
                )
                write_json(seed_dir / "metadata.json", metadata)
        except KeyboardInterrupt as error:
            manifest["status"] = "interrupted"
            manifest["failures"].append(self._failure("execution", error))
            raise
        except Exception as error:
            manifest["status"] = "execution_failed"
            manifest["failures"].append(self._failure("execution", error))
            raise
        finally:
            manifest["rows"] = [row_lookup[row.row_id] for row in self.rows]
            manifest["failures"] = [
                result["failure"] for result in manifest["rows"] if result["failure"]
            ] + [failure for failure in manifest["failures"] if failure.get("row_id") is None]
            completed = sum(result["status"] == "complete" for result in manifest["rows"])
            failed = sum(result["status"] == "failed" for result in manifest["rows"])
            unattempted = len(manifest["rows"]) - completed - failed
            manifest["summary"].update(
                {
                    "completed_rows": completed,
                    "failed_rows": failed,
                    "unattempted_rows": unattempted,
                }
            )
            if manifest["status"] not in {"interrupted", "execution_failed"}:
                manifest["status"] = "complete" if failed == 0 else "partial_failed"
            manifest["execution"]["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
        return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full_study.yaml"),
        help="Frozen study configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/runs/confirmatory_v1"),
        help="Manifest and artifact directory",
    )
    parser.add_argument("--direction", type=Path, help="Unmasked direction tensor artifact")
    parser.add_argument("--model-revision", help="Exact Hugging Face model revision")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run expensive generation; omitted by default so only the plan is written",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = ConfirmatoryRunner(
        args.config,
        args.output,
        direction_path=args.direction,
        model_revision=args.model_revision,
        device=args.device,
    )
    manifest_path = runner.execute() if args.execute else runner.write_plan()
    mode = "Executed" if args.execute else "Planned"
    print(f"{mode} {len(runner.rows)} rows: {manifest_path}")
    if not args.execute:
        print("No model was loaded and no images were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
