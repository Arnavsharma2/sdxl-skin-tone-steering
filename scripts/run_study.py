#!/usr/bin/env python3
"""Run the four-method skin-tone steering study from a frozen YAML manifest.

The runner is intentionally resumable.  Each condition is appended to JSONL
after generation and evaluation, and existing condition keys are skipped on a
subsequent invocation.  Failures are records, not exclusions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import numbers
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from PIL import Image

from src.metrics.skin_tone import SkinToneMetrics
from src.study.config import StudyConfig, StudyConfigError, load_study_config
from src.utils.reproducibility import collect_provenance, seed_everything


def alpha_slug(alpha: float) -> str:
    """Return a filesystem-safe, injective label for a practical alpha value."""

    sign = "p" if alpha >= 0 else "m"
    return f"{sign}{abs(float(alpha)):.3f}".replace(".", "p")


def condition_keys(config: StudyConfig, seeds: Optional[Iterable[int]] = None) -> list[tuple]:
    """Return the complete prespecified condition matrix."""

    selected = tuple(config.seeds if seeds is None else seeds)
    return [
        (int(seed), method, float(alpha))
        for seed in selected
        for method in config.methods
        for alpha in config.alphas_for(method)
    ]


def partition_seeds(
    seeds: Iterable[int],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[int, ...]:
    """Return a deterministic round-robin shard of a frozen seed sequence."""

    if shard_count < 1:
        raise StudyConfigError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise StudyConfigError("shard_index must satisfy 0 <= index < shard_count")
    selected = tuple(
        int(seed)
        for position, seed in enumerate(seeds)
        if position % shard_count == shard_index
    )
    if not selected:
        raise StudyConfigError(
            f"Shard {shard_index}/{shard_count} contains no configured seeds"
        )
    return selected


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def json_safe(value: Any) -> Any:
    """Recursively convert result values to strict, portable JSON values.

    Metric failures can legitimately produce NaN or infinity.  Those values are
    represented as JSON null so a failed metric cannot corrupt the result
    stream.  NumPy scalar values are also normalized without importing NumPy in
    this orchestration module.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file using bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_metric_missing(row: dict, required: Iterable[str]) -> list[str]:
    """Map protocol metric names to result fields and return missing values."""

    aliases = {"pose_difference": "total_pose_diff"}
    return [
        name
        for name in required
        if not _finite(row.get(aliases.get(name, name)))
    ]


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    seed: int
    light_path: Path
    dark_path: Path


class StudyRunner:
    """Resumable executor for the prespecified four-method study."""

    def __init__(
        self,
        config: StudyConfig,
        output_dir: Path,
        *,
        device: Optional[str] = None,
        allow_calibration: bool = False,
    ):
        self.config = config
        self.project_root = config.path.parent.parent
        self.output_dir = output_dir.resolve()
        self.allow_calibration = allow_calibration
        self.device_override = device
        self.results_path = self.output_dir / "results.jsonl"
        self.model = None
        self.evaluator = None
        self.skin_tone = SkinToneMetrics()
        self.raw_direction = None
        self.masked_direction = None

    def validate_execution(self) -> None:
        if self.allow_calibration:
            if self.config.status not in {"planned", "calibration", "preregistered"}:
                raise StudyConfigError("Calibration execution requires a study configuration")
        else:
            self.config.assert_confirmatory_ready()
        self._load_pair_manifest(validate_only=True)

    def _resolve_project_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _load_pair_manifest(self, *, validate_only: bool = False) -> list[PairRecord]:
        manifest_path = self._resolve_project_path(self.config.data["training_manifest"])
        if not manifest_path.exists():
            raise StudyConfigError(
                f"Training manifest not found: {manifest_path}. Generate paired data first."
            )
        if not self.allow_calibration:
            expected_manifest_hash = self.config.data.get("training_manifest_sha256")
            observed_manifest_hash = sha256_file(manifest_path)
            if observed_manifest_hash != expected_manifest_hash:
                raise StudyConfigError(
                    "Training manifest SHA-256 does not match the preregistered config"
                )
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected_metadata = {
            "model_id": str(self.config.model["id"]),
            "model_revision": str(self.config.model["revision"]),
            "inference_steps": int(self.config.model["inference_steps"]),
            "guidance_scale": float(self.config.model["guidance_scale"]),
            "height": int(self.config.model["height"]),
            "width": int(self.config.model["width"]),
        }
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in expected_metadata.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise StudyConfigError(
                f"Pair manifest generation metadata does not match study: {mismatches}"
            )
        if (
            not self.allow_calibration
            and (
                manifest.get("generation_observed_in_this_run") is not True
                or manifest.get("generation_observed_in_campaign") is not True
            )
        ):
            raise StudyConfigError(
                "Confirmatory data manifest must attest that every image was "
                "observed in the resumable generation campaign"
            )
        if not self.allow_calibration:
            ledger_value = manifest.get("generation_ledger")
            ledger_hash = manifest.get("generation_ledger_sha256")
            if not ledger_value or not isinstance(ledger_hash, str):
                raise StudyConfigError(
                    "Confirmatory data manifest must include its generation ledger hash"
                )
            ledger_path = self._resolve_project_path(ledger_value)
            if not ledger_path.is_file() or sha256_file(ledger_path) != ledger_hash:
                raise StudyConfigError(
                    "Generation ledger is missing or does not match the paired-data manifest"
                )
        pairs = []
        for item in manifest.get("pairs", []):
            light = self._resolve_project_path(item["light"]["path"])
            dark = self._resolve_project_path(item["dark"]["path"])
            pairs.append(
                PairRecord(
                    pair_id=str(item["pair_id"]),
                    seed=int(item["seed"]),
                    light_path=light,
                    dark_path=dark,
                )
            )
        needed = int(self.config.direction["train_pairs"]) + int(
            self.config.direction["held_out_pairs"]
        )
        if len(pairs) < needed:
            raise StudyConfigError(
                f"Training manifest has {len(pairs)} pairs; study requires {needed}"
            )
        selected = pairs[:needed]
        missing = [
            str(path)
            for pair in selected
            for path in (pair.light_path, pair.dark_path)
            if not path.is_file()
        ]
        if missing:
            preview = ", ".join(missing[:3])
            raise StudyConfigError(f"Paired image files are missing: {preview}")
        if len({pair.seed for pair in selected}) != len(selected):
            raise StudyConfigError("Pair manifest seeds must be unique before splitting")
        overlap = sorted({pair.seed for pair in selected} & set(self.config.seeds))
        if overlap:
            raise StudyConfigError(
                f"Direction-pair seeds overlap evaluation seeds: {overlap}"
            )
        for item, pair in zip(manifest.get("pairs", [])[:needed], selected):
            for group, path in (("light", pair.light_path), ("dark", pair.dark_path)):
                expected = item.get(group, {}).get("sha256")
                if not expected:
                    raise StudyConfigError(
                        f"Pair {pair.pair_id} {group} image has no recorded SHA-256"
                    )
                observed = sha256_file(path)
                if observed != expected:
                    raise StudyConfigError(
                        f"Pair {pair.pair_id} {group} image hash mismatch: {path}"
                    )
        if validate_only:
            return selected
        return selected[: int(self.config.direction["train_pairs"])]

    def _prepare_output(
        self,
        execution_seeds: tuple[int, ...],
        *,
        partition: Optional[dict] = None,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "run_manifest.json"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("config_fingerprint") != self.config.fingerprint:
                raise StudyConfigError(
                    "Output directory belongs to a different configuration fingerprint"
                )
            if tuple(existing.get("execution_seeds", ())) != execution_seeds:
                raise StudyConfigError(
                    "Output directory belongs to a different execution seed set"
                )
            return

        snapshot_path = self.output_dir / "study_config.yaml"
        with snapshot_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(self.config.raw), handle, sort_keys=False)
        manifest = {
            "schema_version": "1.0",
            "study_id": self.config.study_id,
            "study_status": self.config.status,
            "execution_mode": "calibration" if self.allow_calibration else "confirmatory",
            "config_fingerprint": self.config.fingerprint,
            "execution_seeds": list(execution_seeds),
            "execution_partition": partition,
            "condition_count": len(condition_keys(self.config, execution_seeds)),
            "generation_device": self._select_device(),
            "metric_device": self._metric_device(),
            "provenance": collect_provenance(self.project_root),
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write("\n")

    def _select_device(self) -> str:
        if self.device_override:
            return self.device_override
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _metric_device(self) -> str:
        """Keep auxiliary metric models off MPS beside the large SDXL model."""

        generation_device = self._select_device()
        return "cpu" if generation_device == "mps" else generation_device

    def _load_model(self) -> None:
        if self.model is not None:
            return
        import torch

        from src.models.stable_diffusion import StableDiffusionWrapper

        device = self._select_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        revision = self.config.model.get("revision") or None
        self.model = StableDiffusionWrapper(
            device=device,
            dtype=dtype,
            model_id=str(self.config.model["id"]),
            model_revision=revision,
            local_files_only=bool(self.config.model.get("local_files_only", False)),
            enable_xformers=device == "cuda",
            enable_cpu_offload=device == "cpu",
        )

    def _load_evaluator(self) -> None:
        if self.evaluator is not None:
            return
        from src.metrics.evaluator import CounterfactualEvaluator

        self.evaluator = CounterfactualEvaluator(device=self._metric_device())

    def _release_condition_memory(self) -> None:
        """Release temporary tensors without unloading the reusable SDXL model."""

        gc.collect()
        if self._select_device() == "mps":
            import torch

            torch.mps.empty_cache()

    def _direction_paths(self) -> tuple[Path, Path]:
        direction_dir = self.output_dir / "direction"
        return direction_dir / "raw.pt", direction_dir / "masked.pt"

    def _load_or_estimate_directions(self) -> None:
        import torch

        from src.latent.vector_discovery import SkinToneDirectionExtractor

        raw_path, masked_path = self._direction_paths()
        if raw_path.exists() and masked_path.exists():
            self.raw_direction = torch.load(
                raw_path, map_location=self._select_device(), weights_only=True
            )
            self.masked_direction = torch.load(
                masked_path, map_location=self._select_device(), weights_only=True
            )
            return

        self._load_model()
        pairs = self._load_pair_manifest()
        image_size = int(self.config.data["image_size"])
        light_latents = []
        dark_latents = []
        for index, pair in enumerate(pairs, start=1):
            print(f"Encoding direction pair {index}/{len(pairs)}: {pair.pair_id}")
            light = Image.open(pair.light_path).convert("RGB")
            dark = Image.open(pair.dark_path).convert("RGB")
            light_latents.append(
                self.model.encode_image(light, size=(image_size, image_size))
            )
            dark_latents.append(
                self.model.encode_image(dark, size=(image_size, image_size))
            )

        extractor = SkinToneDirectionExtractor(device=self._select_device())
        self.raw_direction = extractor.extract_from_pairs(light_latents, dark_latents)
        height, width = self.raw_direction.shape[-2:]
        mask_spec = self.config.direction["spatial_mask"]
        mask = extractor.create_center_mask(
            height,
            width,
            center_weight=float(mask_spec["center_weight"]),
            edge_weight=float(mask_spec["edge_weight"]),
            radius=float(mask_spec["radius"]),
        )
        self.masked_direction = self.raw_direction * mask.view(
            *((1,) * (self.raw_direction.ndim - 2)), height, width
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.raw_direction.detach().cpu(), raw_path)
        torch.save(self.masked_direction.detach().cpu(), masked_path)

    def _existing_keys(self) -> set[tuple[int, str, float]]:
        keys = set()
        if not self.results_path.exists():
            return keys
        self._repair_incomplete_tail()
        with self.results_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    keys.add((int(row["seed"]), str(row["method"]), float(row["alpha"])))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise StudyConfigError(
                        f"Malformed result row {line_number} in {self.results_path}: {exc}"
                    ) from exc
        return keys

    def _repair_incomplete_tail(self) -> None:
        """Remove only an unterminated, malformed final JSONL fragment.

        A process interruption may leave the final record incomplete.  Earlier
        malformed records, or malformed newline-terminated records, remain hard
        errors because silently discarding those could bias the study.
        """

        payload = self.results_path.read_bytes()
        if not payload or payload.endswith(b"\n"):
            return
        final_start = payload.rfind(b"\n") + 1
        fragment = payload[final_start:]
        try:
            json.loads(fragment)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.results_path.write_bytes(payload[:final_start])
            print(f"Repaired incomplete final result fragment in {self.results_path}")

    def _append_result(self, row: dict) -> None:
        # Serialize before opening the stream so conversion errors cannot leave
        # a partially written record.
        payload = json.dumps(json_safe(row), allow_nan=False, default=str)
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()

    def _seed_dir(self, seed: int) -> Path:
        return self.output_dir / "images" / f"seed_{seed}"

    def _load_or_generate_base(self, seed: int) -> tuple[Image.Image, Any]:
        import torch

        seed_dir = self._seed_dir(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        image_path = seed_dir / "base.png"
        latent_path = seed_dir / "base_latent.pt"
        if image_path.exists() and latent_path.exists():
            return (
                Image.open(image_path).convert("RGB"),
                torch.load(
                    latent_path,
                    map_location=self._select_device(),
                    weights_only=True,
                ),
            )
        self._load_model()
        image, latent = self.model.generate_from_prompt(
            str(self.config.prompts["base"]),
            negative_prompt=str(self.config.prompts["negative"]),
            seed=seed,
            num_inference_steps=int(self.config.model["inference_steps"]),
            guidance_scale=float(self.config.model["guidance_scale"]),
            height=int(self.config.model["height"]),
            width=int(self.config.model["width"]),
        )
        image.save(image_path)
        torch.save(latent.detach().cpu(), latent_path)
        return image, latent

    def _condition_path(self, seed: int, method: str, alpha: float) -> Path:
        return self._seed_dir(seed) / method / f"alpha_{alpha_slug(alpha)}.png"

    def _generate_condition(
        self,
        seed: int,
        method: str,
        alpha: float,
        base_image: Image.Image,
        base_latent: Any,
    ) -> tuple[Image.Image, Path]:
        output_path = self._condition_path(seed, method, alpha)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            return Image.open(output_path).convert("RGB"), output_path
        if abs(alpha) < 1e-12:
            shutil.copyfile(self._seed_dir(seed) / "base.png", output_path)
            return base_image.copy(), output_path

        self._load_model()
        if method == "prompt_only":
            image, _ = self.model.generate_from_prompt(
                self.config.prompt_for(method, alpha),
                negative_prompt=str(self.config.prompts["negative"]),
                seed=seed,
                num_inference_steps=int(self.config.model["inference_steps"]),
                guidance_scale=float(self.config.model["guidance_scale"]),
                height=int(self.config.model["height"]),
                width=int(self.config.model["width"]),
            )
        elif method == "posthoc_latent":
            from src.latent.manipulator import LatentManipulator

            modified = LatentManipulator(device=self._select_device()).apply_vector(
                base_latent, self.masked_direction, alpha
            )
            image = self.model.decode_latent(modified)
        elif method in {"stepwise_unmasked", "stepwise_masked"}:
            direction = (
                self.raw_direction if method == "stepwise_unmasked" else self.masked_direction
            )
            image, _ = self.model.generate_steered(
                prompt=str(self.config.prompts["base"]),
                race_vector=direction,
                alpha=alpha,
                seed=seed,
                negative_prompt=str(self.config.prompts["negative"]),
                num_inference_steps=int(self.config.model["inference_steps"]),
                guidance_scale=float(self.config.model["guidance_scale"]),
                height=int(self.config.model["height"]),
                width=int(self.config.model["width"]),
            )
        else:
            raise StudyConfigError(f"Unsupported method: {method}")
        image.save(output_path)
        return image, output_path

    def _evaluate_condition(
        self,
        base_image: Image.Image,
        edited: Image.Image,
    ) -> dict:
        self._load_evaluator()
        preservation = self.evaluator.evaluate_pair(base_image, edited).to_dict()
        target = self.skin_tone.compare(base_image, edited)
        row = {**preservation, **target}
        missing = required_metric_missing(
            row, self.config.evaluation["required_metrics"]
        )
        row["missing_required_metrics"] = missing
        row["evaluation_complete"] = not missing
        return row

    def run(
        self,
        *,
        max_seeds: Optional[int] = None,
        seeds_override: Optional[Iterable[int]] = None,
        shard_index: Optional[int] = None,
        shard_count: Optional[int] = None,
    ) -> None:
        self.validate_execution()
        if max_seeds is not None and not self.allow_calibration:
            raise StudyConfigError(
                "--max-seeds is calibration-only; use deterministic sharding for "
                "confirmatory execution"
            )
        if (shard_index is None) != (shard_count is None):
            raise StudyConfigError("--shard-index and --shard-count must be used together")
        if seeds_override is not None and shard_index is not None:
            raise StudyConfigError("--seeds cannot be combined with deterministic sharding")
        if seeds_override is not None:
            if not self.allow_calibration:
                raise StudyConfigError("Seed overrides are forbidden for confirmatory runs")
            override = tuple(int(seed) for seed in seeds_override)
            if not override or len(override) != len(set(override)):
                raise StudyConfigError("Calibration seed overrides must be non-empty and unique")
            overlap = sorted(set(override) & set(self.config.seeds))
            if overlap:
                raise StudyConfigError(
                    f"Calibration seeds overlap confirmatory seeds: {overlap}"
                )
            seeds = override
        else:
            seeds = self.config.seeds
        partition = None
        if shard_index is not None and shard_count is not None:
            seeds = partition_seeds(
                seeds,
                shard_index=shard_index,
                shard_count=shard_count,
            )
            partition = {
                "strategy": "round_robin_config_order",
                "shard_index": int(shard_index),
                "shard_count": int(shard_count),
            }
        seeds = seeds[:max_seeds] if max_seeds else seeds
        self._prepare_output(seeds, partition=partition)
        self._load_model()
        self._load_or_estimate_directions()
        completed = self._existing_keys()
        total = len(condition_keys(self.config, seeds))
        index = 0
        for seed in seeds:
            seed_everything(seed)
            base_image, base_latent = self._load_or_generate_base(seed)
            for method in self.config.methods:
                for alpha in self.config.alphas_for(method):
                    index += 1
                    key = (seed, method, alpha)
                    if key in completed:
                        print(f"[{index}/{total}] resume skip {key}")
                        continue
                    print(f"[{index}/{total}] seed={seed} method={method} alpha={alpha:+g}")
                    row = {
                        "schema_version": "1.0",
                        "study_id": self.config.study_id,
                        "config_fingerprint": self.config.fingerprint,
                        "seed": seed,
                        "method": method,
                        "alpha": alpha,
                        "generation_complete": False,
                        "evaluation_complete": False,
                    }
                    try:
                        edited, path = self._generate_condition(
                            seed, method, alpha, base_image, base_latent
                        )
                        row["image_path"] = str(path.relative_to(self.output_dir))
                        row["generation_complete"] = True
                        row.update(self._evaluate_condition(base_image, edited))
                    except Exception as exc:  # failures are prespecified outcomes
                        row.update(
                            {
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "missing_required_metrics": list(
                                    self.config.evaluation["required_metrics"]
                                ),
                                "traceback": traceback.format_exc(limit=8),
                            }
                        )
                    self._append_result(row)
                    self._release_condition_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/runs/full_study"))
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Zero-based deterministic shard index; requires --shard-count",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help="Number of deterministic seed shards; requires --shard-index",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Disjoint calibration seeds; forbidden for confirmatory execution",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-calibration",
        action="store_true",
        help="Permit a planned/calibration manifest; never label the output confirmatory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    runner = StudyRunner(
        config,
        args.output,
        device=args.device,
        allow_calibration=args.allow_calibration,
    )
    if args.validate_only:
        runner.validate_execution()
        print(
            f"Valid {config.status} study {config.study_id}: "
            f"{len(condition_keys(config))} conditions, fingerprint={config.fingerprint}"
        )
        return
    runner.run(
        max_seeds=args.max_seeds,
        seeds_override=args.seeds,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )


if __name__ == "__main__":
    main()
