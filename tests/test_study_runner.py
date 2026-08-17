import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_study import (
    StudyRunner,
    alpha_slug,
    condition_keys,
    json_safe,
    partition_seeds,
    required_metric_missing,
    sha256_file,
)
from src.study.config import StudyConfigError, load_study_config

ROOT = Path(__file__).resolve().parents[1]


def test_condition_matrix_covers_every_seed_method_and_alpha():
    config = load_study_config(ROOT / "configs/full_study.yaml")
    keys = condition_keys(config)
    expected_per_seed = sum(len(config.alphas_for(method)) for method in config.methods)
    assert len(keys) == len(config.seeds) * expected_per_seed
    assert len(keys) == len(set(keys))


def test_alpha_slug_is_sign_sensitive():
    assert alpha_slug(-0.75) == "m0p750"
    assert alpha_slug(0.75) == "p0p750"
    assert alpha_slug(0.0) != alpha_slug(-0.001)


def test_seed_shards_are_disjoint_and_cover_frozen_order():
    seeds = tuple(range(10))
    shards = [
        partition_seeds(seeds, shard_index=index, shard_count=3)
        for index in range(3)
    ]

    assert shards == [(0, 3, 6, 9), (1, 4, 7), (2, 5, 8)]
    assert set().union(*(set(shard) for shard in shards)) == set(seeds)
    assert sum(len(shard) for shard in shards) == len(seeds)


def test_seed_shard_rejects_invalid_or_empty_partitions():
    with pytest.raises(StudyConfigError, match="0 <= index"):
        partition_seeds([1, 2], shard_index=2, shard_count=2)
    with pytest.raises(StudyConfigError, match="contains no configured seeds"):
        partition_seeds([1, 2], shard_index=2, shard_count=3)


def test_required_metrics_map_pose_and_reject_nonfinite():
    row = {
        "skin_tone_change": 4.0,
        "face_similarity": 0.9,
        "lpips": 0.1,
        "background_ssim": 0.95,
        "total_pose_diff": math.inf,
    }
    missing = required_metric_missing(
        row,
        [
            "skin_tone_change",
            "face_similarity",
            "lpips",
            "background_ssim",
            "pose_difference",
        ],
    )
    assert missing == ["pose_difference"]


def test_json_safe_normalizes_nonfinite_and_numpy_scalars():
    value = {
        "finite": np.float32(1.25),
        "missing": [math.inf, -math.inf, math.nan],
        "count": np.int64(3),
    }
    normalized = json_safe(value)
    assert normalized == {
        "finite": 1.25,
        "missing": [None, None, None],
        "count": 3,
    }
    json.dumps(normalized, allow_nan=False)


def test_existing_keys_repairs_only_incomplete_final_fragment(tmp_path):
    runner = StudyRunner.__new__(StudyRunner)
    runner.results_path = tmp_path / "results.jsonl"
    valid = {"seed": 1, "method": "prompt_only", "alpha": 0.0}
    runner.results_path.write_text(json.dumps(valid) + "\n{\"seed\": 2", encoding="utf-8")

    assert runner._existing_keys() == {(1, "prompt_only", 0.0)}
    assert runner.results_path.read_text(encoding="utf-8") == json.dumps(valid) + "\n"


def test_pair_manifest_rejects_evaluation_seed_overlap(tmp_path):
    light = tmp_path / "light.png"
    dark = tmp_path / "dark.png"
    light.write_bytes(b"light")
    dark.write_bytes(b"dark")
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": "test/model",
                "model_revision": "abc123",
                "inference_steps": 2,
                "guidance_scale": 1.0,
                "height": 8,
                "width": 8,
                "pairs": [
                    {
                        "pair_id": "pair-1",
                        "seed": 42,
                        "light": {"path": str(light), "sha256": sha256_file(light)},
                        "dark": {"path": str(dark), "sha256": sha256_file(dark)},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = StudyRunner.__new__(StudyRunner)
    runner.project_root = tmp_path
    runner.config = SimpleNamespace(
        data={"training_manifest": str(manifest_path)},
        direction={"train_pairs": 1, "held_out_pairs": 0},
        model={
            "id": "test/model",
            "revision": "abc123",
            "inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 8,
            "width": 8,
        },
        seeds=(42,),
    )
    runner.allow_calibration = True

    with pytest.raises(StudyConfigError, match="overlap evaluation seeds"):
        runner._load_pair_manifest(validate_only=True)


def test_confirmatory_pair_manifest_verifies_manifest_and_ledger_hashes(tmp_path):
    light = tmp_path / "light.png"
    dark = tmp_path / "dark.png"
    ledger = tmp_path / "generation.jsonl"
    manifest_path = tmp_path / "pairs.json"
    light.write_bytes(b"light")
    dark.write_bytes(b"dark")
    ledger.write_text('{"path":"generated"}\n', encoding="utf-8")
    manifest = {
        "model_id": "test/model",
        "model_revision": "abc123",
        "inference_steps": 2,
        "guidance_scale": 1.0,
        "height": 8,
        "width": 8,
        "generation_observed_in_this_run": True,
        "generation_observed_in_campaign": True,
        "generation_ledger": str(ledger),
        "generation_ledger_sha256": sha256_file(ledger),
        "pairs": [
            {
                "pair_id": "pair-1",
                "seed": 42,
                "light": {"path": str(light), "sha256": sha256_file(light)},
                "dark": {"path": str(dark), "sha256": sha256_file(dark)},
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runner = StudyRunner.__new__(StudyRunner)
    runner.project_root = tmp_path
    runner.config = SimpleNamespace(
        data={
            "training_manifest": str(manifest_path),
            "training_manifest_sha256": sha256_file(manifest_path),
        },
        direction={"train_pairs": 1, "held_out_pairs": 0},
        model={
            "id": "test/model",
            "revision": "abc123",
            "inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 8,
            "width": 8,
        },
        seeds=(100,),
    )
    runner.allow_calibration = False

    pairs = runner._load_pair_manifest(validate_only=True)
    assert len(pairs) == 1

    ledger.write_text('{"path":"tampered"}\n', encoding="utf-8")
    with pytest.raises(StudyConfigError, match="Generation ledger"):
        runner._load_pair_manifest(validate_only=True)
