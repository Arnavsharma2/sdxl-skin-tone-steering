import json
from pathlib import Path

from scripts.audit_study_run import audit_run
from src.study.config import load_study_config


def _make_run(tmp_path: Path) -> tuple[Path, Path]:
    config_path = Path("configs/full_study_preregistered.yaml")
    config = load_study_config(config_path)
    run_dir = tmp_path / "run"
    (run_dir / "direction").mkdir(parents=True)
    (run_dir / "direction" / "raw.pt").write_bytes(b"raw")
    (run_dir / "direction" / "masked.pt").write_bytes(b"masked")
    (run_dir / "study_config.yaml").write_bytes(config_path.read_bytes())
    rows = []
    for seed in config.seeds:
        for method in config.methods:
            for alpha in config.alphas_for(method):
                relative = Path("images") / str(seed) / method / f"{alpha}.png"
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"png")
                rows.append(
                    {
                        "study_id": config.study_id,
                        "config_fingerprint": config.fingerprint,
                        "seed": seed,
                        "method": method,
                        "alpha": alpha,
                        "generation_complete": True,
                        "evaluation_complete": True,
                        "image_path": str(relative),
                    }
                )
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "study_id": config.study_id,
        "study_status": config.status,
        "execution_mode": "confirmatory",
        "config_fingerprint": config.fingerprint,
        "condition_count": len(rows),
        "execution_seeds": config.seeds,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return config_path, run_dir


def test_audit_accepts_exact_complete_run(tmp_path):
    config_path, run_dir = _make_run(tmp_path)
    audit = audit_run(config_path, run_dir)
    assert audit["passed"] is True
    assert audit["observed_rows"] == 570
    assert audit["evaluation_incomplete_count"] == 0


def test_audit_rejects_missing_condition(tmp_path):
    config_path, run_dir = _make_run(tmp_path)
    rows = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    (run_dir / "results.jsonl").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    audit = audit_run(config_path, run_dir)
    assert audit["passed"] is False
    assert audit["checks"]["exact_condition_keys"] is False
