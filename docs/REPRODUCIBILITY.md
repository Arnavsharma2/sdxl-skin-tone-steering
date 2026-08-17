# Reproducibility guide

## Environments

Use Python 3.10 or 3.11. Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[evaluation,dev]'
```

For the recorded Apple Silicon/Python 3.10 study environment, prefer the exact
lock after creating a clean virtual environment:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements-lock.txt
python3 -m pip install -e . --no-deps --no-build-isolation
```

The package ranges support development; `requirements-lock.txt` records the
tested Apple Silicon stack and must be archived with the paper. Record the SDXL
model revision and weight hash after accepting its license. Hardware backends may not be bitwise
identical, so do not mix CUDA, MPS, and CPU within one statistical comparison.

This path was retested from a new Python 3.10 virtual environment on
2026-08-17. All 63 tests and Ruff passed. The frozen run-integrity,
replication-decision, and manuscript-claim audit outputs reproduced
byte-for-byte; see
`docs/FRESH_ENVIRONMENT_AUDIT.md`.

## Workflow

```bash
make test
python3 generate_training_data.py --n 8
make pilot
make summarize
make paper-audit
make paper
```

Each new run belongs in `experiments/runs/<study>_<seed>/`. The pipeline writes
settings, package versions, git commit, dirty-state flag, and hardware
availability into `metadata.json`. The base-image cache is keyed by model,
prompt, negative prompt, seed, step count, and guidance scale.

For the preregistered study, use the config-driven workflow:

```bash
python3 generate_training_data.py --config configs/full_study.yaml --n 96
make measurement-validate
python3 scripts/run_study.py configs/calibration_candidate_v2.yaml \
  --allow-calibration \
  --output experiments/runs/calibration_candidate_v2
# Freeze immutable model revision, paired-data manifest hash, measurement report
# hash, method grids, matched ITA target, and status into a new file; commit it
# before generating confirmatory images.
python3 scripts/freeze_confirmatory_config.py configs/full_study.yaml \
  --manifest data/generated/training_manifest_study_v1.json \
  --validation-report experiments/measurement_validation/validation_report.json \
  --output configs/full_study_preregistered.yaml
make study-validate
make study-run
make study-audit
make study-analyze
make study-plot
make study-example
```

The prospective independent-seed campaign uses the analogous locked targets:

```bash
make replication-data
make replication-validate
make replication-freeze
make replication-run
make replication-audit
make replication-analyze
make replication-assess
make replication-plot
```

`study-audit` and `replication-audit` require the exact frozen condition-key
set, matching manifest metadata, complete generation flags, every referenced
image, and both direction tensors before statistical analysis.

The runner snapshots the manifest, records its fingerprint and provenance,
writes one `results.jsonl` row per attempted condition, and resumes by exact
`(seed, method, alpha)` key. A confirmatory invocation refuses a merely planned
manifest. Calibration output must never be relabeled as confirmatory.

Paired portrait generation is also resumable. Each image is appended to a
strict-JSON campaign ledger only after it is saved and hashed. On resume, an
image is skipped only when its content hash, seed, descriptor, path, and
generation signature agree. The final config freezes the resulting paired-data
manifest hash; the manifest in turn freezes the campaign-ledger hash.

## Determinism

- VAE encoding uses the posterior mode by default.
- Python, NumPy, and PyTorch are seeded by the CLI seed.
- Generation seed is shared within a counterfactual set.
- Full bitwise determinism is not promised across hardware or library versions.

## Artifact policy

Generated runs are ignored by git. For a release, archive the following under a
versioned DOI or release asset and record SHA-256 hashes:

- exact configuration and source commit;
- long-form result table and aggregate table;
- metadata for every attempted run, including failures;
- vector artifact if released, with training-data manifest;
- paper figures and the script or command that generated each figure;
- environment lock file or container digest.

The repository currently has no license. Select and add one only after checking
the licenses of the model, metrics, data, and generated artifacts.

`CITATION.cff` intentionally uses “Project contributors” until the author list
and order are confirmed. Replace that placeholder before creating a release.
