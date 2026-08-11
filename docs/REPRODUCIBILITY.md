# Reproducibility guide

## Environments

Use Python 3.10 through 3.12. Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[evaluation,dev]'
```

The package ranges support development; a paper release must also archive an
exact lock file or container digest. Record the SDXL model revision and weight
hash after accepting its license. Hardware backends may not be bitwise
identical, so do not mix CUDA, MPS, and CPU within one statistical comparison.

## Workflow

```bash
make test
python3 generate_training_data.py --n 8
make pilot
make summarize
make paper
```

Each new run belongs in `experiments/runs/<study>_<seed>/`. The pipeline writes
settings, package versions, git commit, dirty-state flag, and hardware
availability into `metadata.json`. The base-image cache is keyed by model,
prompt, negative prompt, seed, step count, and guidance scale.

Plan the confirmatory matrix without loading a model or generating images:

```bash
make confirmatory-plan
```

The resulting `study_manifest.json` records the parsed configuration and its
SHA-256 hash, exact matrix, prompts, settings, thresholds, requested and
resolved model revisions, direction-artifact hash, package and hardware
environment, git commit and dirty state, per-image hashes, and machine-readable
failures. Each `seeds/<seed>/metadata.json` is checkpointed independently.
It also embeds `tmlr_evaluation_protocol_v1` and the actual SHA-256 of that
protocol file. Recording the protocol in plan-only mode does not load SDXL.
Actual generation additionally requires explicit `--execute`,
`--model-revision`, and `--direction` arguments and must not be started before
the protocol-validation and approval gates are satisfied.

Audited metric execution is supported only on Linux with Python
`>=3.10,<3.13`, MediaPipe `0.10.21`, and the CPU Face Landmarker delegate.
Headless macOS is rejected before graph construction because the pinned wheel
still requires an unavailable OpenGL pixel format there. Use Linux for
evaluation; do not substitute a detector or whole-image metric.

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
- every metric artifact verification record, including path, expected and
  actual SHA-256, byte size, and status;
- paper figures and the script or command that generated each figure;
- environment lock file or container digest.

The repository currently has no license. Select and add one only after checking
the licenses of the model, metrics, data, and generated artifacts.

`CITATION.cff` intentionally uses “Project contributors” until the author list
and order are confirmed. Replace that placeholder before creating a release.
