# Fresh-environment reproducibility audit

Audit date: 2026-08-17

## Scope

A new Python 3.10 virtual environment was created outside the repository. The
recorded `requirements-lock.txt` was installed, followed by the local project
without dependency resolution or build isolation. No packages from the
development virtual environment were visible to the audit environment.

This was a fresh dependency environment, not originally a fresh Git clone. The
run-integrity checks used the author's retained local image and direction
archives. The frozen JSON/JSONL generation manifests are now tracked so a clean
clone can reproduce the retained aggregate analyses; the large image and tensor
archives remain separate as documented in `ARTIFACT_AVAILABILITY.md`.

## Commands

```bash
python3.10 -m venv <fresh-environment>
<fresh-environment>/bin/python -m pip install --upgrade pip setuptools wheel
<fresh-environment>/bin/python -m pip install -r requirements-lock.txt
<fresh-environment>/bin/python -m pip install -e . --no-deps --no-build-isolation
<fresh-environment>/bin/python -m pytest
<fresh-environment>/bin/python -m ruff check \
  generate_training_data.py run_race_vector_extraction.py scripts tests src
<fresh-environment>/bin/python scripts/verify_manuscript_claims.py
```

The exact frozen replication auditor and decision script were then executed
against the archived local run and analysis inputs, writing only to a temporary
output directory.

## Results

- Python: 3.10
- Tests: 63 passed
- Ruff: passed
- Frozen replication run-integrity audit: passed
- Recreated run-integrity audit SHA-256:
  `9a815f5eff1a8ad18d635adac9e869f696f960e6b9bd24ea9df34a8cad53f7fb`
- Recreated replication-decision audit SHA-256:
  `2085c0a391d9407a4d242cc7f1c6963dce01743e98b0cbd1fc6bff6ed95e81bd`
- Recreated replication family SHA-256:
  `8b9979a244ea5929892286543ff8b66bbc2c01f9ea3ed7e9912b7cb4cec28756`
- Recreated 16-group manuscript-claim audit SHA-256:
  `34b28746ba0d8acca134cc99dc42a16a11df84fd4067aec01f3cbbe8d2c37887`
- Locked decision: `inconclusive_coverage`

All four recreated files were byte-for-byte identical to their retained study
counterparts. This verifies the non-generation tests, run-integrity checks,
locked replication decision, and central manuscript numbers under a fresh
environment. This does not claim
bitwise regeneration of CUDA images on Apple Silicon; the full image campaign
is content-attested and retained separately. The run-integrity result requires
that separate archive and is not reproducible from the Git checkout alone.

## Clean-checkout follow-up

After the manuscript package was made self-contained, a detached clean
worktree reproduced the tests, Ruff, both deterministic PDFs, and the 16-group
claim audit. Venue-specific packaging checks were later separated from the
public research repository. On 2026-08-18, the public checkout passed 61 tests,
Ruff, and all 16 manuscript-claim groups.
