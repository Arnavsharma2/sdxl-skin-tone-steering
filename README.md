# Denoising-Time Skin-Tone Steering in SDXL

Research code for estimating a paired **skin-tone direction** in Stable
Diffusion XL's VAE latent space and injecting it during denoising to generate
controlled portrait sweeps.

This project studies a visual attribute. It does **not** infer or change race or
ethnicity, and the current evidence does not establish bias mitigation or
identity preservation.

## Research status

**Confirmatory study complete (2026-08-16).** The calibration grids, exact SDXL
revision, 192-image direction dataset, generation ledger, and held-out
measurement-validation report have been frozen in
`configs/full_study_preregistered.yaml` (fingerprint `6e7e5a62a18ebfe6`). The
measurement instrument passed all prespecified gates on 32 held-out pairs. The
30-seed, four-method CUDA campaign completed all 570 frozen conditions. All
images generated and 569 conditions were metric-complete; the sole missing
measurement was a retained post-hoc face/skin-region detection failure. The
result ledger SHA-256 is
`c15eecc46241ae945d8da5f1e7265bc225878926f279283208aa15e54cf17a4c`.

**Prospective replication complete (2026-08-17).** A new 96-pair campaign and
30 evaluation seeds were frozen before outcome generation. All 32 new held-out
measurement pairs passed the original gates; the immutable replication config
fingerprint is `fa694f8bb214c219`. The independent 570-condition A100 run
completed with all generations and 569 metric-complete rows. Its result-ledger
SHA-256 is
`26bb0806a87089ea6ba1d3c4edb0a670a1aad67a6cf5ba55011c8e0d05755a69`.

The locked replication decision was **inconclusive coverage**. Masked versus
unmasked LPIPS advantages retained the parent signs: 0.0238 lighter (95% CI
[0.0049, 0.0415], two-test Holm p=.0555) and 0.0409 darker ([0.0176,
0.0655], p=.0192). However, only 8 and 10 shared matched seeds were available,
below the prespecified minimum of 12. The masked cross-campaign direction
cosine was 0.919. These are encouraging but not a strict replication claim.

| Confirmatory result | Interpretation |
|---|---|
| Every face-similarity contrast had a 95% CI spanning zero. | The primary identity-preservation hypothesis was not supported. |
| Masking improved LPIPS versus unmasked steering in lighter and darker matched strata (Holm-adjusted p=.012 and .002). | Spatial masking improves perceptual preservation under the tested protocol. |
| In darker edits, masking also improved background SSIM versus prompt-only and unmasked steering, and pose versus prompt-only. | Several secondary preservation effects support a narrower, direction-dependent claim. |
| Matched coverage was 63%/87% for masked, 43%/50% for unmasked, and 23%/60% for prompt-only (lighter/darker). | Target reliability differs materially by method and direction; matched sample sizes must be reported. |

Two post-confirmatory analyses are explicitly exploratory. Repeating the
matched analysis at one- and two-degree ITA tolerances preserved the direction
of every estimable LPIPS advantage, but shared samples fell to 1--5 and 3--12
seeds, respectively, and no secondary contrast survived Holm correction. A
200-resample split-half analysis found that direction agreement increased with
pair count and masking: for two disjoint 32-pair estimates, median cosine was
0.75 before masking and 0.82 after masking. Re-encoding all 64 frozen pairs
exactly reproduced the confirmatory direction tensor.

The earlier historical pilot did not record valid face-similarity, landmark, or pose
values. Its values labeled “background SSIM” used an all-background fallback
after face-detection failure and therefore behave as whole-image SSIM. See the
[claims and evidence register](docs/CLAIMS_AND_EVIDENCE.md) for the authoritative
audit.

![Pilot counterfactual grid](experiments/results/final_grid.png)

## Method

For matched portrait pairs, the direction is the mean latent difference:

```text
v = mean_i(E(dark_descriptor_i) - E(light_descriptor_i))
```

A Gaussian center mask attenuates the direction outside the face region. For
steering strength `alpha` and `T` denoising steps, the callback adds
`alpha / T * v` after each scheduler step. VAE encoding uses the deterministic
posterior mode for reproducibility.

## Repository map

```text
paper/                         LaTeX confirmatory manuscript and figures
docs/                          protocol, ethics, claims, reproducibility
configs/                       pilot, calibration, and prospectively frozen manifests
scripts/summarize_results.py   long-form aggregation + seed bootstrap intervals
scripts/run_study.py           resumable four-method calibration/confirmatory runner
scripts/audit_study_run.py     exact-key, manifest, image, and tensor run-integrity audit
scripts/analyze_study.py       matched-change contrasts and missingness audit
scripts/analyze_robustness.py  exploratory matching-tolerance sensitivity analysis
scripts/analyze_direction_stability.py split-half direction reproducibility analysis
scripts/analyze_replication.py prospective independent-seed replication decision
scripts/plot_study.py          publication dose-response/contrast/completion figures
scripts/make_qualitative_grid.py deterministic median-case confirmatory example
scripts/synthesize_calibration.py  adaptive calibration provenance/coverage audit
scripts/validate_skin_tone_metric.py  held-out colour-metric validation gates
scripts/freeze_confirmatory_config.py  immutable pre-outcome freeze and hash checks
tests/                         lightweight unit tests
src/models/                    SDXL generation, encoding, and stepwise steering
src/latent/                    direction estimation and latent manipulation
src/metrics/                   identity, perceptual, structural, and pose metrics
generate_training_data.py      paired synthetic portrait generation
run_race_vector_extraction.py  legacy-named end-to-end CLI
```

The legacy CLI and compatibility class retain “race vector” in their names so
existing links and notebooks do not break. New code uses
`SkinToneDirectionExtractor`.

## Setup

Python 3.10 or 3.11 is recommended. SDXL requires substantial memory and model
access under its upstream license.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-lock.txt
python3 -m pip install -e . --no-deps
make test
```

Generate paired synthetic inputs and run the historical pilot:

```bash
python3 generate_training_data.py --n 8
make pilot
make summarize
```

Direct CLI use remains available:

```bash
python3 run_race_vector_extraction.py \
  --steps 25 \
  --alphas -1.5 -0.75 0 0.75 1.5 \
  --seed 999 \
  --output experiments/runs/pilot_seed_999
```

Each run records settings, results, package versions, git commit and dirty
state, and hardware availability in `metadata.json`. Cached base images are
keyed by model, prompts, seed, step count, and guidance scale.

The confirmatory workflow is deliberately gated:

```bash
# 64 direction-training pairs + 32 held-out measurement-validation pairs
python3 generate_training_data.py --config configs/full_study.yaml --n 96

# Safe after interruption: only images with matching per-image hash, seed,
# descriptor, and generation signature are skipped. Unattested files regenerate.

# Must pass before the full manifest can be frozen for outcome collection
make measurement-validate

# Calibration output must use seeds disjoint from final 500000–500029.
python3 scripts/run_study.py configs/calibration_candidate_v2.yaml \
  --allow-calibration \
  --output experiments/runs/calibration_candidate_v2

# Freeze target changes, method-specific grids, the paired-data manifest hash,
# validation-report hash, and frozen status into a new immutable config:
python3 scripts/freeze_confirmatory_config.py configs/full_study.yaml \
  --manifest data/generated/training_manifest_study_v1.json \
  --validation-report experiments/measurement_validation/validation_report.json \
  --output configs/full_study_preregistered.yaml

# These targets default to configs/full_study_preregistered.yaml. Override with
# STUDY_CONFIG=... only for an explicitly separate campaign.
make study-validate
make study-run
make study-audit
make study-analyze
make study-robustness
make study-plot
make study-example
```

A prospective independent-seed replication is specified in
[`docs/REPLICATION_PROTOCOL.md`](docs/REPLICATION_PROTOCOL.md). It uses a new
96-pair direction/validation campaign and seeds 600000--600029 while copying
the frozen method grids and matched-change analysis unchanged.

After the frozen outcome run, execute `make replication-audit` before
`make replication-analyze`, `make replication-assess`, and
`make replication-plot`. The audit refuses missing, duplicate, or extra
conditions and verifies the run manifest, image references, and direction
tensors.

`scripts/run_study.py` appends one JSONL record per attempted seed/method/alpha
condition. Generation or detector failures remain in the table and are never
silently excluded. The current target outcome is a white-reference-normalized
bilateral-cheek CIE Lab/ITA proxy. It is an image-colour measurement, not a
demographic classifier, and must pass the held-out illumination-sensitivity
gates before confirmatory use.

The paired-data manifest records a resumable generation campaign ledger. A
confirmatory run verifies the frozen manifest hash, the ledger hash, and
every selected image hash before loading the model.

The archived configs retain the machine status value `preregistered` because
that literal is part of the executed fingerprints. The paper uses the more
precise phrase *prospectively frozen*: the author-controlled pre-outcome
archives are content-hashed, but were not deposited with an independent
timestamping service. See `docs/ARTIFACT_AVAILABILITY.md` for the exact boundary
between checkout-reproducible analysis and retained large-run artifacts.

## Reproduce the confirmatory result

The [research protocol](docs/RESEARCH_PROTOCOL.md) fixes the full-study
hypotheses, 30 held-out seeds, four baselines/ablations, required metrics,
failure handling, and bootstrap analysis. Before making confirmatory claims:

1. pass the held-out validation gates for the illumination-aware continuous
   skin-tone outcome and freeze the report hash;
2. use train/held-out seed splits and run every prespecified method;
3. report face-detector and metric missingness as outcomes;
4. compare preservation at matched target-attribute change;
5. archive exact dependencies, model revision, manifests, and result hashes;
6. select a repository license after checking all model/data licenses.

The manuscript source is [paper/main.tex](paper/main.tex), and the visually
verified seven-page parent-plus-replication build is
[output/pdf/denoising_time_skin_tone_steering_replication.pdf](output/pdf/denoising_time_skin_tone_steering_replication.pdf).
Build it with `make paper` after installing Tectonic. Run `make paper-audit` to
verify 16 central numerical claim groups against the retained validation,
analysis, robustness, and replication artifacts.

An anonymized WACV 2027 Evaluations & Dataset-track draft using the official
author kit is in [paper/wacv/main.tex](paper/wacv/main.tex). Build it with
`make paper-wacv`; build the identity-scanned supplementary ZIP with
`make anonymous-supplement`. The enrollment and policy gates are tracked in
[docs/WACV2027_SUBMISSION_CHECKLIST.md](docs/WACV2027_SUBMISSION_CHECKLIST.md).

## Responsible use

Read [docs/ETHICS.md](docs/ETHICS.md) before using or releasing artifacts. Do
not use this code for race classification, profiling, identity decisions,
deceptive edits, or editing real people without consent. Face embeddings are
sensitive biometric data and should not be committed.

Release documentation includes the [model card](docs/MODEL_CARD.md),
[data card](docs/DATA_CARD.md), and
[third-party license audit](docs/RELEASE_LICENSE_AUDIT.md). The repository
license and final citation authors remain intentionally unset until approved by
the copyright holder.
