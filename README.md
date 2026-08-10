# Denoising-Time Skin-Tone Steering in SDXL

Research code for estimating a paired **skin-tone direction** in Stable
Diffusion XL's VAE latent space and injecting it during denoising to generate
controlled portrait sweeps.

This project studies a visual attribute. It does **not** infer or change race or
ethnicity, and the current evidence does not establish bias mitigation or
identity preservation.

## Research status

**Pilot / pre-confirmatory.** The repository now contains a fail-closed metric
pipeline, a re-analysis of the historical single-seed sweep, a paper draft,
and a preregisterable evaluation protocol. The pilot is useful engineering
evidence but is not a statistically powered result.

| What the pilot establishes | What remains unestablished |
|---|---|
| A paired VAE-latent direction can be injected at every denoising step. | That the direction isolates skin tone rather than lighting or correlated features. |
| The legacy sweep has a complete, monotonic rendered-colour response (Spearman ρ = -1.00). | Generalization beyond one seed or superiority to any baseline. |
| Moderate steering (α = ±0.8) passed all five engineering gates with a 0.908 mean quality rubric. | Preservation at extreme settings: both ±1.5 edits failed at least one gate. |

The original grid labels remain invalid historical output: its “background
SSIM” silently fell back to whole-image SSIM. The new
[legacy re-analysis](experiments/legacy_reanalysis/README.md) reconstructs the
panels, verifies every input hash, records model/package provenance, and makes
the score unavailable whenever a required metric is missing. See the
[metric specification](docs/METRIC_SPEC.md) and
[claims register](docs/CLAIMS_AND_EVIDENCE.md).

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
paper/                         LaTeX pilot manuscript and bibliography
docs/                          protocol, ethics, claims, reproducibility
configs/                       pilot and planned confirmatory study manifests
scripts/summarize_results.py   long-form aggregation + seed bootstrap intervals
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
python3 -m pip install -e '.[evaluation,dev]'
make metric-models
make test
```

Generate paired synthetic inputs and run the pilot:

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

Re-evaluate any saved sweep without regeneration:

```bash
python3 scripts/evaluate_sweep.py experiments/runs/pilot_seed_999 \
  --output experiments/evaluation --operating-max-alpha 0.75 --strict
```

`--strict` exits nonzero when face embeddings, LPIPS, masked background SSIM,
pose, or the illumination-audited target response is missing.

## From pilot to paper result

The [research protocol](docs/RESEARCH_PROTOCOL.md) fixes the full-study
hypotheses, 30 held-out seeds, four baselines/ablations, required metrics,
failure handling, and bootstrap analysis. Before making confirmatory claims:

1. externally validate the implemented illumination-audited skin-tone outcome;
2. use train/held-out seed splits and run every prespecified method;
3. report face-detector and metric missingness as outcomes;
4. compare preservation at matched target-attribute change;
5. archive exact dependencies, model revision, manifests, and result hashes;
6. select a repository license after checking all model/data licenses.

The manuscript source is [paper/main.tex](paper/main.tex). Build it with
`make paper` after installing a TeX distribution with `latexmk`.

## Responsible use

Read [docs/ETHICS.md](docs/ETHICS.md) before using or releasing artifacts. Do
not use this code for race classification, profiling, identity decisions,
deceptive edits, or editing real people without consent. Face embeddings are
sensitive biometric data and should not be committed.
