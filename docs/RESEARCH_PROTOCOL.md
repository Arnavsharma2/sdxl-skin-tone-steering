# Research protocol

## Research question

Can a linear direction estimated from paired SDXL VAE latents produce a
monotonic change in visible skin tone during denoising while preserving
identity, pose, composition, and background better than prompt-only and
post-hoc latent baselines?

The project estimates **visible skin tone**, not race or ethnicity. Those are
social identities that cannot be inferred from a pixel-space continuum.

## Hypotheses

- H1: Stepwise masked steering produces a monotonic skin-tone response as
  alpha moves from negative to positive.
- H2: At a matched magnitude of skin-tone change, stepwise masked steering
  preserves face embedding similarity better than prompt-only steering.
- H3: The spatial mask improves background SSIM relative to unmasked stepwise
  steering without materially reducing the target-attribute response.
- H4: Stepwise steering improves perceptual quality and identity preservation
  relative to adding the same total direction after denoising.

## Design

The confirmatory design is specified in `configs/full_study.yaml`.

1. Generate paired portraits with the same seeds and prompts, changing only
   the skin-tone descriptor. Split pairs by seed before estimating anything.
2. Estimate one direction on the training split using the paired mean
   difference. Do not tune alpha, masks, or thresholds on the held-out split.
3. Evaluate 30 held-out generation seeds at five alpha values for four methods:
   prompt-only, post-hoc latent addition, unmasked stepwise injection, and
   masked stepwise injection.
4. Measure the target attribute independently of preservation. A study run is
   invalid unless a face is detected and every required metric is present.
5. Compare methods at matched target-attribute change. Bootstrap by generation
   seed—not by image—to preserve within-seed dependence.

The declared matrix contains 30 seeds × 4 methods × 5 alpha values = 600
rows. Of these, 480 have nonzero alpha. Configuration loading rejects a matrix
whose declared counts do not match its dimensions, any overlap among the 64
direction-training seeds, 32 held-out pair seeds, and 30 evaluation seeds, or
noncanonical metric names. All engineering thresholds are frozen explicitly
in the study configuration.

### Executable generation methods

`scripts/run_confirmatory.py` expands the configuration into the complete
matrix and writes an auditable manifest without loading SDXL by default. For
each seed, all methods use the same base portrait prompt, negative prompt,
random seed, scheduler, inference-step count, guidance scale, dimensions, and
alpha. The prompt-only baseline changes only a directional visible-skin-tone
descriptor: the inner magnitude uses “subtly” and the outer magnitude uses
“distinctly.” The three latent methods use one unmasked direction artifact:

- `posthoc_latent` adds the total scaled direction to the final base latent;
- `stepwise_unmasked` divides the same total scale across denoising steps; and
- `stepwise_masked` applies the frozen spatial mask before the same stepwise
  injection.

Alpha zero reuses the exact base image for all four methods. Generation is
disabled unless `--execute` is supplied with a direction artifact and model
revision. This execution gate does not itself constitute approval to collect
the confirmatory data.

## Outcomes

The primary outcome is face-embedding similarity at a matched, prespecified
skin-tone change. Secondary outcomes are LPIPS, background SSIM, pose drift,
face-detection failure rate, and monotonicity of the target response.

The repository contains an auditable target-attribute measurement,
`relative_cheek_CIELAB_Lstar_v1`, specified in `docs/METRIC_SPEC.md`. It uses
geometric cheek regions, CIELAB colour, a neutral-background reference, and
explicit illumination QC. It has deterministic synthetic tests and a complete
legacy pilot re-analysis, but it has not been externally validated.

Before a confirmatory run, validate either:

- a calibrated skin-region color measure reported in a perceptually meaningful
  color space, with illumination sensitivity characterized; or
- an externally validated, continuous skin-tone estimator with its model card,
  error profile, and license recorded.

Pixel brightness alone is not an acceptable primary measurement because it is
confounded by lighting, exposure, and background. The current relative-L*
metric reduces first-order exposure sensitivity but does not eliminate local
illumination or rendering confounds.

## Statistics

- Report means, standard deviations, medians, and 95% subject-level bootstrap
  confidence intervals.
- Use paired method contrasts within each seed.
- Correct confirmatory secondary tests using Holm's method.
- Report effect sizes and intervals; do not interpret threshold crossing alone
  as evidence.
- Report missing values and detector failures by method and alpha. Never drop
  them silently or treat missing metrics as passes.

## Ablations

- mask versus no mask;
- stepwise versus post-hoc injection;
- paired difference versus unpaired difference of means;
- raw direction versus the experimental refinement (exploratory only);
- training-pair count and seed sensitivity.

## Stopping and exclusions

The seed list, alpha grid, methods, and primary analysis are fixed before the
confirmatory run. Exclude an image only for a machine-readable generation or
file-integrity error. Face-detection failure is an outcome, not an exclusion.

## Claim policy

Until the confirmatory study is complete, allowed wording is “pilot,”
“demonstration,” “suggests,” or “feasibility.” Disallowed wording includes
“bias mitigation,” “identity preserved,” “disentangled,” “changes race,” and
“state of the art.”
