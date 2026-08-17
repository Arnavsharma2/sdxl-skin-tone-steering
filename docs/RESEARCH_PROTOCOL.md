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
- H4 (feasibility): Post-hoc latent addition has lower monotonicity and target
  coverage than stepwise steering. Preservation contrasts are reported only
  in seed/direction strata where the post-hoc method reaches the frozen target.

## Design

The editable pre-confirmatory design is specified in `configs/full_study.yaml`;
the executed design is frozen in `configs/full_study_preregistered.yaml`.

1. Generate noise-coupled prompt pairs with the same seed and prompt template,
   changing only the skin-tone descriptor. Changed conditioning can still
   alter apparent identity, so these are not described as identity-matched
   photographs. Split pairs by seed before estimating anything.
2. Estimate one direction on the training split using the paired mean
   difference. Do not tune alpha, masks, or thresholds on the held-out split.
3. Evaluate 30 held-out generation seeds on separately calibrated, frozen
   alpha grids for four methods: prompt-only, post-hoc latent addition,
   unmasked stepwise injection, and masked stepwise injection.
4. Measure the target attribute independently of preservation. A study run is
   invalid unless a face is detected and every required metric is present.
5. Compare methods at matched target-attribute change. Bootstrap by generation
   seed—not by image—to preserve within-seed dependence.
6. Require direction-data, calibration, and confirmatory generation seeds to
   be mutually disjoint. Before loading the model, verify every selected paired
   image against the SHA-256 digest recorded in its manifest. The frozen
   config freezes the manifest hash, and the manifest freezes the resumable
   per-image generation-ledger hash.

Calibration may assign different prespecified alpha grids to different methods
because the numerical scale of prompt descriptors, one-shot latent addition,
and denoising-time addition is not commensurate. The matched ITA target—not an
equal raw alpha—is the controlled comparison. All method-specific grids and
prompt descriptors must be frozen before confirmatory generation.

The post-hoc latent baseline failed to achieve symmetric ±5 ITA target coverage
across calibration identities even after a wider endpoint probe. It therefore
has a prespecified feasibility/dose-response role in the confirmatory study and
is excluded from the primary matched-change contrast set. Its detector failure,
non-monotonicity, and target-coverage rates remain reportable outcomes. This
role was frozen before confirmatory generation; it is not a post hoc exclusion.

## Outcomes

The primary outcome is face-embedding similarity at a matched, prespecified
skin-tone change. Secondary outcomes are LPIPS, background SSIM, pose drift,
face-detection failure rate, and monotonicity of the target response.

The implemented target outcome is
`white_reference_bilateral_cheek_median_cielab_ita_v2`. It detects a face,
samples conservative bilateral cheek ellipses, normalizes exposure and white
balance from bright neutral pixels outside the face box, and reports median CIE
L*a*b* values and Individual Typology Angle (ITA). Positive
`skin_tone_change` means ITA decreased relative to the same-seed base image.
It is an image-colour proxy, not a demographic classifier.

Before a confirmatory run, `scripts/validate_skin_tone_metric.py` must pass the
prespecified gates on the 32 held-out direction pairs: face/region detection,
paired light-above-dark ordering, median pair separation, and sensitivity to
prespecified exposure and white-balance perturbations. The SHA-256 of that
report must be frozen in `configs/full_study_preregistered.yaml`. Failure requires revising
the metric in a new calibration protocol, not weakening gates after inspection.
Pixel brightness alone is not an acceptable primary measurement.

## Statistics

- Report means, standard deviations, medians, and 95% seed-level bootstrap
  confidence intervals.
- Match each method to the prespecified absolute ITA changes separately for
  lighter and darker directions using the closest observed alpha. A match
  outside the frozen ITA tolerance is missing, not extrapolated.
- Use paired method contrasts within each seed. Positive reported advantage is
  always oriented in favor of the masked stepwise reference method.
- Correct confirmatory secondary tests using Holm's method.
- Use two-sided paired sign-flip randomization tests; bootstrap and randomize by
  generation seed, not image row.
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

The seed list, alpha grid, methods, match targets/tolerance, measurement report,
model revision, and primary analysis are fixed before the manifest changes to
`status: preregistered`. Exclude an image only for a machine-readable generation
or file-integrity error. Face-detection failure is an outcome, not an exclusion.

## Claim policy

Until the confirmatory study is complete, allowed wording is “pilot,”
“demonstration,” “suggests,” or “feasibility.” Disallowed wording includes
“bias mitigation,” “identity preserved,” “disentangled,” “changes race,” and
“state of the art.”
