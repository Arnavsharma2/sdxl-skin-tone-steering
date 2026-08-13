# Frozen statistical analysis specification

This document freezes `tmlr_statistical_analysis_v1` before confirmatory image
collection. Its machine-readable authority is the `analysis` block in
[`configs/full_study.yaml`](../configs/full_study.yaml). The estimand concerns
preservation metrics at matched **absolute rendered-skin-tone change**. It does
not estimate race, bias mitigation, identity guarantees, representation
disentanglement, or state-of-the-art performance.

## Confirmatory estimands

The primary contrast is H2: `stepwise_masked - prompt_only` for FaceNet cosine
similarity. The Holm family contains these prespecified secondary contrasts:

1. H3: `stepwise_masked - stepwise_unmasked` for background SSIM;
2. H4: `stepwise_masked - posthoc_latent` for FaceNet cosine similarity; and
3. H4: `stepwise_masked - posthoc_latent` for LPIPS.

Higher FaceNet similarity and background SSIM are favorable. Lower LPIPS is
favorable. Output always retains the literal `method_a - method_b` effect and a
separate sign-oriented `favorable_effect`; it never silently reverses a metric.

For seed (s), method (m), nonzero alpha (a), target change is
(x_{sma}=|\Delta L^*_{sma}|) and the preservation metric is (y_{sma}). The
complete five-alpha target sweep must be present, unique, finite, and strictly
decreasing in the frozen alpha order. Every nonzero-alpha evaluation row must
also be complete under `tmlr_evaluation_protocol_v1`. A failed detector or
missing required metric therefore invalidates the seed-method curve; it is not
an exclusion or a pass.

Within each seed and method, exact target-change ties (absolute tolerance
`1e-12`, zero relative tolerance) are replaced by the arithmetic mean target
change and arithmetic mean preservation value. At least two distinct target
changes must remain. A piecewise-linear curve is then constructed. No
extrapolation is allowed.

For each seed and method pair, common support is

```text
[max(2.0, min(x_method_a), min(x_method_b)),
 min(max(x_method_a), max(x_method_b))].
```

The endpoints are included. Support must have strictly positive width; a
single touching boundary is not sufficient. Both curves are evaluated at 101
equally spaced points including the endpoints. The seed effect is the
arithmetic mean of the 101 pointwise `method_a - method_b` differences. The
confirmatory effect is the unweighted arithmetic mean of the 30 seed effects.

The support interval is seed-specific. This defines a seed-average effect over
the region in which that seed's two methods both demonstrate an observed
target response of at least 2 relative-L* points. It is not an effect at a
single pooled or post-hoc selected alpha.

## Fail-closed availability

A confirmatory contrast is computable only when all 30 prespecified generation
seeds produce a valid paired seed effect. Missing methods, alphas, seeds,
nonfinite values, incomplete evaluation rows, nonmonotonic target sweeps,
insufficient unique target changes, lack of positive-width common support, or
any attempted extrapolation leave the estimate, confidence interval, and
p-value null with a machine-readable reason.

The output may additionally show an `exploratory_valid_seed_estimate`. That is
an explicitly labeled diagnostic over available valid seed effects, not a
replacement confirmatory analysis and not a basis for the frozen claims. No
complete-case result is silently promoted.

## Pairing, uncertainty, and tests

Generation seed is the pairing, cluster, and sampling unit. Image rows are
never independent observations. Each bootstrap draw samples 30 seeds with
replacement and retains the deterministic seed effect derived from all
within-seed methods and alphas. The percentile 95% interval uses exactly 10,000
resamples and recorded analysis RNG seed `20260813`. RNG streams are derived
from the comparison identifier, so changing output order does not change an
interval.

Two-sided p-values use 10,000 paired seed-level sign-flip randomizations with a
plus-one Monte Carlo correction. The primary p-value is reported separately.
Secondary p-values use Holm correction in their configuration order as the
stable tie breaker. Missing, nonfinite, or out-of-range p-values are never
corrected or rejected, but they remain in the frozen family size so a missing
test cannot make the other tests less conservative.

Effect estimates and confidence intervals are the reporting focus. A threshold
or corrected p-value alone is not evidence of bias mitigation or an identity
guarantee.

## Missingness and failure outputs

`scripts/summarize_results.py` treats the study manifest as the canonical
matrix and never drops a planned cell. It verifies the config hash, metric
protocol hash, study identifier, matrix cells, per-seed config hashes, and
source commit before analysis. For every completed generation row it also
requires an in-root output path, reads the generated image and seed base image,
and verifies their recorded SHA-256 hashes. Each long-form row records stable
failure codes, including required-metric missingness and any recorded detector,
generation, or file-integrity failure.

The analysis writes:

- `results_long.csv`: every planned row plus validity and provenance;
- `aggregate_metrics.csv`: explicitly exploratory method/alpha descriptions,
  missingness, detector failures, generation failures, and integrity failures;
- `seed_matched_contrasts.csv`: every valid seed effect or exact failure reason;
- `confirmatory_contrasts.csv`: frozen effects, seed counts, intervals, tests,
  Holm results, and not-computable reasons;
- `failure_counts.csv`: every machine-readable reason by method and alpha; and
- `audit.json`: hashes, source commit, analysis settings, expected/valid row
  counts, failure totals, Holm family, and output inventory.

Outputs distinguish the generation `source_commit` recorded in the study
manifest from the analysis code commit and its dirty-worktree state.

Face-detection failure must be recorded explicitly by evaluation as
`face_detection_failed: true` or `failure_reasons: ["face_detection_failure"]`.
It remains in the planned denominator. The analysis does not guess that every
missing face-related metric is necessarily a detector failure; it records the
specific missing metric when the upstream cause is unavailable.

## Exploratory summaries

Method/alpha means, standard deviations, medians, intervals, the valid-seed
diagnostic, monotonicity diagnostics, failure rates, composite engineering
scores, and any unregistered contrasts are exploratory unless explicitly
listed above. They must be labeled as such in tables and prose. The composite
score is never a primary inferential outcome.
