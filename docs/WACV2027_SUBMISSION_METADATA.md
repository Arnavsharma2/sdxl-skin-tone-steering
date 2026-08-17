# WACV 2027 Round 2 enrollment metadata

## Track

Evaluations & Dataset Track

## Working title

**Skin-Tone Steering at Matched Change: A Preregistered Evaluation of
Denoising-Time Latent Control in SDXL**

The title foregrounds the evaluation contribution and avoids implying that a
new semantic-direction concept or a race-editing system is being claimed.

## Enrollment abstract

We evaluate whether a linear direction estimated from paired portrait latents
can control visible skin tone in Stable Diffusion XL while preserving other
image properties. The intervention computes a paired mean difference in VAE
latent space, applies a fixed spatial Gaussian mask, and distributes the update
across denoising steps. Our central contribution is a preregistered
matched-change protocol that separates target response from preservation,
retains detector and metric failures, and compares methods only after matching
their measured image-colour change. We estimate the direction from 64 of 96
noise-coupled synthetic prompt pairs, validate a continuous image-colour
instrument on 32 held-out pairs, and evaluate 570 conditions over 30 held-out
seeds. At a matched five-degree Individual Typology Angle change, masked
steering provides no detectable face-embedding-similarity advantage. It does,
however, improve LPIPS relative to unmasked steering in both directions and
improves several darker-direction background and pose contrasts. Target
coverage varies materially across methods and directions, demonstrating why
unmatched preservation comparisons can be misleading. Post-confirmatory
tolerance and split-half analyses expose limited strict-match coverage and
sample sensitivity of the learned direction. A prospectively frozen
independent-seed replication retained both LPIPS effect signs and
interval-positive estimates, but missed its preregistered matched-pair coverage
threshold; its locked decision is therefore inconclusive. The study supports a
bounded parent-campaign perceptual-preservation result and an auditable
evaluation practice—not identity preservation, causal disentanglement,
demographic inference, bias mitigation, or a strict replication claim.

## Suggested subject areas

- generative models and diffusion models;
- evaluation methodology and stress testing;
- responsible, fair, and accountable computer vision;
- face and portrait analysis;
- reproducibility and negative results.

## Keywords

latent diffusion; matched-change evaluation; skin tone; perceptual
preservation; preregistration; reproducibility; missingness

## Author-dependent enrollment fields

- Complete author names and order: **required before 2026-08-21 AoE**
- OpenReview IDs and conflict-complete profiles: **required**
- Corresponding author: **required**
- Paper ID: assigned after enrollment and then inserted into
  `paper/wacv/main.tex`
