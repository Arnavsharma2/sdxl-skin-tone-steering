# Model card: paired SDXL skin-tone steering direction

## Summary

This artifact is not a newly trained image generator. It is a paired mean
difference estimated in the deterministic VAE latent space of the immutable
SDXL 1.0 base revision
`462165984030d82259a11f4367a4eed129e94a7b`. The intervention adds a scaled
version of that direction after each denoising scheduler step. A fixed Gaussian
center mask attenuates the update near image edges.

The construct is **visible skin tone in synthetic portraits**. It is not race,
ethnicity, ancestry, identity, or a demographic classifier.

## Intended uses

- controlled generative-model research;
- evaluation of denoising-time latent interventions;
- reproducibility, missingness, and construct-validity research;
- synthetic, consent-free method development before any real-person study.

## Out-of-scope uses

- inferring, classifying, or modifying race or ethnicity;
- profiling, ranking, eligibility, identity verification, surveillance, or
  other consequential decisions;
- claims of fairness or bias mitigation without a defined downstream harm and
  disparity evaluation;
- deceptive editing or editing identifiable people without consent;
- deployment as a skin-color or biometric measurement device.

## Training and intervention data

The confirmatory direction uses 64 of 96 synthetic SDXL prompt pairs. Each pair
shares the initial noise seed and prompt template; only the light/dark skin-tone
descriptor changes. The remaining 32 pairs are held out for measurement
validation. Seed reuse reduces one source of variation but does not guarantee
the same apparent identity. See `DATA_CARD.md` and the content-addressed
manifest for exact seeds, prompts, and hashes.

## Evaluation

The parent and independent-seed campaigns each evaluate 570 conditions over 30
held-out seeds. The target
is a white-reference-normalized bilateral-cheek CIE Lab/ITA image-colour proxy.
Preservation outcomes are FaceNet cosine similarity, LPIPS, background-only
SSIM, and a five-landmark pose proxy. All failures remain in the denominator.

In the parent campaign, masked steering achieved higher matched-target coverage
and lower LPIPS than unmasked steering in both directions. The prospective
replication retained both LPIPS effect signs and had intervals above zero, but
its locked decision was inconclusive because only 8 and 10 shared matched
seeds were available, below the minimum of 12; the lighter two-test Holm value
was .0555. The preregistered face-similarity hypothesis was not supported in
either broader corrected family. Therefore the artifact must not be described
as strictly replicated, identity-preserving, or disentangled.

## Robustness and known limitations

- Re-encoding the fixed 64-pair campaign exactly reproduced the direction.
- A new 64-pair seed campaign had raw/masked cross-campaign direction cosine
  0.884/0.919 and a direction-norm ratio of 1.04.
- Across independent split-halves, median masked-direction cosine was 0.53,
  0.69, and 0.82 at 8, 16, and 32 pairs per half, respectively. The estimator
  is sample-sensitive.
- Stricter one- and two-degree matching tolerances retained LPIPS effect signs
  but produced very small shared samples.
- Closest-grid matching selects on a noisy target measurement and leaves
  residual mismatch inside the tolerance; denser grids or prospectively fixed
  interpolation would be preferable.
- One model, VAE, scheduler, resolution, and prompt family were tested.
- The Gaussian mask is spatial rather than semantic.
- Image-colour, face-detection, face-embedding, and pose metrics can all have
  demographic and acquisition-dependent error.
- No human identity, realism, or harm evaluation has been completed.

## License and release

No third-party model weights are distributed. SDXL remains governed by its
[CreativeML Open RAIL++-M license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/e4e60c65aa20ee60092c60ba197f541872cf9373/LICENSE.md).
See `RELEASE_LICENSE_AUDIT.md` before publishing code, tensors, or generated
images.
