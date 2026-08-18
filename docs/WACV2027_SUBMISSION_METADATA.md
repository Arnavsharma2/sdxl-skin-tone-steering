# WACV 2027 Round 2 enrollment metadata

## Track

Evaluations & Dataset Track

## Working title

**Evaluating Skin-Tone Control and Image Preservation in SDXL**

The title foregrounds the evaluation contribution and avoids implying that a
new semantic-direction concept or a race-editing system is being claimed.

## Enrollment abstract

Controlling apparent skin tone in generated portraits can unintentionally alter
apparent identity, pose, or background, yet preservation comparisons are
unreliable when methods produce different amounts of target change. We evaluate
whether a linear direction derived from paired portrait latents can control
apparent skin tone in Stable Diffusion XL while preserving other image
properties. The intervention computes a paired mean difference in VAE latent
space, applies a predefined spatial Gaussian mask, and distributes the
alteration across multiple denoising steps. Our main contribution is an
experimental plan defined before examining the outcomes, with methods compared
only when they achieve similar measured image-color changes and with detector
and metric failures retained. Using 64 direction-estimation pairs, 32 held-out
validation pairs, and 570 conditions across 30 seeds, masking improved LPIPS
preservation over unmasked steering by 0.023 for lighter edits and 0.082 for
darker edits, but produced no detectable face-embedding advantage. An
independent 30-seed replication retained both effect directions and high
direction agreement, but only 8 and 10 matched seeds remained, below the
prespecified minimum of 12. The evidence therefore supports a limited
perceptual-preservation advantage under the tested SDXL setup, not identity
preservation, race editing, demographic inference, or bias mitigation.

## Suggested subject areas

- generative models and diffusion models;
- evaluation methodology and stress testing;
- responsible, fair, and accountable computer vision;
- face and portrait analysis;
- reproducibility and negative results.

## Keywords

latent diffusion; matched-change evaluation; skin tone; perceptual
preservation; prospective design; reproducibility; missingness

## Author-dependent enrollment fields

- Complete author names and order: **required before 2026-08-21 AoE**
- OpenReview IDs and conflict-complete profiles: **required**
- Corresponding author: **required**
- Paper ID: **235**, inserted into `paper/wacv/main.tex`
