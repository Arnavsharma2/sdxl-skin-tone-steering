# Audited SDXL skin-tone steering study

This release archives the first complete research package for evaluating a
paired SDXL VAE skin-tone direction injected during denoising.

## Included artifacts

- author-identified preprint PDF and arXiv source package;
- prospectively frozen parent and independent-seed replication protocols;
- retained seed-level results, manifests, analysis outputs, and figures;
- deterministic anonymous supplementary archive;
- automated audit linking 16 central manuscript claim groups to retained
  evidence; and
- 70-test research and reproducibility suite.

## Evidence statement

The parent campaign found no detectable face-embedding advantage. Spatial
masking improved LPIPS relative to unmasked steering in both target
directions. A prospectively specified independent-seed campaign reproduced
the effect signs and yielded intervals above zero, but its locked replication
decision remained inconclusive because only 8 and 10 shared matched seeds were
available, below the prespecified minimum of 12. This release does not claim
strict replication, identity preservation, disentanglement, race inference,
or bias mitigation.

## Reproducibility

The exact SDXL revision, dependencies, configurations, manifests, result
ledgers, missingness, and analysis procedures are recorded in the repository.
See `docs/REPRODUCIBILITY.md`, `docs/CLAIMS_AND_EVIDENCE.md`, and
`docs/ETHICS.md` before reuse.
