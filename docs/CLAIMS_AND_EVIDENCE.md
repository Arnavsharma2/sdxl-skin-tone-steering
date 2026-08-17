# Claims and evidence register

| Claim | Current evidence | Status | Evidence required for promotion |
|---|---|---|---|
| The code can inject a paired latent direction during every denoising step. | Implementation and one generated sweep. | Supported as an implementation claim. | Unit/integration regression test against a pinned Diffusers version. |
| The methods produce a monotonic measured skin-tone response. | Parent mean Spearman correlations were 0.95 prompt-only, 0.93 unmasked, 0.92 masked, and 0.67 post-hoc; replication values were 0.97, 0.98, 0.95, and 0.92. | Supported within two independently seeded synthetic SDXL campaigns. | Replication with calibrated physical colour references and other generators. |
| Identity is preserved better by masked steering. | Every parent face-similarity interval spanned zero. In replication, both masked-versus-unmasked intervals spanned zero, and no face contrast survived the broader 16-test Holm family. | Prospectively frozen primary hypothesis not supported in either campaign. | Larger, higher-coverage replication and human identity judgments; do not promote from current evidence. |
| Masking improves perceptual preservation versus unmasked steering. | Parent matched LPIPS advantages were 0.0235 lighter and 0.0825 darker. Replication estimates retained the signs at 0.0238 and 0.0409 with intervals above zero, but shared samples were only 8 and 10; the locked decision was inconclusive coverage and the lighter two-test Holm p=.0555. | Supported in the parent; sign-consistent but formally inconclusive in the prospective independent-seed replication. | A higher-coverage, cross-model/scheduler replication and human perceptual validation. |
| The LPIPS result is insensitive to the matching tolerance. | In an exploratory rerun at 1-, 2-, and 3-degree ITA tolerances, every estimable LPIPS effect retained a positive sign, but shared samples fell to 1--5 seeds at one degree and 3--12 at two degrees; no secondary contrast survived Holm correction at the stricter tolerances. | Directionally robust but too sparse for confirmatory inference under stricter tolerances. | A larger campaign with denser calibration grids and adequate strict-tolerance coverage. |
| The paired direction is reproducible. | Re-encoding all 64 frozen parent pairs exactly reproduced the recorded tensor. Across 200 disjoint split-halves, median raw/masked cosine rose from 0.42/0.53 at 8 pairs per half to 0.75/0.82 at 32. A new 64-pair campaign had raw/masked cosine 0.884/0.919 to the parent direction and a norm ratio of 1.04. | Reproducible across fixed execution and a new seed campaign, but still only within one prompt family and model. | Replication using independent prompt templates and model families. |
| Masking improves background/pose preservation. | Darker-direction background SSIM improved versus prompt-only and unmasked; pose improved versus prompt-only. Other strata did not survive Holm correction. | Supported only for the specified darker-direction contrasts. | Replication with semantic masks and a full 3D pose estimator. |
| Post-hoc editing is inferior. | It had the weakest mean monotonicity, one detector failure, and calibration coverage problems, but it was prespecified as feasibility-only and excluded from matched preservation tests. | Descriptive feasibility result only. | A separately powered, prospectively frozen matched-grid comparison. |
| The method mitigates model bias. | No downstream audit or fairness outcome. | Unsupported and removed from project summary. | A defined harm model, audit task, baseline disparity, intervention, and uncertainty analysis. |
| The direction represents race. | Skin-tone prompts and image brightness cannot establish race. | Rejected. | Not promotable; the construct is invalid. |

This file is the source of truth when README, abstract, figures, and metadata
disagree. Update it in the same commit as new experiment results.

`scripts/verify_manuscript_claims.py` additionally checks 16 central numerical
claim groups in both paper variants against content-hashed retained evidence.
The current audit is `experiments/analysis/manuscript_claim_audit.json`, SHA-256
`34b28746ba0d8acca134cc99dc42a16a11df84fd4067aec01f3cbbe8d2c37887`.
