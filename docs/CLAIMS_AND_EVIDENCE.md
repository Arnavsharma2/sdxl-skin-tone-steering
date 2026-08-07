# Claims and evidence register

| Claim | Current evidence | Status | Evidence required for promotion |
|---|---|---|---|
| The code can inject a paired latent direction during every denoising step. | Implementation and one generated sweep. | Supported as an implementation claim. | Unit/integration regression test against a pinned Diffusers version. |
| The sweep changes visible appearance coherently. | One qualitative example. | Pilot only. | Blinded human or validated continuous skin-tone assessment across held-out seeds. |
| Identity is preserved. | Checked-in metadata has no face-similarity values. | Unsupported. | Valid face embeddings for all methods, detector-failure reporting, held-out paired comparison. |
| Background is preserved. | Historical metadata reports 0.798–0.864, but the old fallback used whole-image SSIM when no face mask was detected. | Unsupported. | Valid face masks and background-only SSIM on held-out seeds. |
| Stepwise injection is better than post-hoc editing. | Qualitative development observation only. | Hypothesis. | Prespecified paired baseline and ablation. |
| The method mitigates model bias. | No downstream audit or fairness outcome. | Unsupported and removed from project summary. | A defined harm model, audit task, baseline disparity, intervention, and uncertainty analysis. |
| The direction represents race. | Skin-tone prompts and image brightness cannot establish race. | Rejected. | Not promotable; the construct is invalid. |

This file is the source of truth when README, abstract, figures, and metadata
disagree. Update it in the same commit as new experiment results.
