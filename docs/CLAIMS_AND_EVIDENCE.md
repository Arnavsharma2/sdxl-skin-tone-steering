# Claims and evidence register

| Claim | Current evidence | Status | Evidence required for promotion |
|---|---|---|---|
| The code can inject a paired latent direction during every denoising step. | Implementation and one generated sweep. | Supported as an implementation claim. | Unit/integration regression test against a pinned Diffusers version. |
| The sweep changes visible appearance coherently. | Legacy one-seed re-analysis: complete target measurement, correct direction for 4/4 edits, Spearman ρ = -1.00 across five panels. | Pilot quantitative support. | External metric validation and held-out multi-seed replication. |
| Identity is preserved. | Moderate ±0.8 edits have mean FaceNet similarity 0.920 and pass the engineering gate; both ±1.5 extremes fail at least one gate. | Pilot support only in the post-hoc moderate band; rejected as a blanket claim. | Held-out paired comparison at matched target change with subgroup and failure reporting. |
| Background is preserved. | Corrected union-mask SSIM is valid for 4/4 legacy edits; mean 0.864 overall and 0.873 at ±0.8. | Pilot quantitative support. | Held-out paired comparison against masked and unmasked baselines. |
| Stepwise injection is better than post-hoc editing. | Qualitative development observation only. | Hypothesis. | Prespecified paired baseline and ablation. |
| The method mitigates model bias. | No downstream audit or fairness outcome. | Unsupported and removed from project summary. | A defined harm model, audit task, baseline disparity, intervention, and uncertainty analysis. |
| The direction represents race. | Skin-tone prompts and image brightness cannot establish race. | Rejected. | Not promotable; the construct is invalid. |
| The target metric has deterministic fail-closed color and sensitivity behavior. | Step 6 synthetic uint8 RGB fixtures cover known color changes, uniform and local illumination, exact QC boundaries, invalid masks/detection, nonfinite input, checksum drift, and runtime drift. | Supported only as an implementation claim. | Held-out calibrated-instrument portrait validation under `tmlr_collection_readiness_v1`. |

The legacy re-analysis is derived from composite PNG crops and has one
generation seed; it is not a substitute for the prespecified confirmatory
study. This file is the source of truth when README, abstract, figures, and
metadata disagree. Update it in the same commit as new experiment results.
