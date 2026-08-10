# Legacy sweep re-analysis

This audit recovers five panels from the checked-in historical composite,
standardizes them to 573×573 pixels, and evaluates the four nonzero-alpha edits
against the crop-matched alpha-zero panel. The import and evaluation are fully
reproducible with `scripts/import_legacy_strip.py` and
`scripts/evaluate_sweep.py`.

| Alpha | Face similarity | LPIPS | Background SSIM | Pose drift | Δ relative L* | Gates | Quality rubric |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -1.5 | 0.476 | 0.284 | 0.850 | 4.53° | +16.60 | 4/5 | 0.741 |
| -0.8 | 0.892 | 0.243 | 0.852 | 0.81° | +10.47 | 5/5 | 0.887 |
| +0.8 | 0.948 | 0.129 | 0.894 | 2.24° | -14.79 | 5/5 | 0.930 |
| +1.5 | 0.689 | 0.194 | 0.860 | 6.32° | -30.13 | 3/5 | 0.819 |

Audit outcomes:

- metric completion: 4/4 (100%);
- target-direction accuracy: 4/4 (100%);
- monotonic target response across all five panels: Spearman ρ = -1.00;
- all-gate success: 2/4 overall and 2/2 in the exploratory moderate operating
  band (α = ±0.8);
- moderate-band means: face similarity 0.920, LPIPS 0.186, background SSIM
  0.873, pose drift 1.53°, and quality rubric 0.908.

This remains a one-seed, post-hoc pilot based on composite-image crops. It does
not establish generalization, baseline superiority, identity preservation at
extreme settings, or bias mitigation. Machine-readable evidence is in
`evaluation/pair_metrics.csv`, `evaluation/summary.json`, and
`import_manifest.json`.
