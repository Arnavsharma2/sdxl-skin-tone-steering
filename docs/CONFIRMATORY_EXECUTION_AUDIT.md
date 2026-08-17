# Confirmatory execution audit

This document records operational facts for the CUDA confirmatory campaign. It
is not a results summary and must be updated from the archived run artifacts.

## Frozen inputs

- Study config: `configs/full_study_preregistered.yaml`
- Config fingerprint: `6e7e5a62a18ebfe6`
- SDXL revision: `462165984030d82259a11f4367a4eed129e94a7b`
- Direction-data manifest SHA-256:
  `382a77c27aa6763e83133bb2f89fb249ad02f692e1775c7ef98c9eebf520a24d`
- Measurement-validation report SHA-256:
  `679571f02e57a40fb208d6ffb54391d654cafa6dc996cbd1947941e3bfd2a0e0`
- Author-controlled pre-outcome freeze archive SHA-256:
  `0291ad0335e90739ecd3caeb65b13d5a252581d6de99db021aca791eaacf9267`
- Planned conditions: 30 seeds and 570 `(seed, method, alpha)` rows.

The direction-data manifest attests 96 light-descriptor and 96 dark-descriptor
images. The held-out measurement validation completed all 32 pairs and passed
every frozen gate before the immutable config was created.

## Runtime

- Platform: Google Colab A100 High-RAM
- GPU: NVIDIA A100-SXM4-80GB
- Python: 3.12.13
- PyTorch: 2.11.0+cu128
- Diffusers: 0.27.2
- Transformers: 4.39.3
- NumPy: 1.26.4
- Pillow: 10.2.0

The clean campaign began at `2026-08-17 02:57:35` UTC and writes to
`experiments/runs/confirmatory_cuda` in the Colab workspace.

## Startup incident

During the first status check, reusing a Colab code cell caused the launcher to
execute more than once. All matching workers were stopped within approximately
three minutes. Their partial output and log were moved intact to
`confirmatory_cuda_concurrency_incident_20260817_0256` and are excluded from
analysis. The clean output directory was created only after all duplicate
workers had stopped. A fresh cell then started exactly one worker; a process
audit confirmed only that worker before its first result row.

This was an execution-control incident, not a protocol change. No frozen input,
seed, method grid, metric, or analysis rule was modified. The clean campaign
started from an empty output directory.

## Completion and integrity

- Completion time: `2026-08-17 03:39` UTC (approximately; final row timestamp)
- Worker exit code: 0
- Completed rows: 570 of 570
- Unique `(seed, method, alpha)` keys: 570
- Unique confirmatory seeds: 30
- Generation failures: 0
- Metric-incomplete rows: 1
- Incomplete condition: seed 500019, post-hoc latent, alpha -1.5;
  `edited_face_or_skin_region_missing`
- Result ledger SHA-256:
  `c15eecc46241ae945d8da5f1e7265bc225878926f279283208aa15e54cf17a4c`
- Clean run archive SHA-256:
  `66973e4ce3fa5a43655b694738f60efeee983dd46bdaec4961b9dbf8ff924da4`
- Analysis config fingerprint check: passed, `6e7e5a62a18ebfe6`
- Analysis rows/expected conditions: 570/570
- Matched rows: 180
- Prespecified contrast rows: 16
- Local analysis audit SHA-256:
  `5a796f7dd8b401d6a8f89aa3e92799a3dace8eb8d95c2c8577df5e251889e4cb`
- Independent run-integrity audit: passed
- Run-integrity audit SHA-256:
  `34aa1c73d2a7efddb4cf98a2d6d269844d32966ba96ac39d8b7b89d27a9dd760`
- Final manuscript PDF SHA-256:
  `4b0d45a7c007fefa5dcefc5682f66901a1d5c82824f02ff75722175b7b286e72`

The archive was downloaded, copied into `experiments/artifacts`, and verified
locally before extraction. Local analysis then rechecked the exact frozen
condition set before producing tables.

## Post-confirmatory robustness analyses

These analyses were performed after the frozen confirmatory analysis and are
therefore exploratory rather than part of the prospectively frozen analysis.

The archive is identified by content hash but was not deposited with an
independent timestamping service before collection. The manuscript therefore
uses “prospectively frozen,” not “externally preregistered.”

- Matching-tolerance sensitivity output:
  `experiments/analysis/robustness`
- Robustness audit SHA-256:
  `e4c67cd85ffa0c51643a55d6484cd76c623d5dec0b6751c2b857423f488a5caf`
- Tolerance contrasts SHA-256:
  `008bd3d60505cc9f064c7ec373d1851a2b3658d7f9d4e07a778c21f554c4d186`
- Tolerance coverage SHA-256:
  `213fc9dc155ce1244e30d25117a1c05c1a7af08fce02422782573c86822c0f78`
- Tolerances evaluated: 1, 2, and 3 ITA degrees
- Direction of every estimable LPIPS contrast: positive
- Shared matched samples: 1--5 seeds at tolerance 1 and 3--12 at tolerance 2
- Holm-significant secondary contrasts at tolerances 1 and 2: none
- Direction-stability output:
  `experiments/analysis/direction_stability_cuda`
- Direction-training pairs re-encoded: 64
- Full-direction cosine against the confirmatory tensor: 0.99999988
- Maximum absolute tensor difference: 0.0
- Split-half resamples: 200 for each of 8, 16, and 32 pairs per half
- Median raw/masked cosine at 32 pairs per half: 0.75/0.82
- Direction-stability archive SHA-256:
  `a93aea777e71b643393b2918fd7c2d11ee136e6052355ec342ca88fbd7ad6cd0`
- Stability summary SHA-256:
  `90d3398074e796883e7940b144bcf3900eb58ae6bf0e0a15a4f012dfb50f8a95`
- Stability audit SHA-256:
  `58e264aa22e3ac19641ad7720559f0f7c29f1491993d10731522e7be1c9a4177`
