# CUDA calibration audit

Date: 2026-08-16
Purpose: adaptive calibration only; not confirmatory evidence.

## Execution environment

- GPU: NVIDIA A100-SXM4-80GB
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- Diffusers: 0.27.2
- Transformers: 4.39.3
- NumPy: 1.26.4
- Pillow: 10.2.0
- Immutable SDXL revision: `462165984030d82259a11f4367a4eed129e94a7b`

All 46 repository tests passed locally after the calibration fixes. The Colab
source snapshot passed the then-current 43-test suite before execution.

## Campaigns

| Campaign | Config fingerprint | Rows | Results SHA-256 |
|---|---:|---:|---|
| candidate v2, corrected CUDA | `470e8a06b2a845b3` | 84 | `8ee2d04f6fc33767add489a92e73a6f372a19778c39199244c76146b74712c32` |
| candidate v3 extension, CUDA | `4d3b819cc92a93cd` | 63 | `444cd6c8471a63c199acbbeb905da92cbc0ad6b003df82b9a6de8874c162558a` |

The combined calibration contained 147 input rows, 135 unique condition keys,
12 equivalent repeated keys, 131 fully evaluated rows, and 4 failed extreme
post-hoc endpoints. Failed endpoints remain in `incomplete_conditions.csv` and
were ineligible for target matching.

The complete artifact archive is
`experiments/artifacts/calibration_cuda_artifacts_20260816.tgz`, SHA-256
`c654bbbe9646790b880003427b6dd432f6410af56bb3941e147ff9a6deec872c`.

## CUDA precision defect found during calibration

The first CUDA run produced black images for every nonzero latent edit. The
paired direction had been estimated by encoding 1024px images with SDXL's
numerically unstable fp16 VAE, producing non-finite values. The correction:

1. temporarily upcasts the force-upcast VAE for direct encoding and decoding;
2. restores the pipeline-managed dtype immediately afterward; and
3. rejects any non-finite paired direction before generation.

The failed run is retained in the artifact archive. The corrected direction
was finite with norm 76.5170, mean absolute value 0.217536, and maximum absolute
value 2.02510.

## Target coverage

The calibration target was an absolute 5 ITA-degree change with tolerance 3.

| Method | Lighter direction | Darker direction | Confirmatory role |
|---|---:|---:|---|
| prompt-only | 3/3 seeds | 3/3 seeds | matched-change comparator |
| stepwise unmasked | 3/3 | 3/3 | matched-change comparator |
| stepwise masked | 3/3 | 3/3 | reference method |
| post-hoc latent | 1/3 | 3/3 | feasibility/dose-response only |

The wider post-hoc probe remained identity-dependent and non-monotonic. Two
seeds had no lighter-direction setting near the target despite probing alphas
from -3 to +3. Continuing to tune that method would not justify a symmetric
matched-change claim, so its feasibility-only role was frozen before any
confirmatory generation.

## Frozen confirmatory grids

- prompt-only: `[-2.0, 0.0, 0.2]`
- post-hoc latent: `[-1.5, -1.0, -0.5, -0.25, 0.0, 0.5, 1.0, 1.5]`
- stepwise unmasked: `[-0.75, 0.0, 0.5]`
- stepwise masked: `[-1.0, -0.75, 0.0, 0.375, 0.5]`

The matched-change set is prompt-only, stepwise unmasked, and stepwise masked.
Post-hoc results remain part of descriptive, missingness, monotonicity, and
target-coverage reporting.
