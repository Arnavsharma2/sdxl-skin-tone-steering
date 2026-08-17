# Local artifact archives

Large archives are intentionally ignored by Git and identified by SHA-256.
They are author-retained unless a stable release or anonymous review URL is
recorded. See `docs/ARTIFACT_AVAILABILITY.md`; a hash alone does not make an
archive independently accessible.

| File | Size | SHA-256 | Contents |
|---|---:|---|---|
| `skin_tone_study_colab_20260816.tgz` | 64 MiB | `525736db241d01beef6f310ac210df013fc199ff92a19ef35c6c9b6b637f31e7` | Source/config snapshot and the resumable partial MPS calibration state transferred to Colab. |
| `calibration_cuda_artifacts_20260816.tgz` | 209 MiB | `c654bbbe9646790b880003427b6dd432f6410af56bb3941e147ff9a6deec872c` | CUDA calibration runs, images, manifests, tables, logs, configs, and corrected runtime files. |
| `full_study_prereg_artifacts_20260816.tgz` | 212 MiB | `0291ad0335e90739ecd3caeb65b13d5a252581d6de99db021aca791eaacf9267` | All 192 direction-data images, generation ledger and manifest, held-out measurement validation, and the exact preregistered config used for the CUDA confirmatory run. |
| `confirmatory_cuda_artifacts_20260816.tgz` | 678 MiB | `66973e4ce3fa5a43655b694738f60efeee983dd46bdaec4961b9dbf8ff924da4` | Clean 570-condition CUDA confirmatory run, 600 PNGs including per-seed bases, run manifest, frozen config snapshot, execution log, and first frozen-analysis tables. |
| `direction_stability_cuda_20260816.tgz` | 344 KiB | `a93aea777e71b643393b2918fd7c2d11ee136e6052355ec342ca88fbd7ad6cd0` | Post-confirmatory split-half direction-stability resamples, summary, figure, exact recomputed direction, audit, and executed script. |
| `replication_preregistration_20260817.tgz` | 20 KiB | `7a90e719add37ef445f643c39aa6465866359cf3fadb293e3a25acf666c2c559` | Prospective independent-seed replication config, protocol, generator, validators, freeze logic, and locked analysis scripts archived after launch but before any replication outcomes existed. |
| `replication_frozen_inputs_20260817.tgz` | 52 KiB | `fb352010ea020f43446e30cf4c251506a844bdcb87a3fdfdb21a6553ee931184` | New paired-data manifest and ledger, held-out validation tables/report, planned and immutable replication configs, generation log, and preserved no-op orchestration incident log. |
| `replication_analysis_code_freeze_20260817.tgz` | 16 KiB | `c739cdd07dcad91d5f53f8c712c153e147f6dde97fccf54ed19e54bd4bb324bf` | Complete statistical-analysis dependency freeze created at 05:13:11 UTC, before the replication outcome run completed and before any effect was inspected. |
| `replication_cuda_slim_20260817.tgz` | 628 KiB | `19e309fee99462cf3cdc6a164c120be9bbe5a000c02040719db499fd81a26d07` | Hash-verified ledger, frozen run metadata, direction tensors, validation artifacts, configs, and execution logs for rapid analysis. |
| `replication_cuda_full_20260817.tgz` | 663 MiB | `be587ca138878736c9bad55758701e6d66c4d8d63772361810c6eb5023a42fbd` | Complete 570-condition replication run with all 630 run PNGs, tensors, manifests, configs, audit, and logs. |
| `replication_direction_data_full_20260817.tgz` | 213 MiB | `f638e8be419dfc542b9d4a7b295675ceb9b7bdf0db8ce7dca4c689ff519966c5` | All 192 independently seeded paired direction/validation images plus their content-attested manifest, ledger, validation outputs, configs, and logs. |
| `replication_analysis_20260817.tgz` | 173 KiB | `b1f778749058a1072e2e7f00de2a3400526dd41c2b8366f549588babdbfd30b6` | Post-outcome locked replication tables and assessment, comparison figure, immutable config, environment lock, and exact analysis/audit code. |

Verify an archive with:

```bash
shasum -a 256 experiments/artifacts/<archive>.tgz
```
