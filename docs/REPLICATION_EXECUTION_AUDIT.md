# Independent-seed replication execution audit

This document records operational facts for the prospective replication
campaign. It is separate from the statistical results and is completed only
from archived run artifacts.

## Frozen inputs

- Planned config: `configs/replication_study.yaml`
- Planned config SHA-256:
  `c6121300529a1d218b0891f953dd7d787043feb44aae870b7c0da2722fcde2b3`
- Planned config fingerprint: `5057ea79d01525ba`
- Frozen config: `configs/replication_study_preregistered.yaml`
- Frozen config SHA-256:
  `b940522847ce647ae7a7147d5c33087a72537252345051a13078204f99cecc66`
- Frozen config fingerprint: `fa694f8bb214c219`
- SDXL revision: `462165984030d82259a11f4367a4eed129e94a7b`
- Direction-data manifest SHA-256:
  `71dfb6ee8e8a8ef2381a4521b0d9110a7a5b1f97d515d76413c3e0f8b5fbe12c`
- Measurement-validation report SHA-256:
  `c47967baf10fec5f361c4239c95999ee9a4a293cf8bfcff85141b4b0b2477541`
- Initial author-controlled freeze archive SHA-256:
  `7a90e719add37ef445f643c39aa6465866359cf3fadb293e3a25acf666c2c559`
- Frozen-input archive SHA-256:
  `fb352010ea020f43446e30cf4c251506a844bdcb87a3fdfdb21a6553ee931184`
- Planned outcomes: 30 new seeds and 570 `(seed, method, alpha)` rows.

The new paired-data campaign contains 96 light-descriptor and 96
dark-descriptor images generated from a disjoint direction-data seed schedule.
All 32 held-out validation pairs completed; all five prespecified measurement
gates passed before the frozen config was created. No replication effect
estimate was inspected before freezing or before the outcome run completed.

## Runtime

- Platform: Google Colab A100 High-RAM
- GPU: NVIDIA A100-SXM4-80GB
- Replication launch time: `2026-08-17T04:20:31Z`
- Output directory: `experiments/runs/replication_cuda`
- Frozen analysis script SHA-256:
  `4a6a4f89c462f6547b111cc53bab6d84867e90381f93b9ae58b0b80dc355d29b`
- Frozen analysis-table script SHA-256:
  `319c35eb09dab38001d35aa04c69378439b2c1ee914a9cbe63b1debcbd25949b`
- Frozen statistical helper SHA-256:
  `a4fab0cbea12ee2d4826ef470fed05bbba6bc6b5545938dcff0ad059e24c5e85`
- Frozen config-loader SHA-256:
  `b36262ffa44684124152665f46840d550d559089f73897d9f241941f3c366966`
- Frozen environment-lock SHA-256:
  `89b52505c320b28a5218e049f2f28ecf8823251cfc2a97ef09c111c865ae0ca1`
- Complete analysis-code freeze time: `2026-08-17T05:13:11Z`
- Complete analysis-code archive SHA-256:
  `c739cdd07dcad91d5f53f8c712c153e147f6dde97fccf54ed19e54bd4bb324bf`

The outcome worker was launched only after the data campaign passed validation
and the immutable replication configuration was written and archived.

## Startup incident

The first orchestration wrapper waited on a defunct data-generation child after
generation had already completed successfully. It did not start an outcome
worker and made no changes to frozen inputs or outputs. The no-op wrapper was
terminated, its log was retained as `replication_pipeline_stalled_wait.log`,
and the same frozen gates were then executed by one clean outcome worker. This
was an orchestration incident, not a protocol or analysis change.

## Completion and integrity

- Completion time: `2026-08-17T05:24:01.515350Z`
- Worker exit code: 0
- Completed rows: 570 / 570
- Unique `(seed, method, alpha)` keys: 570
- Unique evaluation seeds: 30
- Generation failures: 0
- Metric-incomplete rows: 1
- Incomplete condition: seed 600020, post-hoc latent, alpha -1.5;
  `edited_face_or_skin_region_missing`
- Result ledger SHA-256:
  `26bb0806a87089ea6ba1d3c4edb0a670a1aad67a6cf5ba55011c8e0d05755a69`
- Clean full-run archive SHA-256:
  `be587ca138878736c9bad55758701e6d66c4d8d63772361810c6eb5023a42fbd`
- Full paired-direction-data archive SHA-256:
  `f638e8be419dfc542b9d4a7b295675ceb9b7bdf0db8ce7dca4c689ff519966c5`
- Analysis input rows/expected conditions: 570/570
- Local run-integrity audit: passed
- Local run-integrity audit SHA-256:
  `9a815f5eff1a8ad18d635adac9e869f696f960e6b9bd24ea9df34a8cad53f7fb`
- Locked replication decision: `inconclusive_coverage`
- Replication assessment SHA-256:
  `2085c0a391d9407a4d242cc7f1c6963dce01743e98b0cbd1fc6bff6ed95e81bd`
- Post-outcome analysis archive SHA-256:
  `b1f778749058a1072e2e7f00de2a3400526dd41c2b8366f549588babdbfd30b6`

The locked lighter/darker LPIPS advantages were 0.0238 and 0.0409 with
ordinary 95% bootstrap intervals [0.0049, 0.0415] and [0.0176, 0.0655]. The
two-test Holm p-values were .0555 and .0192. Shared matched samples were only 8
and 10, so the prespecified coverage override makes the formal outcome
inconclusive despite same-sign estimates and intervals above zero.

## Analysis-code integrity

The actual replication decision must be produced by the exact
`scripts/analyze_replication.py` included in the author-controlled pre-outcome freeze
archive. The working copy was compared byte-for-byte against that archive and
restored to its archived SHA-256 before any outcome analysis. Plotting helpers
may share equivalent functions, but they are not authoritative for the locked
decision. Before the outcome campaign completed, a second archive additionally
froze `scripts/analyze_study.py`, `src/study/analysis.py`, the config loader,
the immutable config, and the environment lock so every dependency used by the
locked decision has a recorded pre-completion hash.

These content hashes document internal artifact identity but are not an
independent timestamp. The manuscript describes the replication as
prospectively specified and frozen rather than externally preregistered.
