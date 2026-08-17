# Prospective independent-seed replication protocol

Frozen before replication outcome generation began on 2026-08-17 UTC.

## Purpose and status

This is a prospective replication of the completed SDXL confirmatory campaign,
not a replacement for it and not a retrospective sensitivity analysis. It asks
whether the masked-versus-unmasked LPIPS result survives a newly generated
direction dataset and 30 new evaluation seeds while every method grid,
measurement, target, matching tolerance, and analysis rule remains unchanged.

- Planned config: `configs/replication_study.yaml`
- Planned-config SHA-256:
  `c6121300529a1d218b0891f953dd7d787043feb44aae870b7c0da2722fcde2b3`
- Planned-config fingerprint: `5057ea79d01525ba`
- Parent config fingerprint: `6e7e5a62a18ebfe6`
- Parent result-ledger SHA-256:
  `c15eecc46241ae945d8da5f1e7265bc225878926f279283208aa15e54cf17a4c`
- Replication launch time: `2026-08-17T04:20:31Z`
- Replication-data generator SHA-256:
  `c4808aa921bbb0f65967943c48456429a7789b988399640a47958f3e7ed883bf`
- Config validator SHA-256:
  `b36262ffa44684124152665f46840d550d559089f73897d9f241941f3c366966`
- Frozen replication-decision script SHA-256:
  `4a6a4f89c462f6547b111cc53bab6d84867e90381f93b9ae58b0b80dc355d29b`
- Preregistration archive SHA-256:
  `7a90e719add37ef445f643c39aa6465866359cf3fadb293e3a25acf666c2c559`

The paired-data campaign uses a new frozen base seed schedule and the standard
nonrepeating 10,000-seed cycle extension. The 30 evaluation seeds are
600000--600029. These are disjoint from the parent direction, validation,
calibration, and evaluation seeds.

## Locked sequence

1. Generate 96 new noise-coupled light/dark prompt pairs.
2. Reserve pairs 0--63 for the direction and pairs 64--95 for measurement
   validation.
3. Apply the same five held-out measurement gates. If any gate fails, stop and
   report a failed measurement replication; do not alter thresholds or select
   pairs.
4. Freeze the paired-data manifest and validation-report hashes into a new,
   immutable preregistered config.
5. Run all 570 conditions for seeds 600000--600029 exactly once, retaining
   failures and missingness.
6. Apply the same matched $\pm5$ ITA target with a three-degree tolerance and
   the same seed-level bootstrap and Holm correction.

No replication outcome may be inspected before step 4. Calibration grids are
copied from the parent and will not be retuned on replication seeds.

## Replication hypotheses and decision rules

The two directional masked-versus-unmasked LPIPS contrasts are the replication
family. Positive values favor masking.

- **Strict replication:** both lighter and darker contrasts are positive, both
  95% seed-bootstrap intervals exclude zero, and both $p$ values remain below
  .05 after Holm correction across the two tests.
- **Directional replication:** both point estimates are positive, but one or
  both corrected intervals include zero. This supports sign consistency but
  not a claim of independently significant replication.
- **Failure to replicate:** either directional point estimate is zero or
  negative.
- **Inconclusive coverage:** fewer than 12 shared matched seeds are available
  for either contrast. Effect estimates must still be reported, but no strict
  replication claim is allowed.

Face similarity remains a separately reported primary-outcome check inherited
from the parent study; its previous null result does not become an equivalence
claim. Background SSIM, pose, prompt-only comparisons, monotonicity,
missingness, and cross-campaign direction cosine are secondary or descriptive.
They cannot substitute for the locked LPIPS replication family.

## Reporting safeguards

- Report the replication independently before any pooled estimate.
- Label every analysis not specified above as exploratory.
- Do not describe a same-sign estimate with an interval spanning zero as a
  successful strict replication.
- Do not merge parent and replication rows and present them as one
  preregistered 60-seed study.
- Preserve all manifests, ledgers, frozen configs, logs, images, analyses, and
  SHA-256 hashes, including failed conditions.

## Execution record

The new paired campaign completed all 192 images with generator exit code 0.
All 32 held-out pairs were measurable. The frozen gates passed with detection
rate 1.000, pair-order accuracy 0.969, median pair gap 16.08 ITA degrees,
median absolute perturbation shift 0.35 degrees, and 95th-percentile shift 3.83
degrees.

- Replication paired-data manifest SHA-256:
  `71dfb6ee8e8a8ef2381a4521b0d9110a7a5b1f97d515d76413c3e0f8b5fbe12c`
- Replication validation report SHA-256:
  `c47967baf10fec5f361c4239c95999ee9a4a293cf8bfcff85141b4b0b2477541`
- Frozen replication config SHA-256:
  `b940522847ce647ae7a7147d5c33087a72537252345051a13078204f99cecc66`
- Frozen replication config fingerprint: `fa694f8bb214c219`
- Frozen-input archive SHA-256:
  `fb352010ea020f43446e30cf4c251506a844bdcb87a3fdfdb21a6553ee931184`
- Complete analysis-code freeze time: `2026-08-17T05:13:11Z`
- Complete analysis-code archive SHA-256:
  `c739cdd07dcad91d5f53f8c712c153e147f6dde97fccf54ed19e54bd4bb324bf`
- Frozen conditions: 570

The outcome campaign began only after these hashes were written into the
preregistered config. Before the outcome campaign completed and before any
effect was inspected, the full analysis dependency set was archived as an
additional integrity record. Completion and the result-ledger hash will be
appended without changing the decision rules above.

## Locked outcome

The A100 worker exited 0 after completing all 570 exact frozen conditions and
30 evaluation seeds. All images generated. One post-hoc condition (seed
600020, alpha -1.5) retained an edited face/skin-region measurement failure;
569 rows were metric-complete, and all rows in the matched methods were
complete. The independent local run-integrity audit passed every exact-key,
manifest, config, image, and direction-tensor check.

- Completion time: `2026-08-17T05:24:01.515350Z`
- Result-ledger SHA-256:
  `26bb0806a87089ea6ba1d3c4edb0a670a1aad67a6cf5ba55011c8e0d05755a69`
- Run-integrity audit SHA-256:
  `9a815f5eff1a8ad18d635adac9e869f696f960e6b9bd24ea9df34a8cad53f7fb`
- Replication paired-contrast SHA-256:
  `3676ea55caee18f586b2ed9b27cf4a5c3ae2585364f764ffaa03f76d8585202b`
- Replication assessment SHA-256:
  `2085c0a391d9407a4d242cc7f1c6963dce01743e98b0cbd1fc6bff6ed95e81bd`

The locked decision is **inconclusive coverage**. The lighter LPIPS advantage
was 0.0238 (95% CI [0.0049, 0.0415], n=8, two-test Holm p=.0555); the darker
advantage was 0.0409 ([0.0176, 0.0655], n=10, p=.0192). Both estimates retained
the parent sign, but both shared sample sizes fell below the prespecified
minimum of 12. The raw/masked cross-campaign direction cosines were
0.884/0.919. No pooled 60-seed confirmatory claim is made.
