# Step 6 target-metric validation and collection readiness

## Decision

Confirmatory collection is **not approved or ready**. Step 6 adds a
deterministic, fail-closed readiness decision, but it does not supply the
missing external validation, exact generation artifacts, supported execution
host, or collection approval. No 30-seed matrix and no SDXL portrait were
generated while developing this machinery.

The machine-readable authority is
[`configs/step6_readiness.yaml`](../configs/step6_readiness.yaml), readiness ID
`tmlr_collection_readiness_v1`. It was defined before confirmatory collection
and requires all seven criteria below. The report schema is
[`schemas/step6_readiness_report.schema.json`](../schemas/step6_readiness_report.schema.json).
Reproducible post-Step 6 environment, artifact, external-validation,
provenance, and approval instructions are in
[`POST_STEP6_READINESS.md`](POST_STEP6_READINESS.md).

## Acceptance criteria

1. **External target-metric validity.** Supply an independently reviewed,
   held-out validation of `relative_cheek_CIELAB_Lstar_v1` against a calibrated
   colorimeter or spectrophotometer on controlled frontal portraits with a
   neutral background. It must contain at least 30 independent people, cover
   at least 30 CIELAB L* points, report capture and illumination conditions,
   preserve the full failure denominator, and document license/consent and the
   error profile. For reference changes of at least the frozen 2 L* target
   threshold, median absolute delta error must be at most 1 L*, 95th-percentile
   absolute delta error at most 2 L*, and direction agreement at least 0.95,
   with uncertainty intervals. These are conservative precollection
   engineering acceptance limits tied to the frozen 2 L* response gate; they
   are not published universal validity standards.
2. **Illumination and color sensitivity.** Every declared deterministic
   synthetic check must pass. Checks cover known darker/lighter RGB changes,
   uniform exposure, local face-only lighting, both sides of the 8 L*
   reference-shift and 14-chroma boundaries, a colored reference, detector and
   mask failures, nonfinite input, checksum drift, and runtime drift. Passing
   this criterion validates implementation behavior only.
3. **Supported metric runtime.** The actual evaluation process must be Linux,
   Python `>=3.10,<3.13`, MediaPipe `0.10.21`, and the CPU Face Landmarker
   delegate. No unsupported-host fallback is permitted.
4. **Metric artifacts.** Every artifact in the frozen evaluation protocol must
   be a readable regular file with the exact recorded SHA-256: MediaPipe Face
   Landmarker, FaceNet VGGFace2, all three MTCNN networks, the AlexNet backbone,
   and LPIPS Alex v0.1 weights.
5. **Immutable generation provenance.** A supplied provenance record must bind
   the exact SDXL model ID to one lowercase 40-hex requested/resolved revision,
   record model-license acceptance, checksum the direction and its source
   manifest, record an immutable direction code commit, and match the frozen
   estimator, train/held-out counts, deterministic VAE setting, optimization
   setting, study-config hash, and evaluation-protocol hash.
6. **Storage and privacy.** The versioned
   [`collection_policy.yaml`](../configs/collection_policy.yaml) must remain in
   force: confirmatory portraits are synthetic only, real-person inputs are
   prohibited, generated runs are ignored, face embeddings are neither
   persisted nor committed, and publication/upload/release requires a separate
   license, privacy, and misuse review.
7. **Explicit collection approval.** A separate approval record must say
   `decision: approved` for `scope: confirmatory_collection` and bind the exact
   study, evaluation, readiness, external-validation, model-revision, and
   direction hashes. Structural consistency is not cryptographic
   authentication of the approver.

Any failed or absent criterion blocks collection. A report with an edited
`collection_ready` flag is insufficient: the confirmatory runner rechecks the
required criterion set, authority hashes, model ID/revision, and direction
hash. It also rejects a loaded model whose resolved revision differs from the
readiness-bound revision.

## Running the non-generative harness

```bash
python3 -m scripts.validate_readiness \
  --output experiments/readiness/step6_readiness.json
```

This default invocation is expected to write a blocked report because no
external evidence, generation provenance, or approval is supplied. `--strict`
writes the same audit report and exits 2 when blocked. Frozen metric locations
can be overridden explicitly with repeatable `--artifact NAME=PATH` arguments;
every override is still checksum-verified.

`python3 -m scripts.download_metric_models` downloads all seven artifacts from
the upstream registry to an ignored local cache and writes an override
manifest. It does not grant redistribution rights.

A future complete invocation additionally supplies:

```text
--external-validation PATH
--generation-provenance PATH
--approval PATH
```

The external-validation record must contain the fields named in criterion 1
and an `evidence_artifacts` list of `{path, sha256}` entries. The generation
record contains `model` and `direction` mappings described in criterion 5.
The approval record contains the bound fields described in criterion 7. The
harness fails closed on absent, malformed, nonfinite, mismatched, unreadable,
or checksum-drifted inputs.

Confirmatory execution additionally requires the passing report:

```text
python3 scripts/run_confirmatory.py --execute \
  --model-revision <exact-40-hex-revision> \
  --direction <checksum-bound-direction.pt> \
  --readiness-report <passing-step6-report.json>
```

This documents the gate; it is not authorization to run it now.

## What the synthetic evidence validates

- OpenCV's normalized uint8-RGB-to-CIELAB path produces the expected sign and
  more than 2 L* of relative change for the declared patch changes.
- A uniform additive RGB shift leaves at most 1 L* residual in the declared
  fixture, showing first-order exposure correction in that narrow case.
- A face-only lighting change produces more than 2 L* apparent response while
  the border is unchanged. This deliberately passes as detection of a known
  confound, not as robustness.
- The exact 8 L* reference-shift and 14-chroma boundaries are accepted and the
  next representable value above each is rejected.
- Colored references, detector failures, invalid masks, nonfinite arrays,
  corrupted artifacts, and unsupported runtime observations fail closed.
- Face-detection failure still produces an invalid measurement and remains an
  outcome; it is not excluded or replaced.

## What remains unvalidated

- Agreement of the cheek/border image metric with calibrated measurements of
  human skin in the intended portrait domain.
- Generalization of the geometric mask and neutral-border assumption across
  skin appearances, facial geometry, occlusion, makeup, rendering pipelines,
  cameras, white balance, and local illumination.
- Demographic error rates of MediaPipe detection or FaceNet preservation
  signals. Face similarity remains an engineering metric, not an identity
  guarantee.
- The frozen 2 L* response and preservation thresholds as clinical, social,
  fairness, or perceptual guarantees.
- Any claim of race measurement, bias mitigation, identity preservation,
  disentanglement, baseline superiority, or state-of-the-art performance.

## Primary sources, licensing, and evidence gaps

The machine-readable source register is
[`configs/validation_sources.yaml`](../configs/validation_sources.yaml).

- [ISO/CIE 11664-4:2019](https://www.cie.co.at/publications/colorimetry-part-4-cie-1976-lab-colour-space-1)
  defines CIELAB calculation and interpretation. The standard is copyrighted;
  only its public bibliographic summary is cited. It does not validate this
  facial-region procedure.
- [Weatherall and Coombs (1992)](https://doi.org/10.1111/1523-1747.ep12616156)
  is a primary instrument-based study of human-skin CIELAB values. Publisher
  copyright applies; no article text or dataset is redistributed. Its results
  cannot validate an image-derived cheek/border metric.
- [He et al. (2022)](https://doi.org/10.1002/col.22737) evaluates a calibrated
  image-based facial-skin measurement system against instrument readings. Its
  capture system and mappings differ from this repository, and reuse terms for
  its calibration data are not established here.
- [OpenCV color conversions](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
  document the implementation path; OpenCV 4.x is
  [Apache-2.0 licensed](https://github.com/opencv/opencv/blob/4.x/LICENSE).
  Implementation licensing and conformance do not establish construct validity.
- The MediaPipe repository is [Apache-2.0](https://github.com/google-ai-edge/mediapipe),
  facenet-pytorch code is [MIT](https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md),
  and LPIPS code is [BSD-2-Clause](https://github.com/richzhang/PerceptualSimilarity).
  Those code licenses do not by themselves settle redistribution terms for
  separately downloaded task models, pretrained weights, or training data.
- SDXL base 1.0 uses the
  [CreativeML Open RAIL++-M license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/e4e60c65aa20ee60092c60ba197f541872cf9373/LICENSE.md).
  Acceptance must be recorded for the exact revision, and it does not authorize
  publishing a dataset or release artifact.

The source review therefore supports use of CIELAB as a defined color space
and the need for calibrated image validation. It does **not** close the
external-validity gap. Exact licenses for all metric/model weights and any
future validation data also remain a release-review requirement; the project
itself still has no selected repository license.
