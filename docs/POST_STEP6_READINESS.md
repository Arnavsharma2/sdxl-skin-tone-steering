# Post-Step 6 readiness evidence guide

## Scope and non-authorization

This follow-up closes only blockers that can be closed with reproducible
technical evidence. It does not change `tmlr_collection_readiness_v1`, the
frozen study matrix, metric protocol, statistical analysis, thresholds,
estimands, or failure meanings. It does not provide external validity,
generation provenance, license acceptance, independent review, or collection
approval. It never runs SDXL or the confirmatory matrix.

The expected evidence-backed state after running the Linux checks with all
seven local metric artifacts is **4/7 passed**: synthetic sensitivity,
supported Linux runtime, checksum-verified metric artifacts, and
storage/privacy. External target-metric validity, immutable SDXL/direction
provenance, and explicit collection approval remain blocked until real inputs
are supplied.

## Reproducible Linux evaluation path

The evaluation image uses Linux/amd64, the platform-specific official
`python:3.12.11-slim-bookworm` manifest digest
`sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49`,
a dated Debian snapshot, and the fully hashed
`requirements/evaluation-linux-py312.lock`. It pins MediaPipe `0.10.21` and
uses only the CPU Face Landmarker delegate. It is an evaluation/test image; it
does not contain SDXL generation dependencies or model weights.
`containers/evaluation/environment.yaml` records the platform, base manifest,
snapshot, package indexes, and lock-file SHA-256 as machine-readable build
authority. A local image digest is build-output evidence and is not hard-coded
as though it were a registry-published digest.

```bash
make lock-evaluation       # only when intentionally updating the lock
make evaluation-image
make metric-models         # downloads to ignored .artifacts/ and verifies SHA-256
make readiness-linux       # non-generative; a blocked 4/7 report is expected
```

The image build and Linux harness are reproducibility evidence, not a passing
readiness report by themselves. A macOS run remains unsupported and must not
be substituted.

## Metric-artifact sources and licensing

`configs/metric_artifacts.yaml` is the machine-readable upstream registry.
Every source is operated by the upstream project or its official storage, and
every downloaded byte must match the unchanged SHA-256 in
`configs/evaluation_protocol.yaml`. Immutable upstream commits are used for
the MTCNN and LPIPS files; the MediaPipe `latest` alias is accepted only when
its content matches the frozen full hash. The downloader writes an ignored
manifest containing source, revision, path, expected/actual checksum, size,
and verification result.

No weight is committed. Code-license identification does not establish
pretrained-weight or training-data redistribution rights. The registry keeps
each such status as `unresolved_*_do_not_redistribute`, and downloading for
local evaluation does not change that status. A separate release license,
privacy, and misuse review is still mandatory.

## Held-out instrument validation input

Use `templates/external_validation_manifest.yaml` and keep real-person inputs
under the ignored `external_validation/private/` directory with
minimum-necessary access. The checksum-bound CSV must have exactly these
columns in this order:

```text
person_id,pair_id,reference_lstar_before,reference_lstar_after,metric_relative_lstar_before,metric_relative_lstar_after,detector_status,mask_status
```

Each `pair_id` is unique. Instrument L* values are finite and in `[0, 100]`.
Detector and mask statuses are exactly `passed` or `failed`. Metric values are
present exactly when both statuses pass; failed rows stay in the eligible
denominator. The executable analysis is:

```bash
python3 -m scripts.analyze_external_validation \
  external_validation/private/manifest.yaml \
  --output external_validation/output/analysis.json
```

The tool uses the frozen 2 L* minimum reference change and 95% confidence
level, reports the full planned and eligible denominators, bootstraps whole
independent-person clusters, and computes median absolute delta error,
95th-percentile absolute delta error, and direction agreement. If an eligible
detector or mask failure makes a metric outcome undefined, acceptance values
remain unavailable rather than being promoted from a complete-case subset.

Synthetic CSVs may test the analysis implementation only. A real package must
still use at least 30 independent held-out people spanning at least 30 L*
points, document controlled frontal capture and neutral background conditions,
include calibration records, consent/license, limitations and error profile,
attach checksums for the observation data and analysis report, and receive a
real independent `approved` review. Populate
`templates/external_validation_package.yaml` only from that evidence. The
technical analysis deliberately emits `external_validity_approved: false`.

## Immutable SDXL and direction provenance

Do not download SDXL, accept its license, regenerate a direction, or create
portraits merely to complete a template. Reuse an existing direction only if
all of the following are available and checksum-verifiable:

- exact model ID `stabilityai/stable-diffusion-xl-base-1.0`;
- identical requested/resolved lowercase 40-hex model revision;
- a real acceptance record for the exact model license and revision;
- direction tensor and source-manifest paths plus full SHA-256 values;
- immutable direction code commit;
- the unchanged paired-mean estimator, 64 training pairs, 32 held-out pairs,
  deterministic VAE mode, and `optimization: false`;
- exact study-config and evaluation-protocol hashes recorded in
  `templates/generation_provenance.yaml`.

If any item is absent, leave `immutable_generation_provenance` blocked.

## Scoped approval

After approved external validation and valid provenance exist, an operator may
prepare a pending, checksum-bound record:

```bash
python3 -m scripts.prepare_collection_approval \
  --external-validation /path/to/external_validation.yaml \
  --generation-provenance /path/to/generation_provenance.yaml \
  --output /path/to/pending_collection_approval.yaml
```

The tool binds the current study, evaluation, and readiness hashes plus the
validation ID, exact model revision, and verified direction checksum. It
always writes `decision: pending_authorized_approver` and null approver fields.
It cannot approve collection. A real authorized approver must review the exact
bound inputs and supply a stable approval ID, identity, UTC timestamp, and
`decision: approved`. Repository maintainers and automated reviewers must not
infer approval from a structurally valid template.

Only a newly rebuilt 7/7 report on the supported Linux runtime can unlock the
confirmatory runner. Stop before collection while any criterion is blocked.
