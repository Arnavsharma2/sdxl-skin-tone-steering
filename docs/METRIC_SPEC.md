# Frozen metric specification

The machine-readable authority is
[`configs/evaluation_protocol.yaml`](../configs/evaluation_protocol.yaml), protocol
`tmlr_evaluation_protocol_v1`. Confirmatory manifests embed that complete
document plus the actual SHA-256 of the protocol file. Code rejects artifact
or runtime pins that drift from the implementation.

## Validity contract

An evaluation row is complete only when all of the following are present:

1. FaceNet identity cosine similarity from a standardized MTCNN crop;
2. LPIPS-Alex perceptual distance;
3. SSIM averaged only over a valid eroded background mask;
4. pose drift from the MediaPipe facial transformation matrix; and
5. a directional rendered-skin-tone response from geometric cheek regions.

There is no metric fallback. A missing model, failed face detection, invalid
mask, unstable illumination reference, or omitted alpha makes the composite
quality rubric `null` and the row invalid. Face-detection failures remain in
the denominator and must be included in the failure rate; they are never
silently excluded.

## Target-attribute metric

`relative_cheek_CIELAB_Lstar_v1` uses this deterministic procedure:

1. detect 478 face landmarks with the checksum-verified MediaPipe Tasks Face
   Landmarker model;
2. select bilateral cheek regions inside the face oval and exclude eyes,
   eyebrows, and lips geometrically;
3. reject the lowest and highest 10% of cheek `L*` values;
4. take median CIELAB `L*`, `a*`, and `b*` values;
5. take median CIELAB values from neutral upper-border reference regions; and
6. report `skin L* - reference L*`.

The minimum valid skin and reference areas are 256 and 512 pixels. Masks must
have the image shape, contain only binary values, and not overlap. The target
change is invalid when reference `L*` changes by more than 8 points or reference
chroma exceeds 14. Positive change means lighter rendered skin; negative means
darker. The steering convention expects `alpha * skin_tone_change < 0` and at
least 2 `L*` points of magnitude.

The reference corrects global exposure only to first order. Local lighting on
the face is indistinguishable from a rendered-colour response and remains a
documented sensitivity rather than being corrected by an unvalidated
heuristic. ITA and CIE76 ΔE are secondary diagnostics and are not race or
ethnicity measures.

## Background SSIM

The implementation unions 1.5× expanded face boxes from both images, computes
a full-image grayscale SSIM map on untouched pixels, erodes the inverse union
with a 7×7 kernel, and averages only valid background locations. Both inputs
must be same-size uint8 images. Both detections must succeed when masks are
generated automatically. Explicit masks must be binary and image-shaped. An
empty face mask, all-face mask, or fewer than 49 eroded background pixels
invalidates the metric.

This replaces the invalid legacy method that zeroed the face and computed
global SSIM, thereby rewarding identical artificial pixels and silently
becoming whole-image SSIM when detection failed.

## Identity, perceptual, and pose metrics

Identity similarity is cosine similarity in `[-1, 1]` between FaceNet
InceptionResnetV1 VGGFace2 embeddings. MTCNN produces a 160×160 crop with zero
margin and fixed-image standardization (`post_process=true`). Failed detection,
invalid uint8 RGB input, a non-finite embedding, or a zero-norm embedding makes
the metric missing. This is an engineering preservation signal, not an
identity guarantee or identity-decision system.

LPIPS uses AlexNet v0.1. uint8 RGB pixels are mapped to one NCHW batch in
`[-1, 1]`; tensor inputs must already be finite, contain one RGB batch, and be
in that range. Pair dimensions must match.

Pose uses the finite 4×4 MediaPipe facial transformation matrix only when its
3×3 rotation block is a proper orthonormal rotation. Yaw, pitch, and roll are
reported in degrees; pairwise changes use circular absolute differences and
the required outcome is their Euclidean norm. Detection or matrix-validation
failure makes every pose outcome missing.

## Sweep monotonicity

The frozen alpha order is `[-1.5, -0.75, 0, 0.75, 1.5]`. Spearman rho and the
fraction of strictly decreasing adjacent `relative-L*` responses are calculated
only when every unique expected alpha has a finite measurement. A duplicate,
missing, unexpected, non-finite, or constant response invalidates the complete
sweep. Monotonicity is never calculated from a complete-case subset.

## Quality rubric and thresholds

The five gates are FaceNet cosine similarity ≥0.85, LPIPS ≤0.3, background
SSIM ≥0.75, total pose drift ≤5°, and a correctly directed target change of at
least 2 `relative-L*` points. All five must pass.

The fixed-weight 0–1 engineering rubric uses identity (30%), LPIPS similarity
(20%), background SSIM (20%), pose (10%), and directional target response
(20%). It is a dashboard convenience, not an inferential statistic, fairness
score, or disentanglement measure. Raw outcomes, completion, failures, and
uncertainty remain primary.

## Required artifacts and checksum verification

Run `make metric-models` to download the official MediaPipe model. Before use,
each required file is read and its actual SHA-256 is compared with the frozen
checksum. Provenance records path, expected checksum, actual checksum, byte
size, status, and verification result for:

- MediaPipe Face Landmarker;
- FaceNet VGGFace2;
- MTCNN P-Net, R-Net, and O-Net;
- the AlexNet backbone; and
- LPIPS AlexNet v0.1 linear weights.

Missing, non-file, unreadable, and mismatched inputs fail closed. Exact paths
and checksums live in the machine-readable protocol rather than being repeated
as unevaluated provenance strings.

## Supported evaluation runtime

The frozen metric runtime is Python `>=3.10,<3.13` on Linux with MediaPipe
`0.10.21` and its CPU delegate. This is enforced before MediaPipe graph
construction. On a headless Apple Silicon macOS host, the pinned wheel still
requested `kGpuService` and failed to create an `NSOpenGLPixelFormat`, even
with `BaseOptions.Delegate.CPU`. Headless macOS is therefore unsupported for
audited evaluation. It receives a direct diagnostic, with no detector,
skin-mask, pose, or whole-image fallback.

## Limitations

- The colour metric remains sensitive to rendering, camera response, local
  lighting, makeup, and occlusion.
- The neutral-border reference assumes the controlled studio-background prompt.
- The operational background can contain hair, shoulders, or clothing.
- Face embeddings can perform differently across demographic groups and must
  not be used for identity decisions.
- Engineering thresholds require external validation before confirmatory use.
- None of these metrics establishes bias mitigation, representation
  disentanglement, identity preservation, or state-of-the-art performance.
