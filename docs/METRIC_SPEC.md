# Metric specification

## Validity contract

An evaluation row is complete only when all of the following are present:

1. FaceNet identity cosine similarity from a standardized MTCNN crop;
2. LPIPS-Alex perceptual distance;
3. SSIM averaged only over a valid eroded background mask;
4. pose drift from the MediaPipe facial transformation matrix; and
5. a directional rendered-skin-tone response from geometric cheek regions.

There is no metric fallback. A missing model, failed face detection, invalid
mask, unstable illumination reference, or omitted alpha makes the composite
quality rubric `null` and the row invalid.

## Target-attribute metric

`relative_cheek_CIELAB_Lstar_v1` uses the following deterministic procedure:

1. detect 478 face landmarks with the checksum-pinned MediaPipe Tasks Face
   Landmarker model;
2. select bilateral cheek regions inside the face oval and exclude eyes,
   eyebrows, and lips geometrically;
3. reject the lowest and highest 10% of cheek L* values to limit shadow and
   specular-highlight influence;
4. take median CIELAB L*, a*, and b* values;
5. take median CIELAB values from neutral upper-border reference regions; and
6. report `skin L* - reference L*` so first-order exposure changes are removed.

The target change is invalid when the reference L* changes by more than 8
points or its chroma exceeds 14. Positive target change means lighter rendered
skin; negative means darker. The current steering convention therefore expects
`alpha * skin_tone_change < 0` and at least 2 L* points of magnitude.

ITA and CIE76 ΔE are retained as secondary diagnostics. They are not treated as
race or ethnicity measures.

## Background SSIM correction

The old implementation zeroed the face region and then computed global SSIM.
That inflated the result because the artificial zeroed region matched exactly;
when face detection failed, the method became whole-image SSIM. The corrected
implementation unions face masks from both images, computes a full-image SSIM
map on untouched pixels, erodes the background to avoid window-edge leakage,
and averages only valid background locations.

## Quality rubric

The fixed-weight 0–1 engineering rubric uses identity (30%), LPIPS similarity
(20%), background SSIM (20%), pose (10%), and directional target response
(20%). It is a dashboard convenience, not an inferential statistic or a
fairness score. Raw outcomes, completion, failures, and uncertainty remain the
primary report.

## Reproducibility

Run `make metric-models` to download the official MediaPipe model and verify its
SHA-256 checksum. Every sweep report records package versions, model identity,
FaceNet and AlexNet weight checksums, metric definitions, and SHA-256 hashes
for evaluated images.

## Limitations

- The colour metric remains sensitive to rendering, camera response, local
  lighting, makeup, and occlusion.
- The neutral-border reference assumes the controlled studio-background prompt.
- The operational “background” is the region outside the expanded face box; it
  can still contain hair, shoulders, or clothing and is best read as non-face
  structural preservation.
- Face embeddings can perform differently across demographic groups and must
  not be used for identity decisions.
- Engineering thresholds require external validation before confirmatory use.
- None of these metrics establishes bias mitigation.
