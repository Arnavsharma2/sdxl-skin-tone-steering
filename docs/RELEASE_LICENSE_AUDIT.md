# Release and third-party license audit

Status: reviewed and author-approved 2026-08-17. This is an engineering release
audit, not legal advice. The project copyright holder approved Apache-2.0 for
the project-owned source code.

## Release decision

The source repository is released under Apache-2.0 for project-owned source
code. Apache-2.0 supplies explicit patent terms and is compatible with the
permissive software dependencies used here. Third-party dependencies, weights,
and generated artifacts remain governed by their own terms and the boundaries
below.

The release must not bundle SDXL, FaceNet/MTCNN, LPIPS, AlexNet, or other
third-party weights. Installation and execution should fetch those components
from their original distributors under their own terms. The 677 MiB
confirmatory image archive and other generated-image archives should remain DOI
or release assets rather than Git objects.

## Governing components

| Component | Upstream terms/provenance | Project treatment |
|---|---|---|
| SDXL 1.0 base | [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/e4e60c65aa20ee60092c60ba197f541872cf9373/LICENSE.md), immutable revision `462165984030d82259a11f4367a4eed129e94a7b` | Never redistribute the checkpoint. Preserve the license link and prohibited-use notice. The license states that Stability AI claims no rights in generated output, while making the operator responsible for output and downstream uses. |
| Diffusers | [Apache-2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE) | Dependency only; not vendored. |
| facenet-pytorch code | [MIT](https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md) | Dependency only; retain citation and version. |
| FaceNet/MTCNN weights | facenet-pytorch documents automatic pretrained weights, including an InceptionResnetV1 model trained on VGGFace2 | Do not redistribute weights. Report the exact `vggface2` model identifier and treat its real-face training provenance as an evaluation limitation. No source VGGFace2 images are included. |
| LPIPS | [BSD-2-Clause](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE) | Dependency only; cite Zhang et al. and do not vendor weights. |
| PyTorch/torchvision and scientific Python dependencies | Their upstream package licenses | Installed dependencies only; record versions in the lockfile and run metadata. |

## Project-owned and generated materials

- Project Python, test, analysis, and LaTeX source: release under the selected
  repository license after author approval.
- Tables, aggregate statistics, manifests, hashes, and audit JSON: release with
  the paper and code.
- Synthetic SDXL images: no real-person source dataset was collected, but the
  images remain model outputs governed by the SDXL use restrictions and can
  resemble real people. Release only the minimum qualitative examples in the
  paper by default. Place a separate, clearly labeled output-use notice on any
  complete image archive.
- Direction tensors: derived from SDXL VAE activations. Treat them
  conservatively as model-related research artifacts; do not bundle them in
  the source package. If released, provide the SDXL license alongside the
  artifact and its training-manifest hash.
- Face embeddings: never release. Only scalar cosine similarities are retained.

## Release-file status

1. The copyright-holder-approved root `LICENSE` is present.
2. The confirmed author and affiliation are recorded in `CITATION.cff`; no
   release date is declared before an actual release.
3. `MODEL_CARD.md`, `DATA_CARD.md`, the ethics statement, and claims/evidence
   register are present.
4. A public artifact release still needs an archive manifest containing filenames, byte sizes, SHA-256 hashes,
   upstream model revision, config fingerprints, and whether each artifact is
   public or restricted.
5. A public artifact release still needs a release note stating that third-party weights are not distributed and
   that generated outputs cannot be used for profiling, identity decisions,
   discrimination, deceptive editing, or editing real people without consent.

## Current disposition

| Material | Git repository | External archive | Public by default |
|---|---:|---:|---:|
| Source, tests, configs, paper source | Yes | Optional | Yes |
| Aggregate CSV/JSON analyses | Yes or DOI | Yes | Yes |
| Selected paper figures | Yes | Yes | Yes with output notice |
| Full generated image runs | No | Yes | No; explicit release decision required |
| SDXL/evaluation weights | No | No | No; fetch upstream |
| Face embeddings | No | No | Never |
