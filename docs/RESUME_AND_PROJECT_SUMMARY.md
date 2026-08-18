# Resume and project summary

Use these statements only after replacing placeholders with the actual venue
status. Do not describe a manuscript as published or accepted before it is.

## Resume bullet before submission

- Built a prospectively frozen evaluation pipeline for controllable SDXL generation,
  executing two independently seeded 570-condition studies with immutable
  configurations, failure-retaining analysis, bootstrap uncertainty, Holm
  correction, content-attested artifacts, and a reproducible anonymous paper
  package.

## Resume bullet after submission

- Developed and submitted a prospectively frozen study of denoising-time latent
  control in SDXL ([venue], under review), comparing four interventions at
  matched measured image-colour change across two independent 30-seed
  campaigns; released reproducible analysis, provenance audits, and tests.

## Resume bullet after acceptance

- Published a prospectively frozen evaluation of denoising-time latent control in SDXL
  at [venue], showing a bounded perceptual-preservation benefit in the parent
  campaign and an interval-positive but coverage-inconclusive independent
  replication; built a 70-test reproducibility and artifact-integrity system.

## Short project description

This project asks a narrower and more defensible question than whether a model
contains a “race vector”: can a paired direction in SDXL's VAE latent space
change measured apparent skin colour while limiting unrelated visual change?
It introduces matched-change evaluation so preservation methods are compared
only at similar target response, retains detection and metric failures, and
separates calibration, parent confirmation, exploratory robustness, and a
prospectively frozen independent-seed replication.

The parent campaign found no face-embedding-similarity advantage, but masking
reduced LPIPS relative to unmasked steering in both directions. The replication
retained both signs and had bootstrap intervals above zero, yet its locked
decision was inconclusive because only 8 and 10 matched pairs survived versus
the prespecified minimum of 12. This distinction is a strength of the work:
the pipeline prevents a promising pattern from being overstated as successful
replication.

## Interview talking points

- Why matching matters: a method that changes the target less can look better
  on every preservation metric without actually preserving more at equal edit
  strength.
- Why prospective freezing matters: calibration decisions, grids, hypotheses,
  missingness handling, multiplicity correction, and the replication rule were
  frozen before confirmatory outcomes were inspected.
- What failed: the primary face-similarity hypothesis was unsupported, strict
  matching coverage was sparse, and the replication missed its coverage gate.
- What worked: both campaigns showed monotonic target response; the parent
  masked method improved matched LPIPS; direction agreement across campaigns
  was high after masking; every row and artifact is auditable.
- What comes next: denser prespecified grids or interpolation, semantic masks,
  cross-model/scheduler testing, calibrated physical colour references, and
  human perceptual and identity judgments.

## Claims to avoid

- Do not say the method isolates race, removes bias, preserves identity, proves
  disentanglement, or achieved strict replication.
- Do not describe synthetic face-embedding measurements as evidence about real
  demographic groups.
- Do not pool the two 30-seed campaigns into an unregistered 60-seed
  confirmatory claim.
