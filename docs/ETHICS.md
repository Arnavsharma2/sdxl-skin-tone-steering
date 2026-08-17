# Ethics and intended use

## Construct boundary

This project manipulates a visible image attribute described as skin tone. It
does not determine or alter race, ethnicity, nationality, culture, ancestry, or
identity. Treating skin color as a race label is scientifically invalid and can
reinforce racial essentialism.

## Intended use

- controlled research on latent steering and generative-model auditing;
- creation of synthetic counterfactual sets for testing, with clear provenance;
- study of preservation–control tradeoffs in image generation.

## Out-of-scope and harmful uses

- inferring protected traits or identity from a face;
- ranking, classifying, authenticating, or making decisions about people;
- deceptive impersonation or editing a real person without consent;
- “race transformation,” demographic targeting, surveillance, or profiling;
- claiming fairness improvement without a downstream disparity evaluation.

## Data governance

Prefer synthetic portraits. For any real images, document consent, provenance,
license, retention, and deletion. Do not commit face images or embeddings that
could identify people. Treat face embeddings as sensitive biometric data.

## Measurement risks

Skin-tone measurements are sensitive to illumination, white balance, makeup,
camera processing, segmentation, and display conditions. Face-recognition
metrics can have demographic performance differences. Report failures and
stratified error rather than assuming a metric is neutral.

## Release checklist

- complete a misuse review and document model/data licenses;
- verify that all figures are synthetic or properly consented and licensed;
- publish detector failure rates and limitations beside performance numbers;
- include a contact and takedown process before releasing a dataset;
- do not release a model artifact until its behavior outside the study domain
  has been characterized.
