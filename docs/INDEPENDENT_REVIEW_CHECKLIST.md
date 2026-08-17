# Independent technical and anonymity review

Use this form for the required human review before submission. The reviewer
should not be an author and should inspect the rendered WACV PDF and anonymous
supplement, not only the source files.

## Materials

- `output/pdf/wacv2027_submission_draft.pdf`
- `output/supplement/wacv2027_anonymous_supplement.zip`
- `experiments/analysis/manuscript_claim_audit.json`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/ARTIFACT_AVAILABILITY.md`

Verify the artifact hashes against `docs/WACV2027_SUBMISSION_CHECKLIST.md`
before beginning.

## Required checks

- [ ] The title, abstract, contributions, results, limitations, and conclusion
  make the same bounded claim.
- [ ] The paper consistently treats skin tone as apparent image colour and
  never as race, ethnicity, identity, or a demographic label.
- [ ] Parent and replication campaigns are reported separately; the
  replication is called inconclusive because its shared matched-pair coverage
  missed the frozen threshold.
- [ ] Matched-change sample sizes, confidence intervals, multiplicity
  corrections, missing measurements, and exploratory analyses are described
  accurately and without causal overstatement.
- [ ] Each central number checked by the 16-group manuscript audit agrees with
  the rendered paper.
- [ ] All figures, legends, tables, equations, citations, and line numbers are
  legible at 100% zoom and no content is clipped or overlapping.
- [ ] The review PDF contains no author name, affiliation, acknowledgement,
  grant identifier, repository URL, email address, username, or local path.
- [ ] The extracted supplement contains no identity leak and its documented
  installation, `make check`, and manuscript-claim audit commands work.
- [ ] The artifact-availability statement accurately distinguishes tracked
  evidence from author-retained images and tensors.
- [ ] No result appears only in the supplement as a new or improved result.

## Disposition

Record every finding with a severity of `blocking`, `major`, or `minor`. The
submission may proceed only when no blocking or major finding remains.

- Reviewer:
- Review date:
- PDF SHA-256:
- Supplement SHA-256:
- Findings:
- Final disposition: `approved` / `changes required`

The corresponding author records the completed review in the Git-ignored
`submission/author_inputs.yaml`; do not add reviewer identity to the anonymous
supplement.
