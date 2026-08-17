# WACV 2027 Round 2 submission checklist

Official-source check: 2026-08-17. All deadlines are Anywhere on Earth.

## Immediate user actions — due 2026-08-21

- [ ] Confirm the complete author list and order.
- [ ] Ensure every author has a valid OpenReview profile with institutional and
  personal conflicts completed. Non-institutional-email profile approval can
  take up to two weeks.
- [ ] Create the new Round 2 paper enrollment in OpenReview before the hard
  enrollment deadline.
- [ ] Select the Evaluations & Dataset track and add every author before the
  enrollment deadline; authors cannot later be added or removed.
- [ ] Send the assigned paper ID for `\wacvPaperID` in
  `paper/wacv/main.tex`.

## Research gates before submission — due 2026-08-28

- [x] Complete and integrity-check the 30-seed parent confirmatory study.
- [x] Complete matching-tolerance and direction-stability robustness analyses.
- [x] Complete the prospective independent-seed replication and apply its
  locked decision rule.
- [x] Integrate replication results in the main paper, not only the supplement.
- [ ] Obtain at least one technically independent read for statistical claims,
  construct language, figure legibility, and anonymity.
- [x] Confirm all citations against primary sources.
- [x] Run the complete test/lint suite and rebuild both paper variants.
- [x] Reproduce the tests and frozen decision once from a fresh environment.
- [x] Verify central manuscript numbers against retained analysis artifacts.

## Formatting and policy gates

- [x] Use the official WACV 2027 author kit with `review,datasets`.
- [x] Keep the main paper at or below eight pages including figures and tables;
  references may follow.
- [ ] Replace the `*****` paper-ID placeholder.
- [x] Keep the review PDF under 50 MB.
- [x] Remove author names, affiliations, acknowledgements, grant IDs, repository
  URLs, PDF metadata, and other identifying content.
- [x] Do not cite the public repository; say code will be made available.
- [x] Verify that no submitted source or supplemental file leaks a username,
  email address, local path, Git remote, or Colab account.
- [x] Include the synthetic-data provenance, face-model provenance, potential
  misuse, and gated-artifact-release discussion.
- [ ] Confirm that no substantially similar work is simultaneously under
  review elsewhere.
- [ ] Every author must accept the WACV reviewer obligation.

## Supplementary material — due 2026-08-30

- [x] Create an anonymized ZIP/PDF under 200 MB.
- [x] Include setup and reproduction instructions, locked configs, manifests,
  aggregate tables, and enough code to reproduce the main claims.
- [x] Exclude SDXL, FaceNet/MTCNN, LPIPS, and other third-party weights.
- [x] Exclude face embeddings and full generated-image archives.
- [x] Include only results already present in the submitted main paper; WACV
  forbids using the supplement for new datasets or improved-method results.
- [x] Render and inspect the supplementary PDF or list every ZIP member and
  scan all text files for identity leaks.

## Current artifact

- Source: `paper/wacv/main.tex`
- Draft PDF: `output/pdf/wacv2027_submission_draft.pdf`
- Draft status: official review format, Datasets track, seven pages, 1.1 MB,
  replication integrated, visually audited; paper ID and human review pending.
- Draft PDF SHA-256:
  `1de3063b98c4278ad10f677786392ddfe67f09c09f156f2bee5ebaf712213640`.
- Supplement: `output/supplement/wacv2027_anonymous_supplement.zip`, 1.9 MB,
  SHA-256 `18f2e585843803bbd8a5e24a7e2e04f0cee6ecd2c85543e4c3514c13521ae38e`.
