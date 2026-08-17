# WACV 2027 Round 2 submission checklist

Official-source check: 2026-08-17. All deadlines are Anywhere on Earth.

## Immediate user actions — due 2026-08-21

- [x] Confirm the complete author list and order.
- [x] Ensure every author has a valid OpenReview profile with institutional and
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
  construct language, figure legibility, and anonymity using
  `docs/INDEPENDENT_REVIEW_CHECKLIST.md`.
- [x] Choose and document the large-artifact review option in
  `docs/ARTIFACT_AVAILABILITY.md`: anonymous reviewer link, or explicit
  unavailability during review.
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
- [x] Confirm that no substantially similar work is simultaneously under
  review elsewhere.
- [x] Every author must accept the WACV reviewer obligation.

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
- Draft status: official review format, Datasets track, replication integrated;
  paper ID, independent human review, and large-artifact availability decision
  pending.
- Draft PDF SHA-256:
  `d81b4da0e5b7702bf5cd1e331bacd59f96d974ac63fe2d37fc464057b54d0785`.
- Supplement: `output/supplement/wacv2027_anonymous_supplement.zip`, 1.9 MB,
  SHA-256 `ca51adca21bd74f228b3f371ddaec51f36a2f0a48a45956d0eb63d556dbfe0f2`.
