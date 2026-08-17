# Venue plan

Checked 2026-08-17. Recheck every deadline and policy on the official venue
site before submission; dates and tracks can change.

## Current decision

Prepare a conditional WACV 2027 Round 2 submission to the new Evaluations &
Dataset track. The official author kit is now staged under `paper/wacv`, the
core confirmatory study and robustness analyses are complete, and a prospective
independent-seed replication is complete. The replication retained both parent
LPIPS effect signs and interval-positive estimates, but its locked decision is
inconclusive because matched-pair coverage fell below the prespecified minimum.
This route is viable only if the user
creates the OpenReview enrollment with the complete author list by 2026-08-21
and every author has a valid, conflict-complete profile. The paper deadline is
2026-08-28 and Round 2 has no rebuttal. Do not submit merely to meet the date if
the replication, independent review, or ethics/release checks expose a material
problem.

WACV's eight-page limit, double-blind rules, 50 MB PDF limit, and required
Datasets-track style are hard gates. The current anonymized WACV draft is seven
pages including references, uses the official 2027 author kit, has passed
visual and numerical-claim audits, and has a 1.9 MB identity-scanned supplement.
Its paper ID remains a placeholder until enrollment.

- [WACV 2027 call for papers](https://wacv.thecvf.com/Conferences/2027/CallForPapers)
- [WACV 2027 author guide](https://wacv.thecvf.com/Conferences/2027/AuthorGuides)

AAAI-27 is closed: abstracts were due 2026-07-21 and full papers 2026-07-28.

- [AAAI-27 official page](https://aaai.org/conference/aaai/aaai-27/)

## Alternative targets

TMLR is the strongest fallback if WACV enrollment or independent review cannot
be completed responsibly. It uses rolling, double-blind OpenReview submission,
explicitly welcomes experimental studies that expose strengths and weaknesses,
and emphasizes whether claims are supported by accurate evidence rather than
subjective significance. That maps well to this paper's preregistration,
failure retention, robustness analysis, and formally inconclusive replication.
It would require reformatting with the mandatory TMLR template and disclosure
of LLM assistance under the current author policy. Do not submit to TMLR while
the work is under review at WACV or another archival venue.

- [TMLR scope and editorial policies](https://jmlr.org/tmlr/editorial-policies.html)
- [TMLR acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html)
- [TMLR author guide](https://jmlr.org/tmlr/author-guide.html)

FAccT 2027 has an abstract deadline of 2026-10-27 and paper deadline of
2026-11-03 (Anywhere on Earth). It is appropriate only if the completed paper's
central contribution is a responsible-computing result—for example, a rigorous
construct-validity and claim-evidence audit of apparent skin-color control—not
merely a diffusion editing method. Do not claim fairness or bias mitigation
without a defined downstream harm and disparity evaluation.

FAccT also requires deep engagement with the social components of a computing
system. The present technical intervention study is not yet a natural FAccT
paper without a stronger sociotechnical analysis or human-centered component.

- [FAccT 2027 call for papers](https://facctconference.org/2027/cfp.html)

## Preferred technical positioning

For a computer-vision venue, position the work as a controlled comparison of a
paired VAE direction and denoising-time intervention against prompt-only,
post-hoc, and unmasked baselines. Novelty must be stated narrowly because
semantic directions and continuous controls in diffusion models already exist.
The strongest technical evidence would be:

1. improved target-response coverage beyond the current incomplete grids;
2. replication of the observed LPIPS and darker-background advantages;
3. a semantic-mask ablation against the current Gaussian mask;
4. sensitivity to direction-training pair count and seeds;
5. a fully reproducible failure/missingness and data-provenance audit.

The completed study supplies item 5 and a prespecified Gaussian-mask ablation.
It does not support a face-identity advantage, and target coverage ranges from
23% to 87% by method and direction. The technically defensible result is the
masked method's LPIPS advantage in both directions and its darker-direction
background advantage, not a general disentanglement claim.

CVPR 2027 dates were still not available on an official conference page at the
check date. Do not rely on an inferred deadline. The current one-model,
coverage-limited evidence is better aligned with WACV's evaluation track or
TMLR's evidence-centered criteria than with a novelty-driven CVPR submission
unless a future extension adds cross-model evaluation and substantially better
matched-target coverage.

## Release sequence

1. complete calibration, paired data, measurement validation, and the frozen
   preregistration commit;
2. run the confirmatory matrix without changing the config;
3. produce the full analysis, limitations, model/data cards, and paper;
4. create an archival release and preprint only after the result hashes and
   claim-evidence register agree;
5. submit to the venue whose scope matches the observed result.
