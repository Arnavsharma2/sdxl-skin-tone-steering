from scripts.verify_manuscript_claims import ROOT, verify


def test_retained_evidence_matches_generic_and_preprint_manuscripts():
    result = verify(ROOT)
    assert result["passed"], result["failures"]
    assert result["claim_count"] >= 15
