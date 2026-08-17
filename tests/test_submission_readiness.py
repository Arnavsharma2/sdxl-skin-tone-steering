from pathlib import Path

from scripts.check_submission_readiness import human_gate_failures


def complete_inputs() -> dict:
    return {
        "authors": [
            {
                "name": "Researcher One",
                "openreview_id": "~Researcher_One1",
                "reviewer_obligation_accepted": True,
            }
        ],
        "corresponding_author": "Researcher One",
        "paper_id": "1234",
        "enrollment_completed": True,
        "no_concurrent_review_confirmed": True,
        "repository_license": "Apache-2.0",
        "independent_review": {
            "completed": True,
            "reviewer": "Independent Reviewer",
            "date": "2026-08-18",
            "technical_claims_checked": True,
            "statistics_checked": True,
            "figure_legibility_checked": True,
            "anonymity_checked": True,
        },
        "large_artifact_review": {"choice": "unavailable_during_review", "url": None},
    }


def test_missing_author_inputs_report_one_actionable_gate(tmp_path: Path) -> None:
    assert human_gate_failures(None, tmp_path) == [
        "submission/author_inputs.yaml has not been populated"
    ]


def test_complete_inputs_pass_when_license_exists(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_text("Approved license\n", encoding="utf-8")
    assert human_gate_failures(complete_inputs(), tmp_path) == []


def test_anonymous_artifact_choice_requires_a_url(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_text("Approved license\n", encoding="utf-8")
    data = complete_inputs()
    data["large_artifact_review"] = {"choice": "anonymous_url", "url": None}
    assert "anonymous large-artifact URL is missing" in human_gate_failures(data, tmp_path)
