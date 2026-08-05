"""Fairness regression tests.

These are the tests the Responsible Use note in the brief actually asks for.
If any of them fail, the ranking has started conditioning on something it must
not — do not skip them to get a build green.

They test the deterministic path, which is where a guarantee is possible. The
LLM half is constrained by redaction and by prompt rules; see the limitations
section of docs/BIAS_AND_FAIRNESS.md for what these tests cannot prove.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from recruiter.ingest.extract import build_candidate
from recruiter.ingest.parser import ParsedResume
from recruiter.models import JobRequisition
from recruiter.rank.ranker import score_candidate
from recruiter.redact import leaked_terms, redact

from .conftest import make_resume

# Pairs chosen to differ only in the demographic signal the name carries.
NAME_PAIRS = [
    ("Ahmed Khan", "Ayesha Khan"),  # gendered
    ("Muhammad Bilal", "Michael Bell"),  # ethnic / national origin
    ("Priya Sharma", "Peter Shaw"),
    ("Chen Wei", "Charlie West"),
]

INSTITUTION_PAIRS = [
    ("Lahore University of Management Sciences", "Harvard University"),
    ("University of Karachi", "Massachusetts Institute of Technology"),
]


def _score_for(text: str, job: JobRequisition, settings, name: str = ""):
    parsed = ParsedResume(path=Path(f"{name or 'candidate'}.txt"), text=text)
    candidate = build_candidate(parsed, job.id, llm=None, vocabulary=None)
    return score_candidate(candidate, job, llm=None, settings=settings)


@pytest.mark.parametrize("name_a,name_b", NAME_PAIRS)
def test_identical_resumes_score_identically_under_different_names(
    name_a: str, name_b: str, job: JobRequisition, settings
) -> None:
    """The core fairness guarantee: the name must not move the score."""
    score_a = _score_for(make_resume(name_a), job, settings, name_a)
    score_b = _score_for(make_resume(name_b), job, settings, name_b)

    assert score_a.total_score == score_b.total_score, (
        f"'{name_a}' scored {score_a.total_score} but '{name_b}' scored "
        f"{score_b.total_score} on an identical resume"
    )
    assert [g.met for g in score_a.hard_gates] == [g.met for g in score_b.hard_gates]


@pytest.mark.parametrize("institution_a,institution_b", INSTITUTION_PAIRS)
def test_school_prestige_does_not_change_the_score(
    institution_a: str, institution_b: str, job: JobRequisition, settings
) -> None:
    """Institution name is an excluded criterion. It must not be scored."""
    score_a = _score_for(make_resume("Sam Taylor", institution=institution_a), job, settings)
    score_b = _score_for(make_resume("Sam Taylor", institution=institution_b), job, settings)
    assert score_a.total_score == score_b.total_score


def test_employment_gap_is_not_penalised(job: JobRequisition, settings) -> None:
    """A gap may raise a question. It must not lower the score by itself."""
    continuous = make_resume("Alex Doe")
    with_gap = continuous.replace(
        "Backend Engineer - Another Company\nAug 2019 - Dec 2020",
        "Career break\nJan 2020 - Jun 2021\n- Family caregiving.\n\n"
        "Backend Engineer - Another Company\nAug 2019 - Dec 2020",
    )

    gap_score = _score_for(with_gap, job, settings)
    base_score = _score_for(continuous, job, settings)

    assert gap_score.total_score >= base_score.total_score, (
        "the employment gap lowered the deterministic score, which it must never do"
    )


def test_redaction_removes_identity_before_scoring() -> None:
    """Nothing the scorer sees may identify the candidate."""
    text = make_resume(
        "Zainab Qureshi",
        email="zainab.qureshi@example.com",
        institution="Lahore University of Management Sciences",
    )
    text += "\nGender: Female\nDate of Birth: 14 March 1997\nShe led the migration.\n"

    result = redact(text, known_name="Zainab Qureshi")

    leaked = leaked_terms(
        result.text,
        [
            "Zainab",
            "Qureshi",
            "zainab.qureshi@example.com",
            "Lahore University of Management Sciences",
            "14 March 1997",
        ],
    )
    assert not leaked, f"redaction leaked: {leaked}"
    assert "Gender:" not in result.text
    assert " she " not in f" {result.text.lower()} "


def test_redaction_keeps_the_job_relevant_content() -> None:
    """Redaction must not be so aggressive that it destroys what is being assessed."""
    result = redact(make_resume("Zainab Qureshi"), known_name="Zainab Qureshi")
    for skill in ("Python", "FastAPI", "PostgreSQL", "REST", "Docker"):
        assert skill in result.text, f"redaction removed a job-relevant term: {skill}"
    assert "p95 latency" in result.text
    assert "Mentor two engineers" in result.text


def test_scoring_records_that_it_was_redacted(job: JobRequisition, settings) -> None:
    """The audit trail has to show whether the score was blind."""
    score = _score_for(make_resume("Sam Taylor"), job, settings)
    assert score.redacted is True


def test_disabling_redaction_is_flagged_loudly(job: JobRequisition, settings) -> None:
    """Turning the control off must be visible on every affected candidate."""
    unblinded = dataclasses.replace(settings, redact_for_ranking=False)
    score = _score_for(make_resume("Sam Taylor"), job, unblinded)

    assert score.redacted is False
    assert any("REDACTION DISABLED" in flag for flag in score.flags), score.flags
