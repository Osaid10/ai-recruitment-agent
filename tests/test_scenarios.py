"""The ten scenarios from PLAN.md, including the edge cases.

Scenario 7 (name-swap) lives in test_fairness.py with the other bias tests.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from recruiter.ingest.parser import parse_resume
from recruiter.models import (
    Candidate,
    CandidateScore,
    HardGate,
    JobRequisition,
    RunReport,
    Stage,
    StageStatus,
    TimeSlot,
)
from recruiter.pipeline import JobSpecError, RecruitmentAgent, load_job
from recruiter.rank.ranker import build_shortlist
from recruiter.recommend import ApprovalRequired, require_shortlist_approval
from recruiter.schedule.calendar import NoSlotAvailable, find_slot

from .conftest import SAMPLE_JOB, SAMPLE_RESUMES


def _ingested(agent: RecruitmentAgent, job: JobRequisition) -> RunReport:
    agent.register_job(job)
    report = RunReport(job_id=job.id)
    agent.ingest(job, SAMPLE_RESUMES, report)
    agent.rank(job, report)
    return report


def _by_name(agent: RecruitmentAgent, job: JobRequisition, fragment: str):
    for candidate in agent.store.list_candidates(job.id):
        if fragment.lower() in candidate.full_name.lower():
            return candidate
    raise AssertionError(f"no candidate matching '{fragment}'")


# -- 1: strong candidate, clean PDF ---------------------------------------


def test_strong_candidate_ranks_top_and_is_shortlisted(agent, job) -> None:
    _ingested(agent, job)
    shortlist = agent.store.get_shortlist(job.id)

    assert shortlist.ranked, "nobody was shortlisted"
    top = agent.store.get_candidate(shortlist.ranked[0].candidate_id)
    assert top.full_name == "Ahmed Nadeem"
    assert shortlist.ranked[0].all_gates_met
    assert shortlist.ranked[0].total_score > 90


def test_pdf_with_a_text_layer_parses(agent, job) -> None:
    parsed = parse_resume(SAMPLE_RESUMES / "01_ahmed_nadeem.pdf")
    assert parsed.usable
    assert "FastAPI" in parsed.text
    assert not parsed.warnings


# -- 2: clearly unqualified ------------------------------------------------


def test_unqualified_candidate_is_not_shortlisted_but_keeps_a_reason(agent, job) -> None:
    _ingested(agent, job)
    designer = _by_name(agent, job, "Maryam")
    score = agent.store.get_score(designer.id, job.id)

    assert score.shortlisted is False
    assert score.reason, "a candidate was cut with no recorded reason"
    assert score.total_score < job.shortlist_threshold


def test_a_single_passing_skill_mention_is_flagged_as_weak_evidence(agent, job) -> None:
    """The designer's only Python is an evening course. The keyword gate cannot
    tell the difference, so it must say so rather than imply precision."""
    _ingested(agent, job)
    score = agent.store.get_score(_by_name(agent, job, "Maryam").id, job.id)
    assert any("weak evidence" in flag for flag in score.flags), score.flags


# -- 3: borderline ---------------------------------------------------------


def test_borderline_candidate_is_flagged_not_dropped(agent, job) -> None:
    """Meets every hard requirement, lands outside the shortlist size. Must be
    recorded as waitlisted rather than rejected."""
    _ingested(agent, job)
    shortlist = agent.store.get_shortlist(job.id)

    waitlisted = [s for s in shortlist.cut if s.all_gates_met]
    assert waitlisted, "expected at least one qualified-but-not-shortlisted candidate"
    for score in waitlisted:
        assert "not rejected" in score.reason or "human look" in score.reason
        assert any("waitlisted" in f or "borderline" in f for f in score.flags)


# -- 4: scanned PDF with no text layer ------------------------------------


def test_scanned_pdf_is_flagged_for_human_review_and_the_run_continues(agent, job) -> None:
    report = _ingested(agent, job)

    scanned = [
        r
        for r in report.results
        if r.stage is Stage.INGEST and "scanned_no_text_layer" in r.detail
    ]
    assert scanned, "the text-free PDF produced no ingest result at all"
    assert scanned[0].status is StageStatus.NEEDS_HUMAN
    assert "OCR" in scanned[0].detail or "manual review" in scanned[0].detail

    # The important part: it did not stop the batch.
    assert len(agent.store.list_candidates(job.id)) >= 6
    assert not report.failures()


# -- 5: DOCX --------------------------------------------------------------


def test_docx_parses_the_same_as_any_other_format(agent, job) -> None:
    parsed = parse_resume(SAMPLE_RESUMES / "02_fatima_malik.docx")
    assert parsed.usable
    assert "Fatima Malik" in parsed.text
    assert "PostgreSQL" in parsed.text

    _ingested(agent, job)
    score = agent.store.get_score(_by_name(agent, job, "Fatima").id, job.id)
    assert score.all_gates_met


# -- 6: career gap --------------------------------------------------------


def test_career_gap_candidate_is_shortlisted_on_merit(agent, job) -> None:
    _ingested(agent, job)
    zainab = _by_name(agent, job, "Zainab")
    score = agent.store.get_score(zainab.id, job.id)

    assert score.shortlisted, "the candidate with an employment gap was cut"
    assert score.all_gates_met
    # The gap must not have eaten her experience: the break is 18 months and she
    # has roughly six and a half years of actual work.
    assert zainab.total_years_experience >= 6


# -- 8: no calendar slot --------------------------------------------------


def test_no_available_slot_escalates_instead_of_dropping_the_candidate(
    agent, job, calendar
) -> None:
    _ingested(agent, job)
    agent.approve(job.id, "Test Approver")

    # Block the whole window for one interviewer.
    start = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    calendar.block_out(
        job.interview_panel[0].email,
        TimeSlot(start=start, end=start + timedelta(days=60)),
    )

    report = RunReport(job_id=job.id)
    bookings = agent.schedule(job, report)

    assert not bookings
    escalations = [r for r in report.results if r.status is StageStatus.NEEDS_HUMAN]
    assert escalations, "no slot was available and nothing was escalated"
    assert any("still active" in r.detail for r in escalations)
    assert not report.failures(), "a full calendar should not be a stage failure"


def test_find_slot_avoids_lunch_and_weekends(job) -> None:
    from recruiter.schedule.calendar import MockCalendar

    empty = MockCalendar()
    empty._synthetic_busy = lambda *a, **k: []  # type: ignore[assignment]

    monday = datetime(2026, 8, 10, 9, 0)  # a Monday
    slot = find_slot(empty, job.interview_panel, monday, monday + timedelta(days=5), 45)

    assert slot.start.weekday() < 5
    assert not (13 <= slot.start.hour < 14)
    assert 9 <= slot.start.hour and slot.end.hour <= 17


def test_scheduling_with_no_panel_asks_a_human(agent, job) -> None:
    _ingested(agent, job)
    agent.approve(job.id, "Test Approver")
    job.interview_panel = []

    report = RunReport(job_id=job.id)
    assert agent.schedule(job, report) == []
    assert any("interview panel" in r.detail for r in report.results)


# -- 9: no API key --------------------------------------------------------


def test_the_whole_pipeline_works_with_no_model_configured(agent, job) -> None:
    """Everything in this suite runs offline, but this asserts it explicitly."""
    report = _ingested(agent, job)
    shortlist = agent.store.get_shortlist(job.id)

    assert shortlist.ranked, "ranking produced nothing without an LLM"
    for score in shortlist.ranked:
        assert score.llm_available is False
        assert score.gate_score > 0
        # With no model, the deterministic score must be the whole score rather
        # than being blended against a zeroed fit score.
        assert score.total_score == score.gate_score
        assert any("deterministic score only" in f for f in score.flags)
    assert not report.failures()


def test_questions_fall_back_to_targeted_templates_without_a_model(agent, job) -> None:
    _ingested(agent, job)
    agent.approve(job.id, "Test Approver")

    report = RunReport(job_id=job.id)
    sets = agent.questions(job, report)

    assert sets
    for question_set in sets:
        assert question_set.llm_available is False
        assert question_set.questions
        # Still candidate-specific: every question cites what prompted it.
        assert all(q.linked_evidence for q in question_set.questions)


def test_recommendation_without_a_model_refuses_to_guess(agent, job) -> None:
    _ingested(agent, job)
    agent.approve(job.id, "Test Approver")

    report = RunReport(job_id=job.id)
    recommendations = agent.recommend(job, report)

    assert recommendations
    for rec in recommendations:
        assert rec.verdict.value == "insufficient_evidence"
        assert rec.confidence == 0.0
        assert rec.dissenting_signals, "a recommendation with no contrary evidence"
        assert rec.requires_human_decision is True


# -- 10: empty and near-empty resumes -------------------------------------


def test_near_empty_resume_is_flagged_not_scored_as_zero(agent, job) -> None:
    report = _ingested(agent, job)
    short = [r for r in report.results if "08_short_submission" in r.detail]

    assert short, "the one-line resume produced no result"
    assert short[0].status is StageStatus.NEEDS_HUMAN
    assert "too short" in short[0].detail


def test_empty_file_does_not_crash(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    parsed = parse_resume(empty)
    assert not parsed.usable
    assert parsed.warnings


def test_missing_file_reports_rather_than_raises(tmp_path: Path) -> None:
    parsed = parse_resume(tmp_path / "nope.pdf")
    assert not parsed.usable
    assert any("not found" in w for w in parsed.warnings)


def test_unsupported_format_is_reported(tmp_path: Path) -> None:
    weird = tmp_path / "resume.pages"
    weird.write_text("hello", encoding="utf-8")
    parsed = parse_resume(weird)
    assert not parsed.usable
    assert any("unsupported" in w for w in parsed.warnings)


# -- malformed inputs ------------------------------------------------------


def test_malformed_job_spec_gives_a_usable_error(tmp_path: Path) -> None:
    bad = tmp_path / "job.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(JobSpecError, match="not valid JSON"):
        load_job(bad)


def test_missing_job_spec_gives_a_usable_error(tmp_path: Path) -> None:
    with pytest.raises(JobSpecError, match="not found"):
        load_job(tmp_path / "absent.json")


def test_rubric_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        JobRequisition(title="X", rubric={"skills": 0.9, "experience": 0.9, "impact": 0.1, "communication": 0.1})


def test_empty_resume_folder_is_reported_not_crashed(agent, job, tmp_path: Path) -> None:
    agent.register_job(job)
    report = RunReport(job_id=job.id)
    agent.ingest(job, tmp_path, report)
    assert report.failures()
    assert "no supported resume files" in report.failures()[0].detail


def test_the_shipped_job_requisition_is_valid() -> None:
    job = load_job(SAMPLE_JOB)
    assert job.must_haves and job.interview_panel
    assert abs(sum(job.rubric.as_dict().values()) - 1.0) < 1e-9


# -- a resume the model could not assess ----------------------------------


def _score(cid: str, *, gate: float, fit: float, assessed: bool) -> CandidateScore:
    """A score as ranker.score_candidate would have produced it."""
    total = (0.55 * gate + 0.45 * fit) if assessed else gate
    return CandidateScore(
        candidate_id=cid,
        job_id="job_x",
        hard_gates=[HardGate(requirement="Python", met=True, detail="found")],
        gate_score=gate,
        fit_score=fit,
        total_score=round(total, 1),
        llm_available=assessed,
    )


def test_an_unassessed_candidate_never_outranks_an_assessed_one() -> None:
    """The bug a real CV hit through the dashboard.

    The model failed to return structured output for one resume, so that
    candidate kept their full gate score (100.0) while everyone the model did
    read was blended down by their fit score. The unassessed resume came out top
    of the shortlist at 100.0, above a genuine 84.2 — the least-analysed
    candidate ranked highest.
    """
    job = load_job(SAMPLE_JOB)
    good = _score("cand_assessed", gate=92.5, fit=74.0, assessed=True)
    broken = _score("cand_failed", gate=100.0, fit=0.0, assessed=False)

    assert broken.total_score > good.total_score, "precondition: the raw scores do invert"

    shortlist = build_shortlist([good, broken], job)

    ids = [s.candidate_id for s in shortlist.ranked]
    assert "cand_failed" not in ids, (
        "a candidate the model could not assess was ranked against candidates it "
        "did assess, on an incompatible scale"
    )
    assert "cand_assessed" in ids

    held = next(s for s in shortlist.cut if s.candidate_id == "cand_failed")
    assert "could not be assessed" in held.reason
    assert any("needs human" in f for f in held.flags)


def test_a_batch_with_no_model_at_all_still_ranks_normally() -> None:
    """The no-API-key path must not be caught by the mixed-batch rule.

    With nothing assessed, every candidate is on the same rules-only scale, so
    comparing them is meaningful and the whole batch should still rank.
    """
    job = load_job(SAMPLE_JOB)
    a = _score("cand_a", gate=90.0, fit=0.0, assessed=False)
    b = _score("cand_b", gate=70.0, fit=0.0, assessed=False)

    shortlist = build_shortlist([a, b], job)

    assert [s.candidate_id for s in shortlist.ranked] == ["cand_a", "cand_b"]
    assert not shortlist.cut


def test_an_unnamed_candidate_shows_its_filename_not_its_id() -> None:
    """`cand_9a64078a` tells a reviewer nothing they can act on."""
    unnamed = Candidate(job_id="job_x", full_name="", source_file="/tmp/up/resume.pdf")
    assert unnamed.display_name == "resume.pdf (name not extracted)"

    named = Candidate(job_id="job_x", full_name="Ada Lovelace", source_file="/tmp/a.pdf")
    assert named.display_name == "Ada Lovelace"
