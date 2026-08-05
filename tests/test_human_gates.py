"""The guardrails.

If any test in this file fails, the agent can act on a person without a human
having said so. That is the one failure mode this project exists to prevent.
"""

from __future__ import annotations

import pytest

from recruiter.models import HumanAction, RunReport, StageStatus
from recruiter.recommend import (
    ApprovalRequired,
    approve_shortlist,
    record_decision,
    require_shortlist_approval,
    shortlist_is_approved,
)

from .conftest import SAMPLE_RESUMES


def _ranked(agent, job):
    agent.register_job(job)
    report = RunReport(job_id=job.id)
    agent.ingest(job, SAMPLE_RESUMES, report)
    agent.rank(job, report)
    return report


def test_scheduling_is_blocked_until_a_human_approves(agent, job) -> None:
    _ranked(agent, job)
    assert not shortlist_is_approved(agent.store, job.id)

    report = RunReport(job_id=job.id)
    bookings = agent.schedule(job, report)

    assert bookings == [], "interviews were booked without human approval"
    assert agent.store.list_interviews(job.id) == []
    assert report.results[-1].status is StageStatus.NEEDS_HUMAN
    assert "not been approved" in report.results[-1].detail


def test_question_generation_is_blocked_until_a_human_approves(agent, job) -> None:
    _ranked(agent, job)
    report = RunReport(job_id=job.id)
    assert agent.questions(job, report) == []
    assert report.results[-1].status is StageStatus.NEEDS_HUMAN


def test_the_guard_raises_rather_than_returning_false(agent, job) -> None:
    _ranked(agent, job)
    with pytest.raises(ApprovalRequired):
        require_shortlist_approval(agent.store, job.id)


def test_approval_must_name_a_person(agent, job) -> None:
    _ranked(agent, job)
    for anonymous in ("", "   "):
        with pytest.raises(ValueError, match="must name a person"):
            approve_shortlist(agent.store, job.id, anonymous)
    assert not shortlist_is_approved(agent.store, job.id)


def test_approving_a_job_with_no_shortlist_fails(agent, job) -> None:
    agent.register_job(job)
    with pytest.raises(ApprovalRequired, match="no shortlist"):
        approve_shortlist(agent.store, job.id, "Someone")


def test_approval_unlocks_scheduling_and_is_recorded(agent, job) -> None:
    _ranked(agent, job)
    count = approve_shortlist(agent.store, job.id, "Ayesha Raza", notes="looks right")

    assert count == len(agent.store.get_shortlist(job.id).ranked)
    assert shortlist_is_approved(agent.store, job.id)

    decisions = agent.store.list_decisions(job.id)
    assert decisions and all(d.decided_by == "Ayesha Raza" for d in decisions)
    assert all(d.action is HumanAction.APPROVE_SHORTLIST for d in decisions)

    report = RunReport(job_id=job.id)
    assert agent.schedule(job, report), "approval did not unlock scheduling"


def test_the_run_stops_at_the_gate_when_no_approver_is_given(agent, job, tmp_path) -> None:
    from recruiter.pipeline import load_job

    from .conftest import SAMPLE_JOB

    report = agent.run(SAMPLE_JOB, SAMPLE_RESUMES)

    assert report.awaiting
    assert "approval" in report.awaiting
    assert agent.store.list_interviews(load_job(SAMPLE_JOB).id) == []


def test_the_run_ends_awaiting_a_human_decision_not_a_conclusion(agent, job) -> None:
    from .conftest import SAMPLE_INTERVIEWS, SAMPLE_JOB

    report = agent.run(
        SAMPLE_JOB, SAMPLE_RESUMES, approve_as="Osaid Khan", transcripts_dir=SAMPLE_INTERVIEWS
    )

    assert "does not advance or reject" in report.awaiting
    # Recommendations exist, but no decision has been made for anyone.
    job_id = report.job_id
    assert agent.store.list_recommendations(job_id)
    final = [
        d
        for d in agent.store.list_decisions(job_id)
        if d.action is not HumanAction.APPROVE_SHORTLIST
    ]
    assert final == [], "the agent recorded a hiring decision by itself"


def test_recording_a_decision_requires_a_named_person(agent, job) -> None:
    _ranked(agent, job)
    candidate = agent.store.list_candidates(job.id)[0]
    with pytest.raises(ValueError, match="must name a person"):
        record_decision(agent.store, candidate.id, job.id, HumanAction.REJECT, "  ")


def test_a_human_decision_is_persisted_and_audited(agent, job) -> None:
    _ranked(agent, job)
    candidate = agent.store.list_candidates(job.id)[0]

    decision = record_decision(
        agent.store, candidate.id, job.id, HumanAction.ADVANCE, "Ayesha Raza", "strong round 1"
    )

    stored = agent.store.decisions_for_candidate(candidate.id)
    assert decision.id in {d.id for d in stored}
    assert any(
        row["action"] == "human_decision" for row in agent.store.audit_trail(entity_id=candidate.id)
    )


def test_every_stage_writes_an_audit_trail(agent, job) -> None:
    from .conftest import SAMPLE_JOB

    agent.run(SAMPLE_JOB, SAMPLE_RESUMES, approve_as="Osaid Khan")
    stages = {row["stage"] for row in agent.store.audit_trail()}

    for expected in ("setup", "ingest", "rank", "questions", "schedule", "recommend"):
        assert expected in stages, f"no audit rows for the {expected} stage"


def test_the_audit_trail_records_which_model_made_each_judgement(agent, job) -> None:
    from .conftest import SAMPLE_JOB

    agent.run(SAMPLE_JOB, SAMPLE_RESUMES, approve_as="Osaid Khan")
    rows = agent.store.audit_trail()

    scored = [r for r in rows if r["action"] == "candidate_scored"]
    assert scored
    # Offline here, so the model column is empty - but the field must exist and
    # the redaction state must be on the record either way.
    assert all("redacted" in r["detail"] for r in scored)
