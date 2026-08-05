"""Stage 6 — hiring recommendation, plus the human approval gates.

Two things live here, and they belong together:

1. `recommend()` produces an evidence-backed *recommendation*. It is advisory.
2. `shortlist_is_approved()` / `approve_shortlist()` / `record_decision()` are
   the gates. There is no function in this codebase that hires or rejects
   anyone — the pipeline stops at `awaiting_decision` and a person acts.

If you are adding a feature that would let the agent close the loop by itself,
that is the feature to not add.
"""

from __future__ import annotations

from .llm import LLMClient, LLMUnavailable
from .models import (
    Candidate,
    CandidateScore,
    HumanAction,
    HumanDecision,
    InterviewSummary,
    JobRequisition,
    Recommendation,
    Verdict,
)
from .store import ATSStore

SYSTEM = """You write a hiring recommendation for a hiring manager to act on.

You are advisory. The manager decides. Your job is to make their decision
faster and better informed, not to make it for them.

Rules:
1. Base the recommendation on the screening score and the interview evidence.
   Cite specifics in `supporting_evidence` — quotes and concrete facts, not
   impressions.
2. `dissenting_signals` is mandatory. List the evidence that argues against your
   verdict. If you genuinely found none, say "no contrary evidence found" and
   explain why the case is one-sided — do not leave it empty.
3. `confidence` reflects the strength of the *evidence*, not the strength of
   your opinion. Thin evidence means low confidence even for a clear-looking
   candidate. If the interview did not happen or covered little, the verdict is
   `insufficient_evidence`.
4. Never reference or infer age, gender, nationality, race, religion, family
   status, health, or disability. Never use school or employer prestige as a
   reason.
5. `suggested_next_step` should be actionable: a specific second interview
   focus, a take-home, a reference check, or a clear close-out."""

USER_TEMPLATE = """ROLE: {title}
{summary}

MUST HAVE: {must_haves}

--- SCREENING ---
Total score: {score}/100 (deterministic {gate_score}, model-judged fit {fit_score})
Hard requirements met: {gates_met}
Hard requirements NOT met: {gates_failed}
Screening strengths: {strengths}
Screening concerns: {concerns}
Screening flags: {flags}

--- INTERVIEW EVIDENCE ---
{interviews}

Write the recommendation."""


def _format_summaries(summaries: list[InterviewSummary]) -> str:
    if not summaries:
        return "No interview has been completed yet."

    blocks: list[str] = []
    for i, s in enumerate(summaries, start=1):
        signals = "\n".join(
            f"  - {sig.dimension}: {sig.rating.value} — \"{sig.evidence}\"" for sig in s.signals
        )
        blocks.append(
            f"Interview {i}:\n"
            f"  Overall: {s.overall_impression}\n"
            f"  Signals:\n{signals or '  (none recorded)'}\n"
            f"  Strengths: {'; '.join(s.strengths) or 'none recorded'}\n"
            f"  Concerns: {'; '.join(s.concerns) or 'none recorded'}\n"
            f"  Unanswered: {'; '.join(s.unanswered_questions) or 'none'}"
        )
    return "\n\n".join(blocks)


def _fallback_recommendation(
    candidate: Candidate,
    job: JobRequisition,
    score: CandidateScore,
    summaries: list[InterviewSummary],
) -> Recommendation:
    """No-model path. Reports the evidence and hands over, rather than guessing
    a verdict from a number."""
    evidence = [
        f"deterministic screening score {score.gate_score:.1f}/100",
        f"hard requirements met: {sum(1 for g in score.hard_gates if g.met)}"
        f"/{len(score.hard_gates)}",
    ]
    dissent = [g.detail for g in score.hard_gates if not g.met] or [
        "no contrary evidence found in the deterministic screen, which does not "
        "assess judgement, collaboration, or depth"
    ]

    return Recommendation(
        candidate_id=candidate.id,
        job_id=job.id,
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        rationale=(
            "No model was available, so this is a screening readout, not an assessment. "
            f"The candidate met {sum(1 for g in score.hard_gates if g.met)} of "
            f"{len(score.hard_gates)} hard requirements"
            + (
                f" and completed {len(summaries)} interview(s) that have not been assessed."
                if summaries
                else " and has not been interviewed."
            )
            + " A human must review the resume and any transcripts directly."
        ),
        supporting_evidence=evidence,
        dissenting_signals=dissent,
        suggested_next_step="Human review of the full file — no automated assessment was possible.",
        llm_available=False,
    )


def recommend(
    candidate: Candidate,
    job: JobRequisition,
    score: CandidateScore,
    summaries: list[InterviewSummary] | None = None,
    llm: LLMClient | None = None,
) -> Recommendation:
    """Produce an advisory recommendation. Always returns one — never raises."""
    summaries = summaries or []

    if llm and llm.available:
        user = USER_TEMPLATE.format(
            title=job.title,
            summary=job.summary,
            must_haves=", ".join(r.skill for r in job.must_haves) or "none",
            score=f"{score.total_score:.1f}",
            gate_score=f"{score.gate_score:.1f}",
            fit_score=f"{score.fit_score:.1f}",
            gates_met=", ".join(g.requirement for g in score.hard_gates if g.met) or "none",
            gates_failed=", ".join(g.requirement for g in score.hard_gates if not g.met) or "none",
            strengths="; ".join(score.strengths) or "none recorded",
            concerns="; ".join(score.concerns) or "none recorded",
            flags="; ".join(score.flags) or "none",
            interviews=_format_summaries(summaries),
        )

        try:
            rec = llm.structured(Recommendation, SYSTEM, user)
            rec.candidate_id = candidate.id
            rec.job_id = job.id
            rec.llm_available = True
            rec.requires_human_decision = True

            # Guardrails on the model's output.
            if not summaries and rec.verdict not in {
                Verdict.INSUFFICIENT_EVIDENCE,
                Verdict.NO_HIRE,
            }:
                rec.dissenting_signals.append(
                    "no interview has taken place — this verdict rests on the resume alone"
                )
                rec.confidence = min(rec.confidence, 0.4)
            if not rec.dissenting_signals:
                rec.dissenting_signals = [
                    "the model returned no contrary evidence; treat the case as unexamined "
                    "rather than one-sided"
                ]
                rec.confidence = min(rec.confidence, 0.5)
            if not score.all_gates_met and rec.verdict in {Verdict.STRONG_HIRE, Verdict.HIRE}:
                rec.dissenting_signals.append(
                    "candidate does not meet every declared hard requirement: "
                    + ", ".join(g.requirement for g in score.hard_gates if not g.met)
                )
            return rec
        except LLMUnavailable:
            pass

    return _fallback_recommendation(candidate, job, score, summaries)


# --------------------------------------------------------------------------
# Human gates — the part of this file that must never be bypassed
# --------------------------------------------------------------------------


class ApprovalRequired(PermissionError):
    """Raised when an action needs a human sign-off that has not happened."""


def shortlist_is_approved(store: ATSStore, job_id: str) -> bool:
    shortlist = store.get_shortlist(job_id)
    return bool(shortlist and shortlist.is_approved)


def approve_shortlist(store: ATSStore, job_id: str, approved_by: str, notes: str = "") -> int:
    """Record a human's approval of the shortlist. Returns how many were approved.

    Nothing downstream — scheduling, questions, interviews — runs until this has
    been called by a named person.
    """
    if not approved_by.strip():
        raise ValueError("approved_by is required: approvals must name a person")

    shortlist = store.get_shortlist(job_id)
    if shortlist is None:
        raise ApprovalRequired(f"no shortlist exists for {job_id} — run ranking first")

    from .models import utcnow

    shortlist.approved_by = approved_by.strip()
    shortlist.approved_at = utcnow()
    store.save_shortlist(shortlist)

    for score in shortlist.ranked:
        store.save_decision(
            HumanDecision(
                candidate_id=score.candidate_id,
                job_id=job_id,
                action=HumanAction.APPROVE_SHORTLIST,
                decided_by=approved_by.strip(),
                notes=notes,
            )
        )

    store.audit(
        "recommend",
        "shortlist_approved",
        job_id=job_id,
        approved_by=approved_by,
        count=len(shortlist.ranked),
        notes=notes,
    )
    return len(shortlist.ranked)


def require_shortlist_approval(store: ATSStore, job_id: str) -> None:
    """Guard called by scheduling. Raises rather than proceeding."""
    if not shortlist_is_approved(store, job_id):
        raise ApprovalRequired(
            f"shortlist for {job_id} has not been approved by a human. "
            f"Run: python -m recruiter approve --job {job_id} --by \"Your Name\""
        )


def record_decision(
    store: ATSStore,
    candidate_id: str,
    job_id: str,
    action: HumanAction,
    decided_by: str,
    notes: str = "",
) -> HumanDecision:
    """Persist a human's final call. The agent never calls this by itself."""
    if not decided_by.strip():
        raise ValueError("decided_by is required: decisions must name a person")

    decision = HumanDecision(
        candidate_id=candidate_id,
        job_id=job_id,
        action=action,
        decided_by=decided_by.strip(),
        notes=notes,
    )
    store.save_decision(decision)
    store.audit(
        "recommend",
        "human_decision",
        entity_id=candidate_id,
        job_id=job_id,
        action=action.value,
        decided_by=decided_by,
        notes=notes,
    )
    return decision
