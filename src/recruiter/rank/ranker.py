"""Stage 2b — LLM fit assessment and shortlist assembly.

The model sees redacted text and the declared rubric dimensions, and must quote
evidence for every score. Its judgement is blended with the deterministic score;
a failed hard gate can never be overridden by a good write-up.

Nobody is rejected here. Candidates who miss the shortlist land in `Shortlist.cut`
with a stated reason, and a human decides what happens to them.
"""

from __future__ import annotations

from ..config import Settings
from ..llm import LLMClient, LLMUnavailable, truncate
from ..models import (
    Candidate,
    CandidateScore,
    DimensionScore,
    FitAssessment,
    JobRequisition,
    Shortlist,
)
from ..redact import redact
from .rubric import deterministic_score

# How much of the final score is rules vs. model judgement.
GATE_WEIGHT = 0.55
FIT_WEIGHT = 0.45

SYSTEM = """You assess a candidate's fit for a role. You are one input into a human's
hiring decision, not the decision-maker.

The resume has been redacted: names, contact details, schools, ages, and
demographic fields are replaced with placeholders like [NAME] or [INSTITUTION].
This is deliberate.

Rules you must follow:
1. Score ONLY the dimensions listed in the rubric you are given. Nothing else.
2. Every dimension score needs an `evidence` field containing a short quote
   copied from the resume. If you cannot find supporting text, score the
   dimension 0 and write "insufficient evidence" as the evidence.
3. Never speculate about the candidate's identity, gender, age, nationality, or
   background, and never treat a redaction placeholder as informative. Employment
   gaps are not evidence of anything — do not penalise them.
4. Judge whether writing is clear, not whether it is idiomatic or native-sounding.
5. Prestige is not a dimension. Assess demonstrated work.

Scores are 0-10 where 0 = no evidence, 5 = meets the bar, 8 = clearly exceeds it,
10 = exceptional and well evidenced."""

USER_TEMPLATE = """ROLE: {title}
{summary}

RESPONSIBILITIES:
{responsibilities}

MUST HAVE: {must_haves}
NICE TO HAVE: {nice_to_haves}

Score these rubric dimensions (weights shown for context):
{rubric}

Dimension meanings:
- skills: depth and relevance of the technologies and methods evidenced
- experience: relevance and seniority of the work described, versus this role
- impact: evidence of outcomes — shipped systems, measured results, ownership
- communication: how clearly the work is explained (clarity, not fluency)

--- REDACTED RESUME ---
{resume}
--- END RESUME ---

Return a score, an evidence quote, and a one-sentence rationale for each
dimension, plus overall strengths and concerns."""


def _blank_dimensions(job: JobRequisition, reason: str) -> list[DimensionScore]:
    return [
        DimensionScore(dimension=name, score=0.0, evidence="insufficient evidence", rationale=reason)
        for name in job.rubric.as_dict()
    ]


def _weighted_fit(dimensions: list[DimensionScore], job: JobRequisition) -> float:
    """Weighted mean of dimension scores, rescaled to 0-100."""
    weights = job.rubric.as_dict()
    total_weight = 0.0
    accumulated = 0.0
    for dim in dimensions:
        weight = weights.get(dim.dimension.strip().lower())
        if weight is None:
            continue  # model invented a dimension — ignore it rather than score it
        accumulated += dim.score * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return round((accumulated / total_weight) * 10.0, 1)


def assess_fit(
    candidate: Candidate, job: JobRequisition, llm: LLMClient, resume_text: str
) -> FitAssessment:
    rubric_lines = "\n".join(f"- {k}: weight {v:.2f}" for k, v in job.rubric.as_dict().items())
    user = USER_TEMPLATE.format(
        title=job.title,
        summary=job.summary,
        responsibilities="\n".join(f"- {r}" for r in job.responsibilities) or "- (not specified)",
        must_haves=", ".join(f"{r.skill} ({r.min_years:g}y+)" for r in job.must_haves) or "none",
        nice_to_haves=", ".join(r.skill for r in job.nice_to_haves) or "none",
        rubric=rubric_lines,
        resume=truncate(resume_text, 10_000),
    )
    return llm.structured(FitAssessment, SYSTEM, user)


def score_candidate(
    candidate: Candidate,
    job: JobRequisition,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> CandidateScore:
    """Score one candidate. Always returns a score — never raises."""
    settings = settings or Settings.load()
    redact_on = settings.redact_for_ranking

    if redact_on:
        result = redact(candidate.raw_text, known_name=candidate.full_name)
        scoring_text = result.text
        candidate.redacted_text = scoring_text
        redaction_note = f"redacted before scoring ({result.summary()})"
    else:
        scoring_text = candidate.raw_text
        redaction_note = "REDACTION DISABLED — scoring saw full PII"

    gate_score, gates, flags = deterministic_score(candidate, job, scoring_text)
    if not redact_on:
        flags.append(redaction_note)

    dimensions: list[DimensionScore] = []
    strengths: list[str] = []
    concerns: list[str] = []
    llm_available = bool(llm and llm.available)

    if llm_available and scoring_text.strip():
        try:
            fit = assess_fit(candidate, job, llm, scoring_text)
            dimensions = fit.dimensions
            strengths = fit.strengths
            concerns = fit.concerns
            missing_evidence = [d.dimension for d in dimensions if not d.evidence.strip()]
            if missing_evidence:
                flags.append(
                    "scores without evidence, treated as unsupported: "
                    + ", ".join(missing_evidence)
                )
                for dim in dimensions:
                    if not dim.evidence.strip():
                        dim.score = 0.0
                        dim.evidence = "insufficient evidence"
        except LLMUnavailable as exc:
            llm_available = False
            dimensions = _blank_dimensions(job, "LLM assessment unavailable")
            flags.append(f"LLM assessment failed — deterministic score only ({exc})")
    else:
        reason = (
            "resume text was empty" if not scoring_text.strip() else "no LLM configured"
        )
        dimensions = _blank_dimensions(job, reason)
        flags.append(f"deterministic score only — {reason}")

    fit_score = _weighted_fit(dimensions, job)

    # With no model, the deterministic score is the whole score. Blending against
    # a zeroed fit score would make every candidate look bad for the wrong reason.
    if llm_available:
        total = GATE_WEIGHT * gate_score + FIT_WEIGHT * fit_score
    else:
        total = gate_score

    return CandidateScore(
        candidate_id=candidate.id,
        job_id=job.id,
        hard_gates=gates,
        gate_score=gate_score,
        fit_score=fit_score,
        total_score=round(total, 1),
        dimensions=dimensions,
        strengths=strengths,
        concerns=concerns,
        llm_available=llm_available,
        redacted=redact_on,
        flags=flags,
    )


def build_shortlist(scores: list[CandidateScore], job: JobRequisition) -> Shortlist:
    """Rank, then split into shortlisted and not — with a reason on every row.

    A candidate who fails a hard gate is never shortlisted, however well they
    score elsewhere. A candidate who passes every gate but lands just under the
    threshold is flagged for human review rather than quietly dropped.
    """
    ranked = sorted(scores, key=lambda s: (s.all_gates_met, s.total_score), reverse=True)

    shortlisted: list[CandidateScore] = []
    cut: list[CandidateScore] = []

    for score in ranked:
        failed = [g.requirement for g in score.hard_gates if not g.met]
        room_left = len(shortlisted) < job.shortlist_size

        if failed:
            score.shortlisted = False
            # Quote the detail, not just the requirement name: "must-have: Python"
            # reads as "has no Python" when the real reason is the years minimum.
            score.reason = "does not meet - " + "; ".join(
                g.detail or g.requirement for g in score.hard_gates if not g.met
            )
            cut.append(score)
        elif score.total_score < job.shortlist_threshold:
            score.shortlisted = False
            score.reason = (
                f"score {score.total_score:.1f} is below the {job.shortlist_threshold:g} "
                "threshold, but all hard requirements are met — worth a human look"
            )
            score.flags.append("borderline: meets every requirement, below threshold")
            cut.append(score)
        elif not room_left:
            score.shortlisted = False
            score.reason = (
                f"qualified (score {score.total_score:.1f}) but outside the top "
                f"{job.shortlist_size} — waitlisted, not rejected"
            )
            score.flags.append("waitlisted: qualified, shortlist was full")
            cut.append(score)
        else:
            score.shortlisted = True
            score.reason = f"meets all hard requirements; score {score.total_score:.1f}"
            shortlisted.append(score)

    return Shortlist(job_id=job.id, ranked=shortlisted, cut=cut)
