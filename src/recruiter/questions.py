"""Stage 4 — tailored interview questions.

The point of this stage is that the questions are *about this candidate*: the
claims worth verifying, the gaps worth probing, the concerns the ranking raised.
A generic question bank would be cheaper and useless.

Falls back to a targeted template set built from the score's gaps and concerns
when no model is available — still candidate-specific, just less fluent.
"""

from __future__ import annotations

from .llm import LLMClient, LLMUnavailable, truncate
from .models import (
    Candidate,
    CandidateScore,
    InterviewQuestion,
    JobRequisition,
    QuestionCategory,
    QuestionSet,
)

DEFAULT_QUESTION_COUNT = 8

SYSTEM = """You write interview questions for a hiring panel.

The questions must be specific to this candidate — they should be unusable for
anyone else. Ground each one in something the resume actually says, or in a
concern the screening raised.

Rules:
1. Mix the categories: technical depth, behavioral, verification of a specific
   resume claim, probing a genuine gap, and role fit.
2. `linked_evidence` must quote the resume claim or name the gap that prompted
   the question. No question without a reason to ask it.
3. `what_good_looks_like` describes the substance of a strong answer, so an
   interviewer can score consistently across candidates.
4. Ask about work, not about the person. Never ask about age, family, marital or
   parental status, nationality, religion, health, or disability. If there is an
   employment gap, the only acceptable framing is a neutral, optional one about
   what they worked on, and it must never imply the gap is a negative.
5. No brain-teasers and no trivia. Every question should predict job performance.
6. Questions must be answerable in 3-5 minutes each."""

USER_TEMPLATE = """ROLE: {title}
{summary}

KEY RESPONSIBILITIES:
{responsibilities}

MUST HAVE: {must_haves}

--- WHAT SCREENING FOUND ---
Score: {score}/100
Requirements met: {gates_met}
Requirements NOT met: {gates_failed}
Strengths noted: {strengths}
Concerns to probe: {concerns}

--- CANDIDATE PROFILE (redacted) ---
Years of experience: {years}
Skills claimed: {skills}

{resume}
--- END ---

Write exactly {count} questions."""


def _fallback_questions(
    candidate: Candidate, job: JobRequisition, score: CandidateScore
) -> list[InterviewQuestion]:
    """Template questions targeted at this candidate's actual gaps."""
    questions: list[InterviewQuestion] = []

    for gate in score.hard_gates:
        if gate.met:
            continue
        requirement = gate.requirement.replace("must-have: ", "")
        questions.append(
            InterviewQuestion(
                question=(
                    f"This role leans heavily on {requirement}. Your resume does not "
                    f"mention it — have you worked with it, and if not, what is the "
                    f"closest thing you have done?"
                ),
                category=QuestionCategory.GAP_PROBE,
                rationale=f"Screening could not find evidence of {requirement}.",
                what_good_looks_like=(
                    f"Either concrete {requirement} work that the resume omitted, or a "
                    "credible adjacent experience plus a realistic view of the ramp-up."
                ),
                linked_evidence=gate.detail,
            )
        )

    for skill in candidate.skills[:3]:
        questions.append(
            InterviewQuestion(
                question=(
                    f"Walk us through a problem you solved with {skill}. What was the "
                    f"hardest part, and what would you do differently now?"
                ),
                category=QuestionCategory.CLAIM_VERIFICATION,
                rationale=f"The resume lists {skill}; this checks the depth behind it.",
                what_good_looks_like=(
                    "Specific technical detail, a real constraint they hit, and a "
                    "considered retrospective — not a description of the happy path."
                ),
                linked_evidence=f"resume lists '{skill}' under skills",
            )
        )

    for concern in score.concerns[:2]:
        questions.append(
            InterviewQuestion(
                question=f"Screening flagged this: {concern}. How would you respond to it?",
                category=QuestionCategory.ROLE_FIT,
                rationale="Gives the candidate a fair chance to answer a recorded concern.",
                what_good_looks_like="A direct answer with evidence, or an honest acknowledgement.",
                linked_evidence=concern,
            )
        )

    questions.append(
        InterviewQuestion(
            question=(
                f"Tell us about a time you disagreed with a technical decision on your "
                f"team. What did you do?"
            ),
            category=QuestionCategory.BEHAVIORAL,
            rationale="Baseline collaboration signal for a role with shared ownership.",
            what_good_looks_like=(
                "Engaged with the substance, escalated appropriately, and committed to "
                "the outcome regardless of which way it went."
            ),
            linked_evidence="standard for this role",
        )
    )

    if job.responsibilities:
        questions.append(
            InterviewQuestion(
                question=(
                    f"A core part of this job is: {job.responsibilities[0]} "
                    f"How have you done something like that before?"
                ),
                category=QuestionCategory.ROLE_FIT,
                rationale="Tests fit against the role's stated day-to-day work.",
                what_good_looks_like="A directly comparable example with their own contribution clear.",
                linked_evidence=job.responsibilities[0],
            )
        )

    return questions[:DEFAULT_QUESTION_COUNT]


def generate_questions(
    candidate: Candidate,
    job: JobRequisition,
    score: CandidateScore,
    llm: LLMClient | None = None,
    count: int = DEFAULT_QUESTION_COUNT,
) -> QuestionSet:
    """Build a question set. Always returns one — never raises."""
    if llm and llm.available:
        gates_met = [g.requirement for g in score.hard_gates if g.met]
        gates_failed = [g.requirement for g in score.hard_gates if not g.met]
        resume_text = candidate.redacted_text or candidate.raw_text

        user = USER_TEMPLATE.format(
            title=job.title,
            summary=job.summary,
            responsibilities="\n".join(f"- {r}" for r in job.responsibilities) or "- (not specified)",
            must_haves=", ".join(r.skill for r in job.must_haves) or "none",
            score=f"{score.total_score:.1f}",
            gates_met=", ".join(gates_met) or "none",
            gates_failed=", ".join(gates_failed) or "none",
            strengths="; ".join(score.strengths) or "none recorded",
            concerns="; ".join(score.concerns) or "none recorded",
            years=f"{candidate.total_years_experience:g}",
            skills=", ".join(candidate.skills[:25]) or "none extracted",
            resume=truncate(resume_text, 7_000),
            count=count,
        )

        try:
            result = llm.structured(QuestionSet, SYSTEM, user)
            if result.questions:
                result.candidate_id = candidate.id
                result.job_id = job.id
                result.llm_available = True
                return result
        except LLMUnavailable:
            pass  # fall through to templates

    return QuestionSet(
        candidate_id=candidate.id,
        job_id=job.id,
        questions=_fallback_questions(candidate, job, score),
        llm_available=False,
    )


def format_agenda(question_set: QuestionSet, limit: int = 5) -> str:
    """A short agenda block for the calendar invite body."""
    if not question_set.questions:
        return ""
    lines = ["Prepared focus areas:"]
    for i, q in enumerate(question_set.questions[:limit], start=1):
        lines.append(f"{i}. [{q.category.value}] {q.question}")
    return "\n".join(lines)
