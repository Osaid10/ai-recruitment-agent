"""Stage 5 — interview transcript to a structured summary.

Every signal rating must carry a quote from the transcript. The summary is what
the hiring manager actually reads, so an unsupported claim here is worse than a
missing one — `not_assessed` is a valid and useful outcome.
"""

from __future__ import annotations

import re
from pathlib import Path

from .evidence import verify
from .llm import LLMClient, LLMUnavailable, truncate
from .models import (
    InterviewSummary,
    JobRequisition,
    QuestionSet,
    Rating,
    SignalAssessment,
)

SYSTEM = """You summarise a job interview transcript for a hiring panel.

Rules:
1. Report only what is in the transcript. If the interview did not cover a
   dimension, rate it `not_assessed` — do not infer it from tone or from what
   the candidate seems like.
2. **`evidence` must be text copied word-for-word out of the transcript.** It is
   a quotation, not a description. Copy the candidate's actual words.

   WRONG: "The candidate explained the idempotency layer and how it works."
   RIGHT: "Idempotency keys. The client sends a key with the request, we store
           it with the result, and if we see the same key again we return the
           stored result instead of processing again."

   If you cannot find words in the transcript that support a rating, set that
   dimension to `not_assessed`. A dimension marked not_assessed is a useful,
   correct answer. An invented quote is the worst thing you can produce here:
   quotes are checked against the transcript, and one that is not found is
   discarded along with the rating built on it.
3. Record concerns as plainly as strengths. A summary with no concerns is
   almost always an incomplete summary — say so if you genuinely found none.
4. `unanswered_questions` lists what the panel planned to ask but did not get a
   substantive answer to. This drives the next round.
5. Never comment on accent, appearance, personality, background, or anything
   unrelated to the ability to do the job.
6. Do not recommend hiring or rejecting. That is a later step, and a human's."""

USER_TEMPLATE = """ROLE: {title}

Rate these dimensions: {dimensions}

{planned}

--- TRANSCRIPT ---
{transcript}
--- END TRANSCRIPT ---

Summarise the interview."""


def _read_transcript(source: str | Path) -> str:
    """Accept either raw transcript text or a path to a transcript file."""
    if isinstance(source, Path) or (
        isinstance(source, str) and len(source) < 260 and Path(source).is_file()
    ):
        return Path(source).read_text(encoding="utf-8", errors="replace")
    return str(source)


def _heuristic_summary(transcript: str, job: JobRequisition) -> InterviewSummary:
    """No-LLM fallback: extract the candidate's turns so a human still has
    something structured to read. Deliberately makes no judgements."""
    turns = re.findall(
        r"^\s*(?:candidate|interviewee)\s*[:\-]\s*(.+)$", transcript, re.I | re.M
    )
    longest = sorted(turns, key=len, reverse=True)[:3]

    return InterviewSummary(
        overall_impression=(
            "No model available — this is a mechanical extract, not an assessment. "
            f"The transcript contains {len(turns)} candidate responses "
            f"({len(transcript.split())} words total). A human must read it."
        ),
        signals=[
            SignalAssessment(
                dimension=dim,
                rating=Rating.NOT_ASSESSED,
                evidence="no model available to assess",
            )
            for dim in job.rubric.as_dict()
        ],
        notable_quotes=[q.strip()[:400] for q in longest],
        concerns=["summary generated without a model — treat as unreviewed"],
        llm_available=False,
    )


def summarize_interview(
    transcript_source: str | Path,
    job: JobRequisition,
    candidate_id: str = "",
    interview_id: str = "",
    question_set: QuestionSet | None = None,
    llm: LLMClient | None = None,
) -> InterviewSummary:
    """Summarise one interview. Always returns a summary — never raises."""
    transcript = _read_transcript(transcript_source).strip()

    if not transcript:
        return InterviewSummary(
            candidate_id=candidate_id,
            interview_id=interview_id,
            overall_impression="Transcript is empty — nothing to summarise.",
            concerns=["no transcript provided; interview may not have happened yet"],
            llm_available=False,
        )

    if llm and llm.available:
        planned = ""
        if question_set and question_set.questions:
            planned = "The panel planned to ask:\n" + "\n".join(
                f"- {q.question}" for q in question_set.questions
            )

        user = USER_TEMPLATE.format(
            title=job.title,
            dimensions=", ".join(job.rubric.as_dict()),
            planned=planned,
            transcript=truncate(transcript, 14_000),
        )

        try:
            summary = llm.structured(InterviewSummary, SYSTEM, user)
            summary.candidate_id = candidate_id
            summary.interview_id = interview_id
            summary.llm_available = True

            # Checking that `evidence` is non-empty is not enough: the common
            # failure is a fluent description that never appears in the
            # transcript. Verify each quote against the source and discard the
            # rating when it does not hold up.
            unsupported: list[str] = []
            for signal in summary.signals:
                if signal.rating is Rating.NOT_ASSESSED:
                    continue
                supported, reason = verify(signal.evidence, transcript)
                if not supported:
                    unsupported.append(f"{signal.dimension} ({reason})")
                    signal.rating = Rating.NOT_ASSESSED
                    signal.evidence = f"discarded — {reason}"

            if unsupported:
                summary.concerns.append(
                    "ratings whose evidence could not be found in the transcript were "
                    "downgraded to not_assessed: " + "; ".join(unsupported)
                )
            return summary
        except LLMUnavailable:
            pass

    fallback = _heuristic_summary(transcript, job)
    fallback.candidate_id = candidate_id
    fallback.interview_id = interview_id
    return fallback
