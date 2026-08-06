"""Tests for evidence verification.

The strings here are real output from llama-3.3-70b on the sample data, not
invented examples. Measured across one full run, the ranking stage quoted
verbatim 26 times out of 26 while the summarisation stage managed it once in
eight — it wrote fluent descriptions instead. These tests pin the behaviour that
catches that.
"""

from __future__ import annotations

import pytest

from recruiter.evidence import (
    is_explicit_absence,
    looks_paraphrased,
    quote_appears_in,
    verify,
)

TRANSCRIPT = """
Daniyal: You mentioned in your CV that you took p95 on the availability endpoint
from 1.9 seconds to 210 milliseconds. What was the actual problem?

Candidate: It was an N+1 in disguise. For each candidate slot in the requested
window we were running a query to check whether it collided with an existing
booking. I replaced it with one query that pulls all the bookings in the window
as a range, and then did the collision check in memory.

Candidate: ECS for services, RDS for Postgres, S3 for stored documents. I have
not run Kubernetes in production.
"""

RESUME = """
TECHNICAL SKILLS
Python, FastAPI, PostgreSQL, SQLAlchemy, REST API design, Docker, AWS

EXPERIENCE
Senior Backend Engineer - Sehat Kahani
- Cut p95 latency on the availability endpoint from 1.9s to 210ms, mostly by
  replacing a per-slot query with one indexed range query.
"""


# -- genuine quotes are accepted ------------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        "It was an N+1 in disguise",
        "ECS for services, RDS for Postgres, S3 for stored documents",
        "I have not run Kubernetes in production",
    ],
)
def test_verbatim_quotes_are_accepted(quote: str) -> None:
    supported, reason = verify(quote, TRANSCRIPT)
    assert supported, f"rejected a real quote: {reason}"


def test_quote_matching_ignores_insignificant_differences() -> None:
    """Line breaks and punctuation must not fail an otherwise real quote."""
    assert quote_appears_in(
        "I replaced it with one query that pulls all the bookings in the window as a range",
        TRANSCRIPT,
    )
    # A skill list is legitimate evidence for the skills dimension — and the
    # ones the model actually cites are full lines, not three words.
    assert quote_appears_in(
        "Python, FastAPI, PostgreSQL, SQLAlchemy, REST API design, Docker, AWS", RESUME
    )


def test_quote_spanning_a_line_break_is_accepted() -> None:
    assert quote_appears_in(
        "Cut p95 latency on the availability endpoint from 1.9s to 210ms", RESUME
    )


# -- the real failure mode: fluent paraphrase -----------------------------

# Verbatim output from the summarisation stage before this check existed.
OBSERVED_PARAPHRASES = [
    "The candidate explained the idempotency layer they designed for the payments "
    "API and how it handles duplicate charges",
    "The candidate described their experience with mentoring engineers and leading "
    "a significant technical migration",
    "The candidate has experience with FastAPI, Postgres, and Terraform, and has "
    "optimized the availability endpoint",
    "The candidate is able to clearly explain technical concepts and their approach "
    "to problem-solving",
]


@pytest.mark.parametrize("paraphrase", OBSERVED_PARAPHRASES)
def test_observed_paraphrases_are_rejected(paraphrase: str) -> None:
    """These all passed the old non-empty check. They must not pass this one."""
    supported, reason = verify(paraphrase, TRANSCRIPT)
    assert not supported
    assert reason


@pytest.mark.parametrize("paraphrase", OBSERVED_PARAPHRASES)
def test_paraphrase_marker_is_detected(paraphrase: str) -> None:
    assert looks_paraphrased(paraphrase)


def test_a_plausible_but_absent_quote_is_rejected() -> None:
    """The dangerous case: reads like a quote, was never said."""
    supported, reason = verify(
        "I rewrote the entire billing system over a weekend single-handedly", TRANSCRIPT
    )
    assert not supported
    assert "not found" in reason


# -- reporting an absence is correct, not a failure -----------------------


@pytest.mark.parametrize(
    "text",
    [
        "insufficient evidence",
        "No relevant experience found",
        "No evidence of shipped systems or measured results found",
        "not assessed",
        "",
    ],
)
def test_explicit_absence_is_recognised(text: str) -> None:
    assert is_explicit_absence(text) or not text.strip()


def test_absence_is_unsupported_but_not_called_a_fabrication() -> None:
    supported, reason = verify("No relevant experience found", RESUME)
    assert not supported
    assert "no evidence" in reason.lower() or "explicitly" in reason.lower()
    assert "not found in the source" not in reason


# -- guards against false positives ---------------------------------------


def test_a_short_common_phrase_does_not_count_as_evidence() -> None:
    """'Python' appears in the resume, but it supports nothing on its own."""
    assert not quote_appears_in("Python", RESUME)
    assert not quote_appears_in("the", RESUME)


def test_empty_source_never_verifies() -> None:
    assert not quote_appears_in("anything at all here", "")
