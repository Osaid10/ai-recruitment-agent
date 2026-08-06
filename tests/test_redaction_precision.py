"""Redaction must remove identity without removing the work.

Over-redaction is not a safe failure. The DOB pattern once matched the letters
"age" inside "agentic" and its trailing wildcard consumed the rest of the
phrase, so "Agentic AI", "Multi-Agent Systems" and "agent orchestration" were
deleted from an AI engineer's resume before it was scored — erasing the single
most relevant qualification on the page, for every candidate in that field.

Found by running a real CV through the pipeline. These tests hold both edges:
identity out, skills in.
"""

from __future__ import annotations

import pytest

from recruiter.redact import guess_name_tokens, leaked_terms, redact

AI_RESUME = """AMARA OKONKWO
Islamabad, Pakistan | someone@example.com | +92 300 1234567

SUMMARY
AI/ML Engineer with hands-on experience building agentic AI, computer vision
and LLM-powered systems. Specialized in RAG architectures, multi-agent
orchestration and real-time inference.

CORE SKILLS
AI/ML: Deep Learning, Computer Vision, Generative AI, RAG, LLMs, Agentic AI,
Multi-Agent Systems, Transformers, Fine-tuning (LoRA), Prompt Engineering
Frameworks: Python, PyTorch, LangChain, LangGraph, Hugging Face, FastAPI

EDUCATION
BS Artificial Intelligence | Some Institute of Technology 2022 - 2026
"""

SKILLS_THAT_MUST_SURVIVE = [
    "Agentic AI",
    "Multi-Agent Systems",
    "multi-agent",
    "orchestration",
    "LangGraph",
    "LangChain",
    "PyTorch",
    "FastAPI",
    "RAG",
    "Prompt Engineering",
    "Computer Vision",
]


@pytest.mark.parametrize("skill", SKILLS_THAT_MUST_SURVIVE)
def test_redaction_preserves_job_relevant_skills(skill: str) -> None:
    result = redact(AI_RESUME, known_name="Amara Okonkwo")
    assert skill.lower() in result.text.lower(), (
        f"redaction destroyed '{skill}' — the scorer can no longer see the "
        "candidate's most relevant qualification"
    )


def test_redaction_still_removes_identity_from_the_same_resume() -> None:
    result = redact(AI_RESUME, known_name="Amara Okonkwo")
    leaked = leaked_terms(
        result.text,
        ["Amara", "Okonkwo", "someone@example.com", "1234567", "Some Institute"],
    )
    assert not leaked, f"redaction leaked: {leaked}"


# -- the name heuristic must handle both resume conventions ---------------


def test_all_caps_name_header_is_detected() -> None:
    """Plenty of CVs set the name in capitals. Missing it leaves the name in
    the text the scorer sees."""
    assert guess_name_tokens("AMARA OKONKWO CHUKWU\nIslamabad, Pakistan") == [
        "AMARA",
        "OKONKWO",
        "CHUKWU",
    ]


def test_title_case_name_header_is_detected() -> None:
    assert guess_name_tokens("Zainab Qureshi\nLahore, Pakistan") == ["Zainab", "Qureshi"]


def test_a_document_heading_is_not_mistaken_for_a_name() -> None:
    assert guess_name_tokens("CURRICULUM VITAE\nSomeone") == []
    assert guess_name_tokens("RESUME\nSomeone") == []


# -- date-of-birth patterns: catch the real ones, only the real ones ------


@pytest.mark.parametrize(
    "line",
    [
        "Date of Birth: 14 March 1997",
        "DOB: 1997",
        "Age: 25",
        "born in 1990",
    ],
)
def test_genuine_age_and_dob_fields_are_removed(line: str) -> None:
    result = redact(line)
    assert "dob" in result.counts, f"failed to redact: {line}"


@pytest.mark.parametrize(
    "line",
    [
        "Built agentic AI systems in production",
        "Agent orchestration across multiple services",
        "Experience with multi-agent workflows",
        "Managed the storage layer",
        "Language model fine-tuning",
    ],
)
def test_ordinary_technical_prose_is_not_mistaken_for_a_dob(line: str) -> None:
    result = redact(line)
    assert "dob" not in result.counts, f"wrongly redacted as DOB: {line}"
    assert result.text.strip(), "redaction emptied a line of technical prose"
