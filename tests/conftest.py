"""Shared fixtures.

Every test runs with the LLM switched off. That is deliberate: the tests must be
deterministic and must pass on a machine with no API key, and it means the
deterministic half of the system is what is actually under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recruiter.config import Settings
from recruiter.models import Interviewer, JobRequisition, RubricWeights, SkillRequirement
from recruiter.pipeline import RecruitmentAgent
from recruiter.schedule.calendar import MockCalendar
from recruiter.store import ATSStore

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RESUMES = REPO_ROOT / "data" / "resumes"
SAMPLE_INTERVIEWS = REPO_ROOT / "data" / "interviews"
SAMPLE_JOB = REPO_ROOT / "data" / "jobs" / "senior_python_engineer.json"


class OfflineLLM:
    """Stands in for `LLMClient` with no model configured."""

    available = False
    unavailable_reason = "disabled for tests"
    model_name = "none"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        groq_api_key="",
        db_path=tmp_path / "ats.db",
        out_dir=tmp_path / "out",
        redact_for_ranking=True,
    )


@pytest.fixture
def store(settings: Settings) -> ATSStore:
    s = ATSStore(settings.db_path)
    yield s
    s.close()


@pytest.fixture
def calendar() -> MockCalendar:
    return MockCalendar()


@pytest.fixture
def agent(settings: Settings, store: ATSStore, calendar: MockCalendar) -> RecruitmentAgent:
    return RecruitmentAgent(
        settings=settings, store=store, llm=OfflineLLM(), calendar=calendar
    )


@pytest.fixture
def job() -> JobRequisition:
    return JobRequisition(
        id="job_test",
        title="Senior Python Engineer",
        summary="Backend services for the data platform.",
        responsibilities=["Design and ship production Python services."],
        must_haves=[
            SkillRequirement(skill="Python", min_years=4, weight=2.0),
            SkillRequirement(skill="SQL", min_years=3, weight=1.5),
            SkillRequirement(skill="REST", min_years=2, weight=1.0),
        ],
        nice_to_haves=[
            SkillRequirement(skill="Docker", weight=1.0),
            SkillRequirement(skill="AWS", weight=1.0),
        ],
        min_years_experience=4,
        rubric=RubricWeights(),
        shortlist_size=3,
        shortlist_threshold=60.0,
        interview_panel=[
            Interviewer(name="Ayesha Raza", email="ayesha@example.com"),
            Interviewer(name="Daniyal Ahmed", email="daniyal@example.com"),
        ],
    )


RESUME_BODY = """{name}
{email} | +92 300 1234567 | Lahore, Pakistan

SUMMARY
Backend engineer with six years building Python services.

TECHNICAL SKILLS
Python, FastAPI, PostgreSQL, SQL, REST API design, Docker, AWS, pytest

EXPERIENCE

Senior Backend Engineer - Example Corp
Jan 2021 - Present
- Own a FastAPI service on PostgreSQL handling 200,000 requests a day.
- Cut p95 latency from 1.9s to 210ms by replacing an N+1 with a range query.
- Mentor two engineers and run the weekly design review.

Backend Engineer - Another Company
Aug 2019 - Dec 2020
- Built REST APIs in Python and Django.
- Containerised four services and moved them onto AWS.

EDUCATION
BS Computer Science, {institution}, 2019
"""


def make_resume(name: str, email: str = "candidate@example.com", institution: str = "Some University") -> str:
    """Identical resume body under a different identity. Used by the fairness tests."""
    return RESUME_BODY.format(name=name, email=email, institution=institution)
