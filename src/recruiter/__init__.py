"""AI Recruitment Agent — runs the top of the hiring funnel.

The agent ingests resumes, ranks them, schedules interviews, generates questions,
summarises outcomes and recommends. It does not hire or reject anyone: both of
those require a human decision recorded through `recruiter.recommend`.
"""

__version__ = "0.1.0"

from .config import Settings
from .models import (
    Candidate,
    CandidateScore,
    InterviewBooking,
    InterviewSummary,
    JobRequisition,
    QuestionSet,
    Recommendation,
    RunReport,
    Shortlist,
)
from .store import ATSStore

__all__ = [
    "Settings",
    "ATSStore",
    "JobRequisition",
    "Candidate",
    "CandidateScore",
    "Shortlist",
    "InterviewBooking",
    "QuestionSet",
    "InterviewSummary",
    "Recommendation",
    "RunReport",
    "__version__",
]
