"""Domain models — the typed contract between every pipeline stage.

Every stage takes and returns these. LLM calls bind directly to them via
`with_structured_output`, so a malformed model response is a validation error
rather than a parsing bug that surfaces three stages later.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Stage(str, Enum):
    INGEST = "ingest"
    RANK = "rank"
    SCHEDULE = "schedule"
    QUESTIONS = "questions"
    SUMMARIZE = "summarize"
    RECOMMEND = "recommend"


class StageStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_HUMAN = "needs_human"


class Verdict(str, Enum):
    """The agent's recommendation. Advisory only — a human decides."""

    STRONG_HIRE = "strong_hire"
    HIRE = "hire"
    LEAN_HIRE = "lean_hire"
    LEAN_NO_HIRE = "lean_no_hire"
    NO_HIRE = "no_hire"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class HumanAction(str, Enum):
    APPROVE_SHORTLIST = "approve_shortlist"
    ADVANCE = "advance"
    REJECT = "reject"
    REQUEST_MORE_INFO = "request_more_info"


class Rating(str, Enum):
    STRONG = "strong"
    ADEQUATE = "adequate"
    WEAK = "weak"
    NOT_ASSESSED = "not_assessed"


class QuestionCategory(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    CLAIM_VERIFICATION = "claim_verification"
    GAP_PROBE = "gap_probe"
    ROLE_FIT = "role_fit"


# --------------------------------------------------------------------------
# Job requisition — the ranking criteria, declared by a human up front
# --------------------------------------------------------------------------


class SkillRequirement(BaseModel):
    skill: str
    min_years: float = 0.0
    weight: float = 1.0


class RubricWeights(BaseModel):
    """Weights for the LLM-judged dimensions. Must sum to 1.0."""

    skills: float = 0.40
    experience: float = 0.30
    impact: float = 0.20
    communication: float = 0.10

    @model_validator(mode="after")
    def _sums_to_one(self) -> RubricWeights:
        total = self.skills + self.experience + self.impact + self.communication
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rubric weights must sum to 1.0, got {total:.4f}")
        return self

    def as_dict(self) -> dict[str, float]:
        return {
            "skills": self.skills,
            "experience": self.experience,
            "impact": self.impact,
            "communication": self.communication,
        }


class Interviewer(BaseModel):
    name: str
    email: str
    role: str = "Interviewer"


class JobRequisition(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    title: str
    department: str = ""
    location: str = ""
    employment_type: str = "Full-time"
    summary: str = ""
    responsibilities: list[str] = Field(default_factory=list)

    must_haves: list[SkillRequirement] = Field(default_factory=list)
    nice_to_haves: list[SkillRequirement] = Field(default_factory=list)
    min_years_experience: float = 0.0

    rubric: RubricWeights = Field(default_factory=RubricWeights)
    shortlist_size: int = 5
    shortlist_threshold: float = 60.0
    interview_panel: list[Interviewer] = Field(default_factory=list)
    interview_minutes: int = 45


# --------------------------------------------------------------------------
# Candidate — the output of stage 1 (ingest)
# --------------------------------------------------------------------------


class WorkExperience(BaseModel):
    company: str = ""
    title: str = ""
    start: str = ""  # free text as written on the resume, e.g. "Mar 2021"
    end: str = ""  # "Present" is common
    description: str = ""
    years: float = 0.0


class Education(BaseModel):
    institution: str = ""
    degree: str = ""
    field: str = ""
    year: str = ""


class ExtractedProfile(BaseModel):
    """The LLM's structured read of a resume. Kept separate from `Candidate` so
    the extraction prompt binds to exactly these fields and nothing else."""

    full_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[str] = Field(default_factory=list)
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    total_years_experience: float = 0.0


class Candidate(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cand"))
    job_id: str = ""
    source_file: str = ""
    ingested_at: str = Field(default_factory=utcnow)

    full_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[str] = Field(default_factory=list)
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    total_years_experience: float = 0.0

    raw_text: str = ""
    redacted_text: str = ""

    # Two different things, deliberately separated. A parse warning means the
    # file itself is a problem and a person should look at it. A note records a
    # correction the pipeline made and handled — worth auditing, not worth
    # escalating, and flagging every one of them would train reviewers to
    # ignore the flag that matters.
    parse_warnings: list[str] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)
    extraction_method: str = "llm"  # "llm" | "heuristic"

    @property
    def display_name(self) -> str:
        return self.full_name or f"<unnamed {self.id}>"


# --------------------------------------------------------------------------
# Ranking — the output of stage 2
# --------------------------------------------------------------------------


class HardGate(BaseModel):
    """A deterministic pass/fail check. The LLM cannot override these."""

    requirement: str
    met: bool
    detail: str = ""


class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(ge=0.0, le=10.0)
    evidence: str = Field(
        default="",
        description="A short quote from the resume supporting this score.",
    )
    rationale: str = ""

    @field_validator("score")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(10.0, v))


class FitAssessment(BaseModel):
    """What the LLM returns when judging a redacted resume against the rubric."""

    dimensions: list[DimensionScore] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class CandidateScore(BaseModel):
    candidate_id: str
    job_id: str
    scored_at: str = Field(default_factory=utcnow)

    hard_gates: list[HardGate] = Field(default_factory=list)
    gate_score: float = 0.0  # 0-100, deterministic
    fit_score: float = 0.0  # 0-100, LLM-judged
    total_score: float = 0.0  # 0-100, weighted blend

    dimensions: list[DimensionScore] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    llm_available: bool = True
    redacted: bool = True
    flags: list[str] = Field(default_factory=list)
    shortlisted: bool = False
    reason: str = ""

    @property
    def all_gates_met(self) -> bool:
        return all(g.met for g in self.hard_gates)


class Shortlist(BaseModel):
    job_id: str
    generated_at: str = Field(default_factory=utcnow)
    ranked: list[CandidateScore] = Field(default_factory=list)
    cut: list[CandidateScore] = Field(default_factory=list)
    approved_by: str = ""
    approved_at: str = ""

    @property
    def is_approved(self) -> bool:
        return bool(self.approved_by)


# --------------------------------------------------------------------------
# Scheduling — stage 3
# --------------------------------------------------------------------------


class TimeSlot(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _ordered(self) -> TimeSlot:
        if self.end <= self.start:
            raise ValueError("slot end must be after start")
        return self

    def overlaps(self, other: TimeSlot) -> bool:
        return self.start < other.end and other.start < self.end

    def __str__(self) -> str:
        return f"{self.start:%a %d %b %Y %H:%M}–{self.end:%H:%M}"


class InterviewBooking(BaseModel):
    id: str = Field(default_factory=lambda: new_id("intv"))
    candidate_id: str
    job_id: str
    slot: TimeSlot
    interviewers: list[Interviewer] = Field(default_factory=list)
    mode: str = "Video call"
    location: str = ""
    ics_path: str = ""
    status: str = "scheduled"  # scheduled | completed | cancelled | unscheduled
    note: str = ""
    created_at: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Interview questions — stage 4
# --------------------------------------------------------------------------


class InterviewQuestion(BaseModel):
    question: str
    category: QuestionCategory = QuestionCategory.TECHNICAL
    rationale: str = Field(
        default="", description="Why this candidate specifically is being asked this."
    )
    what_good_looks_like: str = ""
    linked_evidence: str = Field(
        default="", description="The resume claim or gap that prompted the question."
    )


class QuestionSet(BaseModel):
    candidate_id: str = ""
    job_id: str = ""
    generated_at: str = Field(default_factory=utcnow)
    questions: list[InterviewQuestion] = Field(default_factory=list)
    llm_available: bool = True


# --------------------------------------------------------------------------
# Interview summary — stage 5
# --------------------------------------------------------------------------


class SignalAssessment(BaseModel):
    dimension: str
    rating: Rating = Rating.NOT_ASSESSED
    evidence: str = Field(
        default="", description="A quote from the transcript supporting this rating."
    )


class InterviewSummary(BaseModel):
    candidate_id: str = ""
    interview_id: str = ""
    generated_at: str = Field(default_factory=utcnow)

    overall_impression: str = ""
    signals: list[SignalAssessment] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    notable_quotes: list[str] = Field(default_factory=list)
    llm_available: bool = True


# --------------------------------------------------------------------------
# Recommendation — stage 6. Advisory only.
# --------------------------------------------------------------------------


class Recommendation(BaseModel):
    candidate_id: str = ""
    job_id: str = ""
    generated_at: str = Field(default_factory=utcnow)

    verdict: Verdict = Verdict.INSUFFICIENT_EVIDENCE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    supporting_evidence: list[str] = Field(default_factory=list)
    dissenting_signals: list[str] = Field(
        default_factory=list,
        description="Evidence that argues against the verdict. Never leave empty "
        "without saying why — a one-sided case is a red flag.",
    )
    suggested_next_step: str = ""
    llm_available: bool = True

    # Structural reminder that this is not a decision. Nothing reads this to
    # decide anything; it exists so the field shows up in every export and UI.
    requires_human_decision: bool = True


class HumanDecision(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dec"))
    candidate_id: str
    job_id: str
    action: HumanAction
    decided_by: str
    notes: str = ""
    decided_at: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Pipeline plumbing
# --------------------------------------------------------------------------


class StageResult(BaseModel):
    """A stage never raises at the pipeline boundary — it returns one of these."""

    stage: Stage
    status: StageStatus
    detail: str = ""
    candidate_id: str = ""
    data: dict = Field(default_factory=dict)
    at: str = Field(default_factory=utcnow)

    @property
    def ok(self) -> bool:
        return self.status is StageStatus.OK


class RunReport(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    job_id: str = ""
    started_at: str = Field(default_factory=utcnow)
    finished_at: str = ""
    results: list[StageResult] = Field(default_factory=list)
    awaiting: str = ""  # what the agent is blocked on, if anything

    def add(self, result: StageResult) -> StageResult:
        self.results.append(result)
        return result

    def failures(self) -> list[StageResult]:
        return [r for r in self.results if r.status is StageStatus.FAILED]

    def needs_human(self) -> list[StageResult]:
        return [r for r in self.results if r.status is StageStatus.NEEDS_HUMAN]
