"""The agent loop.

Owns all I/O — the stages stay pure so they can be tested without a database.
Two properties matter more than anything else here:

1. **One bad candidate cannot end a run.** Every per-candidate step is wrapped;
   a failure becomes a `StageResult(status=FAILED)` on the report and the loop
   continues with the rest.
2. **The agent stops at the human gates.** `run()` will not schedule anyone until
   a named person has approved the shortlist, and it finishes at
   `awaiting_decision` rather than concluding anything.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .config import Settings
from .ingest.extract import build_candidate
from .ingest.parser import discover_resumes, parse_resume
from .llm import LLMClient
from .models import (
    Candidate,
    InterviewBooking,
    InterviewSummary,
    JobRequisition,
    QuestionSet,
    Recommendation,
    RunReport,
    Shortlist,
    Stage,
    StageResult,
    StageStatus,
    utcnow,
)
from .questions import format_agenda, generate_questions
from .rank.ranker import build_shortlist, score_candidate
from .recommend import (
    ApprovalRequired,
    approve_shortlist,
    recommend,
    require_shortlist_approval,
    shortlist_is_approved,
)
from .schedule.calendar import (
    CalendarAdapter,
    MockCalendar,
    NoSlotAvailable,
    default_window,
    find_slot,
)
from .schedule.ics import write_ics
from .store import ATSStore
from .summarize import summarize_interview

log = logging.getLogger(__name__)


class JobSpecError(ValueError):
    """The requisition file is missing or malformed."""


def load_job(path: Path | str) -> JobRequisition:
    """Read a job requisition JSON file. Raises `JobSpecError` with a usable message."""
    path = Path(path)
    if not path.is_file():
        raise JobSpecError(f"job requisition not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JobSpecError(f"{path.name} is not valid JSON: {exc}") from exc

    try:
        return JobRequisition.model_validate(raw)
    except Exception as exc:
        raise JobSpecError(f"{path.name} is not a valid job requisition: {exc}") from exc


class RecruitmentAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        store: ATSStore | None = None,
        llm: LLMClient | None = None,
        calendar: CalendarAdapter | None = None,
    ) -> None:
        self.settings = settings or Settings.load()
        self.store = store or ATSStore(self.settings.db_path)
        self.llm = llm if llm is not None else LLMClient(self.settings)
        self.calendar = calendar or MockCalendar()

    # -- helpers ----------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.llm.model_name if self.llm else "none"

    def _skill_vocabulary(self, job: JobRequisition) -> list[str]:
        return [r.skill for r in (*job.must_haves, *job.nice_to_haves)]

    def register_job(self, job: JobRequisition) -> JobRequisition:
        self.store.save_job(job)
        self.store.audit(
            "setup",
            "job_registered",
            entity_id=job.id,
            job_id=job.id,
            title=job.title,
            must_haves=[r.skill for r in job.must_haves],
            rubric=job.rubric.as_dict(),
        )
        return job

    def get_job(self, job_id: str) -> JobRequisition:
        job = self.store.get_job(job_id)
        if job is None:
            known = ", ".join(j.id for j in self.store.list_jobs()) or "none"
            raise JobSpecError(f"unknown job id '{job_id}'. Known jobs: {known}")
        return job

    # -- stage 1: ingest --------------------------------------------------

    def ingest(self, job: JobRequisition, resumes_dir: Path | str, report: RunReport) -> list[Candidate]:
        try:
            files = discover_resumes(resumes_dir)
        except NotADirectoryError as exc:
            report.add(
                StageResult(stage=Stage.INGEST, status=StageStatus.FAILED, detail=str(exc))
            )
            return []

        if not files:
            report.add(
                StageResult(
                    stage=Stage.INGEST,
                    status=StageStatus.FAILED,
                    detail=f"no supported resume files found in {resumes_dir}",
                )
            )
            return []

        vocabulary = self._skill_vocabulary(job)
        candidates: list[Candidate] = []

        for path in files:
            try:
                parsed = parse_resume(path)

                if not parsed.usable:
                    reason = "; ".join(parsed.warnings) or "no readable text"
                    self.store.audit(
                        "ingest",
                        "unreadable",
                        job_id=job.id,
                        source_file=str(path),
                        reason=reason,
                    )
                    report.add(
                        StageResult(
                            stage=Stage.INGEST,
                            status=StageStatus.NEEDS_HUMAN,
                            detail=f"{path.name}: {reason}",
                        )
                    )
                    continue

                existing = self.store.find_candidate_by_source(job.id, str(path))
                candidate = build_candidate(parsed, job.id, self.llm, vocabulary)
                if existing:
                    candidate.id = existing.id  # re-ingesting updates, not duplicates

                self.store.save_candidate(candidate)
                self.store.audit(
                    "ingest",
                    "candidate_parsed",
                    entity_id=candidate.id,
                    job_id=job.id,
                    model=self.model_name if candidate.extraction_method == "llm" else "",
                    source_file=path.name,
                    method=candidate.extraction_method,
                    skills_found=len(candidate.skills),
                    years=candidate.total_years_experience,
                    warnings=candidate.parse_warnings,
                    notes=candidate.extraction_notes,
                )
                candidates.append(candidate)

                # Only genuine parse problems escalate. Corrections the pipeline
                # already handled are audited and reported, not flagged.
                status = (
                    StageStatus.NEEDS_HUMAN if candidate.parse_warnings else StageStatus.OK
                )
                detail = (
                    f"{path.name} -> {candidate.display_name} "
                    f"({len(candidate.skills)} skills, "
                    f"{candidate.total_years_experience:g}y)"
                )
                if candidate.parse_warnings:
                    detail += " | " + "; ".join(candidate.parse_warnings)
                if candidate.extraction_notes:
                    detail += " | note: " + "; ".join(candidate.extraction_notes)

                report.add(
                    StageResult(
                        stage=Stage.INGEST,
                        status=status,
                        candidate_id=candidate.id,
                        detail=detail,
                    )
                )
            except Exception as exc:  # a genuinely unexpected failure
                log.exception("ingest failed for %s", path)
                self.store.audit("ingest", "error", job_id=job.id, source_file=str(path), error=str(exc))
                report.add(
                    StageResult(
                        stage=Stage.INGEST,
                        status=StageStatus.FAILED,
                        detail=f"{path.name}: unexpected error — {exc}",
                    )
                )

        return candidates

    # -- stage 2: rank ----------------------------------------------------

    def rank(self, job: JobRequisition, report: RunReport) -> Shortlist:
        candidates = self.store.list_candidates(job.id)
        if not candidates:
            report.add(
                StageResult(
                    stage=Stage.RANK,
                    status=StageStatus.FAILED,
                    detail="no candidates ingested for this job",
                )
            )
            return Shortlist(job_id=job.id)

        scores = []
        for candidate in candidates:
            try:
                score = score_candidate(candidate, job, self.llm, self.settings)
                scores.append(score)
                self.store.save_candidate(candidate)  # persists redacted_text
                self.store.audit(
                    "rank",
                    "candidate_scored",
                    entity_id=candidate.id,
                    job_id=job.id,
                    model=self.model_name if score.llm_available else "",
                    gate_score=score.gate_score,
                    fit_score=score.fit_score,
                    total=score.total_score,
                    redacted=score.redacted,
                    gates_failed=[g.requirement for g in score.hard_gates if not g.met],
                    flags=score.flags,
                )
            except Exception as exc:
                log.exception("ranking failed for %s", candidate.id)
                report.add(
                    StageResult(
                        stage=Stage.RANK,
                        status=StageStatus.FAILED,
                        candidate_id=candidate.id,
                        detail=f"{candidate.display_name}: {exc}",
                    )
                )

        shortlist = build_shortlist(scores, job)
        for score in (*shortlist.ranked, *shortlist.cut):
            self.store.save_score(score)
        self.store.save_shortlist(shortlist)

        self.store.audit(
            "rank",
            "shortlist_built",
            job_id=job.id,
            shortlisted=[s.candidate_id for s in shortlist.ranked],
            cut=len(shortlist.cut),
            threshold=job.shortlist_threshold,
        )

        report.add(
            StageResult(
                stage=Stage.RANK,
                status=StageStatus.NEEDS_HUMAN,
                detail=(
                    f"{len(shortlist.ranked)} shortlisted, {len(shortlist.cut)} not — "
                    "awaiting human approval before anyone is contacted"
                ),
                data={"shortlisted": len(shortlist.ranked), "cut": len(shortlist.cut)},
            )
        )
        return shortlist

    # -- the human gate ---------------------------------------------------

    def approve(self, job_id: str, approved_by: str, notes: str = "") -> int:
        """Record a human's approval. Nothing downstream runs without this."""
        return approve_shortlist(self.store, job_id, approved_by, notes)

    # -- stage 3: schedule ------------------------------------------------

    def schedule(
        self,
        job: JobRequisition,
        report: RunReport,
        window: tuple[datetime, datetime] | None = None,
    ) -> list[InterviewBooking]:
        try:
            require_shortlist_approval(self.store, job.id)
        except ApprovalRequired as exc:
            report.add(
                StageResult(stage=Stage.SCHEDULE, status=StageStatus.NEEDS_HUMAN, detail=str(exc))
            )
            report.awaiting = "shortlist approval"
            return []

        shortlist = self.store.get_shortlist(job.id)
        if not shortlist or not shortlist.ranked:
            report.add(
                StageResult(
                    stage=Stage.SCHEDULE,
                    status=StageStatus.SKIPPED,
                    detail="shortlist is empty — nobody to schedule",
                )
            )
            return []

        if not job.interview_panel:
            report.add(
                StageResult(
                    stage=Stage.SCHEDULE,
                    status=StageStatus.NEEDS_HUMAN,
                    detail=(
                        "no interview panel defined on the requisition — "
                        "add `interview_panel` before scheduling"
                    ),
                )
            )
            return []

        window_start, window_end = window or default_window()
        bookings: list[InterviewBooking] = []

        for score in shortlist.ranked:
            candidate = self.store.get_candidate(score.candidate_id)
            if candidate is None:
                continue

            if self.store.interviews_for_candidate(candidate.id):
                report.add(
                    StageResult(
                        stage=Stage.SCHEDULE,
                        status=StageStatus.SKIPPED,
                        candidate_id=candidate.id,
                        detail=f"{candidate.display_name} already has an interview booked",
                    )
                )
                continue

            try:
                slot = find_slot(
                    self.calendar,
                    job.interview_panel,
                    window_start,
                    window_end,
                    job.interview_minutes,
                )
            except NoSlotAvailable as exc:
                self.store.audit(
                    "schedule", "no_slot", entity_id=candidate.id, job_id=job.id, reason=str(exc)
                )
                report.add(
                    StageResult(
                        stage=Stage.SCHEDULE,
                        status=StageStatus.NEEDS_HUMAN,
                        candidate_id=candidate.id,
                        detail=(
                            f"{candidate.display_name}: {exc}. Candidate is still active — "
                            "widen the window or change the panel."
                        ),
                    )
                )
                continue

            try:
                attendees = [p.email for p in job.interview_panel]
                if candidate.email:
                    attendees.append(candidate.email)

                event_id = self.calendar.book(
                    slot,
                    attendees,
                    f"Interview: {candidate.display_name} — {job.title}",
                    f"Interview for {job.title}.",
                )

                booking = InterviewBooking(
                    candidate_id=candidate.id,
                    job_id=job.id,
                    slot=slot,
                    interviewers=list(job.interview_panel),
                    location=job.location or "Video call",
                    note=f"calendar event {event_id}",
                )

                questions = self.store.get_questions(candidate.id, job.id)
                ics_path = write_ics(
                    booking,
                    self.settings.out_dir / "invites",
                    candidate.display_name,
                    candidate.email,
                    job.title,
                    company=self.settings.company_name,
                    agenda=format_agenda(questions) if questions else "",
                )
                booking.ics_path = str(ics_path)

                self.store.save_interview(booking)
                self.store.audit(
                    "schedule",
                    "interview_booked",
                    entity_id=candidate.id,
                    job_id=job.id,
                    interview_id=booking.id,
                    slot=str(slot),
                    panel=[p.email for p in job.interview_panel],
                    ics=str(ics_path),
                )
                bookings.append(booking)

                report.add(
                    StageResult(
                        stage=Stage.SCHEDULE,
                        status=StageStatus.OK,
                        candidate_id=candidate.id,
                        detail=f"{candidate.display_name}: {slot} — invite at {ics_path.name}",
                    )
                )
            except Exception as exc:
                log.exception("scheduling failed for %s", candidate.id)
                report.add(
                    StageResult(
                        stage=Stage.SCHEDULE,
                        status=StageStatus.FAILED,
                        candidate_id=candidate.id,
                        detail=f"{candidate.display_name}: {exc}",
                    )
                )

        return bookings

    # -- stage 4: questions -----------------------------------------------

    def questions(self, job: JobRequisition, report: RunReport) -> list[QuestionSet]:
        try:
            require_shortlist_approval(self.store, job.id)
        except ApprovalRequired as exc:
            report.add(
                StageResult(stage=Stage.QUESTIONS, status=StageStatus.NEEDS_HUMAN, detail=str(exc))
            )
            report.awaiting = "shortlist approval"
            return []

        shortlist = self.store.get_shortlist(job.id)
        if not shortlist or not shortlist.ranked:
            report.add(
                StageResult(
                    stage=Stage.QUESTIONS, status=StageStatus.SKIPPED, detail="shortlist is empty"
                )
            )
            return []

        sets: list[QuestionSet] = []
        for score in shortlist.ranked:
            candidate = self.store.get_candidate(score.candidate_id)
            if candidate is None:
                continue
            try:
                question_set = generate_questions(candidate, job, score, self.llm)
                self.store.save_questions(question_set)
                self.store.audit(
                    "questions",
                    "generated",
                    entity_id=candidate.id,
                    job_id=job.id,
                    model=self.model_name if question_set.llm_available else "",
                    count=len(question_set.questions),
                    categories=[q.category.value for q in question_set.questions],
                )
                sets.append(question_set)
                report.add(
                    StageResult(
                        stage=Stage.QUESTIONS,
                        status=StageStatus.OK,
                        candidate_id=candidate.id,
                        detail=(
                            f"{candidate.display_name}: {len(question_set.questions)} questions"
                            + ("" if question_set.llm_available else " (template fallback)")
                        ),
                    )
                )
            except Exception as exc:
                log.exception("question generation failed for %s", candidate.id)
                report.add(
                    StageResult(
                        stage=Stage.QUESTIONS,
                        status=StageStatus.FAILED,
                        candidate_id=candidate.id,
                        detail=f"{candidate.display_name}: {exc}",
                    )
                )
        return sets

    # -- stage 5: summarize -----------------------------------------------

    def summarize(
        self,
        job: JobRequisition,
        candidate_id: str,
        transcript: str | Path,
        report: RunReport,
        interview_id: str = "",
    ) -> InterviewSummary | None:
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            report.add(
                StageResult(
                    stage=Stage.SUMMARIZE,
                    status=StageStatus.FAILED,
                    detail=f"unknown candidate '{candidate_id}'",
                )
            )
            return None

        if not interview_id:
            booked = self.store.interviews_for_candidate(candidate_id)
            interview_id = booked[-1].id if booked else "unscheduled"

        try:
            summary = summarize_interview(
                transcript,
                job,
                candidate_id=candidate_id,
                interview_id=interview_id,
                question_set=self.store.get_questions(candidate_id, job.id),
                llm=self.llm,
            )
            self.store.save_summary(summary)

            booking = self.store.get_interview(interview_id)
            if booking:
                booking.status = "completed"
                self.store.save_interview(booking)

            self.store.audit(
                "summarize",
                "interview_summarized",
                entity_id=candidate_id,
                job_id=job.id,
                model=self.model_name if summary.llm_available else "",
                interview_id=interview_id,
                signals={s.dimension: s.rating.value for s in summary.signals},
                concerns=len(summary.concerns),
            )
            report.add(
                StageResult(
                    stage=Stage.SUMMARIZE,
                    status=StageStatus.OK,
                    candidate_id=candidate_id,
                    detail=(
                        f"{candidate.display_name}: {len(summary.signals)} signals, "
                        f"{len(summary.concerns)} concerns"
                        + ("" if summary.llm_available else " (no model — mechanical extract)")
                    ),
                )
            )
            return summary
        except Exception as exc:
            log.exception("summarization failed for %s", candidate_id)
            report.add(
                StageResult(
                    stage=Stage.SUMMARIZE,
                    status=StageStatus.FAILED,
                    candidate_id=candidate_id,
                    detail=f"{candidate.display_name}: {exc}",
                )
            )
            return None

    # -- stage 6: recommend -----------------------------------------------

    def recommend(self, job: JobRequisition, report: RunReport) -> list[Recommendation]:
        shortlist = self.store.get_shortlist(job.id)
        if not shortlist or not shortlist.ranked:
            report.add(
                StageResult(
                    stage=Stage.RECOMMEND, status=StageStatus.SKIPPED, detail="shortlist is empty"
                )
            )
            return []

        recommendations: list[Recommendation] = []
        for score in shortlist.ranked:
            candidate = self.store.get_candidate(score.candidate_id)
            if candidate is None:
                continue
            try:
                summaries = self.store.list_summaries(candidate.id)
                rec = recommend(candidate, job, score, summaries, self.llm)
                self.store.save_recommendation(rec)
                self.store.audit(
                    "recommend",
                    "recommendation_generated",
                    entity_id=candidate.id,
                    job_id=job.id,
                    model=self.model_name if rec.llm_available else "",
                    verdict=rec.verdict.value,
                    confidence=rec.confidence,
                    interviews_considered=len(summaries),
                )
                recommendations.append(rec)
                report.add(
                    StageResult(
                        stage=Stage.RECOMMEND,
                        status=StageStatus.NEEDS_HUMAN,
                        candidate_id=candidate.id,
                        detail=(
                            f"{candidate.display_name}: {rec.verdict.value} "
                            f"(confidence {rec.confidence:.0%}) — human decision required"
                        ),
                    )
                )
            except Exception as exc:
                log.exception("recommendation failed for %s", candidate.id)
                report.add(
                    StageResult(
                        stage=Stage.RECOMMEND,
                        status=StageStatus.FAILED,
                        candidate_id=candidate.id,
                        detail=f"{candidate.display_name}: {exc}",
                    )
                )
        return recommendations

    # -- the whole thing --------------------------------------------------

    def run(
        self,
        job_path: Path | str,
        resumes_dir: Path | str,
        approve_as: str = "",
        transcripts_dir: Path | str | None = None,
    ) -> RunReport:
        """Ingest → rank → (gate) → schedule → questions → summarize → recommend.

        Without `approve_as`, the run stops after ranking and reports what it is
        waiting for. Passing `approve_as` is a named human approving the
        shortlist up front; it is recorded as their decision, not the agent's.
        """
        job = load_job(job_path)
        report = RunReport(job_id=job.id)
        self.register_job(job)

        self.ingest(job, resumes_dir, report)
        self.rank(job, report)

        if approve_as:
            try:
                count = self.approve(job.id, approve_as, notes="approved at run time via CLI")
                report.add(
                    StageResult(
                        stage=Stage.RANK,
                        status=StageStatus.OK,
                        detail=f"shortlist of {count} approved by {approve_as}",
                    )
                )
            except (ApprovalRequired, ValueError) as exc:
                report.add(
                    StageResult(stage=Stage.RANK, status=StageStatus.FAILED, detail=str(exc))
                )

        if not shortlist_is_approved(self.store, job.id):
            report.awaiting = (
                "human approval of the shortlist — nobody is contacted until then. "
                f"Run: python -m recruiter approve --job {job.id} --by \"Your Name\""
            )
            report.finished_at = utcnow()
            return report

        # Questions first so the invite can carry the agenda.
        self.questions(job, report)
        self.schedule(job, report)

        if transcripts_dir:
            self._summarize_folder(job, Path(transcripts_dir), report)

        self.recommend(job, report)

        report.awaiting = (
            "the hiring manager's decision — the agent does not advance or reject anyone"
        )
        report.finished_at = utcnow()
        return report

    def _summarize_folder(self, job: JobRequisition, folder: Path, report: RunReport) -> None:
        """Match transcript files to candidates by id or by name in the filename."""
        if not folder.is_dir():
            report.add(
                StageResult(
                    stage=Stage.SUMMARIZE,
                    status=StageStatus.SKIPPED,
                    detail=f"transcript folder not found: {folder}",
                )
            )
            return

        candidates = self.store.list_candidates(job.id)
        for path in sorted(folder.glob("*.txt")):
            stem = path.stem.lower()
            match = next(
                (
                    c
                    for c in candidates
                    if c.id.lower() in stem
                    or (c.full_name and c.full_name.split()[0].lower() in stem)
                ),
                None,
            )
            if match is None:
                report.add(
                    StageResult(
                        stage=Stage.SUMMARIZE,
                        status=StageStatus.NEEDS_HUMAN,
                        detail=f"{path.name}: could not match to a candidate — assign it manually",
                    )
                )
                continue
            self.summarize(job, match.id, path, report)

    def close(self) -> None:
        self.store.close()
