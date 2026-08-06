"""ATS tracking database — SQLite, stdlib only.

Every table stores the full Pydantic model as JSON in `payload` plus a few
promoted columns for querying. That keeps the schema stable while the models
evolve, which matters on a 3-week project.

`audit` is append-only and is the reason any past decision can be reconstructed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import (
    Candidate,
    CandidateScore,
    HumanDecision,
    InterviewBooking,
    InterviewSummary,
    JobRequisition,
    QuestionSet,
    Recommendation,
    Shortlist,
    utcnow,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id          TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL,
    full_name   TEXT,
    email       TEXT,
    source_file TEXT,
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_job ON candidates(job_id);

CREATE TABLE IF NOT EXISTS scores (
    candidate_id TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    total_score  REAL NOT NULL,
    shortlisted  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (candidate_id, job_id)
);

CREATE TABLE IF NOT EXISTS shortlists (
    job_id      TEXT PRIMARY KEY,
    approved_by TEXT DEFAULT '',
    approved_at TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interviews (
    id           TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    starts_at    TEXT,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interviews_job ON interviews(job_id);

CREATE TABLE IF NOT EXISTS question_sets (
    candidate_id TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (candidate_id, job_id)
);

CREATE TABLE IF NOT EXISTS summaries (
    candidate_id TEXT NOT NULL,
    interview_id TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (candidate_id, interview_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    candidate_id TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    verdict      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (candidate_id, job_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id           TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    job_id       TEXT NOT NULL,
    action       TEXT NOT NULL,
    decided_by   TEXT NOT NULL,
    decided_at   TEXT NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_job ON decisions(job_id);

CREATE TABLE IF NOT EXISTS audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    stage     TEXT NOT NULL,
    action    TEXT NOT NULL,
    entity_id TEXT DEFAULT '',
    job_id    TEXT DEFAULT '',
    model     TEXT DEFAULT '',
    detail    TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit(entity_id);
"""


class ATSStore:
    """Thin persistence layer. Stages never touch it — the pipeline does."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # `check_same_thread=False` because Streamlit re-runs the script on a
        # different worker thread for every interaction, while the agent (and
        # therefore this connection) is cached across those re-runs. Without it,
        # the first button click raises ProgrammingError.
        #
        # Turning that check off makes thread-safety our problem, so every
        # statement goes through `self._lock`. Access is serialised rather than
        # concurrent, which is the right trade for an ATS: writes are small and
        # a recruiter dashboard has no throughput requirement worth the risk of
        # interleaved transactions.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ATSStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Serialised write transaction. Holds the lock so a rollback can never
        race another thread's statements on the same connection."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Serialised read. Reads share the connection, so they take the lock too."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # -- jobs -------------------------------------------------------------

    def save_job(self, job: JobRequisition) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO jobs (id, title, created_at, payload) VALUES (?,?,?,?)",
                (job.id, job.title, utcnow(), job.model_dump_json()),
            )

    def get_job(self, job_id: str) -> JobRequisition | None:
        row = self._query_one("SELECT payload FROM jobs WHERE id=?", (job_id,))
        return JobRequisition.model_validate_json(row["payload"]) if row else None

    def list_jobs(self) -> list[JobRequisition]:
        rows = self._query("SELECT payload FROM jobs ORDER BY created_at DESC")
        return [JobRequisition.model_validate_json(r["payload"]) for r in rows]

    # -- candidates -------------------------------------------------------

    def save_candidate(self, cand: Candidate) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO candidates "
                "(id, job_id, full_name, email, source_file, created_at, payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    cand.id,
                    cand.job_id,
                    cand.full_name,
                    cand.email,
                    cand.source_file,
                    cand.ingested_at,
                    cand.model_dump_json(),
                ),
            )

    def get_candidate(self, candidate_id: str) -> Candidate | None:
        row = self._query_one(
            "SELECT payload FROM candidates WHERE id=?", (candidate_id,)
        )
        return Candidate.model_validate_json(row["payload"]) if row else None

    def list_candidates(self, job_id: str) -> list[Candidate]:
        rows = self._query(
            "SELECT payload FROM candidates WHERE job_id=? ORDER BY created_at", (job_id,)
        )
        return [Candidate.model_validate_json(r["payload"]) for r in rows]

    def find_candidate_by_source(self, job_id: str, source_file: str) -> Candidate | None:
        row = self._query_one(
            "SELECT payload FROM candidates WHERE job_id=? AND source_file=?",
            (job_id, source_file),
        )
        return Candidate.model_validate_json(row["payload"]) if row else None

    # -- scores & shortlist ----------------------------------------------

    def save_score(self, score: CandidateScore) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO scores "
                "(candidate_id, job_id, total_score, shortlisted, created_at, payload) "
                "VALUES (?,?,?,?,?,?)",
                (
                    score.candidate_id,
                    score.job_id,
                    score.total_score,
                    int(score.shortlisted),
                    score.scored_at,
                    score.model_dump_json(),
                ),
            )

    def get_score(self, candidate_id: str, job_id: str) -> CandidateScore | None:
        row = self._query_one(
            "SELECT payload FROM scores WHERE candidate_id=? AND job_id=?",
            (candidate_id, job_id),
        )
        return CandidateScore.model_validate_json(row["payload"]) if row else None

    def list_scores(self, job_id: str) -> list[CandidateScore]:
        rows = self._query(
            "SELECT payload FROM scores WHERE job_id=? ORDER BY total_score DESC", (job_id,)
        )
        return [CandidateScore.model_validate_json(r["payload"]) for r in rows]

    def save_shortlist(self, shortlist: Shortlist) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO shortlists "
                "(job_id, approved_by, approved_at, created_at, payload) VALUES (?,?,?,?,?)",
                (
                    shortlist.job_id,
                    shortlist.approved_by,
                    shortlist.approved_at,
                    shortlist.generated_at,
                    shortlist.model_dump_json(),
                ),
            )

    def get_shortlist(self, job_id: str) -> Shortlist | None:
        row = self._query_one(
            "SELECT payload FROM shortlists WHERE job_id=?", (job_id,)
        )
        return Shortlist.model_validate_json(row["payload"]) if row else None

    # -- interviews -------------------------------------------------------

    def save_interview(self, booking: InterviewBooking) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO interviews "
                "(id, candidate_id, job_id, starts_at, status, created_at, payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    booking.id,
                    booking.candidate_id,
                    booking.job_id,
                    booking.slot.start.isoformat(),
                    booking.status,
                    booking.created_at,
                    booking.model_dump_json(),
                ),
            )

    def get_interview(self, interview_id: str) -> InterviewBooking | None:
        row = self._query_one(
            "SELECT payload FROM interviews WHERE id=?", (interview_id,)
        )
        return InterviewBooking.model_validate_json(row["payload"]) if row else None

    def list_interviews(self, job_id: str) -> list[InterviewBooking]:
        rows = self._query(
            "SELECT payload FROM interviews WHERE job_id=? ORDER BY starts_at", (job_id,)
        )
        return [InterviewBooking.model_validate_json(r["payload"]) for r in rows]

    def interviews_for_candidate(self, candidate_id: str) -> list[InterviewBooking]:
        rows = self._query(
            "SELECT payload FROM interviews WHERE candidate_id=? ORDER BY starts_at",
            (candidate_id,),
        )
        return [InterviewBooking.model_validate_json(r["payload"]) for r in rows]

    # -- questions / summaries / recommendations --------------------------

    def save_questions(self, qs: QuestionSet) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO question_sets "
                "(candidate_id, job_id, created_at, payload) VALUES (?,?,?,?)",
                (qs.candidate_id, qs.job_id, qs.generated_at, qs.model_dump_json()),
            )

    def get_questions(self, candidate_id: str, job_id: str) -> QuestionSet | None:
        row = self._query_one(
            "SELECT payload FROM question_sets WHERE candidate_id=? AND job_id=?",
            (candidate_id, job_id),
        )
        return QuestionSet.model_validate_json(row["payload"]) if row else None

    def save_summary(self, summary: InterviewSummary) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO summaries "
                "(candidate_id, interview_id, created_at, payload) VALUES (?,?,?,?)",
                (
                    summary.candidate_id,
                    summary.interview_id,
                    summary.generated_at,
                    summary.model_dump_json(),
                ),
            )

    def get_summary(self, candidate_id: str, interview_id: str) -> InterviewSummary | None:
        row = self._query_one(
            "SELECT payload FROM summaries WHERE candidate_id=? AND interview_id=?",
            (candidate_id, interview_id),
        )
        return InterviewSummary.model_validate_json(row["payload"]) if row else None

    def list_summaries(self, candidate_id: str) -> list[InterviewSummary]:
        rows = self._query(
            "SELECT payload FROM summaries WHERE candidate_id=? ORDER BY created_at",
            (candidate_id,),
        )
        return [InterviewSummary.model_validate_json(r["payload"]) for r in rows]

    def save_recommendation(self, rec: Recommendation) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO recommendations "
                "(candidate_id, job_id, verdict, created_at, payload) VALUES (?,?,?,?,?)",
                (
                    rec.candidate_id,
                    rec.job_id,
                    rec.verdict.value,
                    rec.generated_at,
                    rec.model_dump_json(),
                ),
            )

    def get_recommendation(self, candidate_id: str, job_id: str) -> Recommendation | None:
        row = self._query_one(
            "SELECT payload FROM recommendations WHERE candidate_id=? AND job_id=?",
            (candidate_id, job_id),
        )
        return Recommendation.model_validate_json(row["payload"]) if row else None

    def list_recommendations(self, job_id: str) -> list[Recommendation]:
        rows = self._query(
            "SELECT payload FROM recommendations WHERE job_id=? ORDER BY created_at",
            (job_id,),
        )
        return [Recommendation.model_validate_json(r["payload"]) for r in rows]

    # -- human decisions (the gates) --------------------------------------

    def save_decision(self, decision: HumanDecision) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO decisions "
                "(id, candidate_id, job_id, action, decided_by, decided_at, payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    decision.id,
                    decision.candidate_id,
                    decision.job_id,
                    decision.action.value,
                    decision.decided_by,
                    decision.decided_at,
                    decision.model_dump_json(),
                ),
            )

    def list_decisions(self, job_id: str) -> list[HumanDecision]:
        rows = self._query(
            "SELECT payload FROM decisions WHERE job_id=? ORDER BY decided_at", (job_id,)
        )
        return [HumanDecision.model_validate_json(r["payload"]) for r in rows]

    def decisions_for_candidate(self, candidate_id: str) -> list[HumanDecision]:
        rows = self._query(
            "SELECT payload FROM decisions WHERE candidate_id=? ORDER BY decided_at",
            (candidate_id,),
        )
        return [HumanDecision.model_validate_json(r["payload"]) for r in rows]

    # -- audit ------------------------------------------------------------

    def audit(
        self,
        stage: str,
        action: str,
        /,
        *,
        entity_id: str = "",
        job_id: str = "",
        model: str = "",
        **detail: Any,
    ) -> None:
        """Append an audit row. `stage` and `action` are positional-only so that
        a detail key of the same name (`action="advance"`, say) lands in the
        payload instead of colliding with the parameter."""
        with self._tx() as c:
            c.execute(
                "INSERT INTO audit (at, stage, action, entity_id, job_id, model, detail) "
                "VALUES (?,?,?,?,?,?,?)",
                (utcnow(), stage, action, entity_id, job_id, model, json.dumps(detail, default=str)),
            )

    def audit_trail(self, entity_id: str = "", job_id: str = "") -> list[dict]:
        if entity_id:
            rows = self._query(
                "SELECT * FROM audit WHERE entity_id=? ORDER BY id", (entity_id,)
            )
        elif job_id:
            rows = self._query(
                "SELECT * FROM audit WHERE job_id=? ORDER BY id", (job_id,)
            )
        else:
            rows = self._query("SELECT * FROM audit ORDER BY id")

        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detail"] = json.loads(d["detail"])
            except (json.JSONDecodeError, TypeError):
                d["detail"] = {}
            out.append(d)
        return out
