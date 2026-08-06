"""The store is used from more than one thread.

Streamlit re-runs the script on a different worker thread for every interaction
while the agent — and therefore its database connection — is cached across those
re-runs. A stdlib SQLite connection refuses that by default, so the first button
click in the dashboard raised:

    sqlite3.ProgrammingError: SQLite objects created in a thread can only be
    used in that same thread.

Found by clicking the button, not by the test suite. These tests exist so it is
found by the test suite next time.
"""

from __future__ import annotations

import threading
from pathlib import Path

from recruiter.models import Candidate, JobRequisition
from recruiter.store import ATSStore


def _run_in_new_thread(fn) -> list:
    """Run `fn` on a separate thread and re-raise anything it throws here."""
    box: list = []

    def target() -> None:
        try:
            box.append(fn())
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=15)
    assert not thread.is_alive(), "worker thread hung"

    result = box[0] if box else None
    if isinstance(result, BaseException):
        raise result
    return result


def test_store_can_be_written_from_another_thread(tmp_path: Path) -> None:
    """The exact failure from the dashboard: create here, write there."""
    store = ATSStore(tmp_path / "ats.db")
    job = JobRequisition(title="Senior Python Engineer")

    _run_in_new_thread(lambda: store.save_job(job))

    assert store.get_job(job.id) is not None
    store.close()


def test_store_can_be_read_from_another_thread(tmp_path: Path) -> None:
    store = ATSStore(tmp_path / "ats.db")
    job = JobRequisition(title="Senior Python Engineer")
    store.save_job(job)

    fetched = _run_in_new_thread(lambda: store.get_job(job.id))

    assert fetched is not None
    assert fetched.title == "Senior Python Engineer"
    store.close()


def test_audit_trail_is_readable_across_threads(tmp_path: Path) -> None:
    store = ATSStore(tmp_path / "ats.db")
    store.audit("rank", "candidate_scored", entity_id="cand_x", job_id="job_y")

    rows = _run_in_new_thread(lambda: store.audit_trail(job_id="job_y"))

    assert rows and rows[0]["action"] == "candidate_scored"
    store.close()


def test_concurrent_writes_do_not_corrupt_or_lose_rows(tmp_path: Path) -> None:
    """Serialised access should mean every write lands, none interleave."""
    store = ATSStore(tmp_path / "ats.db")
    job = JobRequisition(title="Senior Python Engineer")
    store.save_job(job)

    errors: list[BaseException] = []

    def writer(index: int) -> None:
        try:
            store.save_candidate(
                Candidate(job_id=job.id, full_name=f"Candidate {index}", raw_text="x" * 200)
            )
            store.audit("ingest", "candidate_parsed", job_id=job.id, index=index)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, f"concurrent writes raised: {errors[0]!r}"
    assert len(store.list_candidates(job.id)) == 12
    assert len(store.audit_trail(job_id=job.id)) == 12
    store.close()
