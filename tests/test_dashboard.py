"""Headless UI tests for the Streamlit dashboard.

The dashboard was the least-tested surface in the project and it showed: the
first button click in a real browser raised a SQLite cross-thread error that
none of the unit tests could see, because the pipeline is normally driven from
a single thread by the CLI.

Streamlit's AppTest runs the real script and exercises real widgets, so these
catch what unit tests structurally cannot — a widget that raises on click, a
tab that breaks on empty data, a deprecated API that has passed its removal
date.

Every test runs against a temporary database. None of them touch the demo data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from recruiter.models import HumanAction
from recruiter.pipeline import RecruitmentAgent, load_job
from recruiter.store import ATSStore

from .conftest import SAMPLE_JOB, SAMPLE_RESUMES, OfflineLLM

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
TIMEOUT = 120


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app at a scratch database and no API key.

    `@st.cache_resource` outlives an individual AppTest run, so without clearing
    it the agent — and the database connection inside it — leaks from one test
    into the next and writes land in the previous test's file.
    """
    import streamlit as st

    st.cache_resource.clear()

    db = tmp_path / "ats.db"
    monkeypatch.setenv("RECRUITER_DB_PATH", str(db))
    monkeypatch.setenv("RECRUITER_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("GROQ_API_KEY", "")
    yield db
    st.cache_resource.clear()


@pytest.fixture
def seeded(app_env: Path) -> Path:
    """A database already carrying a ranked, approved, scheduled job."""
    job = load_job(SAMPLE_JOB)
    store = ATSStore(app_env)
    agent = RecruitmentAgent(store=store, llm=OfflineLLM())
    agent.register_job(job)

    from recruiter.models import RunReport

    report = RunReport(job_id=job.id)
    agent.ingest(job, SAMPLE_RESUMES, report)
    agent.rank(job, report)
    agent.approve(job.id, "Test Approver")
    agent.questions(job, report)
    agent.schedule(job, report)
    agent.recommend(job, report)
    store.close()
    return app_env


def _run(script_path: Path = APP) -> "AppTest":
    at = AppTest.from_file(str(script_path), default_timeout=TIMEOUT)
    at.run()
    return at


# -- the app loads at all --------------------------------------------------


def test_app_loads_with_an_empty_database(app_env: Path) -> None:
    at = _run()
    assert not at.exception, at.exception
    # With nothing ingested it should say so rather than crash on missing data.
    assert any("Nothing in the database" in i.value for i in at.info)


def test_app_loads_with_seeded_data(seeded: Path) -> None:
    at = _run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Senior Python Engineer"


def test_no_deprecated_streamlit_api_is_used(seeded: Path) -> None:
    """`use_container_width` was already past its removal date in the log."""
    source = APP.read_text(encoding="utf-8")
    assert "use_container_width" not in source, (
        "use_container_width is deprecated and past its stated removal date"
    )


# -- the crash the user hit in the browser --------------------------------


def test_ingest_and_rank_button_does_not_raise(app_env: Path) -> None:
    """The exact interaction that raised sqlite3.ProgrammingError in a browser.

    AppTest re-runs the script the way the real server does, so the cached
    agent is reused across runs — which is what surfaced the threading bug.
    """
    at = _run()
    buttons = [b for b in at.sidebar.button if "Ingest and rank" in b.label]
    assert buttons, "the Ingest and rank button is missing"

    buttons[0].click().run()
    assert not at.exception, f"clicking Ingest and rank raised: {at.exception}"


def test_ingest_and_rank_actually_populates_the_store(app_env: Path) -> None:
    at = _run()
    [b for b in at.sidebar.button if "Ingest and rank" in b.label][0].click().run()
    assert not at.exception, at.exception

    store = ATSStore(app_env)
    jobs = store.list_jobs()
    assert jobs, "the button ran but stored no job"
    assert store.list_candidates(jobs[0].id), "the button ran but ingested nobody"
    store.close()


# -- every tab renders -----------------------------------------------------


def test_all_tabs_render_without_error(seeded: Path) -> None:
    at = _run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6, f"expected 6 tabs, found {len(at.tabs)}"


def _rendered(at) -> str:
    """All markdown the page emitted, concatenated.

    Asserting on content rather than on which widget produced it means the
    tests survive a visual redesign — they check the information is on the
    page, which is what actually matters.
    """
    return "\n".join(m.value for m in at.markdown)


def test_headline_counts_are_shown(seeded: Path) -> None:
    at = _run()
    page = _rendered(at)
    for label in ("Candidates", "Shortlisted", "Interviews", "Recommendations"):
        assert label in page, f"the {label} figure is missing from the page"
    assert "Shortlist approved" in page


def test_audit_trail_renders(seeded: Path) -> None:
    at = _run()
    assert not at.exception, at.exception
    assert at.dataframe, "audit trail table did not render"


# -- the human gates, through the UI --------------------------------------


def test_approval_requires_a_name_in_the_ui(app_env: Path) -> None:
    """Clicking Approve with an empty name field must error, not approve."""
    job = load_job(SAMPLE_JOB)
    store = ATSStore(app_env)
    agent = RecruitmentAgent(store=store, llm=OfflineLLM())
    agent.register_job(job)
    from recruiter.models import RunReport

    report = RunReport(job_id=job.id)
    agent.ingest(job, SAMPLE_RESUMES, report)
    agent.rank(job, report)
    store.close()

    at = _run()
    approve = [b for b in at.button if "Approve shortlist" in b.label]
    assert approve, "approval control missing while a shortlist is pending"

    approve[0].click().run()
    assert not at.exception, at.exception
    assert any("must name a person" in e.value for e in at.error), (
        "an anonymous approval was not rejected"
    )

    store = ATSStore(app_env)
    assert not (store.get_shortlist(job.id) or {}).is_approved
    store.close()


def test_downstream_controls_are_blocked_until_approval(app_env: Path) -> None:
    job = load_job(SAMPLE_JOB)
    store = ATSStore(app_env)
    agent = RecruitmentAgent(store=store, llm=OfflineLLM())
    agent.register_job(job)
    from recruiter.models import RunReport

    report = RunReport(job_id=job.id)
    agent.ingest(job, SAMPLE_RESUMES, report)
    agent.rank(job, report)
    store.close()

    at = _run()
    assert any("Blocked until the shortlist is approved" in i.value for i in at.info)


def test_no_hiring_decision_is_recorded_by_rendering_the_page(seeded: Path) -> None:
    """Opening the recommendations tab must not decide anything."""
    at = _run()
    assert not at.exception, at.exception

    store = ATSStore(seeded)
    job = store.list_jobs()[0]
    final = [
        d
        for d in store.list_decisions(job.id)
        if d.action is not HumanAction.APPROVE_SHORTLIST
    ]
    store.close()
    assert final == [], "viewing the dashboard recorded a hiring decision"


def test_recommendations_tab_states_that_it_does_not_decide(seeded: Path) -> None:
    """The advisory-only disclaimer must be on the page, however it is styled."""
    at = _run()
    page = _rendered(at).lower()
    assert "recommendations, not decisions" in page, (
        "the advisory-only statement is missing from the UI"
    )
    assert "no code path" in page


# -- degraded conditions ---------------------------------------------------


def test_missing_model_is_surfaced_in_the_sidebar(seeded: Path) -> None:
    """Running without a key must be stated, not hidden."""
    at = _run()
    assert any("No model configured" in w.value for w in at.sidebar.warning), (
        "the dashboard did not disclose that it is running without a model"
    )


def test_a_job_with_no_resumes_does_not_crash_the_page(
    app_env: Path, tmp_path: Path
) -> None:
    empty = tmp_path / "empty_resumes"
    empty.mkdir()

    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    folder_inputs = [t for t in at.sidebar.text_input if "Resume folder" in t.label]
    assert folder_inputs, "resume folder input missing"
    folder_inputs[0].set_value(str(empty)).run()

    [b for b in at.sidebar.button if "Ingest and rank" in b.label][0].click().run()
    assert not at.exception, f"an empty resume folder crashed the page: {at.exception}"


def test_a_corrupt_resume_file_does_not_crash_the_page(
    app_env: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "mixed"
    folder.mkdir()
    shutil.copy(SAMPLE_RESUMES / "03_sana_iqbal.txt", folder / "good.txt")
    (folder / "broken.pdf").write_bytes(b"this is not a pdf at all")

    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    [t for t in at.sidebar.text_input if "Resume folder" in t.label][0].set_value(
        str(folder)
    ).run()
    [b for b in at.sidebar.button if "Ingest and rank" in b.label][0].click().run()

    assert not at.exception, f"a corrupt file crashed the page: {at.exception}"

    store = ATSStore(app_env)
    jobs = store.list_jobs()
    assert jobs and store.list_candidates(jobs[0].id), (
        "the good resume should still have been ingested alongside the broken one"
    )
    store.close()
