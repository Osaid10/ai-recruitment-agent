"""ATS-style dashboard for the AI Recruitment Agent.

Run from the repo root:

    streamlit run app/streamlit_app.py

The screen exists to make the agent's reasoning inspectable: not just who ranked
where, but which evidence drove each score, who was cut and why, and where a
human has to act. The two approval controls here are the only way a candidate
moves forward.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recruiter.config import Settings  # noqa: E402
from recruiter.ingest.parser import SUPPORTED_SUFFIXES  # noqa: E402
from recruiter.models import HumanAction, Rating, RunReport, StageStatus  # noqa: E402
from recruiter.pipeline import JobSpecError, RecruitmentAgent, load_job  # noqa: E402
from recruiter.recommend import record_decision, shortlist_is_approved  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui  # noqa: E402

st.set_page_config(
    page_title="AI Recruitment Agent",
    page_icon="::",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

STAGE_MARK = {
    StageStatus.OK: ("good", "done"),
    StageStatus.FAILED: ("critical", "failed"),
    StageStatus.SKIPPED: ("neutral", "skipped"),
    StageStatus.NEEDS_HUMAN: ("warning", "needs a human"),
}

RATING_TONE = {
    Rating.STRONG: "good",
    Rating.ADEQUATE: "serious",
    Rating.WEAK: "critical",
    Rating.NOT_ASSESSED: "neutral",
}

VERDICT_TONE = {
    "strong_hire": "good",
    "hire": "good",
    "lean_hire": "serious",
    "lean_no_hire": "warning",
    "no_hire": "critical",
    "insufficient_evidence": "neutral",
}


@st.cache_resource
def get_agent() -> RecruitmentAgent:
    return RecruitmentAgent(Settings.load())


def show_report(report: RunReport) -> None:
    for result in report.results:
        tone, word = STAGE_MARK[result.status]
        st.markdown(
            ui.pill(f"{result.stage.value} · {word}", tone) + f"  {result.detail}",
            unsafe_allow_html=True,
        )
    if report.awaiting:
        ui.gate("Waiting on a person", report.awaiting)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

# Write the header before building the agent. Constructing it imports the
# LangChain stack and takes ~15s on a cold start, and @st.cache_resource means
# that cost lands on the first visitor. With the header after it the browser
# showed a blank white page for the whole wait, which reads as a crash rather
# than as loading.
st.sidebar.markdown("### AI Recruitment Agent")
st.sidebar.caption("Screens the top of the funnel. A human decides.")

with st.spinner("Starting the agent…"):
    agent = get_agent()
settings = agent.settings

if agent.llm.available:
    st.sidebar.success(f"Model: {agent.model_name}")
else:
    st.sidebar.warning(
        "No model configured — deterministic ranking only.\n\n"
        "Hard requirements, scoring against declared criteria and both human "
        "gates all still work. Fit assessment, tailored questions, interview "
        "summaries and recommendations will report themselves unavailable "
        "rather than inventing output."
    )

st.sidebar.divider()
st.sidebar.markdown("**Run the pipeline**")

job_files = sorted((ROOT / "data" / "jobs").glob("*.json"))
job_choice = st.sidebar.selectbox(
    "Job requisition", job_files, format_func=lambda p: p.stem.replace("_", " ")
)
resume_dir = st.sidebar.text_input("Resume folder", value=str(ROOT / "data" / "resumes"))
transcript_dir = st.sidebar.text_input(
    "Transcript folder (optional)", value=str(ROOT / "data" / "interviews")
)

uploads = st.sidebar.file_uploader(
    "…or drop resumes in",
    type=[s.lstrip(".") for s in sorted(SUPPORTED_SUFFIXES)],
    accept_multiple_files=True,
    help=(
        "Uploaded files are ranked instead of the folder above. They are written "
        "to a temporary directory that is deleted when the run finishes; the "
        "resume text itself is persisted to the ATS store exactly as a folder "
        "run would persist it."
    ),
)

if st.sidebar.button("Ingest and rank", type="primary", width="stretch"):
    if not job_choice:
        st.sidebar.error("No job requisition found in data/jobs/")
    else:
        try:
            job = agent.register_job(load_job(job_choice))
            report = RunReport(job_id=job.id)
            with st.spinner("Reading resumes and ranking…"):
                if uploads:
                    # Uploads arrive as in-memory buffers, but every downstream
                    # stage — parser, redaction, audit — is written against real
                    # paths. Staging them keeps the uploaded path identical to
                    # the folder path rather than forking the pipeline.
                    with tempfile.TemporaryDirectory(prefix="ra_upload_") as staged:
                        for upload in uploads:
                            name = Path(upload.name).name  # ignore any client path
                            (Path(staged) / name).write_bytes(upload.getvalue())
                        agent.ingest(job, staged, report)
                else:
                    agent.ingest(job, resume_dir, report)
                agent.rank(job, report)
            st.session_state["report"] = report.model_dump()
            st.session_state["job_id"] = job.id
        except JobSpecError as exc:
            st.sidebar.error(str(exc))

st.sidebar.divider()

jobs = agent.store.list_jobs()
if jobs:
    def _progress(job) -> tuple[int, int, int]:
        return (
            len(agent.store.list_recommendations(job.id)),
            len(agent.store.list_interviews(job.id)),
            len(agent.store.list_candidates(job.id)),
        )

    jobs = sorted(jobs, key=_progress, reverse=True)
    job_ids = [j.id for j in jobs]

    # Default to the furthest-along job, except right after a run — then show
    # the job that was just ranked. Without this the screen silently stays on
    # another requisition and the run looks like it did nothing.
    just_ran = st.session_state.get("job_id")
    default_index = job_ids.index(just_ran) if just_ran in job_ids else 0

    job_id = st.sidebar.selectbox(
        "Viewing",
        job_ids,
        format_func=lambda jid: next(j.title for j in jobs if j.id == jid),
        index=default_index,
    )
else:
    job_id = ""

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if not job_id:
    ui.eyebrow("Nothing loaded")
    st.title("AI Recruitment Agent")
    ui.meta(
        "Reads resumes, ranks candidates against a role, books interviews, drafts "
        "tailored questions, summarises the interviews and produces a "
        "recommendation — then stops, because the hiring decision is a person's."
    )
    st.info(
        "Nothing in the database yet. Pick a job requisition in the sidebar and "
        "press **Ingest and rank** to start."
    )
    st.stop()

job = agent.store.get_job(job_id)
candidates = {c.id: c for c in agent.store.list_candidates(job_id)}
shortlist = agent.store.get_shortlist(job_id)
approved = shortlist_is_approved(agent.store, job_id)
interviews = agent.store.list_interviews(job_id)
recommendations = agent.store.list_recommendations(job_id)

ui.eyebrow(f"{job.department or 'Open role'} · {job.employment_type}")
st.title(job.title)
ui.meta(
    f"{job.location}  ·  must have: "
    f"{', '.join(r.skill for r in job.must_haves) or 'none declared'}"
    f"  ·  shortlist of {job.shortlist_size} at {job.shortlist_threshold:g}+"
)

ui.stat_tiles(
    [
        ("Candidates", str(len(candidates)), "resumes ingested"),
        (
            "Shortlisted",
            str(len(shortlist.ranked) if shortlist else 0),
            f"{len(shortlist.cut) if shortlist else 0} not advanced",
        ),
        ("Interviews", str(len(interviews)), "slots booked"),
        ("Recommendations", str(len(recommendations)), "advisory only"),
        (
            "Shortlist approved",
            "Yes" if approved else "No",
            shortlist.approved_by if approved and shortlist else "blocks scheduling",
        ),
    ]
)

tabs = st.tabs(
    ["Ranking", "Human gate", "Interviews", "Questions", "Recommendations", "Audit trail"]
)

# -- Ranking ---------------------------------------------------------------

with tabs[0]:
    if "report" in st.session_state:
        with st.expander("Last run", expanded=False):
            show_report(RunReport.model_validate(st.session_state["report"]))

    if not shortlist:
        st.info("Nothing ranked yet.")
    else:
        ui.eyebrow(f"Shortlisted — {len(shortlist.ranked)}")
        ui.note(
            "Hard requirements are scored by rules, not by the model. The model "
            "scores only the rubric dimensions, from redacted text, and every "
            "score has to quote the resume — a quote that cannot be found in the "
            "source is discarded and the score zeroed."
        )
        st.write("")

        for score in shortlist.ranked:
            candidate = candidates.get(score.candidate_id)
            name = candidate.display_name if candidate else score.candidate_id
            with st.expander(f"**{name}** · {score.total_score:.0f}/100", expanded=False):
                left, right = st.columns([1, 2])

                with left:
                    ui.eyebrow("Total")
                    ui.headline_score(score.total_score)
                    st.write("")
                    ui.meters(
                        [
                            ("Rules", score.gate_score, 100),
                            ("Model fit", score.fit_score, 100),
                        ]
                    )
                    ui.note(
                        "Rules and model blended 55/45."
                        if score.llm_available
                        else "No model — the rules score is the whole score."
                    )

                with right:
                    ui.eyebrow("Hard requirements")
                    ui.pills(
                        [
                            (g.requirement.replace("must-have: ", ""), "good" if g.met else "critical")
                            for g in score.hard_gates
                        ]
                    )
                    for g in score.hard_gates:
                        if not g.met:
                            ui.note(f"✗ {g.requirement} — {g.detail}")

                if score.dimensions and score.llm_available:
                    st.write("")
                    ui.eyebrow("Rubric dimensions — scored on redacted text")
                    ui.meters([(d.dimension, d.score, 10) for d in score.dimensions])
                    for dim in score.dimensions:
                        ui.quote(dim.evidence, f"{dim.dimension} — evidence")

                if score.strengths:
                    ui.eyebrow("Strengths")
                    for item in score.strengths:
                        st.markdown(f"- {item}")
                if score.concerns:
                    ui.eyebrow("Concerns")
                    for item in score.concerns:
                        st.markdown(f"- {item}")
                if score.flags:
                    ui.eyebrow("Flagged for a human")
                    for item in score.flags:
                        ui.flag(item)

                ui.note(
                    f"Scored on {'redacted' if score.redacted else 'UNREDACTED'} text · "
                    f"{score.reason}"
                )

        ui.rule()
        ui.eyebrow(f"Not shortlisted — {len(shortlist.cut)}")
        ui.note(
            "Nobody here has been rejected. Each row records why they were not "
            "advanced; the decision is still a person's."
        )
        st.write("")

        for score in shortlist.cut:
            candidate = candidates.get(score.candidate_id)
            name = candidate.display_name if candidate else score.candidate_id
            met = sum(1 for g in score.hard_gates if g.met)
            tone = "warning" if score.all_gates_met else "neutral"
            with st.expander(f"{name} · {score.total_score:.0f}/100"):
                ui.pills([(f"{met}/{len(score.hard_gates)} requirements met", tone)])
                ui.meta(score.reason)
                st.write("")
                for gate_result in score.hard_gates:
                    if not gate_result.met:
                        ui.note(f"✗ {gate_result.requirement} — {gate_result.detail}")
                for item in score.flags:
                    ui.flag(item)

# -- Human gate ------------------------------------------------------------

with tabs[1]:
    ui.eyebrow("Gate 1 of 2")
    st.markdown("#### Approve the shortlist")
    ui.note(
        "Nothing downstream runs until a named person approves. Scheduling and "
        "question generation refuse to execute without this — the guard raises, "
        "it does not warn."
    )
    st.write("")

    if approved:
        ui.gate(
            "Approved",
            f"{shortlist.approved_by} approved this shortlist at {shortlist.approved_at}. "
            "Scheduling is unlocked.",
            done=True,
        )
    elif not shortlist or not shortlist.ranked:
        st.info("Rank some candidates first.")
    else:
        ui.gate(
            "Awaiting approval",
            f"{len(shortlist.ranked)} candidates are waiting. Nobody has been "
            "contacted and nothing has been scheduled.",
        )
        approver = st.text_input("Your name", key="approver")
        notes = st.text_area("Notes (optional)", key="approval_notes")
        if st.button("Approve shortlist", type="primary"):
            if not approver.strip():
                st.error("Approvals must name a person.")
            else:
                count = agent.approve(job_id, approver, notes)
                st.success(f"{count} candidates approved. Scheduling is unlocked.")
                st.rerun()

    ui.rule()
    st.markdown("#### Run the rest of the pipeline")

    if not approved:
        st.info("Blocked until the shortlist is approved.")
    else:
        days = st.slider("Scheduling window (days ahead)", 3, 30, 10)
        if st.button("Generate questions, schedule interviews, recommend"):
            report = RunReport(job_id=job_id)
            start = (datetime.now() + timedelta(days=1)).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            with st.spinner("Working…"):
                agent.questions(job, report)
                agent.schedule(job, report, window=(start, start + timedelta(days=days)))
                if transcript_dir and Path(transcript_dir).is_dir():
                    agent._summarize_folder(job, Path(transcript_dir), report)
                agent.recommend(job, report)
            show_report(report)

# -- Interviews ------------------------------------------------------------

with tabs[2]:
    if not interviews:
        st.info("No interviews booked yet.")

    for booking in interviews:
        candidate = candidates.get(booking.candidate_id)
        name = candidate.display_name if candidate else booking.candidate_id
        with st.expander(f"{name} · {booking.slot}"):
            ui.pills(
                [
                    (booking.status, "good" if booking.status == "completed" else "neutral"),
                    (booking.mode, "neutral"),
                ]
            )
            ui.meta(
                f"Panel: {', '.join(p.name for p in booking.interviewers)} · {booking.location}"
            )

            if booking.ics_path and Path(booking.ics_path).is_file():
                st.download_button(
                    "Download calendar invite (.ics)",
                    Path(booking.ics_path).read_bytes(),
                    file_name=Path(booking.ics_path).name,
                    mime="text/calendar",
                    key=f"ics_{booking.id}",
                )

            for summary in agent.store.list_summaries(booking.candidate_id):
                ui.rule()
                ui.eyebrow("Interview summary")
                ui.meta(summary.overall_impression)
                st.write("")
                ui.pills(
                    [
                        (f"{s.dimension}: {s.rating.value}", RATING_TONE[s.rating])
                        for s in summary.signals
                    ]
                )
                for signal in summary.signals:
                    if signal.evidence:
                        ui.quote(signal.evidence, f"{signal.dimension} — transcript")
                if summary.concerns:
                    ui.eyebrow("Concerns")
                    for concern in summary.concerns:
                        st.markdown(f"- {concern}")
                if summary.unanswered_questions:
                    ui.eyebrow("Not covered")
                    for item in summary.unanswered_questions:
                        st.markdown(f"- {item}")

# -- Questions -------------------------------------------------------------

with tabs[3]:
    any_questions = False
    for score in shortlist.ranked if shortlist else []:
        question_set = agent.store.get_questions(score.candidate_id, job_id)
        if not question_set:
            continue
        any_questions = True
        candidate = candidates.get(score.candidate_id)
        name = candidate.display_name if candidate else score.candidate_id
        with st.expander(f"{name} · {len(question_set.questions)} questions"):
            if not question_set.llm_available:
                ui.note("Template fallback — no model was available.")
            for i, q in enumerate(question_set.questions, start=1):
                st.markdown(f"**{i}. {q.question}**")
                ui.pills([(q.category.value.replace("_", " "), "neutral")])
                if q.linked_evidence:
                    ui.quote(q.linked_evidence, "prompted by")
                if q.what_good_looks_like:
                    ui.note(f"A strong answer: {q.what_good_looks_like}")
                st.write("")

    if not any_questions:
        st.info("No questions generated yet.")

# -- Recommendations -------------------------------------------------------

with tabs[4]:
    ui.gate(
        "These are recommendations, not decisions",
        "There is no code path in this system that hires or rejects anyone. The "
        "agent stops here and waits for you.",
    )

    if not recommendations:
        st.info("No recommendations yet.")

    for rec in recommendations:
        candidate = candidates.get(rec.candidate_id)
        name = candidate.display_name if candidate else rec.candidate_id
        tone = VERDICT_TONE.get(rec.verdict.value, "neutral")
        with st.expander(f"{name} · {rec.verdict.value.replace('_', ' ')}"):
            ui.pills(
                [
                    (rec.verdict.value.replace("_", " "), tone),
                    (f"confidence {rec.confidence:.0%}", "neutral"),
                ]
            )
            st.write("")
            ui.meta(rec.rationale)
            st.write("")

            if rec.supporting_evidence:
                ui.eyebrow("Supporting evidence")
                for item in rec.supporting_evidence:
                    st.markdown(f"- {item}")

            ui.eyebrow("Evidence against — always recorded")
            for item in rec.dissenting_signals:
                st.markdown(f"- {item}")

            if rec.suggested_next_step:
                st.info(f"Suggested next step: {rec.suggested_next_step}")

            ui.rule()
            ui.eyebrow("Gate 2 of 2 — your decision")

            decided = [
                d
                for d in agent.store.decisions_for_candidate(rec.candidate_id)
                if d.action is not HumanAction.APPROVE_SHORTLIST
            ]
            if decided:
                latest = decided[-1]
                ui.gate(
                    f"Decided: {latest.action.value.replace('_', ' ')}",
                    f"{latest.decided_by} on {latest.decided_at}"
                    + (f" — {latest.notes}" if latest.notes else ""),
                    done=True,
                )
            else:
                decider = st.text_input("Your name", key=f"decider_{rec.candidate_id}")
                action = st.radio(
                    "Decision",
                    [HumanAction.ADVANCE, HumanAction.REJECT, HumanAction.REQUEST_MORE_INFO],
                    format_func=lambda a: a.value.replace("_", " "),
                    key=f"action_{rec.candidate_id}",
                    horizontal=True,
                )
                note_text = st.text_area("Reasoning", key=f"note_{rec.candidate_id}")
                if st.button("Record decision", key=f"decide_{rec.candidate_id}"):
                    if not decider.strip():
                        st.error("Decisions must name a person.")
                    else:
                        record_decision(
                            agent.store, rec.candidate_id, job_id, action, decider, note_text
                        )
                        st.rerun()

# -- Audit trail -----------------------------------------------------------

with tabs[5]:
    ui.eyebrow("Append-only")
    ui.note(
        "Every stage writes a row here: what ran, what it decided, which model "
        "and when. This is how a decision gets reconstructed months later."
    )
    st.write("")

    rows = agent.store.audit_trail(job_id=job_id)
    if not rows:
        st.info("No audit entries yet.")
    else:
        st.dataframe(
            [
                {
                    "when": r["at"],
                    "stage": r["stage"],
                    "action": r["action"],
                    "entity": r["entity_id"],
                    "model": r["model"] or "—",
                    "detail": str(r["detail"])[:200],
                }
                for r in rows
            ],
            width="stretch",
            hide_index=True,
        )
