"""ATS-style dashboard for the AI Recruitment Agent.

Run from the repo root:

    streamlit run app/streamlit_app.py

The dashboard exists to make the agent's reasoning visible: not just who was
ranked where, but which evidence drove each score, who was cut and why, and
where the human gates are. The two approval controls here are the only way to
move a candidate forward.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recruiter.config import Settings  # noqa: E402
from recruiter.models import HumanAction, RunReport, StageStatus  # noqa: E402
from recruiter.pipeline import JobSpecError, RecruitmentAgent, load_job  # noqa: E402
from recruiter.recommend import record_decision, shortlist_is_approved  # noqa: E402

st.set_page_config(page_title="AI Recruitment Agent", page_icon="::", layout="wide")

STATUS_ICON = {
    StageStatus.OK: ":white_check_mark:",
    StageStatus.FAILED: ":x:",
    StageStatus.SKIPPED: ":heavy_minus_sign:",
    StageStatus.NEEDS_HUMAN: ":warning:",
}


@st.cache_resource
def get_agent() -> RecruitmentAgent:
    return RecruitmentAgent(Settings.load())


def show_report(report: RunReport) -> None:
    for result in report.results:
        st.write(
            f"{STATUS_ICON[result.status]} **{result.stage.value}** — {result.detail}"
        )
    if report.awaiting:
        st.warning(f"Waiting on: {report.awaiting}")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

agent = get_agent()
settings = agent.settings

st.sidebar.title("AI Recruitment Agent")
st.sidebar.caption("Screens the top of the funnel. A human decides.")

if agent.llm.available:
    st.sidebar.success(f"Model: {agent.model_name}")
else:
    st.sidebar.warning(
        "No model configured — deterministic ranking only.\n\n"
        "Add `GROQ_API_KEY` to `.env` for fit assessment, tailored questions, "
        "interview summaries and recommendations."
    )

st.sidebar.divider()
st.sidebar.subheader("Run the pipeline")

job_files = sorted((ROOT / "data" / "jobs").glob("*.json"))
job_choice = st.sidebar.selectbox(
    "Job requisition", job_files, format_func=lambda p: p.name, index=0 if job_files else None
)
resume_dir = st.sidebar.text_input("Resume folder", value=str(ROOT / "data" / "resumes"))
transcript_dir = st.sidebar.text_input(
    "Transcript folder (optional)", value=str(ROOT / "data" / "interviews")
)

if st.sidebar.button("Ingest and rank", type="primary", use_container_width=True):
    if not job_choice:
        st.sidebar.error("No job requisition found in data/jobs/")
    else:
        try:
            job = agent.register_job(load_job(job_choice))
            report = RunReport(job_id=job.id)
            with st.spinner("Reading resumes and ranking..."):
                agent.ingest(job, resume_dir, report)
                agent.rank(job, report)
            st.session_state["report"] = report.model_dump()
            st.session_state["job_id"] = job.id
        except JobSpecError as exc:
            st.sidebar.error(str(exc))

st.sidebar.divider()
jobs = agent.store.list_jobs()
if jobs:
    selected = st.sidebar.selectbox(
        "Viewing",
        [j.id for j in jobs],
        format_func=lambda jid: next(j.title for j in jobs if j.id == jid),
        index=0,
    )
    st.session_state["job_id"] = st.session_state.get("job_id") or selected
    job_id = selected
else:
    job_id = ""

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if not job_id:
    st.title("AI Recruitment Agent")
    st.info(
        "Nothing in the database yet. Pick a job requisition in the sidebar and "
        "press **Ingest and rank** to start."
    )
    st.stop()

job = agent.store.get_job(job_id)
candidates = {c.id: c for c in agent.store.list_candidates(job_id)}
shortlist = agent.store.get_shortlist(job_id)
approved = shortlist_is_approved(agent.store, job_id)

st.title(job.title)
st.caption(
    f"{job.department} · {job.location} · {job.employment_type} · "
    f"must have: {', '.join(r.skill for r in job.must_haves)}"
)

cols = st.columns(5)
cols[0].metric("Candidates", len(candidates))
cols[1].metric("Shortlisted", len(shortlist.ranked) if shortlist else 0)
cols[2].metric("Interviews", len(agent.store.list_interviews(job_id)))
cols[3].metric("Recommendations", len(agent.store.list_recommendations(job_id)))
cols[4].metric("Shortlist approved", "Yes" if approved else "No")

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
        st.subheader("Shortlisted")
        for score in shortlist.ranked:
            candidate = candidates.get(score.candidate_id)
            name = candidate.display_name if candidate else score.candidate_id
            with st.expander(f"**{name}** — {score.total_score:.1f}/100", expanded=False):
                a, b, c = st.columns(3)
                a.metric("Total", f"{score.total_score:.1f}")
                b.metric("Rules", f"{score.gate_score:.1f}")
                c.metric("Model fit", f"{score.fit_score:.1f}" if score.llm_available else "n/a")

                st.markdown("**Hard requirements**")
                for gate in score.hard_gates:
                    st.write(
                        f"{':white_check_mark:' if gate.met else ':x:'} "
                        f"{gate.requirement} — {gate.detail}"
                    )

                if score.dimensions and score.llm_available:
                    st.markdown("**Rubric dimensions** (scored on redacted text)")
                    for dim in score.dimensions:
                        st.write(f"**{dim.dimension}** — {dim.score:.1f}/10")
                        st.caption(f"evidence: “{dim.evidence}”")

                if score.strengths:
                    st.markdown("**Strengths**")
                    for item in score.strengths:
                        st.write(f"- {item}")
                if score.concerns:
                    st.markdown("**Concerns**")
                    for item in score.concerns:
                        st.write(f"- {item}")
                if score.flags:
                    st.markdown("**Flags for a human**")
                    for flag in score.flags:
                        st.warning(flag)

                st.caption(
                    f"Scored on {'redacted' if score.redacted else 'UNREDACTED'} text · "
                    f"{score.reason}"
                )

        st.subheader("Not shortlisted")
        st.caption(
            "Nobody here has been rejected. Each row records why they were not "
            "advanced, and the decision is still a human's."
        )
        for score in shortlist.cut:
            candidate = candidates.get(score.candidate_id)
            name = candidate.display_name if candidate else score.candidate_id
            with st.expander(f"{name} — {score.total_score:.1f}/100"):
                st.write(score.reason)
                for gate in score.hard_gates:
                    if not gate.met:
                        st.write(f":x: {gate.requirement} — {gate.detail}")
                for flag in score.flags:
                    st.caption(flag)

# -- Human gate ------------------------------------------------------------

with tabs[1]:
    st.subheader("Gate 1 — approve the shortlist")
    st.caption(
        "Nothing downstream runs until a named person approves. Scheduling and "
        "question generation refuse to execute without this."
    )

    if approved:
        st.success(
            f"Approved by **{shortlist.approved_by}** at {shortlist.approved_at}."
        )
    elif not shortlist or not shortlist.ranked:
        st.info("Rank some candidates first.")
    else:
        approver = st.text_input("Your name", key="approver")
        notes = st.text_area("Notes (optional)", key="approval_notes")
        if st.button("Approve shortlist", type="primary"):
            if not approver.strip():
                st.error("Approvals must name a person.")
            else:
                count = agent.approve(job_id, approver, notes)
                st.success(f"{count} candidates approved. Scheduling is unlocked.")
                st.rerun()

    st.divider()
    st.subheader("Run the rest of the pipeline")
    if not approved:
        st.info("Blocked until the shortlist is approved.")
    else:
        days = st.slider("Scheduling window (days ahead)", 3, 30, 10)
        if st.button("Generate questions, schedule interviews, recommend"):
            report = RunReport(job_id=job_id)
            start = (datetime.now() + timedelta(days=1)).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            with st.spinner("Working..."):
                agent.questions(job, report)
                agent.schedule(job, report, window=(start, start + timedelta(days=days)))
                if transcript_dir and Path(transcript_dir).is_dir():
                    agent._summarize_folder(job, Path(transcript_dir), report)
                agent.recommend(job, report)
            show_report(report)

# -- Interviews ------------------------------------------------------------

with tabs[2]:
    interviews = agent.store.list_interviews(job_id)
    if not interviews:
        st.info("No interviews booked yet.")
    for booking in interviews:
        candidate = candidates.get(booking.candidate_id)
        name = candidate.display_name if candidate else booking.candidate_id
        with st.expander(f"{name} — {booking.slot}"):
            st.write(f"**Panel:** {', '.join(p.name for p in booking.interviewers)}")
            st.write(f"**Mode:** {booking.mode} · {booking.location}")
            st.write(f"**Status:** {booking.status}")
            if booking.ics_path and Path(booking.ics_path).is_file():
                st.download_button(
                    "Download calendar invite (.ics)",
                    Path(booking.ics_path).read_bytes(),
                    file_name=Path(booking.ics_path).name,
                    mime="text/calendar",
                    key=f"ics_{booking.id}",
                )
            for summary in agent.store.list_summaries(booking.candidate_id):
                st.markdown("**Interview summary**")
                st.write(summary.overall_impression)
                for signal in summary.signals:
                    st.write(f"- **{signal.dimension}**: {signal.rating.value}")
                    if signal.evidence:
                        st.caption(f"  “{signal.evidence}”")
                if summary.concerns:
                    st.markdown("**Concerns**")
                    for concern in summary.concerns:
                        st.write(f"- {concern}")
                if summary.unanswered_questions:
                    st.markdown("**Not covered**")
                    for item in summary.unanswered_questions:
                        st.write(f"- {item}")

# -- Questions -------------------------------------------------------------

with tabs[3]:
    if not shortlist or not shortlist.ranked:
        st.info("Nothing to show yet.")
    for score in (shortlist.ranked if shortlist else []):
        question_set = agent.store.get_questions(score.candidate_id, job_id)
        if not question_set:
            continue
        candidate = candidates.get(score.candidate_id)
        name = candidate.display_name if candidate else score.candidate_id
        with st.expander(f"{name} — {len(question_set.questions)} questions"):
            if not question_set.llm_available:
                st.caption("Template fallback — no model was available.")
            for i, q in enumerate(question_set.questions, start=1):
                st.markdown(f"**{i}. {q.question}**")
                st.caption(f"category: {q.category.value}")
                if q.linked_evidence:
                    st.caption(f"prompted by: {q.linked_evidence}")
                if q.what_good_looks_like:
                    st.caption(f"a strong answer: {q.what_good_looks_like}")

# -- Recommendations -------------------------------------------------------

with tabs[4]:
    recommendations = agent.store.list_recommendations(job_id)
    if not recommendations:
        st.info("No recommendations yet.")

    st.warning(
        "These are recommendations, not decisions. The agent has no code path "
        "that hires or rejects anyone."
    )

    for rec in recommendations:
        candidate = candidates.get(rec.candidate_id)
        name = candidate.display_name if candidate else rec.candidate_id
        with st.expander(f"{name} — {rec.verdict.value} ({rec.confidence:.0%} confidence)"):
            st.write(rec.rationale)

            if rec.supporting_evidence:
                st.markdown("**Supporting evidence**")
                for item in rec.supporting_evidence:
                    st.write(f"- {item}")

            st.markdown("**Evidence against** (always recorded)")
            for item in rec.dissenting_signals:
                st.write(f"- {item}")

            if rec.suggested_next_step:
                st.info(f"Suggested next step: {rec.suggested_next_step}")

            st.divider()
            st.markdown("**Gate 2 — your decision**")
            decided = [
                d
                for d in agent.store.decisions_for_candidate(rec.candidate_id)
                if d.action is not HumanAction.APPROVE_SHORTLIST
            ]
            if decided:
                latest = decided[-1]
                st.success(
                    f"{latest.decided_by} chose **{latest.action.value}** on "
                    f"{latest.decided_at}"
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
                note = st.text_area("Reasoning", key=f"note_{rec.candidate_id}")
                if st.button("Record decision", key=f"decide_{rec.candidate_id}"):
                    if not decider.strip():
                        st.error("Decisions must name a person.")
                    else:
                        record_decision(
                            agent.store, rec.candidate_id, job_id, action, decider, note
                        )
                        st.rerun()

# -- Audit -----------------------------------------------------------------

with tabs[5]:
    st.caption(
        "Every stage writes a row here: what ran, what it decided, which model, "
        "and when. This is how a decision gets reconstructed months later."
    )
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
                    "model": r["model"] or "-",
                    "detail": str(r["detail"])[:200],
                }
                for r in rows
            ],
            use_container_width=True,
            hide_index=True,
        )
