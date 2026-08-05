"""Command line interface.

argparse rather than a CLI framework - the command surface is small, and stdlib
means one less thing that can break on a demo machine. `rich` is used when it is
installed and degrades to plain text when it is not.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from .config import Settings
from .models import HumanAction, RunReport, StageStatus, utcnow
from .pipeline import JobSpecError, RecruitmentAgent, load_job
from .recommend import ApprovalRequired, record_decision

try:
    from rich.console import Console
    from rich.table import Table

    _console = Console()
except ImportError:  # pragma: no cover
    _console = None
    Table = None  # type: ignore[assignment]

STATUS_STYLE = {
    StageStatus.OK: ("[green]ok[/green]", "ok"),
    StageStatus.FAILED: ("[red]failed[/red]", "FAILED"),
    StageStatus.SKIPPED: ("[dim]skipped[/dim]", "skipped"),
    StageStatus.NEEDS_HUMAN: ("[yellow]needs human[/yellow]", "NEEDS HUMAN"),
}

MARKUP_RE = re.compile(r"\[/?[a-z ]+\]")

GATE_NOTICE = "[!] "


def out(message: str = "") -> None:
    if _console:
        _console.print(message)
    else:
        print(MARKUP_RE.sub("", message))


def print_report(report: RunReport) -> None:
    out("")
    if _console and Table:
        table = Table(title=f"Run {report.run_id}")
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Detail")
        for result in report.results:
            table.add_row(result.stage.value, STATUS_STYLE[result.status][0], result.detail)
        _console.print(table)
    else:
        for result in report.results:
            print(
                f"  {result.stage.value:<10} "
                f"{STATUS_STYLE[result.status][1]:<12} {result.detail}"
            )

    failures = report.failures()
    if failures:
        out(f"\n[red]{len(failures)} stage failure(s)[/red] - see above.")
    if report.awaiting:
        out(f"\n[yellow]{GATE_NOTICE}Waiting on:[/yellow] {report.awaiting}")


def _agent(args: argparse.Namespace) -> RecruitmentAgent:
    agent = RecruitmentAgent(Settings.load())
    if not agent.llm.available:
        out(
            f"[yellow]note:[/yellow] running without a model - {agent.llm.unavailable_reason}\n"
            "      Deterministic ranking still works; LLM stages will say so."
        )
    return agent


def close_agent(agent: RecruitmentAgent) -> None:
    try:
        agent.close()
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        report = agent.run(
            args.job,
            args.resumes,
            approve_as=args.approve_as or "",
            transcripts_dir=args.transcripts,
        )
    except JobSpecError as exc:
        out(f"[red]error:[/red] {exc}")
        close_agent(agent)
        return 2

    print_report(report)
    out(f"\nJob id: [bold]{report.job_id}[/bold]  (use it with the other commands)")
    close_agent(agent)
    return 1 if report.failures() else 0


def cmd_ingest(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        job = load_job(args.job)
    except JobSpecError as exc:
        out(f"[red]error:[/red] {exc}")
        close_agent(agent)
        return 2

    agent.register_job(job)
    report = RunReport(job_id=job.id)
    agent.ingest(job, args.resumes, report)
    report.finished_at = utcnow()
    print_report(report)
    out(f"\nJob id: [bold]{job.id}[/bold]")
    close_agent(agent)
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        job = agent.get_job(args.job)
    except JobSpecError as exc:
        out(f"[red]error:[/red] {exc}")
        close_agent(agent)
        return 2

    report = RunReport(job_id=job.id)
    shortlist = agent.rank(job, report)
    print_report(report)

    if _console and Table:
        table = Table(title=f"Ranking - {job.title}")
        table.add_column("#", justify="right")
        table.add_column("Candidate")
        table.add_column("Total", justify="right")
        table.add_column("Rules", justify="right")
        table.add_column("Fit", justify="right")
        table.add_column("Gates", justify="center")
        table.add_column("Outcome")
        for i, score in enumerate([*shortlist.ranked, *shortlist.cut], start=1):
            candidate = agent.store.get_candidate(score.candidate_id)
            met = sum(1 for g in score.hard_gates if g.met)
            table.add_row(
                str(i),
                candidate.display_name if candidate else score.candidate_id,
                f"{score.total_score:.1f}",
                f"{score.gate_score:.1f}",
                f"{score.fit_score:.1f}",
                f"{met}/{len(score.hard_gates)}",
                "[green]shortlisted[/green]" if score.shortlisted else f"[dim]{score.reason}[/dim]",
            )
        _console.print(table)

    out(
        f"\n[yellow]{GATE_NOTICE}Nobody is contacted yet.[/yellow] Approve with:\n"
        f'   python -m recruiter approve --job {job.id} --by "Your Name"'
    )
    close_agent(agent)
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        count = agent.approve(args.job, args.by, args.notes or "")
    except (ApprovalRequired, ValueError) as exc:
        out(f"[red]error:[/red] {exc}")
        close_agent(agent)
        return 2
    out(f"[green]Shortlist of {count} approved by {args.by}.[/green] Scheduling is now unlocked.")
    close_agent(agent)
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        job = agent.get_job(args.job)
    except JobSpecError as exc:
        out(f"[red]error:[/red] {exc}")
        close_agent(agent)
        return 2
    report = RunReport(job_id=job.id)
    agent.schedule(job, report)
    print_report(report)
    close_agent(agent)
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        job = agent.get_job(args.job)
    except JobSpecError as exc:
        out(f"[red]error:[/red] {exc}")
        close_agent(agent)
        return 2

    report = RunReport(job_id=job.id)
    sets = agent.questions(job, report)
    print_report(report)

    if args.show:
        for question_set in sets:
            candidate = agent.store.get_candidate(question_set.candidate_id)
            name = candidate.display_name if candidate else question_set.candidate_id
            out(f"\n[bold cyan]{name}[/bold cyan]")
            for i, q in enumerate(question_set.questions, start=1):
                out(f"  {i}. ({q.category.value}) {q.question}")
                if q.linked_evidence:
                    out(f"     [dim]why: {q.linked_evidence}[/dim]")
    close_agent(agent)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    agent = _agent(args)
    candidate = agent.store.get_candidate(args.candidate)
    if candidate is None:
        out(f"[red]error:[/red] unknown candidate '{args.candidate}'")
        close_agent(agent)
        return 2
    try:
        job = agent.get_job(candidate.job_id)
    except JobSpecError as exc:
        out(f"[red]error:[/red] {exc}")
        close_agent(agent)
        return 2

    report = RunReport(job_id=job.id)
    summary = agent.summarize(job, candidate.id, Path(args.transcript), report)
    print_report(report)

    if summary:
        out(f"\n[bold]{candidate.display_name}[/bold] - {summary.overall_impression}")
        for signal in summary.signals:
            out(f"  {signal.dimension}: [bold]{signal.rating.value}[/bold] - {signal.evidence}")
        if summary.concerns:
            out("  concerns: " + "; ".join(summary.concerns))
    close_agent(agent)
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    agent = _agent(args)
    try:
        job = agent.get_job(args.job)
    except JobSpecError as exc:
        out(f"[red]error:[/red] {exc}")
        close_agent(agent)
        return 2

    report = RunReport(job_id=job.id)
    recommendations = agent.recommend(job, report)
    print_report(report)

    for rec in recommendations:
        candidate = agent.store.get_candidate(rec.candidate_id)
        name = candidate.display_name if candidate else rec.candidate_id
        out(f"\n[bold cyan]{name}[/bold cyan]")
        out(f"  verdict: [bold]{rec.verdict.value}[/bold]  confidence: {rec.confidence:.0%}")
        out(f"  {rec.rationale}")
        for item in rec.supporting_evidence:
            out(f"   [green]+[/green] {item}")
        for item in rec.dissenting_signals:
            out(f"   [yellow]-[/yellow] {item}")
        if rec.suggested_next_step:
            out(f"  next: {rec.suggested_next_step}")

    out(
        f"\n[yellow]{GATE_NOTICE}These are recommendations, not decisions.[/yellow] "
        "Record yours with:\n"
        '   python -m recruiter decide --candidate <id> --action advance --by "Your Name"'
    )
    close_agent(agent)
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    agent = _agent(args)
    candidate = agent.store.get_candidate(args.candidate)
    if candidate is None:
        out(f"[red]error:[/red] unknown candidate '{args.candidate}'")
        close_agent(agent)
        return 2
    try:
        decision = record_decision(
            agent.store,
            candidate.id,
            candidate.job_id,
            HumanAction(args.action),
            args.by,
            args.notes or "",
        )
    except ValueError as exc:
        out(f"[red]error:[/red] {exc}")
        close_agent(agent)
        return 2
    out(
        f"[green]Recorded:[/green] {decision.decided_by} chose "
        f"[bold]{decision.action.value}[/bold] for {candidate.display_name} "
        f"at {decision.decided_at}."
    )
    close_agent(agent)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    agent = _agent(args)
    rows = agent.store.audit_trail(entity_id=args.candidate or "", job_id=args.job or "")
    if not rows:
        out("[dim]no audit entries[/dim]")
        close_agent(agent)
        return 0

    if _console and Table:
        table = Table(title="Audit trail")
        table.add_column("When", no_wrap=True)
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Action", no_wrap=True)
        table.add_column("Entity", no_wrap=True)
        table.add_column("Model", no_wrap=True)
        table.add_column("Detail")
        for row in rows:
            detail = ", ".join(f"{k}={v}" for k, v in row["detail"].items())
            table.add_row(
                row["at"],
                row["stage"],
                row["action"],
                row["entity_id"],
                row["model"] or "-",
                detail[:120],
            )
        _console.print(table)
    else:
        for row in rows:
            print(f"{row['at']} {row['stage']:<10} {row['action']:<24} {row['entity_id']}")
    close_agent(agent)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    agent = _agent(args)
    jobs = agent.store.list_jobs()
    if not jobs:
        out("[dim]no jobs in the database yet - run `ingest` or `run` first[/dim]")
        close_agent(agent)
        return 0

    for job in jobs:
        shortlist = agent.store.get_shortlist(job.id)
        candidates = agent.store.list_candidates(job.id)
        interviews = agent.store.list_interviews(job.id)
        decisions = agent.store.list_decisions(job.id)
        approved = (
            f"[green]approved by {shortlist.approved_by}[/green]"
            if shortlist and shortlist.is_approved
            else "[yellow]awaiting approval[/yellow]"
        )
        out(
            f"[bold]{job.title}[/bold] ({job.id})\n"
            f"  candidates: {len(candidates)}   "
            f"shortlisted: {len(shortlist.ranked) if shortlist else 0}   "
            f"interviews: {len(interviews)}   decisions: {len(decisions)}\n"
            f"  shortlist: {approved}"
        )
    close_agent(agent)
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recruiter",
        description="AI Recruitment Agent - screens the top of the funnel. A human decides.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the whole pipeline")
    p.add_argument("--job", required=True, help="path to a job requisition JSON file")
    p.add_argument("--resumes", required=True, help="folder of resume files")
    p.add_argument("--transcripts", help="folder of interview transcripts (optional)")
    p.add_argument(
        "--approve-as",
        dest="approve_as",
        help="your name - approves the shortlist so the run continues past the gate",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("ingest", help="stage 1 - parse resumes into the ATS")
    p.add_argument("--job", required=True, help="path to a job requisition JSON file")
    p.add_argument("--resumes", required=True)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("rank", help="stage 2 - score and shortlist")
    p.add_argument("--job", required=True, help="job id")
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("approve", help="human gate - approve the shortlist")
    p.add_argument("--job", required=True, help="job id")
    p.add_argument("--by", required=True, help="your name - approvals must name a person")
    p.add_argument("--notes")
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("schedule", help="stage 3 - book interviews and write invites")
    p.add_argument("--job", required=True, help="job id")
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("questions", help="stage 4 - generate tailored questions")
    p.add_argument("--job", required=True, help="job id")
    p.add_argument("--show", action="store_true", help="print the questions")
    p.set_defaults(func=cmd_questions)

    p = sub.add_parser("summarize", help="stage 5 - summarise an interview transcript")
    p.add_argument("--candidate", required=True, help="candidate id")
    p.add_argument("--transcript", required=True, help="path to the transcript file")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("recommend", help="stage 6 - produce recommendations")
    p.add_argument("--job", required=True, help="job id")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("decide", help="human gate - record the final decision")
    p.add_argument("--candidate", required=True)
    p.add_argument(
        "--action",
        required=True,
        choices=[a.value for a in HumanAction],
        help="what you decided",
    )
    p.add_argument("--by", required=True, help="your name")
    p.add_argument("--notes")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("audit", help="show the audit trail")
    p.add_argument("--job")
    p.add_argument("--candidate")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("status", help="what is in the ATS right now")
    p.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        out("\n[yellow]interrupted[/yellow]")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
