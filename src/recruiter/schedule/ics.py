"""Stage 3b — write RFC 5545 calendar invites.

Hand-rolled rather than pulled from a library: the format is small, and a real
.ics file that opens in Outlook or Google Calendar is the most convincing part
of the scheduling demo.

Times are written as floating local time (RFC 5545 form 1), so the invite shows
at the same wall-clock time for everyone — correct for a mock calendar and free
of a tzdata dependency on Windows.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..models import InterviewBooking

MAX_LINE = 73  # RFC 5545 caps content lines at 75 octets including the CRLF


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    """Fold long lines; continuations start with a single space."""
    if len(line) <= MAX_LINE:
        return line
    chunks = [line[:MAX_LINE]]
    rest = line[MAX_LINE:]
    while rest:
        chunks.append(" " + rest[: MAX_LINE - 1])
        rest = rest[MAX_LINE - 1 :]
    return "\r\n".join(chunks)


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def build_ics(
    booking: InterviewBooking,
    candidate_name: str,
    candidate_email: str,
    job_title: str,
    organizer_email: str = "recruiting@example.com",
    company: str = "Mercurial Minds",
    agenda: str = "",
) -> str:
    # Plain hyphen, not an em dash: some older calendar clients mangle non-ASCII
    # in SUMMARY even when the file is valid UTF-8.
    summary = f"Interview: {candidate_name or 'Candidate'} - {job_title}"
    description_parts = [
        f"Interview for {job_title} at {company}.",
        f"Candidate: {candidate_name or '(name not parsed)'}",
        f"Mode: {booking.mode}",
    ]
    if agenda:
        description_parts.append("")
        description_parts.append(agenda)
    description_parts += [
        "",
        "Scheduled by the AI Recruitment Agent. The hiring decision remains with the panel.",
    ]

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Mercurial Minds//AI Recruitment Agent//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{booking.id}@ai-recruitment-agent",
        f"DTSTAMP:{_stamp(datetime.now())}",
        f"DTSTART:{_stamp(booking.slot.start)}",
        f"DTEND:{_stamp(booking.slot.end)}",
        f"SUMMARY:{_escape(summary)}",
        f"DESCRIPTION:{_escape(chr(10).join(description_parts))}",
        f"LOCATION:{_escape(booking.location or booking.mode)}",
        "STATUS:CONFIRMED",
        "SEQUENCE:0",
        f"ORGANIZER;CN={_escape(company)}:mailto:{organizer_email}",
    ]

    if candidate_email:
        lines.append(
            f"ATTENDEE;CN={_escape(candidate_name or 'Candidate')};ROLE=REQ-PARTICIPANT;"
            f"PARTSTAT=NEEDS-ACTION;RSVP=TRUE:mailto:{candidate_email}"
        )
    for person in booking.interviewers:
        lines.append(
            f"ATTENDEE;CN={_escape(person.name)};ROLE=REQ-PARTICIPANT;"
            f"PARTSTAT=NEEDS-ACTION;RSVP=TRUE:mailto:{person.email}"
        )

    lines += [
        "BEGIN:VALARM",
        "TRIGGER:-PT15M",
        "ACTION:DISPLAY",
        "DESCRIPTION:Interview starts in 15 minutes",
        "END:VALARM",
        "END:VEVENT",
        "END:VCALENDAR",
    ]

    return "\r\n".join(_fold(line) for line in lines) + "\r\n"


def write_ics(
    booking: InterviewBooking,
    out_dir: Path | str,
    candidate_name: str,
    candidate_email: str,
    job_title: str,
    **kwargs: str,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in (candidate_name or "candidate"))
    path = out_dir / f"{booking.id}_{safe_name}.ics"
    path.write_text(
        build_ics(booking, candidate_name, candidate_email, job_title, **kwargs),
        encoding="utf-8",
        newline="",
    )
    return path
