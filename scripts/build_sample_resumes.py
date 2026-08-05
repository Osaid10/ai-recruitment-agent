"""Generate the binary sample resumes (PDF and DOCX) from text sources.

Run from the repo root:

    python scripts/build_sample_resumes.py

The generated files are committed so the demo works straight after a clone.
Re-run this only if you change the text below.

The PDF writer is hand-rolled rather than pulled from reportlab. It emits a
minimal but valid PDF with a real text layer, which is all the parser needs, and
it keeps the project's dependency list short. It also lets us produce the one
file we specifically want: a PDF with *no* text layer, standing in for a scanned
resume, so the ingest stage's "needs human review" path can be demonstrated.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "resumes"

# --------------------------------------------------------------------------
# Minimal PDF writer
# --------------------------------------------------------------------------

PAGE_WIDTH, PAGE_HEIGHT = 612, 792  # US Letter, in points
MARGIN = 54
LINE_HEIGHT = 13
FONT_SIZE = 10


def _escape_pdf(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> str:
    """Lay text out top-down, starting a new page's worth of lines is not
    supported — sample resumes are short enough to fit one page."""
    parts = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LINE_HEIGHT} TL", f"1 0 0 1 {MARGIN} {PAGE_HEIGHT - MARGIN} Tm"]
    for line in lines:
        parts.append(f"({_escape_pdf(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts)


def write_pdf(path: Path, text: str = "", title: str = "Resume") -> Path:
    """Write a one-page PDF. Passing text="" produces a page with no text layer,
    which is what a scanned resume looks like to a parser."""
    lines = [ln[:95] for ln in text.splitlines()] if text else []
    stream = _content_stream(lines) if lines else ""

    objects: list[str] = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        None,  # placeholder for the content stream object
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        f"<< /Title ({_escape_pdf(title)}) >>",
    ]

    body = b""
    offsets: list[int] = []
    header = b"%PDF-1.4\n"
    cursor = len(header)

    for i, obj in enumerate(objects, start=1):
        if obj is None:
            encoded = stream.encode("latin-1", errors="replace")
            chunk = (
                f"{i} 0 obj\n<< /Length {len(encoded)} >>\nstream\n".encode("latin-1")
                + encoded
                + b"\nendstream\nendobj\n"
            )
        else:
            chunk = f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1", errors="replace")
        offsets.append(cursor)
        body += chunk
        cursor += len(chunk)

    xref_pos = cursor
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    xref += "".join(f"{off:010d} 00000 n \n" for off in offsets)
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body + xref.encode("latin-1") + trailer.encode("latin-1"))
    return path


def write_docx(path: Path, text: str) -> Path:
    try:
        import docx
    except ImportError:
        print("python-docx is not installed - skipping DOCX. Run: pip install -r requirements.txt")
        return path

    document = docx.Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    return path


# --------------------------------------------------------------------------
# Sample content
# --------------------------------------------------------------------------

STRONG_PDF = """Ahmed Nadeem
ahmed.nadeem@example.com | +92 300 1122334 | Lahore, Pakistan
github.com/ahmednadeem

SUMMARY
Senior backend engineer, seven years building Python services that other teams
depend on. Most recently responsible for the payments API at a company where
downtime is measured in lost transactions.

TECHNICAL SKILLS
Python, FastAPI, Django, PostgreSQL, SQL, REST API design, Redis, Docker,
Kubernetes, AWS (EKS, RDS, S3, Lambda), Terraform, CI/CD, GitHub Actions,
pytest, OpenTelemetry, Grafana

EXPERIENCE

Senior Software Engineer - Safepay
Mar 2022 - Present
- Own the payments API: FastAPI on PostgreSQL, roughly 1.2M requests a day.
- Led the move from a monolith to four services over nine months. Did it with
  the strangler pattern so there was never a big-bang cutover.
- Reduced p99 latency on the authorisation endpoint from 840ms to 190ms by
  replacing an N+1 query pattern and adding two composite indexes.
- Designed the idempotency layer that stopped duplicate charges. This was the
  single most common support ticket before, and is now roughly zero.
- Mentor two mid-level engineers and run the weekly design review.

Software Engineer - Careem
Jun 2019 - Feb 2022
- Built REST services in Python for the driver-earnings platform.
- Migrated batch reporting from nightly cron scripts onto Airflow, which cut
  the reporting delay from 24 hours to under 1.
- Introduced contract tests between our service and two downstream consumers
  after a breaking change caused a Friday incident.

Software Engineer - Systems Limited
Aug 2018 - May 2019
- Django and PostgreSQL work on an enterprise inventory product.

EDUCATION
BS Computer Science, FAST National University, 2018

OTHER
- Speaker, Python Pakistan meetup 2024: "Idempotency is a design problem".
- Maintainer of a small open-source library for Postgres advisory locks.
"""

SOLID_DOCX = """Fatima Malik
fatima.malik@example.com | +92 311 4455667 | Islamabad, Pakistan

SUMMARY
Backend engineer with five years of Python experience, focused on API design
and relational data modelling.

SKILLS
Python, Flask, FastAPI, PostgreSQL, SQL, REST APIs, SQLAlchemy, Docker,
Kubernetes, Git, pytest, RabbitMQ, Linux

EXPERIENCE

Software Engineer - Educative
Apr 2023 - Present
- Own three REST services behind the course platform, written in FastAPI.
- Redesigned the enrolment schema after a data model that had grown by
  accretion started causing duplicate-enrolment bugs. Shipped the migration
  in stages behind a feature flag.
- Introduced Kubernetes readiness probes and structured logging across the
  team's services.

Backend Engineer - Arbisoft
Sep 2021 - Mar 2023
- Built Flask APIs for a US client's logistics product.
- Wrote the reporting queries that backed the client's operations dashboard.
- Containerised the local development environment; onboarding a new engineer
  went from most of a day to about an hour.

Associate Engineer - Netsol Technologies
Jul 2020 - Aug 2021
- Python scripting and SQL work supporting a leasing platform.

EDUCATION
BS Software Engineering, COMSATS University Islamabad, 2020
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    written = [
        write_pdf(OUT / "01_ahmed_nadeem.pdf", STRONG_PDF, "Ahmed Nadeem - Resume"),
        write_docx(OUT / "02_fatima_malik.docx", SOLID_DOCX),
        # No text layer: stands in for a scanned or photographed resume.
        write_pdf(OUT / "07_scanned_no_text_layer.pdf", "", "Scanned Resume"),
    ]

    for path in written:
        if path.exists():
            print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
