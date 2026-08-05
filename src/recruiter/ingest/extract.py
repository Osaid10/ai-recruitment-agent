"""Stage 1b — resume text to a structured `Candidate`.

Primary path is the LLM bound to `ExtractedProfile`. If the model is
unavailable, `heuristic_profile` still produces something usable so the
deterministic rubric can rank the batch with no API key at all.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from ..llm import LLMClient, LLMUnavailable, truncate
from ..models import Candidate, Education, ExtractedProfile, WorkExperience
from ..redact import EMAIL_RE, PHONE_RE, URL_RE, guess_name_tokens
from .parser import ParsedResume

SYSTEM = """You extract structured data from resumes for an applicant tracking system.

Rules:
- Copy only what the resume actually says. Never infer, embellish, or fill gaps.
- If a field is absent, leave it empty. An empty field is correct; a guess is a bug.
- `total_years_experience` is the sum of professional work experience in years,
  excluding internships shorter than 6 months and excluding education. Round to
  one decimal. If dates are missing or unclear, return 0.
- `skills` are concrete technologies, tools, languages and methods named in the
  resume. Do not add skills implied by a job title.
- For each role, `years` is its own duration in years. "Present" means today.
"""

USER_TEMPLATE = """Extract the structured profile from this resume.

--- RESUME ---
{resume}
--- END RESUME ---"""

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        ],
        start=1,
    )
}

DATE_RANGE_RE = re.compile(
    r"(?P<m1>[A-Za-z]{3,9})?\s*(?P<y1>(?:19|20)\d{2})\s*[-–—to]{1,3}\s*"
    r"(?:(?P<present>present|current|now)|(?P<m2>[A-Za-z]{3,9})?\s*(?P<y2>(?:19|20)\d{2}))",
    re.I,
)

SECTION_RE = re.compile(
    r"^\s*(technical\s+skills|core\s+skills|skills|technologies|tech\s+stack|competencies)\s*:?\s*$",
    re.I | re.M,
)

EDU_KEYWORDS = re.compile(
    r"\b(b\.?s\.?c?|m\.?s\.?c?|b\.?tech|m\.?tech|bachelor|master|phd|doctorate|diploma)\b", re.I
)


def _months_between(m1: str | None, y1: str, m2: str | None, y2: str) -> int:
    start_m = MONTHS.get((m1 or "").lower()[:3], 1)
    end_m = MONTHS.get((m2 or "").lower()[:3], 12)
    return max(0, (int(y2) - int(y1)) * 12 + (end_m - start_m) + 1)


def heuristic_years(text: str, today: date | None = None) -> float:
    """Sum non-overlapping date ranges found in the text.

    Deliberately conservative: overlapping ranges (common on resumes listing
    concurrent roles) are merged rather than double-counted, so a candidate who
    held two jobs at once is not credited with double the experience.
    """
    today = today or date.today()
    spans: list[tuple[int, int]] = []

    for m in DATE_RANGE_RE.finditer(text):
        y1 = m.group("y1")
        start = int(y1) * 12 + MONTHS.get((m.group("m1") or "").lower()[:3], 1)

        if m.group("present"):
            end = today.year * 12 + today.month
        else:
            y2, m2 = m.group("y2"), m.group("m2")
            if not y2:
                continue
            end = int(y2) * 12 + MONTHS.get((m2 or "").lower()[:3], 12)

        if end > start:
            spans.append((start, end))

    if not spans:
        return 0.0

    spans.sort()
    merged: list[list[int]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    months = sum(end - start for start, end in merged)
    return round(months / 12.0, 1)


def heuristic_skills(text: str, vocabulary: list[str] | None = None) -> list[str]:
    """Pull skills from an explicit skills section, then from a known vocabulary."""
    found: list[str] = []

    match = SECTION_RE.search(text)
    if match:
        block = text[match.end() : match.end() + 900]
        block = re.split(r"\n\s*\n|\n\s*[A-Z][A-Z\s]{5,}\n", block)[0]
        for token in re.split(r"[,|;\n•·/]+", block):
            token = token.strip(" -–—:\t")
            if 1 < len(token) <= 40 and not token.endswith("."):
                found.append(token)

    lowered = text.lower()
    for term in vocabulary or []:
        if re.search(rf"\b{re.escape(term.lower())}\b", lowered):
            found.append(term)

    seen: dict[str, str] = {}
    for skill in found:
        seen.setdefault(skill.lower(), skill)
    return list(seen.values())[:40]


def heuristic_profile(text: str, vocabulary: list[str] | None = None) -> ExtractedProfile:
    """Regex-only extraction. Used when the LLM is unavailable."""
    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)

    education: list[Education] = []
    for line in text.splitlines():
        if EDU_KEYWORDS.search(line) and len(line) < 160:
            year = re.search(r"(?:19|20)\d{2}", line)
            education.append(Education(degree=line.strip(), year=year.group(0) if year else ""))
        if len(education) >= 4:
            break

    return ExtractedProfile(
        full_name=" ".join(guess_name_tokens(text)),
        email=email_match.group(0) if email_match else "",
        phone=phone_match.group(0).strip() if phone_match else "",
        links=URL_RE.findall(text)[:5],
        summary=text.strip()[:400],
        skills=heuristic_skills(text, vocabulary),
        experience=[],
        education=education,
        total_years_experience=heuristic_years(text),
    )


def extract_profile(
    text: str,
    llm: LLMClient | None = None,
    vocabulary: list[str] | None = None,
) -> tuple[ExtractedProfile, str, list[str]]:
    """Return (profile, method, warnings). Falls back to heuristics on any LLM failure."""
    warnings: list[str] = []

    if llm and llm.available:
        try:
            profile = llm.structured(
                ExtractedProfile,
                SYSTEM,
                USER_TEMPLATE.format(resume=truncate(text)),
            )
            # The model is unreliable at arithmetic across date ranges; if it
            # returns nothing, trust the regex sum instead of a zero.
            if profile.total_years_experience <= 0:
                computed = heuristic_years(text)
                if computed > 0:
                    profile.total_years_experience = computed
                    warnings.append("years of experience computed from dates, not from the model")
            if not profile.skills:
                profile.skills = heuristic_skills(text, vocabulary)
                warnings.append("skills recovered heuristically — model returned none")
            return profile, "llm", warnings
        except LLMUnavailable as exc:
            warnings.append(f"LLM extraction failed, used heuristics instead: {exc}")

    return heuristic_profile(text, vocabulary), "heuristic", warnings


def build_candidate(
    parsed: ParsedResume,
    job_id: str,
    llm: LLMClient | None = None,
    vocabulary: list[str] | None = None,
) -> Candidate:
    """Turn a parsed file into a `Candidate`. Redaction happens later, in ranking."""
    profile, method, warnings = (
        extract_profile(parsed.text, llm, vocabulary)
        if parsed.usable
        else (ExtractedProfile(), "none", ["resume text unusable — nothing extracted"])
    )

    return Candidate(
        job_id=job_id,
        source_file=str(parsed.path),
        full_name=profile.full_name,
        email=profile.email,
        phone=profile.phone,
        location=profile.location,
        links=profile.links,
        summary=profile.summary,
        skills=profile.skills,
        experience=list(profile.experience) or _fallback_experience(parsed.text),
        education=profile.education,
        total_years_experience=profile.total_years_experience,
        raw_text=parsed.text,
        parse_warnings=[*parsed.warnings, *warnings],
        extraction_method=method,
    )


def _fallback_experience(text: str) -> list[WorkExperience]:
    """A single synthetic entry so downstream stages always have something to
    point at when structured extraction produced no roles."""
    years = heuristic_years(text)
    if years <= 0:
        return []
    return [WorkExperience(title="(not parsed)", description="", years=years)]


def source_label(path: Path | str) -> str:
    return Path(path).name
