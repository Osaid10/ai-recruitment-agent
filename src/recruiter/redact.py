"""PII redaction applied before ranking.

The scoring path never sees the candidate's identity. The human reviewer still
sees the full resume — redaction narrows what the *scorer* can condition on, it
does not hide anything from people. See docs/BIAS_AND_FAIRNESS.md.

This is regex- and list-based. It reduces demographic signal; it does not
eliminate it. Treat it as a control, not a guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}"
    r"(?:[\s.-]?\d{2,4})?(?!\w)"
)
URL_RE = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin|github|gitlab)\.com/\S+", re.I)

# Gendered titles and pronouns
TITLE_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Mx|Miss|Sir|Madam)\.?\s+", re.I)
PRONOUN_RE = re.compile(r"\b(?:he|him|his|she|her|hers)\b", re.I)
PRONOUN_DECL_RE = re.compile(r"\((?:he|she|they)/(?:him|her|them)[^)]*\)", re.I)

# Age / date of birth.
#
# The word boundary after the alternation is load-bearing. Without it, `age`
# matched the first three letters of "agentic" and the trailing wildcard ate the
# rest of the phrase — so "Agentic AI", "Multi-Agent Systems" and "agent
# orchestration" were deleted from an AI engineer's resume before scoring.
# Redaction that removes the candidate's most relevant skill does more damage
# than the bias it is there to prevent.
#
# A separator or a digit is also required, so a stray "born" or "age" in prose
# does not swallow the line it sits on.
DOB_RE = re.compile(
    r"\b(?:date\s+of\s+birth|d\.o\.b\.?|dob|age|born)\b\s*[:\-]\s*[^\n,;]{1,32}"
    r"|\b(?:date\s+of\s+birth|d\.o\.b\.?|dob)\b\s+\d[^\n,;]{0,30}"
    r"|\bborn\s+(?:on\s+|in\s+)?\d[^\n,;]{0,30}"
    r"|\bage\s+\d{1,2}\b",
    re.I,
)

# Single-line demographic declarations common on South Asian / EU CVs
DEMOGRAPHIC_LINE_RE = re.compile(
    r"^\s*(?:gender|sex|nationality|citizenship|marital\s+status|religion|caste|"
    r"race|ethnicity|father'?s\s+name|photo(?:graph)?|passport)\s*[:\-].*$",
    re.I | re.M,
)

# Institution names: capture the name, keep the qualification.
INSTITUTION_RE = re.compile(
    r"\b(?:[A-Z][\w&'.-]*\s+){0,4}"
    r"(?:University|College|Institute|Polytechnic|Academy|Schule|Universidad|"
    r"School\s+of\s+[A-Z]\w+)"
    r"(?:\s+of\s+(?:[A-Z][\w&'.-]*\s*){1,3})?"
)

ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Block|Sector|Phase)\b"
    r"[^\n]{0,40}",
    re.I,
)

_LABELS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "url": "[LINK]",
    "name": "[NAME]",
    "title": "",
    "pronoun": "they",
    "dob": "[AGE/DOB REMOVED]",
    "demographic": "[DEMOGRAPHIC FIELD REMOVED]",
    "institution": "[INSTITUTION]",
    "address": "[ADDRESS]",
}

# Words that look like names but are structural resume vocabulary.
_NAME_STOPWORDS = {
    "resume",
    "curriculum",
    "vitae",
    "cv",
    "profile",
    "summary",
    "engineer",
    "developer",
    "manager",
    "senior",
    "junior",
    "lead",
}


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> str:
        if not self.counts:
            return "no PII patterns matched"
        return ", ".join(f"{k}×{v}" for k, v in sorted(self.counts.items()))


def _sub(pattern: re.Pattern[str], repl: str, text: str, key: str, counts: dict) -> str:
    new_text, n = pattern.subn(repl, text)
    if n:
        counts[key] = counts.get(key, 0) + n
    return new_text


def guess_name_tokens(text: str) -> list[str]:
    """Heuristic: the first line of a resume is usually the candidate's name.

    Used only as a fallback when the extracted name is unknown, so that the
    header name still gets stripped before scoring.
    """
    for line in text.splitlines():
        # Markdown resumes lead with "# Name"; plain ones sometimes with a bullet.
        line = line.strip().lstrip("#*-• ").strip()
        if not line or len(line) > 60:
            continue
        words = [w for w in re.split(r"[\s,|]+", line) if w]
        if not 1 < len(words) <= 4:
            return []
        if any(w.lower().strip(".") in _NAME_STOPWORDS for w in words):
            return []

        # Two conventions, both common: "Amara Okonkwo" and the all-caps header
        # "AMARA OKONKWO". Missing the second one leaves the name unredacted in
        # the text the scorer sees, which is the failure this whole module
        # exists to prevent.
        title_case = r"[A-Z][a-z'.-]{1,}|[A-Z]\."
        all_caps = r"[A-Z][A-Z'.-]{1,}"
        if all(re.fullmatch(title_case, w) for w in words):
            return words
        if all(re.fullmatch(all_caps, w) for w in words):
            return words
        return []
    return []


def redact(text: str, known_name: str = "", extra_terms: list[str] | None = None) -> RedactionResult:
    """Strip identity and demographic signal from resume text.

    `known_name` is the name extracted in stage 1, if available. Passing it makes
    redaction far more reliable than the header heuristic alone.
    """
    if not text:
        return RedactionResult(text="", counts={})

    counts: dict[str, int] = {}
    out = text

    # Order matters: structured fields first, free text last.
    out = _sub(DEMOGRAPHIC_LINE_RE, _LABELS["demographic"], out, "demographic", counts)
    out = _sub(EMAIL_RE, _LABELS["email"], out, "email", counts)
    out = _sub(URL_RE, _LABELS["url"], out, "url", counts)
    out = _sub(DOB_RE, _LABELS["dob"], out, "dob", counts)
    out = _sub(ADDRESS_RE, _LABELS["address"], out, "address", counts)
    out = _sub(PHONE_RE, _LABELS["phone"], out, "phone", counts)
    out = _sub(INSTITUTION_RE, _LABELS["institution"], out, "institution", counts)

    # Names: the extracted name, then any caller-supplied aliases, then the
    # header heuristic as a backstop.
    name_tokens: list[str] = []
    if known_name:
        name_tokens += [t for t in re.split(r"[\s,]+", known_name) if len(t) > 1]
    if extra_terms:
        name_tokens += [t for t in extra_terms if len(t) > 1]
    if not name_tokens:
        name_tokens = guess_name_tokens(text)

    for token in dict.fromkeys(name_tokens):  # dedupe, keep order
        pattern = re.compile(rf"\b{re.escape(token)}\b", re.I)
        out = _sub(pattern, _LABELS["name"], out, "name", counts)

    # Collapse "[NAME] [NAME] [NAME]" into one token so the model can't count words.
    out, n = re.subn(r"(?:\[NAME\][\s,]*){2,}", "[NAME] ", out)
    if n:
        counts.setdefault("name", 0)

    out = _sub(PRONOUN_DECL_RE, "", out, "pronoun", counts)
    out = _sub(TITLE_RE, _LABELS["title"], out, "title", counts)
    out = _sub(PRONOUN_RE, _LABELS["pronoun"], out, "pronoun", counts)

    return RedactionResult(text=out, counts=counts)


def leaked_terms(redacted_text: str, terms: list[str]) -> list[str]:
    """Return any `terms` still present. Used by the redaction coverage test."""
    lowered = redacted_text.lower()
    return [t for t in terms if t and t.lower() in lowered]
