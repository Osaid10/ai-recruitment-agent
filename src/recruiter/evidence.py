"""Verification that a cited quote actually appears in its source document.

The project's central claim is that every score and rating is backed by evidence
quoted from the resume or transcript. Checking that the `evidence` field is
merely non-empty does not enforce that — a model will happily write "The
candidate explained their approach clearly", which is a description, not a quote.

Measured on the sample set, the ranking stage quoted verbatim 26 times out of 26
while the summarisation stage did so once in eight. The prompt alone is not a
control. This module is the control.
"""

from __future__ import annotations

import re

# A quote shorter than this can match by coincidence ("Python", "the team").
MIN_QUOTE_WORDS = 4

# How much of a long quote must line up before we accept it. Models often stitch
# a quote across a line break or drop a parenthetical, and rejecting those would
# throw away good evidence.
PREFIX_WORDS = 6

# Phrases that mark a description rather than a quotation.
PARAPHRASE_MARKERS = re.compile(
    r"^\s*(the\s+)?(candidate|applicant|interviewee)\b|"
    r"^\s*(they|he|she)\s+(described|explained|demonstrated|showed|mentioned|discussed)\b",
    re.I,
)

NOT_EVIDENCE = {
    "insufficient evidence",
    "no evidence",
    "not assessed",
    "n/a",
    "none",
    "",
}

# Models phrase an absence many ways — "No relevant experience found", "No
# evidence of shipped systems", "Unable to find any mention". Match the shape
# (a negation, then an evidence-ish noun) rather than listing the variants.
ABSENCE_RE = re.compile(
    r"^\s*(no|none|not|insufficient|unable to find|could not find|couldn't find)\b"
    r"[^.]*\b(evidence|experience|mention|information|data|detail|found|assessed|"
    r"support|record|example)\b",
    re.I,
)


def normalise(text: str) -> str:
    """Collapse to comparable form: lowercase, alphanumerics and single spaces.

    Deliberately lossy — it forgives differences in punctuation, hyphenation and
    whitespace that are not meaningful, while still requiring the same words in
    the same order.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_explicit_absence(quote: str) -> bool:
    """True when the model correctly reported that it found nothing.

    This is a valid, useful answer — not a failed quotation — and must not be
    penalised as a fabrication.
    """
    stripped = quote.strip().lower().rstrip(".")
    if stripped in NOT_EVIDENCE:
        return True
    return bool(ABSENCE_RE.match(stripped))


def quote_appears_in(quote: str, source: str) -> bool:
    """Does this quote actually occur in the source document?"""
    if not quote or not source:
        return False

    needle, haystack = normalise(quote), normalise(source)
    if not needle:
        return False

    # Length check comes first, deliberately. A one-word "quote" like "Python"
    # is present in almost any technical resume, so accepting it would let a
    # score cite a word rather than a claim — a substring match is not evidence.
    words = needle.split()
    if len(words) < MIN_QUOTE_WORDS:
        return False

    if needle in haystack:
        return True

    # Tolerate a quote stitched across a line break or with a middle clause
    # dropped: accept when a substantial run from either end matches.
    if len(words) >= PREFIX_WORDS:
        head = " ".join(words[:PREFIX_WORDS])
        tail = " ".join(words[-PREFIX_WORDS:])
        return head in haystack or tail in haystack

    return False


def looks_paraphrased(quote: str) -> bool:
    """Cheap check for the commonest failure: a description in quote's clothing."""
    return bool(PARAPHRASE_MARKERS.search(quote))


def verify(quote: str, source: str) -> tuple[bool, str]:
    """Return (is_supported, reason).

    `is_supported` false means the rating built on this quote must be discarded
    rather than shown to a human as evidence-backed.
    """
    if is_explicit_absence(quote):
        return False, "explicitly reported no evidence"
    if not quote.strip():
        return False, "no evidence given"
    if quote_appears_in(quote, source):
        return True, "verbatim"
    if looks_paraphrased(quote):
        return False, "paraphrased rather than quoted"
    return False, "quote not found in the source document"
