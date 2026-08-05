"""Stage 2a — deterministic scoring.

Hard requirements are decided by rules, never by the model. This is the part of
ranking that is reproducible, explainable line by line, and identical for two
resumes that differ only in whose name is on top.

The LLM's judgement (`ranker.py`) is blended on top, but it can never flip a
failed hard gate.
"""

from __future__ import annotations

import re

from ..models import Candidate, HardGate, JobRequisition, SkillRequirement

# How the deterministic score is composed.
MUST_HAVE_WEIGHT = 0.70
EXPERIENCE_WEIGHT = 0.15
NICE_TO_HAVE_WEIGHT = 0.15

# Common ways the same technology is written on a resume. Matching on the raw
# string alone is the single biggest source of false negatives in resume
# screening, and false negatives here mean a real person gets dropped.
ALIASES: dict[str, list[str]] = {
    "python": ["python3", "py3"],
    "javascript": ["js", "ecmascript", "es6"],
    "typescript": ["ts"],
    "postgresql": ["postgres", "psql"],
    "kubernetes": ["k8s"],
    "amazon web services": ["aws"],
    "aws": ["amazon web services"],
    "google cloud platform": ["gcp"],
    "gcp": ["google cloud platform"],
    "continuous integration": ["ci/cd", "ci cd", "cicd"],
    "ci/cd": ["continuous integration", "continuous delivery", "cicd"],
    "machine learning": ["ml"],
    "natural language processing": ["nlp"],
    "rest": ["restful", "rest api", "rest apis"],
    "sql": ["mysql", "postgresql", "postgres", "sqlite", "t-sql", "plsql"],
    "docker": ["containerization", "containerisation"],
    "fastapi": ["fast api"],
    "node.js": ["nodejs", "node"],
    "c#": ["csharp", "c sharp"],
    "react": ["react.js", "reactjs"],
}


def _variants(skill: str) -> list[str]:
    key = skill.lower().strip()
    return [key, *ALIASES.get(key, [])]


def skill_present(skill: str, candidate: Candidate, search_text: str = "") -> bool:
    """Look for a skill in the extracted skills list first, then in the resume body.

    `search_text` should be the *redacted* text when redaction is on.
    """
    haystacks = [s.lower() for s in candidate.skills]
    joined = " | ".join(haystacks)
    body = (search_text or candidate.raw_text).lower()

    for variant in _variants(skill):
        escaped = re.escape(variant)
        # \b breaks on tokens ending in punctuation like "c#" and "node.js"
        pattern = rf"(?<![\w#+.]){escaped}(?![\w#+.])"
        if re.search(pattern, joined) or re.search(pattern, body):
            return True
    return False


# A must-have named this many times or fewer counts as thin evidence.
WEAK_EVIDENCE_MENTIONS = 2


def mention_count(skill: str, text: str) -> int:
    """How many times a skill is named in the resume body.

    A must-have named once or twice across a whole resume is usually a passing
    reference - "took an online Python course" - rather than evidence of
    professional depth. We do not change the score on that basis, because a
    terse senior resume can also be sparse. We flag it so a human looks.
    """
    lowered = text.lower()
    total = 0
    for variant in _variants(skill):
        escaped = re.escape(variant)
        total += len(re.findall(rf"(?<![\w#+.]){escaped}(?![\w#+.])", lowered))
    return total


def evaluate_gates(
    candidate: Candidate, job: JobRequisition, search_text: str = ""
) -> list[HardGate]:
    """Pass/fail checks for every declared must-have plus overall experience."""
    gates: list[HardGate] = []

    for req in job.must_haves:
        present = skill_present(req.skill, candidate, search_text)
        if not present:
            detail = f"no mention of {req.skill} in the resume"
        elif req.min_years and candidate.total_years_experience < req.min_years:
            detail = (
                f"{req.skill} found, but total experience "
                f"{candidate.total_years_experience:g}y is under the {req.min_years:g}y minimum"
            )
        else:
            detail = f"{req.skill} found"
        met = present and (
            not req.min_years or candidate.total_years_experience >= req.min_years
        )
        gates.append(HardGate(requirement=f"must-have: {req.skill}", met=met, detail=detail))

    if job.min_years_experience:
        met = candidate.total_years_experience >= job.min_years_experience
        gates.append(
            HardGate(
                requirement=f"minimum {job.min_years_experience:g} years experience",
                met=met,
                detail=f"resume shows {candidate.total_years_experience:g} years",
            )
        )

    return gates


def _coverage(requirements: list[SkillRequirement], candidate: Candidate, text: str) -> float:
    """Weighted fraction of requirements the candidate satisfies. 1.0 if none declared."""
    if not requirements:
        return 1.0
    total = sum(r.weight for r in requirements) or 1.0
    got = sum(r.weight for r in requirements if skill_present(r.skill, candidate, text))
    return got / total


def deterministic_score(
    candidate: Candidate, job: JobRequisition, search_text: str = ""
) -> tuple[float, list[HardGate], list[str]]:
    """Return (score 0-100, gates, flags).

    Flags are things a human should look at — they never change the score.
    """
    gates = evaluate_gates(candidate, job, search_text)
    flags: list[str] = []

    must_ratio = _coverage(job.must_haves, candidate, search_text)
    nice_ratio = _coverage(job.nice_to_haves, candidate, search_text)

    if job.min_years_experience > 0:
        exp_ratio = min(1.0, candidate.total_years_experience / job.min_years_experience)
    else:
        exp_ratio = 1.0

    score = 100.0 * (
        MUST_HAVE_WEIGHT * must_ratio
        + EXPERIENCE_WEIGHT * exp_ratio
        + NICE_TO_HAVE_WEIGHT * nice_ratio
    )

    # Skill matching is keyword-based, and `min_years` is checked against total
    # experience because per-skill years are rarely stated on a resume. That
    # combination can pass a career-changer whose only mention of a must-have is
    # an evening course. Surface it rather than pretend the gate is precise.
    body = search_text or candidate.raw_text
    weak = [
        (req.skill, mention_count(req.skill, body))
        for req in job.must_haves
        if skill_present(req.skill, candidate, body)
        and mention_count(req.skill, body) <= WEAK_EVIDENCE_MENTIONS
    ]
    if weak:
        flags.append(
            "weak evidence - "
            + ", ".join(f"{skill} appears only {n}x" for skill, n in weak)
            + "; the years minimum was checked against total experience "
            f"({candidate.total_years_experience:g}y), not experience with that skill"
        )

    if candidate.extraction_method != "llm":
        flags.append("profile extracted heuristically — verify the parsed fields")
    if candidate.parse_warnings:
        flags.append("resume had parse warnings — check the original file")
    if candidate.total_years_experience == 0:
        flags.append("no dated experience found — years may be unparseable, not absent")
    if len(candidate.raw_text) < 400:
        flags.append("very short resume — limited evidence to assess")
    failed = [g.requirement for g in gates if not g.met]
    if failed and len(failed) < len(gates):
        flags.append(f"partially meets requirements — missing: {', '.join(failed)}")

    return round(score, 1), gates, flags


def explain(score: float, gates: list[HardGate]) -> str:
    """One-line, human-readable justification for the deterministic score."""
    met = sum(1 for g in gates if g.met)
    return f"deterministic score {score:.1f}/100; {met}/{len(gates)} hard requirements met"
