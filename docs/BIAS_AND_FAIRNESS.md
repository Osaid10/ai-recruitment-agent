# Bias & Fairness

The brief's Responsible Use note requires two things: final hiring decisions stay
with a human, and the ranking criteria are documented and tested for discriminatory
outcomes. This file is that documentation.

## 1. Where the ranking criteria come from

Criteria are **not** invented by the LLM. They are declared per role in the job
requisition file (`data/jobs/*.json`) by a human, and each carries a weight:

```json
"must_haves":  [{"skill": "Python", "min_years": 4}],
"nice_to_haves": [{"skill": "Kubernetes", "weight": 0.5}],
"rubric": {"skills": 0.40, "experience": 0.30, "impact": 0.20, "communication": 0.10}
```

If a criterion is not in the requisition, it cannot affect the score. This makes the
criteria reviewable before a single resume is read, which is the point.

### Criteria we deliberately excluded

| Excluded | Reason |
|---|---|
| School / university name | Strong proxy for socioeconomic class and nationality; weak predictor of performance |
| Employer prestige | Same proxy problem; rewards network access, not skill |
| Years since graduation | Proxy for age |
| Continuous employment / "no gaps" | Penalises caregiving, illness, immigration, and layoffs |
| Native-speaker phrasing / grammar polish | Proxy for national origin; we assess whether the writing is *clear*, not whether it is idiomatic |
| Photo, address, hobbies | No job relevance, high demographic signal |

## 2. Redaction before ranking

`src/recruiter/redact.py` strips the following from resume text *before* it reaches
the rubric or any ranking prompt:

- Name, email, phone, postal address, personal URLs (LinkedIn/GitHub handles)
- Gender markers and pronouns, marital status, date of birth, age, nationality,
  religion, photo references
- Institution names (universities and schools)

The unredacted record stays in the ATS store for the recruiter's use — redaction
applies to the *scoring path* only, so the human reviewer still sees the whole
resume when they make the call.

Redaction is on by default. Turning it off requires `redact_for_ranking = false` in
config and is recorded in the audit log for every affected candidate.

## 3. Prompt-level controls

Ranking prompts:
- receive redacted text only,
- are instructed to score against the declared rubric dimensions and nothing else,
- must return an `evidence` quote from the resume for every dimension score.

A dimension score without supporting evidence is treated as invalid and re-requested
once, then dropped to "insufficient evidence" rather than guessed.

## 4. How we test for discriminatory outcomes

`tests/test_fairness.py` runs on every change:

**Name-swap test.** The same resume body is submitted under paired names chosen to
differ only in the demographic signal they carry (gendered pairs, and names common
to different ethnic groups). The deterministic rubric score must be *identical*, and
the LLM fit score must not differ by more than a tolerance of 0.05. A difference
larger than that fails the build.

**Gap test.** Two identical resumes, one with an 18-month employment gap. The gap
must not lower the score by itself; it may only appear as a topic in the generated
interview questions.

**Redaction coverage test.** After redaction, asserts that no seeded PII token
(names, schools, pronouns, DOB) survives in the text handed to the scorer.

**Distribution check.** Over the full sample set, logs mean score by resume format
and by resume length so a systematic advantage to, say, long resumes is visible.

## 5. Known limitations

Be honest about these in the demo:

- The name-swap test uses a small, hand-built set of name pairs. It catches gross
  bias, not subtle bias, and it cannot prove the model is fair.
- Redaction is regex- and list-based. Unusual name formats or an unlisted university
  can leak through. It reduces signal; it does not eliminate it.
- The LLM's judgement of "impact" and "communication" reflects its training data.
  These carry the lowest rubric weights for that reason.
- Rubric weights are a human choice and can themselves encode bias. They are stored
  in the requisition so they can be argued about.

## 6. Human control points

| Gate | Enforced by | What the human sees |
|---|---|---|
| Shortlist approval — before any candidate is contacted | `recommend.require_approval()`; scheduling refuses to run without an `approvals` row | Full ranked list, per-dimension scores, evidence quotes, and everyone who was cut with the reason |
| Final decision — after the recommendation | The pipeline ends at `status="awaiting_decision"`. There is no code path that hires or rejects | Recommendation, confidence, evidence, dissenting signals, and the full audit trail |

Every stage writes an `audit` row recording the model used, the inputs, the output,
and the timestamp, so any decision can be reconstructed months later.
