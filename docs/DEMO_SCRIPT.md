# Demo Walkthrough (5–10 minutes)

A walkthrough script for demoing the agent. The structure is deliberate: show
the pipeline, then the guardrails, then be honest about the limits. The third
part is what makes the first two credible.

## Before you start

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
copy .env.example .env      # add your Groq key
Remove-Item data\ats.db -ErrorAction SilentlyContinue   # start clean
```

Have two terminals open, plus a browser tab for the dashboard.

---

## 1. The problem (45 seconds)

> "A recruiter filling one role reads a hundred CVs, works out who's worth
> talking to, finds a slot everyone's free, and writes up each interview. None
> of that is the judgement call. The judgement call is who to hire. This agent
> does the four jobs and stops before the fifth."

Show the pipeline diagram in the README.

---

## 2. One command, the whole funnel (90 seconds)

```powershell
python -m recruiter run --job data/jobs/senior_python_engineer.json --resumes data/resumes
```

Talk over the output as it scrolls:

- **Nine resumes, four formats** — PDF, DOCX, TXT, Markdown.
- **Point at `07_scanned_no_text_layer.pdf`.** "That's a scan with no text layer.
  It didn't crash the run and it didn't score zero — it's flagged for a human.
  Scoring an unreadable CV as zero is how you silently reject someone."
- **Point at `08_short_submission.txt`.** "Same idea. One line. Flagged, not judged."
- **Then the last line:** the run **stopped**. It ranked everyone and refused to
  go further.

> "It stopped on purpose. Nobody gets contacted until a person approves."

---

## 3. The ranking, and why it's defensible (2 minutes)

```powershell
python -m recruiter rank --job job_seniorpy
```

Three things to point at in the table:

**The Rules column.** "Hard requirements — years, must-have skills — are scored
by plain rules, not by the model. Same input, same output, every time, and I can
explain any number on that line by line."

**Zainab Qureshi, ranked second.** "She has an eighteen-month career break. She's
still second. The gap doesn't cost her anything — there's a test that fails the
build if it ever does."

**Maryam Shah, at the bottom.** Open her row and read the flag aloud:

> *weak evidence - Python appears only 2x; the years minimum was checked against
> total experience (9.4y), not experience with that skill*

> "She's a graphic designer who took an online Python course. Keyword matching
> can't tell that apart from four years of professional Python. Rather than
> pretend the gate is precise, it says exactly what it doesn't know."

**And the ones who were cut** — point out that Sana Iqbal's row says *"qualified
but outside the top 4 — waitlisted, not rejected."*

> "Nobody in this system is rejected by software. They're ranked, with a written
> reason, and a person decides."

---

## 4. Blind screening (1 minute)

This is usually the question that gets asked, so get ahead of it.

```powershell
python -m pytest tests/test_fairness.py -v
```

> "The same CV goes in under paired names — gendered pairs, and names common to
> different ethnic groups. If the scores differ *at all*, the build fails. Same
> for school prestige, and for employment gaps."

Then explain the mechanism: names, emails, universities, ages, pronouns and
demographic fields are stripped *before* scoring. The recruiter still sees the
whole CV — redaction narrows what the **scorer** can condition on, not what
people can read.

Point at `docs/BIAS_AND_FAIRNESS.md`: "the criteria we deliberately excluded, and
why, are written down so they can be argued with."

---

## 5. The gate (1 minute)

Try to skip ahead, and let it refuse:

```powershell
python -m recruiter schedule --job job_seniorpy
```

> *shortlist for job_seniorpy has not been approved by a human*

> "That's not a warning. The function raises."

Now approve, as a named person:

```powershell
python -m recruiter approve --job job_seniorpy --by "Your Name"
python -m recruiter schedule --job job_seniorpy
python -m recruiter questions --job job_seniorpy --show
```

Open one of the generated `.ics` files from `out/invites/` in Outlook or Google
Calendar — it's a real invite, with the interview agenda already in the body.

On the questions: point out that each one cites *what prompted it* — a specific
claim on that CV, or a specific gap. "These questions are unusable for any other
candidate. That's the point."

---

## 6. The dashboard (90 seconds)

```powershell
streamlit run app/streamlit_app.py
```

Walk three tabs:

- **Ranking** — expand a candidate: every dimension score has the evidence quote
  that justifies it. "If the model scores someone without quoting the CV, the
  code overwrites the score to zero. An unsupported score is treated as a bug."
- **Recommendations** — every one carries an **Evidence against** section. "That
  field is mandatory. A one-sided case is a red flag, so the agent is forced to
  argue against itself."
- **Audit trail** — "every stage writes a row: what ran, what it decided, which
  model, when. If someone asks in six months why a candidate was cut, it's here."

---

## 7. Where it falls down (45 seconds)

Do not skip this. It's the most convincing part.

> - "Redaction is regex-based. An unusual name format or an unlisted university
>   can leak through. It reduces signal, it doesn't eliminate it."
> - "The name-swap test uses a small hand-built set of pairs. It catches gross
>   bias. It cannot prove the model is fair."
> - "Skill matching is keywords, so 'took a Python course' and 'four years of
>   Python' look similar to it. That's what the weak-evidence flag is for."
> - "The rubric weights are a human choice, and they could encode bias
>   themselves. They're in the requisition file so they can be argued about
>   before anyone's CV is read."

Close on the scope line:

> "It never decides. It reads, ranks, books, drafts and summarises — and then
> it hands a person a file with the evidence and gets out of the way."

---

## If the API is unreachable mid-demo

Turn it into a feature, because it is one:

> "The model's unreachable. Watch — deterministic ranking still runs, and every
> model-backed stage reports itself unavailable instead of inventing output. It
> degrades, it doesn't guess."

Graceful degradation is a design goal here, and demonstrating it deliberately is
stronger than hitting it by accident.

## Likely questions

| Question | Answer |
|---|---|
| "Could it auto-reject to save time?" | No — and that's a design decision, not a missing feature. There's no code path that rejects. Candidates who miss the cut are recorded with a reason for a person to review. |
| "What if the LLM hallucinates a qualification?" | Every score needs an evidence quote from the CV. No quote, and the code zeroes the score. It's checked, not trusted. |
| "Why not just use embeddings similarity?" | A similarity score can't be explained to a rejected candidate or a regulator. Hard requirements are rules; the model only judges the fuzzy dimensions, and it must cite evidence. |
| "Is this legal to use in hiring?" | It's a screening aid with a human decision-maker, which is the posture most regulation expects. The bias documentation and audit log exist because "we tested for this" needs to be demonstrable. It is not a compliance product. |
| "How would you swap in a real calendar?" | `CalendarAdapter` is a three-method protocol. A Google Calendar adapter implements `busy`, `book`, `cancel` and nothing else changes. |
