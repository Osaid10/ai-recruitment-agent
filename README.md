# AI Recruitment Agent

An agent that runs the top of the hiring funnel — reads resumes, ranks candidates
against a role, books interviews, writes tailored questions, summarises the
interviews, and hands a hiring manager a recommendation.

**It never makes the hiring decision.** A human does, at two enforced gates.

Built for the Mercurial Minds Agentic AI internship.

---

## What it does

Recruiters lose most of their time to the same four jobs: reading a stack of
resumes, working out who is worth talking to, finding a time everyone is free,
and writing up what happened. None of those are the judgement call — the
judgement call is who to hire. This agent does the four, and stops before the fifth.

Give it a job requisition and a folder of resumes, and it works through the
pipeline on its own, only coming back to a human when it needs a decision or
gets stuck:

```
resumes/ ─▶ 1 INGEST ─▶ 2 RANK ─▶ ⛔ human approves shortlist ─▶ 3 SCHEDULE
                                                                     │
                                                                     ▼
 6 RECOMMEND ◀─ 5 SUMMARIZE ◀─ interview happens ◀─ 4 QUESTIONS ─────┘
      │
      ▼
 ⛔ human decides: hire / reject / another round
```

| Stage | What it does |
|---|---|
| **1 Ingest** | Reads PDF, DOCX, TXT and Markdown. Extracts name, skills, roles, education, years of experience into a typed record. Flags scans, encrypted files, and unreadable resumes for manual review instead of scoring them zero. |
| **2 Rank** | Scores against the role. Hard requirements are decided by deterministic rules; the LLM judges the fuzzy dimensions from **redacted** text and must quote evidence for every score. |
| **3 Schedule** | Finds a slot where the whole interview panel is free, books it, and writes a real `.ics` invite you can open in Outlook or Google Calendar. |
| **4 Questions** | Writes questions specific to *that* candidate — verifying their claims, probing their actual gaps, answering the concerns screening raised. |
| **5 Summarize** | Turns a transcript into structured signals, each backed by a quote. Anything the interview did not cover is marked `not_assessed`, not guessed. |
| **6 Recommend** | An evidence-backed recommendation with a mandatory list of contrary evidence, then it stops and waits for a person. |

## Why it's an agent, not a chatbot

You give it a goal, not a prompt. It runs the stages itself, calls tools
(file parsers, a calendar, a database), checks its own output — scores without
evidence are rejected and re-requested, ratings without a transcript quote are
downgraded — and routes to a human when it hits a real decision or a wall it
cannot pass, like no available interview slot.

## Guardrails

These are the parts worth arguing about, so they are written down rather than assumed.

**A human decides.** There is no code path that hires or rejects anyone. Scheduling
refuses to run until a named person approves the shortlist
(`require_shortlist_approval` raises otherwise), and the pipeline terminates at
`awaiting_decision`.

**Nobody is silently dropped.** Candidates who miss the shortlist go into
`Shortlist.cut` with a stated reason, and the ones who were merely borderline or
waitlisted are flagged for a human look.

**Ranking is blind by default.** Names, contact details, schools, ages, and
demographic fields are stripped before scoring. The recruiter still sees the whole
resume; the *scorer* does not.

**Evidence or it did not happen.** Every score and rating carries a quote. Unsupported
ones are downgraded automatically, not trusted.

**Everything is auditable.** Every stage writes an `audit` row — model used, inputs,
output, timestamp — so any decision can be reconstructed later.

**Criteria are declared up front,** in the job requisition file, by a human, before
any resume is read. If a criterion is not in the requisition, it cannot affect the
score. The reasoning behind what we deliberately excluded — school prestige,
employment gaps, native-speaker phrasing — is in
[`docs/BIAS_AND_FAIRNESS.md`](docs/BIAS_AND_FAIRNESS.md), along with how it is tested.

## Quick start

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .            # puts `recruiter` on the import path

copy .env.example .env
# add a free Groq key from https://console.groq.com/keys
```

Run the whole pipeline:

```powershell
python -m recruiter run --job data/jobs/senior_python_engineer.json --resumes data/resumes
```

Or stage by stage:

```powershell
python -m recruiter ingest   --job data/jobs/senior_python_engineer.json --resumes data/resumes
python -m recruiter rank     --job <job_id>
python -m recruiter approve  --job <job_id> --by "Your Name"     # the human gate
python -m recruiter schedule --job <job_id>
python -m recruiter questions --job <job_id>
python -m recruiter summarize --candidate <cand_id> --transcript data/interviews/<file>.txt
python -m recruiter recommend --job <job_id>
python -m recruiter audit    --job <job_id>
```

The dashboard:

```powershell
streamlit run app/streamlit_app.py
```

It shows the ranking with the evidence behind each score, both human gates, the
interview questions, the recommendations, and the audit trail. You can point it
at a resume folder or drop files straight onto the sidebar uploader — uploads go
through the same parser, redaction and audit path as a folder run, so nothing
arriving that way skips the bias controls.

**It runs without an API key.** The deterministic rubric still ranks the whole
batch, and every LLM stage reports itself unavailable rather than inventing
output. You lose the fit assessment, not the pipeline.

## Layout

```
src/recruiter/
  config.py     settings from .env — no key ever in source
  models.py     Pydantic models; the contract between stages
  store.py      ATS database (SQLite) + append-only audit log
  llm.py        Groq wrapper: structured output, retries, graceful degradation
  redact.py     PII stripping applied before ranking
  ingest/       parser.py (file -> text), extract.py (text -> Candidate)
  rank/         rubric.py (deterministic gates), ranker.py (LLM fit + shortlist)
  schedule/     calendar.py (adapter + MockCalendar), ics.py (invite files)
  questions.py  summarize.py  recommend.py
  pipeline.py   the agent loop
  cli.py
app/streamlit_app.py    demo dashboard
data/                   job requisitions, sample resumes, transcripts
docs/                   bias documentation, demo walkthrough
tests/
```

## Tech choices

| | | |
|---|---|---|
| **LLM** | Groq `llama-3.3-70b-versatile` via `langchain-groq` | Free tier, fast enough to demo live |
| **Structured output** | Pydantic v2 + `with_structured_output` | Every stage gets typed data or a validation error — no prompt-output parsing |
| **Calendar** | `MockCalendar` + real `.ics` files | Full end-to-end demo with no OAuth setup. `CalendarAdapter` is a three-method protocol, so a real Google Calendar adapter drops in without touching anything else |
| **Database** | SQLite, stdlib only | The brief asks for ATS-style tracking; one file, no server, easy to inspect |
| **UI** | Streamlit | Fastest path to a dashboard that shows the agent's reasoning |

## Testing

```powershell
python -m pytest -q      # 125 tests, all green
```

Each file answers a different question:

- **`tests/test_fairness.py`** — does ranking condition on anything it shouldn't?
  Identical resumes under paired names (gendered, and names common to different
  ethnic groups) must score *identically*; school prestige must not move the
  score; an employment gap must not lower it; redaction must remove every seeded
  PII token while leaving the job-relevant content intact.
- **`tests/test_human_gates.py`** — can the agent act on a person alone? Scheduling
  and question generation are blocked until a named human approves, anonymous
  approvals are rejected, and a full run ends with recommendations but zero
  recorded hiring decisions.
- **`tests/test_scenarios.py`** — the ten scenarios from the plan, including the
  text-free PDF, the DOCX, the career gap, the near-empty resume, the malformed
  job spec, the fully booked calendar, and a complete run with no model at all.
- **`tests/test_evidence.py`** — is every score actually supported? A quote the
  model invented, or one that cannot be found in the source, must be discarded
  and its score zeroed rather than trusted.
- **`tests/test_redaction_precision.py`** — does redaction remove too much? A
  pattern once ate "Agentic AI" and "Multi-Agent Systems" off an AI engineer's
  resume, deleting the exact qualification the role asked for.
- **`tests/test_years_extraction.py`** — degree dates are not work experience, and
  a three-month internship written "Jun - Aug 2024" still counts as a role.
- **`tests/test_store_threading.py`** — Streamlit re-runs the script on a different
  worker thread while the SQLite connection is cached across those re-runs.
- **`tests/test_dashboard.py`** — drives the real widgets through Streamlit's
  `AppTest`: every tab against seeded and empty data, an anonymous approval being
  refused, downstream controls staying blocked until approval, and the view
  following the job that was just ranked.

The whole suite runs offline, so it passes on a machine with no API key.

## Design notes

A few decisions that are easy to get wrong, and why they went this way:

**Deterministic before probabilistic.** Hard requirements are rules, not model
judgement, so they are reproducible and explainable line by line. The model only
scores the fuzzy dimensions, and it can never override a failed hard gate.

**Skill matching is keyword-based, and it says so.** Because per-skill years are
almost never stated on a resume, the years minimum is checked against *total*
experience. That means a career-changer's evening course can technically clear a
four-year gate. Rather than paper over it, the ranker raises a flag naming the
exact problem, and there is a test asserting that flag fires.

**Redaction is a control, not a guarantee.** It is regex- and list-based, so an
unusual name format or an unlisted institution can leak through. It reduces
demographic signal; it does not eliminate it.

**Failures degrade, they don't crash.** A stage that cannot complete returns a
result with a reason, the pipeline records it, and the run continues with the
other candidates. One unreadable resume never ends a batch.
