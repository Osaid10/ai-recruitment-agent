"""Presentation components for the dashboard.

Kept separate from `streamlit_app.py` so the page reads as a sequence of
sections rather than a wall of markup.

Colour follows the job the value does, which for this screen is only ever two
things:

* **Magnitude** — a score out of 100 or a dimension out of 10. That is one
  sequential hue, light to dark. It is deliberately *not* a categorical palette:
  the four rubric dimensions are four measurements of the same thing, each
  directly labelled, so hue carries no identity and a rainbow would imply a
  distinction that does not exist.
* **State** — shortlisted, cut, needs a human. That is the reserved status
  palette, and every status ships with a label, never colour alone.

The hexes and both surfaces are taken unchanged from the reference palette, so
its published contrast and colour-vision validation applies as written.
"""

from __future__ import annotations

import html

import streamlit as st

# -- tokens ----------------------------------------------------------------

CSS = """
<style>
:root {
  --surface-1: #fcfcfb;
  --page: #f9f9f7;
  --ink-1: #0b0b0b;
  --ink-2: #52514e;
  --ink-muted: #898781;
  --grid: #e1e0d9;
  --baseline: #c3c2b7;
  --hairline: rgba(11,11,11,0.10);
  --track: #ece9e2;

  /* sequential blue - magnitude only */
  --mag-100: #cde2fb;
  --mag-250: #86b6ef;
  --mag-450: #2a78d6;
  --mag-600: #184f95;

  /* reserved status - state only, never a series */
  --good: #0ca30c;
  --warning: #fab219;
  --serious: #ec835a;
  --critical: #d03b3b;
}

@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --ink-1: #ffffff;
    --ink-2: #c3c2b7;
    --ink-muted: #898781;
    --grid: #2c2c2a;
    --baseline: #383835;
    --hairline: rgba(255,255,255,0.10);
    --track: #2c2c2a;
    --mag-450: #3987e5;
    --mag-600: #86b6ef;
  }
}
:root[data-theme="dark"] {
  --surface-1: #1a1a19;
  --page: #0d0d0d;
  --ink-1: #ffffff;
  --ink-2: #c3c2b7;
  --grid: #2c2c2a;
  --baseline: #383835;
  --hairline: rgba(255,255,255,0.10);
  --track: #2c2c2a;
  --mag-450: #3987e5;
  --mag-600: #86b6ef;
}

/* -- page chrome -------------------------------------------------------- */

.block-container { padding-top: 2.4rem; max-width: 1180px; }

.ra-eyebrow {
  font-size: 0.70rem;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--ink-muted);
  font-weight: 600;
  margin-bottom: 0.35rem;
}

.ra-meta {
  color: var(--ink-2);
  font-size: 0.86rem;
  line-height: 1.6;
}

.ra-rule {
  height: 1px;
  background: var(--hairline);
  border: 0;
  margin: 1.6rem 0 1.2rem;
}

/* -- stat tiles --------------------------------------------------------- */

.ra-tiles { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.4rem 0 0.2rem; }

.ra-tile {
  flex: 1 1 150px;
  background: var(--surface-1);
  border: 1px solid var(--hairline);
  border-radius: 10px;
  padding: 0.85rem 0.95rem;
}
.ra-tile-label {
  font-size: 0.70rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-muted);
  font-weight: 600;
}
.ra-tile-value {
  font-size: 1.85rem;
  line-height: 1.15;
  font-weight: 640;
  color: var(--ink-1);
  margin-top: 0.15rem;
}
.ra-tile-hint { font-size: 0.76rem; color: var(--ink-2); margin-top: 0.1rem; }

/* -- score meter (magnitude) -------------------------------------------- */

.ra-meter-row {
  display: grid;
  grid-template-columns: 128px 1fr 54px;
  align-items: center;
  gap: 12px;
  margin: 5px 0;
}
.ra-meter-label { font-size: 0.82rem; color: var(--ink-2); }
.ra-meter-track {
  position: relative;
  height: 8px;
  background: var(--track);
  border-radius: 4px;
  overflow: hidden;
}
.ra-meter-fill {
  height: 100%;
  border-radius: 4px;
  background: var(--mag-450);
}
.ra-meter-fill.is-empty { background: var(--baseline); }
.ra-meter-value {
  font-size: 0.82rem;
  color: var(--ink-1);
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-weight: 560;
}

/* -- headline score ----------------------------------------------------- */

.ra-headline { display: flex; align-items: baseline; gap: 10px; }
.ra-headline-value {
  font-size: 2.3rem;
  font-weight: 660;
  color: var(--ink-1);
  line-height: 1;
}
.ra-headline-unit { font-size: 0.9rem; color: var(--ink-muted); }

/* -- status pills (state, always with a label) -------------------------- */

.ra-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 9px 2px 7px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 560;
  color: var(--ink-1);
  background: var(--surface-1);
  border: 1px solid var(--hairline);
  margin: 2px 4px 2px 0;
}
.ra-dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.ra-dot-good { background: var(--good); }
.ra-dot-warning { background: var(--warning); }
.ra-dot-serious { background: var(--serious); }
.ra-dot-critical { background: var(--critical); }
.ra-dot-neutral { background: var(--baseline); }

/* -- evidence quote ----------------------------------------------------- */

.ra-quote {
  border-left: 2px solid var(--mag-250);
  padding: 2px 0 2px 11px;
  margin: 4px 0 10px;
  color: var(--ink-2);
  font-size: 0.83rem;
  line-height: 1.55;
}
.ra-quote-none {
  border-left-color: var(--baseline);
  color: var(--ink-muted);
  font-style: italic;
}
.ra-quote-src {
  display: block;
  font-size: 0.70rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 2px;
  font-weight: 600;
}

/* -- gate callout ------------------------------------------------------- */

.ra-gate {
  border: 1px solid var(--hairline);
  border-left: 3px solid var(--warning);
  border-radius: 8px;
  padding: 0.75rem 0.9rem;
  background: var(--surface-1);
  margin: 0.5rem 0 1rem;
}
.ra-gate-title { font-weight: 620; color: var(--ink-1); font-size: 0.9rem; }
.ra-gate-body { color: var(--ink-2); font-size: 0.83rem; margin-top: 0.2rem; line-height: 1.55; }
.ra-gate.is-done { border-left-color: var(--good); }

/* -- misc --------------------------------------------------------------- */

.ra-note { font-size: 0.78rem; color: var(--ink-muted); line-height: 1.55; }
.ra-flag {
  font-size: 0.79rem;
  color: var(--ink-2);
  padding: 5px 0 5px 11px;
  border-left: 2px solid var(--warning);
  margin: 4px 0;
  line-height: 1.5;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _esc(text: str) -> str:
    return html.escape(str(text))


# -- components ------------------------------------------------------------


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="ra-eyebrow">{_esc(text)}</div>', unsafe_allow_html=True)


def rule() -> None:
    st.markdown('<hr class="ra-rule" />', unsafe_allow_html=True)


def stat_tiles(tiles: list[tuple[str, str, str]]) -> None:
    """A KPI row. Each tile is (label, value, hint)."""
    cells = "".join(
        f'<div class="ra-tile">'
        f'<div class="ra-tile-label">{_esc(label)}</div>'
        f'<div class="ra-tile-value">{_esc(value)}</div>'
        f'<div class="ra-tile-hint">{_esc(hint)}</div>'
        f"</div>"
        for label, value, hint in tiles
    )
    st.markdown(f'<div class="ra-tiles">{cells}</div>', unsafe_allow_html=True)


def headline_score(value: float, out_of: int = 100) -> None:
    st.markdown(
        f'<div class="ra-headline">'
        f'<span class="ra-headline-value">{value:.0f}</span>'
        f'<span class="ra-headline-unit">/ {out_of}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def meter(label: str, value: float, out_of: float = 10.0, suffix: str = "") -> str:
    """One magnitude bar. Returns markup so several can be emitted together."""
    pct = 0.0 if out_of <= 0 else max(0.0, min(100.0, (value / out_of) * 100))
    empty = " is-empty" if value <= 0 else ""
    shown = f"{value:.1f}{suffix}" if value else f"0{suffix}"
    return (
        f'<div class="ra-meter-row">'
        f'<div class="ra-meter-label">{_esc(label)}</div>'
        f'<div class="ra-meter-track">'
        f'<div class="ra-meter-fill{empty}" style="width:{pct:.1f}%"></div>'
        f"</div>"
        f'<div class="ra-meter-value">{_esc(shown)}</div>'
        f"</div>"
    )


def meters(rows: list[tuple[str, float, float]]) -> None:
    st.markdown("".join(meter(a, b, c) for a, b, c in rows), unsafe_allow_html=True)


def pill(text: str, kind: str = "neutral") -> str:
    """A state chip. Colour is never the only signal — the label always shows."""
    return (
        f'<span class="ra-pill"><span class="ra-dot ra-dot-{kind}"></span>'
        f"{_esc(text)}</span>"
    )


def pills(items: list[tuple[str, str]]) -> None:
    st.markdown("".join(pill(t, k) for t, k in items), unsafe_allow_html=True)


def quote(text: str, source: str = "evidence") -> None:
    missing = not text or text.strip().lower().startswith(
        ("insufficient", "discarded", "no evidence", "not assessed")
    )
    cls = "ra-quote ra-quote-none" if missing else "ra-quote"
    body = _esc(text) if text else "no evidence given"
    st.markdown(
        f'<div class="{cls}"><span class="ra-quote-src">{_esc(source)}</span>'
        f"{body}</div>",
        unsafe_allow_html=True,
    )


def gate(title: str, body: str, done: bool = False) -> None:
    cls = "ra-gate is-done" if done else "ra-gate"
    st.markdown(
        f'<div class="{cls}"><div class="ra-gate-title">{_esc(title)}</div>'
        f'<div class="ra-gate-body">{_esc(body)}</div></div>',
        unsafe_allow_html=True,
    )


def flag(text: str) -> None:
    st.markdown(f'<div class="ra-flag">{_esc(text)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="ra-note">{_esc(text)}</div>', unsafe_allow_html=True)


def meta(text: str) -> None:
    st.markdown(f'<div class="ra-meta">{_esc(text)}</div>', unsafe_allow_html=True)
