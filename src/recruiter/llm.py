"""Groq LLM wrapper.

Two jobs: bind every call to a Pydantic schema so stages get typed data back,
and fail in a way callers can handle. If there is no API key, or Groq is down,
callers get `LLMUnavailable` and fall back to their deterministic path — the
pipeline must still produce a ranked shortlist with no model at all.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from typing import TypeVar

from pydantic import BaseModel

from .config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Groq's free tier is metered on tokens per minute, and a full pipeline run over
# a folder of resumes will exceed it. Pace requests instead of discovering the
# limit through 429s: a rate-limited stage falls back to its template path, and
# a demo that silently degrades is worse than one that takes a few seconds longer.
CHARS_PER_TOKEN = 4  # rough, but only needs to be right to within ~25%
RATE_WINDOW_SECONDS = 60

# "Please try again in 1.08s" / "in 910ms" — Groq puts the wait in the message body.
RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)\s*(ms|s)\b", re.I)


class LLMUnavailable(RuntimeError):
    """No usable model: missing key, missing package, or the call kept failing."""


def _retry_after(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, if it said."""
    match = RETRY_AFTER_RE.search(str(exc))
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000 if match.group(2).lower() == "ms" else value


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate_limit" in text or "429" in text


def _is_daily_quota(exc: Exception) -> bool:
    """A per-day cap, as opposed to a per-minute one.

    Worth distinguishing: a per-minute limit clears in seconds and is worth
    waiting out, while a daily cap does not clear for hours. Retrying into a
    daily cap just burns the retry budget and delays the fallback path.
    """
    text = str(exc).lower()
    return "tokens per day" in text or "tpd" in text or "requests per day" in text


class TokenPacer:
    """Keeps a rolling estimate of tokens spent in the last minute and sleeps
    before a call that would breach the budget.

    Deliberately conservative and approximate — the point is to stay clear of
    the ceiling, not to model the provider's accounting exactly.
    """

    def __init__(self, tokens_per_minute: int) -> None:
        self.budget = max(1000, tokens_per_minute)
        self._spent: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._spent and now - self._spent[0][0] > RATE_WINDOW_SECONDS:
            self._spent.popleft()

    def reserve(self, estimated_tokens: int) -> None:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            used = sum(t for _, t in self._spent)

            if used + estimated_tokens > self.budget and self._spent:
                # Wait until the oldest entry ages out of the window.
                wait = RATE_WINDOW_SECONDS - (now - self._spent[0][0]) + 0.5
                if wait > 0:
                    log.info(
                        "pacing: %d/%d tokens used this minute, sleeping %.1fs",
                        used,
                        self.budget,
                        wait,
                    )
                    time.sleep(wait)
                    now = time.monotonic()
                    self._prune(now)

            self._spent.append((now, estimated_tokens))

    def record_actual(self, tokens: int) -> None:
        """Correct the last estimate once the real usage is known."""
        with self._lock:
            if self._spent:
                stamp, _ = self._spent[-1]
                self._spent[-1] = (stamp, tokens)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self._chat = None
        self._init_error: str = ""
        self._pacer = TokenPacer(self.settings.tokens_per_minute)

        if not self.settings.groq_api_key:
            self._init_error = (
                "GROQ_API_KEY is not set. Copy .env.example to .env and add a key "
                "from https://console.groq.com/keys"
            )
            return

        try:
            from langchain_groq import ChatGroq
        except ImportError:  # pragma: no cover - environment problem, not logic
            self._init_error = (
                "langchain-groq is not installed. Run: "
                "python -m pip install -r requirements.txt"
            )
            return

        try:
            self._chat = ChatGroq(
                api_key=self.settings.groq_api_key,
                model=self.settings.model,
                temperature=self.settings.temperature,
                timeout=60,
                max_retries=0,  # we retry ourselves so we can log each attempt
            )
        except Exception as exc:  # pragma: no cover
            self._init_error = f"could not construct ChatGroq: {exc}"

    # -- capability -------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._chat is not None

    @property
    def unavailable_reason(self) -> str:
        return self._init_error

    @property
    def model_name(self) -> str:
        return self.settings.model if self.available else "none"

    def _require(self) -> None:
        if not self.available:
            raise LLMUnavailable(self._init_error or "LLM unavailable")

    # -- calls ------------------------------------------------------------

    def structured(
        self,
        schema: type[T],
        system: str,
        user: str,
        *,
        attempts: int = 4,
    ) -> T:
        """Call the model and return a validated instance of `schema`.

        Retries on transient errors and on schema-validation failures — a
        validation failure usually means the model returned a near-miss, and
        one more attempt with the same prompt normally fixes it.
        """
        self._require()
        runnable = self._chat.with_structured_output(schema)
        messages = [("system", system), ("human", user)]
        self._pacer.reserve(self._estimate_tokens(system, user))

        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                result = runnable.invoke(messages)
                if isinstance(result, schema):
                    return result
                # Some providers hand back a dict; validate it ourselves.
                return schema.model_validate(result)
            except Exception as exc:
                last_error = exc
                if _is_daily_quota(exc):
                    # Hours away, not seconds — go to the fallback path now
                    # rather than spending three more retries discovering that.
                    raise LLMUnavailable(
                        f"daily token quota exhausted: {self._brief(exc)}"
                    ) from exc
                log.warning(
                    "LLM structured call failed (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    self._brief(exc),
                )
                if attempt < attempts:
                    time.sleep(self._backoff(exc, attempt))

        raise LLMUnavailable(
            f"{schema.__name__} call failed after {attempts} attempts: {self._brief(last_error)}"
        ) from last_error

    # -- retry helpers ----------------------------------------------------

    @staticmethod
    def _estimate_tokens(*parts: str) -> int:
        """Rough prompt size, plus headroom for the completion."""
        chars = sum(len(p) for p in parts)
        return int(chars / CHARS_PER_TOKEN) + 800

    @staticmethod
    def _brief(exc: Exception | None) -> str:
        """Providers return a wall of JSON on a 429; keep the log readable."""
        if exc is None:
            return "unknown error"
        if _is_daily_quota(exc):
            return "daily token quota exhausted — resets on the provider's schedule"
        if _is_rate_limit(exc):
            wait = _retry_after(exc)
            return f"rate limited{f', provider asked for {wait:.1f}s' if wait else ''}"
        return str(exc)[:300]

    @staticmethod
    def _backoff(exc: Exception, attempt: int) -> float:
        """Honour the provider's own retry hint when it gives one.

        Guessing an exponential delay against a token-per-minute limit tends to
        either waste time or retry too early; the server knows when the window
        frees up, so use its number and add a small buffer.
        """
        hinted = _retry_after(exc)
        if hinted is not None:
            return hinted + 1.0
        return min(2 ** (attempt - 1), 8)

    def text(self, system: str, user: str, *, attempts: int = 4) -> str:
        self._require()
        messages = [("system", system), ("human", user)]
        self._pacer.reserve(self._estimate_tokens(system, user))

        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return str(self._chat.invoke(messages).content).strip()
            except Exception as exc:
                last_error = exc
                if _is_daily_quota(exc):
                    raise LLMUnavailable(
                        f"daily token quota exhausted: {self._brief(exc)}"
                    ) from exc
                log.warning(
                    "LLM text call failed (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    self._brief(exc),
                )
                if attempt < attempts:
                    time.sleep(self._backoff(exc, attempt))

        raise LLMUnavailable(
            f"text call failed after {attempts} attempts: {self._brief(last_error)}"
        ) from last_error


def truncate(text: str, limit: int = 12_000) -> str:
    """Keep prompts inside the context window without silently losing the tail.

    Resumes and transcripts have important content at both ends, so we keep the
    head and the tail rather than a prefix.
    """
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n\n[... {len(text) - limit} characters trimmed ...]\n\n{text[-tail:]}"
