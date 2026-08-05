"""Groq LLM wrapper.

Two jobs: bind every call to a Pydantic schema so stages get typed data back,
and fail in a way callers can handle. If there is no API key, or Groq is down,
callers get `LLMUnavailable` and fall back to their deterministic path — the
pipeline must still produce a ranked shortlist with no model at all.
"""

from __future__ import annotations

import logging
import time
from typing import TypeVar

from pydantic import BaseModel

from .config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """No usable model: missing key, missing package, or the call kept failing."""


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self._chat = None
        self._init_error: str = ""

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
        attempts: int = 3,
    ) -> T:
        """Call the model and return a validated instance of `schema`.

        Retries on transient errors and on schema-validation failures — a
        validation failure usually means the model returned a near-miss, and
        one more attempt with the same prompt normally fixes it.
        """
        self._require()
        runnable = self._chat.with_structured_output(schema)
        messages = [("system", system), ("human", user)]

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
                log.warning(
                    "LLM structured call failed (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))

        raise LLMUnavailable(
            f"{schema.__name__} call failed after {attempts} attempts: {last_error}"
        ) from last_error

    def text(self, system: str, user: str, *, attempts: int = 3) -> str:
        self._require()
        messages = [("system", system), ("human", user)]

        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return str(self._chat.invoke(messages).content).strip()
            except Exception as exc:
                last_error = exc
                log.warning("LLM text call failed (attempt %d/%d): %s", attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))

        raise LLMUnavailable(f"text call failed after {attempts} attempts: {last_error}") from last_error


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
