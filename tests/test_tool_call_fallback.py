"""Recovery when the model breaks Groq's function-call format.

A reviewer uploaded a real CV through the dashboard and every model stage failed
on it with `tool_use_failed`: the model opened a `<function=ExtractedProfile>`
block and then carried on transcribing the resume instead of emitting arguments.
Because the prompt is identical on each attempt and the temperature is 0.1, all
four retries reproduced it byte for byte — then the same thing happened again
for `FitAssessment`. Eight calls, one outcome.

The tool-calling wrapper is the fragile part, not the model, so the client now
re-asks for the same schema as plain JSON. These tests drive that at the client
boundary, where the failure actually occurs.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from recruiter.config import Settings
from recruiter.llm import LLMClient, LLMUnavailable, _extract_json


class Person(BaseModel):
    name: str
    years: int


# The real 400 body, trimmed to the part the client matches on.
TOOL_USE_FAILED = (
    "Error code: 400 - {'error': {'message': \"Failed to call a function. "
    "Please adjust your prompt. See 'failed_generation' for more details.\", "
    "'type': 'invalid_request_error', 'code': 'tool_use_failed', "
    "'failed_generation': '<function=Person> and finance sectors.\\n\\nEDUCATION'}}"
)


class _Runnable:
    """What `with_structured_output(schema)` returns."""

    def __init__(self, outcome: Exception | BaseModel) -> None:
        self.outcome = outcome
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChat:
    """Stands in for `ChatGroq`.

    `with_structured_output` fails the way Groq fails; a plain `invoke` — the
    JSON path — returns whatever text the test supplies.
    """

    def __init__(self, structured_outcome: Exception | BaseModel, text_reply: str = "") -> None:
        self._runnable = _Runnable(structured_outcome)
        self.text_reply = text_reply
        self.text_calls = 0
        self.last_system = ""

    def with_structured_output(self, _schema):
        return self._runnable

    def invoke(self, messages):
        self.text_calls += 1
        self.last_system = messages[0][1]
        if isinstance(self.text_reply, Exception):
            raise self.text_reply
        return _Reply(self.text_reply)

    @property
    def structured_calls(self) -> int:
        return self._runnable.calls


def _client(chat: FakeChat) -> LLMClient:
    client = LLMClient(Settings(groq_api_key="test-key"))
    client._chat = chat  # bypass the network
    return client


# -- the recovery ----------------------------------------------------------


def test_a_broken_tool_call_is_retried_as_plain_json() -> None:
    chat = FakeChat(RuntimeError(TOOL_USE_FAILED), '{"name": "Ada", "years": 9}')
    result = _client(chat).structured(Person, "sys", "user")

    assert result.name == "Ada" and result.years == 9
    assert chat.text_calls == 1, "the JSON path should have been used"


def test_it_does_not_burn_four_identical_retries_first() -> None:
    """The whole point: the failure is deterministic, so retrying reproduces it."""
    chat = FakeChat(RuntimeError(TOOL_USE_FAILED), '{"name": "Ada", "years": 9}')
    _client(chat).structured(Person, "sys", "user")

    assert chat.structured_calls == 1, (
        f"tool-call path was retried {chat.structured_calls}x on a failure that "
        "reproduces exactly — that is what wasted eight calls on the real CV"
    )


def test_the_json_retry_states_the_schema() -> None:
    chat = FakeChat(RuntimeError(TOOL_USE_FAILED), '{"name": "Ada", "years": 9}')
    _client(chat).structured(Person, "sys", "user")

    assert "JSON Schema" in chat.last_system
    assert '"years"' in chat.last_system


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n{"name": "Ada", "years": 9}\n```',      # fenced
        'Here is the profile:\n{"name": "Ada", "years": 9}',  # preamble
        '  {"name": "Ada", "years": 9}  ',                 # padded
    ],
)
def test_the_json_retry_tolerates_how_models_actually_reply(reply: str) -> None:
    """Told "JSON only", models still fence it or add a sentence."""
    result = _client(FakeChat(RuntimeError(TOOL_USE_FAILED), reply)).structured(
        Person, "sys", "user"
    )
    assert result.name == "Ada"


# -- when it cannot be recovered ------------------------------------------


def test_unrecoverable_still_degrades_rather_than_crashing() -> None:
    """If JSON fails too, callers must still get LLMUnavailable to fall back on."""
    chat = FakeChat(RuntimeError(TOOL_USE_FAILED), "I could not parse that resume.")
    with pytest.raises(LLMUnavailable) as err:
        _client(chat).structured(Person, "sys", "user")

    assert "tool-call format" in str(err.value)


def test_an_ordinary_failure_is_still_retried_normally() -> None:
    """Only the tool-call format short-circuits; transient errors keep retrying."""
    chat = FakeChat(RuntimeError("Error code: 500 - upstream had a wobble"))
    with pytest.raises(LLMUnavailable):
        _client(chat).structured(Person, "sys", "user", attempts=3)

    assert chat.structured_calls == 3, "a transient error should still use its retries"
    assert chat.text_calls == 0, "and should not divert to the JSON path"


# -- the extractor --------------------------------------------------------


def test_extract_json_finds_the_object() -> None:
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _extract_json('prose {"a": 1} trailing') == '{"a": 1}'
    assert _extract_json('{"a": {"b": 2}}') == '{"a": {"b": 2}}'
