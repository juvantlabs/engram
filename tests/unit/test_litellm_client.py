"""Tests for the LiteLLM client — response parsing and error normalisation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import litellm
import pytest

from engram.llm.litellm_client import LiteLLMClient
from engram.models.config import EngramConfig
from engram.models.errors import EngramLLMError


@dataclass
class _FakeFunction:
    arguments: str


@dataclass
class _FakeToolCall:
    function: _FakeFunction


@dataclass
class _FakeMessage:
    tool_calls: list[_FakeToolCall] | None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage | None = None


def _fake_response(candidates: list[dict[str, Any]]) -> _FakeResponse:
    args = json.dumps({"candidates": candidates})
    return _FakeResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    tool_calls=[_FakeToolCall(function=_FakeFunction(arguments=args))]
                )
            )
        ],
        usage=_FakeUsage(prompt_tokens=100, completion_tokens=20),
    )


async def test_happy_path_returns_sorted_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        return _fake_response(
            [
                {
                    "selector": "#join",
                    "selector_type": "css",
                    "action": "click",
                    "action_value": None,
                    "confidence": 0.6,
                    "reasoning": "id match",
                },
                {
                    "selector": "button[aria-label='Join']",
                    "selector_type": "css",
                    "action": "click",
                    "action_value": None,
                    "confidence": 0.9,
                    "reasoning": "aria-label match",
                },
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    out = await LiteLLMClient(EngramConfig()).resolve_action(
        intent="click join",
        preprocessed_dom="<button id='join' aria-label='Join'/>",
    )
    assert [c.confidence for c in out] == [0.9, 0.6]
    assert out[0].selector == "button[aria-label='Join']"


async def test_litellm_raising_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        raise RuntimeError("network down")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(EngramLLMError, match="LLM call failed"):
        await LiteLLMClient(EngramConfig()).resolve_action("x", "<p/>")


async def test_missing_tool_calls_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(tool_calls=None))])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(EngramLLMError, match="did not return a tool call"):
        await LiteLLMClient(EngramConfig()).resolve_action("x", "<p/>")


async def test_invalid_json_arguments_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        return _FakeResponse(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(
                        tool_calls=[_FakeToolCall(function=_FakeFunction(arguments="{bad"))]
                    )
                )
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(EngramLLMError, match="not valid JSON"):
        await LiteLLMClient(EngramConfig()).resolve_action("x", "<p/>")


async def test_empty_candidates_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        return _fake_response([])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(EngramLLMError, match="no candidates"):
        await LiteLLMClient(EngramConfig()).resolve_action("x", "<p/>")


async def test_invalid_candidate_shape_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        # Missing "reasoning" field.
        return _fake_response(
            [
                {
                    "selector": "#x",
                    "selector_type": "css",
                    "action": "click",
                    "action_value": None,
                    "confidence": 0.5,
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(EngramLLMError, match="Invalid candidate"):
        await LiteLLMClient(EngramConfig()).resolve_action("x", "<p/>")


# ----------------------------------------------------------------- extract


def _fake_tool_response(arguments: dict[str, Any]) -> _FakeResponse:
    return _FakeResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    tool_calls=[
                        _FakeToolCall(function=_FakeFunction(arguments=json.dumps(arguments)))
                    ]
                )
            )
        ],
        usage=_FakeUsage(prompt_tokens=10, completion_tokens=5),
    )


async def test_extraction_returns_tool_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        # Sanity: the user's JSON Schema must be plumbed into the tool params.
        tools = kwargs["tools"]
        assert isinstance(tools, list) and len(tools) == 1
        params = tools[0]["function"]["parameters"]
        assert params["properties"]["name"]["type"] == "string"
        return _fake_tool_response({"name": "Alice", "age": 30})

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    out = await LiteLLMClient(EngramConfig()).resolve_extraction(
        intent="the person",
        preprocessed_dom="<p>Alice, 30</p>",
        schema=schema,
    )
    assert out == {"name": "Alice", "age": 30}


async def test_extraction_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        return _FakeResponse(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(
                        tool_calls=[_FakeToolCall(function=_FakeFunction(arguments="not-json"))]
                    )
                )
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(EngramLLMError, match="not valid JSON"):
        await LiteLLMClient(EngramConfig()).resolve_extraction("x", "<p/>", {"type": "object"})


async def test_extraction_missing_tool_call_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(tool_calls=None))])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(EngramLLMError, match="did not return a tool call"):
        await LiteLLMClient(EngramConfig()).resolve_extraction("x", "<p/>", {"type": "object"})


# ----------------------------------------------------------------- observe


async def test_observation_returns_sorted_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        return _fake_tool_response(
            {
                "actions": [
                    {
                        "description": "Click low",
                        "selector": "#low",
                        "selector_type": "css",
                        "action": "click",
                        "confidence": 0.3,
                    },
                    {
                        "description": "Click high",
                        "selector": "#high",
                        "selector_type": "css",
                        "action": "click",
                        "confidence": 0.95,
                    },
                ]
            }
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    actions = await LiteLLMClient(EngramConfig()).resolve_observation("<button/>")
    assert [a.selector for a in actions] == ["#high", "#low"]


async def test_observation_empty_list_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        return _fake_tool_response({"actions": []})

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    actions = await LiteLLMClient(EngramConfig()).resolve_observation("<p/>")
    assert actions == []


async def test_observation_missing_actions_field_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        return _fake_tool_response({"not_actions": []})

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(EngramLLMError, match="'actions' list"):
        await LiteLLMClient(EngramConfig()).resolve_observation("<p/>")


# --------------------------------------------------------- prompt-injection mitigations


async def test_action_wraps_dom_in_untrusted_delimiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _fake_response(
            [
                {
                    "selector": "#x",
                    "selector_type": "css",
                    "action": "click",
                    "action_value": None,
                    "confidence": 0.9,
                    "reasoning": "r",
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await LiteLLMClient(EngramConfig()).resolve_action(
        intent="click join",
        preprocessed_dom="<button id='x'/>",
    )
    user_msg = captured["messages"][1]["content"]
    assert "<UNTRUSTED_PAGE_CONTENT>" in user_msg
    assert "</UNTRUSTED_PAGE_CONTENT>" in user_msg
    # Intent must appear OUTSIDE the delimiters; otherwise injection from page
    # content could trick the model into treating it as page-supplied data.
    intent_pos = user_msg.index("Intent: click join")
    open_tag = user_msg.index("<UNTRUSTED_PAGE_CONTENT>")
    assert intent_pos < open_tag


async def test_action_system_prompt_warns_about_untrusted_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _fake_response(
            [
                {
                    "selector": "#x",
                    "selector_type": "css",
                    "action": "click",
                    "action_value": None,
                    "confidence": 0.9,
                    "reasoning": "r",
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await LiteLLMClient(EngramConfig()).resolve_action(
        intent="click",
        preprocessed_dom="<p/>",
    )
    system_msg = captured["messages"][0]["content"]
    assert "UNTRUSTED_PAGE_CONTENT" in system_msg
    assert "ignore" in system_msg.lower()


async def test_extraction_wraps_dom_in_untrusted_delimiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _fake_tool_response({"name": "x", "age": 1})

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await LiteLLMClient(EngramConfig()).resolve_extraction(
        intent="the person",
        preprocessed_dom="<p>Alice, 30</p>",
        schema={"type": "object"},
    )
    user_msg = captured["messages"][1]["content"]
    assert "<UNTRUSTED_PAGE_CONTENT>" in user_msg


async def test_observation_wraps_dom_in_untrusted_delimiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _fake_tool_response({"actions": []})

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await LiteLLMClient(EngramConfig()).resolve_observation("<p/>")
    user_msg = captured["messages"][1]["content"]
    assert user_msg.startswith("<UNTRUSTED_PAGE_CONTENT>")
    assert user_msg.endswith("</UNTRUSTED_PAGE_CONTENT>")


# --------------------------------------------------------- prior_failures injection


async def test_action_prior_failures_are_prepended_to_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each retry's failures must surface in the next prompt, or retries are useless."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _fake_response(
            [
                {
                    "selector": "#x",
                    "selector_type": "css",
                    "action": "click",
                    "action_value": None,
                    "confidence": 0.9,
                    "reasoning": "r",
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await LiteLLMClient(EngramConfig()).resolve_action(
        intent="click join",
        preprocessed_dom="<button id='x'/>",
        prior_failures=[
            "selector `#bogus` (css) matched 0 elements on the page",
            "selector `#also-bogus` (css) matched 0 elements on the page",
        ],
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    user_msg = messages[1]["content"]
    # Feedback section must appear *before* the original intent so the model
    # reads "what went wrong" first.
    assert user_msg.index("#bogus") < user_msg.index("Intent: click join")
    assert "#also-bogus" in user_msg


async def test_action_no_prior_failures_means_clean_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-attempt prompts must not carry a 'previous attempts failed' header."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _fake_response(
            [
                {
                    "selector": "#x",
                    "selector_type": "css",
                    "action": "click",
                    "action_value": None,
                    "confidence": 0.9,
                    "reasoning": "r",
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await LiteLLMClient(EngramConfig()).resolve_action(
        intent="click join",
        preprocessed_dom="<button id='x'/>",
    )
    user_msg = captured["messages"][1]["content"]
    assert "Previous attempts" not in user_msg
    assert user_msg.startswith("Intent: click join")


async def test_extraction_prior_failures_are_prepended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _fake_tool_response({"name": "Alice", "age": 30})

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    await LiteLLMClient(EngramConfig()).resolve_extraction(
        intent="the person",
        preprocessed_dom="<p>Alice, 30</p>",
        schema=schema,
        prior_failures=["output failed schema validation: age: field required"],
    )
    user_msg = captured["messages"][1]["content"]
    assert "field required" in user_msg
    assert user_msg.index("field required") < user_msg.index("Intent: the person")


async def test_observation_invalid_action_shape_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_: object) -> _FakeResponse:
        # Missing required confidence.
        return _fake_tool_response(
            {
                "actions": [
                    {
                        "description": "Click x",
                        "selector": "#x",
                        "selector_type": "css",
                        "action": "click",
                    }
                ]
            }
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(EngramLLMError, match="Invalid action"):
        await LiteLLMClient(EngramConfig()).resolve_observation("<p/>")
