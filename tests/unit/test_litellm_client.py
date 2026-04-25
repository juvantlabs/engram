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
