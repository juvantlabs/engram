"""LiteLLM-backed implementation of :class:`AbstractLLMClient`.

Uses LiteLLM's tool-calling interface — portable across Anthropic, OpenAI,
and other providers LiteLLM supports — to force structured output. The tool
schema mirrors :class:`SelectorCandidate` exactly.
"""

from __future__ import annotations

import json
import time
from typing import Any, Final

import litellm
from pydantic import ValidationError

from engram.models.config import EngramConfig
from engram.models.errors import EngramLLMError
from engram.models.resolution import SelectorCandidate
from engram.observability import get_logger

_LOG = get_logger(__name__)

_TOOL_NAME: Final[str] = "resolve_action"

_TOOL_SCHEMA: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": (
            "Return 2 or 3 ranked selector candidates for the given natural-language "
            "intent, based on the provided HTML. Order candidates by confidence, "
            "highest first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "The selector string (CSS, XPath, role, or text).",
                            },
                            "selector_type": {
                                "type": "string",
                                "enum": ["css", "xpath", "role", "text"],
                                "description": (
                                    "css: standard CSS selector. "
                                    "xpath: XPath expression. "
                                    "role: ARIA role (e.g. 'button[name=\"Join\"]'). "
                                    "text: visible text match."
                                ),
                            },
                            "action": {
                                "type": "string",
                                "enum": ["click", "fill"],
                                "description": (
                                    "click: click the element. "
                                    "fill: type text into the element (use action_value)."
                                ),
                            },
                            "action_value": {
                                "type": ["string", "null"],
                                "description": "Required when action is 'fill'; null for 'click'.",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief justification for this candidate.",
                            },
                        },
                        "required": [
                            "selector",
                            "selector_type",
                            "action",
                            "confidence",
                            "reasoning",
                        ],
                    },
                }
            },
            "required": ["candidates"],
        },
    },
}

_SYSTEM_PROMPT: Final[str] = """You are an expert at resolving natural-language browser automation intents to concrete selectors and actions.

You receive:
- A user intent (e.g. "click the join session button" or "type 'Alice' in the name field").
- A preprocessed HTML snippet of the page.

You must return 2-3 ranked selector candidates via the resolve_action tool.

Rules:
- Prefer stable selectors: data-testid > aria-label > role+name > id > text > class > xpath.
- Never invent selectors that are not grounded in the provided HTML.
- Set action to "fill" only when the intent clearly involves typing text; otherwise "click".
- For "fill" actions, action_value MUST be the exact text to type.
- confidence is your honest estimate in [0, 1]. If the intent is ambiguous, say so in reasoning and lower the confidence.
- Always call the resolve_action tool exactly once. Do not produce any free-form response.
"""


class LiteLLMClient:
    """LLM client that talks to any LiteLLM-supported provider."""

    def __init__(self, config: EngramConfig) -> None:
        self._config = config

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
    ) -> list[SelectorCandidate]:
        user_message = f"Intent: {intent}\n\nPreprocessed HTML:\n{preprocessed_dom}"
        kwargs: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "tools": [_TOOL_SCHEMA],
            "tool_choice": {"type": "function", "function": {"name": _TOOL_NAME}},
            "temperature": self._config.llm_temperature,
            "timeout": self._config.llm_timeout_seconds,
        }
        if self._config.llm_api_key is not None:
            kwargs["api_key"] = self._config.llm_api_key.get_secret_value()
        if self._config.llm_api_base is not None:
            kwargs["api_base"] = self._config.llm_api_base
        if self._config.llm_api_version is not None:
            kwargs["api_version"] = self._config.llm_api_version

        start = time.perf_counter()
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            # LiteLLM raises provider-specific exceptions; normalise them.
            raise EngramLLMError(f"LLM call failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        candidates = _parse_response(response)

        _LOG.info(
            "llm.call",
            model=self._config.llm_model,
            intent=intent,
            latency_ms=round(latency_ms, 2),
            candidate_count=len(candidates),
            prompt_tokens=_extract_usage(response, "prompt_tokens"),
            completion_tokens=_extract_usage(response, "completion_tokens"),
        )

        return sorted(candidates, key=lambda c: c.confidence, reverse=True)


def _parse_response(response: Any) -> list[SelectorCandidate]:
    try:
        choices = response.choices
        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
    except (AttributeError, IndexError) as exc:
        raise EngramLLMError(f"Malformed LLM response shape: {exc}") from exc

    if not tool_calls:
        raise EngramLLMError("LLM did not return a tool call; expected resolve_action.")

    tool_call = tool_calls[0]
    try:
        arguments_raw = tool_call.function.arguments
    except AttributeError as exc:
        raise EngramLLMError(f"Tool call missing function.arguments: {exc}") from exc

    try:
        payload = json.loads(arguments_raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise EngramLLMError(f"Tool call arguments are not valid JSON: {exc}") from exc

    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise EngramLLMError("Tool call returned no candidates.")

    candidates: list[SelectorCandidate] = []
    for raw in raw_candidates:
        try:
            candidates.append(SelectorCandidate.model_validate(raw))
        except ValidationError as exc:
            raise EngramLLMError(f"Invalid candidate in LLM response: {exc}") from exc
    return candidates


def _extract_usage(response: Any, field: str) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    value = getattr(usage, field, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(field)
    return int(value) if isinstance(value, int) else None
