"""Tests for the per-session LLM-call budget enforced by ``BudgetedLLMClient``."""

from __future__ import annotations

from typing import Any

import pytest

from engram.llm.budgeted import BudgetedLLMClient
from engram.models.errors import EngramBudgetExceededError, EngramLLMError
from engram.models.resolution import ObservedAction, SelectorCandidate


class _CountingInner:
    """Minimal inner client that records call counts per primitive."""

    def __init__(
        self,
        *,
        action_response: list[SelectorCandidate] | None = None,
        extraction_response: dict[str, Any] | None = None,
        observation_response: list[ObservedAction] | None = None,
        action_error: Exception | None = None,
    ) -> None:
        self._action_response = action_response or []
        self._extraction_response = extraction_response or {}
        self._observation_response = observation_response or []
        self._action_error = action_error
        self.action_calls = 0
        self.extraction_calls = 0
        self.observation_calls = 0

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
        *,
        prior_failures: list[str] | None = None,
    ) -> list[SelectorCandidate]:
        self.action_calls += 1
        if self._action_error is not None:
            raise self._action_error
        return list(self._action_response)

    async def resolve_extraction(
        self,
        intent: str,
        preprocessed_dom: str,
        schema: dict[str, Any],
        *,
        prior_failures: list[str] | None = None,
    ) -> dict[str, Any]:
        self.extraction_calls += 1
        return dict(self._extraction_response)

    async def resolve_observation(
        self, preprocessed_dom: str
    ) -> list[ObservedAction]:
        self.observation_calls += 1
        return list(self._observation_response)


def _candidate(selector: str = "#x") -> SelectorCandidate:
    return SelectorCandidate(
        selector=selector,
        selector_type="css",
        action="click",
        action_value=None,
        confidence=0.9,
        reasoning="r",
    )


def test_zero_budget_rejected_at_construction() -> None:
    """A budget of zero (or negative) is a misconfiguration, not a no-budget signal.

    ``None`` is the no-budget value (set at the EngramConfig layer); zero
    here would mean "every call is denied", which is never useful.
    """
    inner = _CountingInner()
    with pytest.raises(ValueError, match="must be positive"):
        BudgetedLLMClient(inner, 0)
    with pytest.raises(ValueError):
        BudgetedLLMClient(inner, -1)


async def test_calls_within_budget_pass_through() -> None:
    inner = _CountingInner(action_response=[_candidate()])
    wrapped = BudgetedLLMClient(inner, budget=3)

    for _ in range(3):
        out = await wrapped.resolve_action(intent="x", preprocessed_dom="<p/>")
        assert out == [_candidate()]

    assert inner.action_calls == 3
    assert wrapped.calls_made == 3
    assert wrapped.calls_remaining == 0


async def test_call_after_budget_raises() -> None:
    inner = _CountingInner(action_response=[_candidate()])
    wrapped = BudgetedLLMClient(inner, budget=2)

    await wrapped.resolve_action(intent="x", preprocessed_dom="<p/>")
    await wrapped.resolve_action(intent="x", preprocessed_dom="<p/>")

    with pytest.raises(EngramBudgetExceededError) as excinfo:
        await wrapped.resolve_action(intent="x", preprocessed_dom="<p/>")
    assert excinfo.value.budget == 2
    assert excinfo.value.attempted_call == 3
    # Crucially: the inner client was NOT called for the third attempt — the
    # whole point of the budget is to stop calls before they hit the wire.
    assert inner.action_calls == 2


async def test_budget_is_shared_across_primitives() -> None:
    """One LLM-call budget covers act + extract + observe combined.

    A user who configures a session budget cares about total spend, not per-
    primitive spend.
    """
    inner = _CountingInner(
        action_response=[_candidate()],
        extraction_response={"x": 1},
        observation_response=[],
    )
    wrapped = BudgetedLLMClient(inner, budget=2)

    await wrapped.resolve_action(intent="x", preprocessed_dom="<p/>")
    await wrapped.resolve_extraction(
        intent="x", preprocessed_dom="<p/>", schema={"type": "object"}
    )

    with pytest.raises(EngramBudgetExceededError):
        await wrapped.resolve_observation(preprocessed_dom="<p/>")
    assert inner.observation_calls == 0


async def test_failed_inner_call_still_counts() -> None:
    """A flaky provider's failures must not be free.

    Otherwise an LLM that 500s 90% of the time could be retried indefinitely
    and the per-session budget would never bite.
    """
    inner = _CountingInner(action_error=EngramLLMError("provider down"))
    wrapped = BudgetedLLMClient(inner, budget=2)

    with pytest.raises(EngramLLMError):
        await wrapped.resolve_action(intent="x", preprocessed_dom="<p/>")
    with pytest.raises(EngramLLMError):
        await wrapped.resolve_action(intent="x", preprocessed_dom="<p/>")
    # Budget exhausted by failed calls.
    with pytest.raises(EngramBudgetExceededError):
        await wrapped.resolve_action(intent="x", preprocessed_dom="<p/>")


async def test_calls_made_and_remaining_are_observable() -> None:
    inner = _CountingInner(action_response=[_candidate()])
    wrapped = BudgetedLLMClient(inner, budget=5)

    assert wrapped.calls_made == 0
    assert wrapped.calls_remaining == 5

    await wrapped.resolve_action(intent="x", preprocessed_dom="<p/>")
    assert wrapped.calls_made == 1
    assert wrapped.calls_remaining == 4
