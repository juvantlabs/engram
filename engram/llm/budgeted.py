"""Budget-enforcing wrapper around an :class:`AbstractLLMClient`.

A session-level guardrail: the wrapped client refuses to make more than
``budget`` downstream calls before raising :class:`EngramBudgetExceededError`.
Counts every call — failed ones too — because charging only successes would
make a flaky provider's retries invisible to the budget.
"""

from __future__ import annotations

from typing import Any

from engram.llm.base import AbstractLLMClient
from engram.models.errors import EngramBudgetExceededError
from engram.models.resolution import ObservedAction, SelectorCandidate


class BudgetedLLMClient:
    """Decorator that enforces a per-session call budget on any LLM client."""

    def __init__(self, inner: AbstractLLMClient, budget: int) -> None:
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self._inner = inner
        self._budget = budget
        self._calls = 0

    @property
    def calls_made(self) -> int:
        return self._calls

    @property
    def calls_remaining(self) -> int:
        return max(0, self._budget - self._calls)

    def _consume(self) -> None:
        # Reserve the slot before the downstream call. If the call raises,
        # the slot is already consumed — that's intentional: a flaky provider
        # should not be able to burn the budget invisibly through retries.
        if self._calls >= self._budget:
            raise EngramBudgetExceededError(
                budget=self._budget, attempted_call=self._calls + 1
            )
        self._calls += 1

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
        *,
        prior_failures: list[str] | None = None,
    ) -> list[SelectorCandidate]:
        self._consume()
        return await self._inner.resolve_action(
            intent=intent,
            preprocessed_dom=preprocessed_dom,
            prior_failures=prior_failures,
        )

    async def resolve_extraction(
        self,
        intent: str,
        preprocessed_dom: str,
        schema: dict[str, Any],
        *,
        prior_failures: list[str] | None = None,
    ) -> dict[str, Any]:
        self._consume()
        return await self._inner.resolve_extraction(
            intent=intent,
            preprocessed_dom=preprocessed_dom,
            schema=schema,
            prior_failures=prior_failures,
        )

    async def resolve_observation(
        self, preprocessed_dom: str
    ) -> list[ObservedAction]:
        self._consume()
        return await self._inner.resolve_observation(
            preprocessed_dom=preprocessed_dom,
        )
