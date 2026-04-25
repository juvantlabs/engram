"""Abstract LLM client interface used by EngramPage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from engram.models.resolution import SelectorCandidate


@runtime_checkable
class AbstractLLMClient(Protocol):
    """Protocol for clients that resolve an intent to ranked selector candidates."""

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
    ) -> list[SelectorCandidate]:
        """Return a non-empty, confidence-sorted list of candidates.

        Implementations must:

        - Return candidates ordered from highest to lowest confidence.
        - Validate their own output against a schema; malformed LLM responses
          must not leak out as random exceptions. Raise
          :class:`engram.models.errors.EngramLLMError` instead.
        - Never return an empty list; raise :class:`EngramLLMError` if the LLM
          declines to produce candidates.
        """
        ...
