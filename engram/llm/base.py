"""Abstract LLM client interface used by EngramPage."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from engram.models.resolution import ObservedAction, SelectorCandidate


@runtime_checkable
class AbstractLLMClient(Protocol):
    """Protocol for clients backing Engram's three primitive operations."""

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
        *,
        prior_failures: list[str] | None = None,
    ) -> list[SelectorCandidate]:
        """Return a non-empty, confidence-sorted list of action candidates.

        Implementations must:

        - Return candidates ordered from highest to lowest confidence.
        - Validate their own output against a schema; malformed LLM responses
          must not leak out as random exceptions. Raise
          :class:`engram.models.errors.EngramLLMError` instead.
        - Never return an empty list; raise :class:`EngramLLMError` if the LLM
          declines to produce candidates.

        :param prior_failures: Human-readable summaries of why previous attempts
            for this same ``intent`` were unusable (selector matched zero
            elements, candidate fell below the confidence threshold, etc.).
            When non-empty, implementations MUST surface this feedback in the
            prompt so the LLM can correct itself rather than re-emit the same
            broken output. ``None`` and ``[]`` both mean "first attempt".
        """
        ...

    async def resolve_extraction(
        self,
        intent: str,
        preprocessed_dom: str,
        schema: dict[str, Any],
        *,
        prior_failures: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extract data from ``preprocessed_dom`` matching ``schema``.

        :param schema: A JSON Schema (typically produced by
            ``BaseModel.model_json_schema()``) describing the desired output.
        :param prior_failures: As for :meth:`resolve_action` — typically a list
            of Pydantic validation errors from previous attempts.
        :returns: A JSON-serialisable dict that the caller will validate
            against the original Pydantic model. Implementations should not
            attempt their own Pydantic validation — the user-supplied model
            is unknown to the client layer.
        :raises EngramLLMError: On provider errors, malformed responses, or
            when the LLM produces output that does not match the JSON Schema
            structurally (missing required keys, wrong types).
        """
        ...

    async def resolve_observation(
        self,
        preprocessed_dom: str,
    ) -> list[ObservedAction]:
        """Return a confidence-sorted list of meaningful actions on the page.

        Implementations must follow the same error-handling contract as
        :meth:`resolve_action`. May return an empty list if the page truly
        has no actionable elements (e.g. a static error page) — that is a
        legitimate observation, not an LLM failure.
        """
        ...
