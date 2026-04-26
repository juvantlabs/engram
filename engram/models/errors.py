"""Exception hierarchy for Engram.

All errors raised by Engram derive from :class:`EngramError` so callers can
catch the library's failures without also swallowing unrelated exceptions.
"""

from __future__ import annotations


class EngramError(Exception):
    """Base class for every exception raised by Engram."""


class EngramConfigError(EngramError):
    """Raised when configuration is missing, invalid, or inconsistent."""


class EngramCacheError(EngramError):
    """Raised when the cache backend fails to read, write, or invalidate."""


class EngramLLMError(EngramError):
    """Raised when the LLM call fails or returns malformed output."""


class EngramResolutionError(EngramError):
    """Raised when no candidate selector can be validated against the page."""

    def __init__(self, intent: str, attempts: int, last_error: str | None = None) -> None:
        self.intent = intent
        self.attempts = attempts
        self.last_error = last_error
        detail = f" ({last_error})" if last_error else ""
        super().__init__(
            f"Could not resolve intent {intent!r} after {attempts} attempt(s){detail}"
        )


class EngramExtractionError(EngramError):
    """Raised when ``extract`` cannot produce data conforming to the user-supplied schema."""

    def __init__(self, intent: str, reason: str) -> None:
        self.intent = intent
        self.reason = reason
        super().__init__(f"Could not extract intent {intent!r}: {reason}")


class EngramBudgetExceededError(EngramError):
    """Raised when a session exhausts its configured LLM-call budget.

    The budget is a session-level guardrail (configured via
    :attr:`EngramConfig.max_llm_calls_per_session`) so a runaway flow — say,
    a pathological intent that keeps re-prompting on every retry — cannot
    quietly burn through a token budget. Each downstream LLM call counts,
    whether it succeeds or not: charging only successful calls would let a
    flaky provider's failures be invisible.
    """

    def __init__(self, budget: int, attempted_call: int) -> None:
        self.budget = budget
        self.attempted_call = attempted_call
        super().__init__(
            f"LLM-call budget exhausted: attempted call #{attempted_call} but "
            f"max_llm_calls_per_session={budget}"
        )


class EngramBrowserError(EngramError):
    """Raised when the underlying browser driver reports a failure."""


class EngramDOMError(EngramError):
    """Raised when DOM preprocessing fails or the result exceeds configured limits."""
