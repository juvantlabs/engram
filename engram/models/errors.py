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


class EngramBrowserError(EngramError):
    """Raised when the underlying browser driver reports a failure."""


class EngramDOMError(EngramError):
    """Raised when DOM preprocessing fails or the result exceeds configured limits."""
