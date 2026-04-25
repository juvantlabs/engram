"""Pydantic data models and exception hierarchy."""

from engram.models.config import CacheBackend, EngramConfig
from engram.models.errors import (
    EngramBrowserError,
    EngramCacheError,
    EngramConfigError,
    EngramDOMError,
    EngramError,
    EngramLLMError,
    EngramResolutionError,
)
from engram.models.resolution import (
    ActionType,
    CacheEntry,
    SelectorCandidate,
    SelectorResolution,
    SelectorType,
)

__all__ = [
    "ActionType",
    "CacheBackend",
    "CacheEntry",
    "EngramBrowserError",
    "EngramCacheError",
    "EngramConfig",
    "EngramConfigError",
    "EngramDOMError",
    "EngramError",
    "EngramLLMError",
    "EngramResolutionError",
    "SelectorCandidate",
    "SelectorResolution",
    "SelectorType",
]
