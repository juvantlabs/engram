"""Engram — Python-native browser automation with cached LLM-resolved selectors.

Typical usage::

    import asyncio
    from engram import EngramBrowser, EngramConfig

    async def main() -> None:
        async with EngramBrowser(EngramConfig.from_env()) as browser:
            page = await browser.new_page()
            await page.goto("https://example.com")
            await page.act("click the more information link")

    asyncio.run(main())
"""

from engram.browser.context import EngramBrowser
from engram.cache import RedisCache, SQLiteCache, build_cache
from engram.cache.base import AbstractCache
from engram.llm.budgeted import BudgetedLLMClient
from engram.models.config import CacheBackend, EngramConfig
from engram.models.errors import (
    EngramBrowserError,
    EngramBudgetExceededError,
    EngramCacheError,
    EngramConfigError,
    EngramDOMError,
    EngramError,
    EngramExtractionError,
    EngramLLMError,
    EngramResolutionError,
)
from engram.models.resolution import (
    ActionResolution,
    ActionType,
    CacheEntry,
    ExtractionResolution,
    ObservationResolution,
    ObservedAction,
    SelectorCandidate,
    SelectorType,
)
from engram.observability import configure_logging, get_logger
from engram.page import EngramPage

__version__ = "0.1.0"

__all__ = [
    "AbstractCache",
    "ActionResolution",
    "ActionType",
    "BudgetedLLMClient",
    "CacheBackend",
    "CacheEntry",
    "EngramBrowser",
    "EngramBrowserError",
    "EngramBudgetExceededError",
    "EngramCacheError",
    "EngramConfig",
    "EngramConfigError",
    "EngramDOMError",
    "EngramError",
    "EngramExtractionError",
    "EngramLLMError",
    "EngramPage",
    "EngramResolutionError",
    "ExtractionResolution",
    "ObservationResolution",
    "ObservedAction",
    "RedisCache",
    "SQLiteCache",
    "SelectorCandidate",
    "SelectorType",
    "__version__",
    "build_cache",
    "configure_logging",
    "get_logger",
]
