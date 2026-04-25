"""Thin async wrapper around Playwright that produces :class:`EngramPage` instances.

The wrapper exists so that the library's public surface never forces a
caller to reach for Playwright directly before they have an Engram page to
work with. It also centralises browser launch and teardown so that the
cache and LLM client are wired in once, not at every call site.
"""

from __future__ import annotations

from contextlib import suppress
from types import TracebackType

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright

from engram.cache.base import AbstractCache
from engram.cache.sqlite import SQLiteCache
from engram.llm.base import AbstractLLMClient
from engram.llm.litellm_client import LiteLLMClient
from engram.models.config import EngramConfig
from engram.models.errors import EngramBrowserError
from engram.page import EngramPage


class EngramBrowser:
    """Async context manager that owns a Playwright browser + its Engram dependencies."""

    def __init__(
        self,
        config: EngramConfig | None = None,
        *,
        cache: AbstractCache | None = None,
        llm: AbstractLLMClient | None = None,
    ) -> None:
        self._config = config or EngramConfig()
        self._cache: AbstractCache = cache or SQLiteCache(self._config.cache_path)
        self._llm: AbstractLLMClient = llm or LiteLLMClient(self._config)
        self._owns_cache = cache is None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._contexts: list[BrowserContext] = []

    async def start(self) -> None:
        """Launch Playwright and the underlying browser process."""
        if self._browser is not None:
            return
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self._config.headless)
        except Exception as exc:
            await self._teardown_playwright()
            raise EngramBrowserError(f"Failed to start browser: {exc}") from exc

    async def new_page(self) -> EngramPage:
        """Create a fresh browser context + page, wrapped as :class:`EngramPage`."""
        if self._browser is None:
            await self.start()
        assert self._browser is not None  # appeases type checker after start()
        try:
            context = await self._browser.new_context()
            self._contexts.append(context)
            page = await context.new_page()
        except Exception as exc:
            raise EngramBrowserError(f"Failed to create new page: {exc}") from exc
        return EngramPage(page=page, config=self._config, cache=self._cache, llm=self._llm)

    async def close(self) -> None:
        """Shut down every context, the browser, and Playwright itself.

        Also closes the cache if this instance created it (i.e. the caller
        did not supply their own cache). If the caller supplied a cache,
        they own its lifecycle.
        """
        for ctx in self._contexts:
            with suppress(Exception):
                await ctx.close()
        self._contexts.clear()

        if self._browser is not None:
            try:
                await self._browser.close()
            finally:
                self._browser = None

        await self._teardown_playwright()

        if self._owns_cache:
            await self._cache.close()

    async def _teardown_playwright(self) -> None:
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            finally:
                self._playwright = None

    async def __aenter__(self) -> EngramBrowser:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
