"""Live integration test: real Playwright + real SQLite cache + counting fake LLM, for extract().

Mirrors :mod:`test_act_live` but covers the ``extract`` primitive. Proves:

1. A cache miss triggers exactly one LLM call.
2. The second extract on the same page + intent + schema hits the cache (zero
   additional LLM calls).
3. Mutating visible page text causes a cache miss and a fresh LLM call.

The LLM is faked rather than mocked at the HTTP layer because the assertions
here are about the cache contract, not the LLM's own quality — a real LLM call
would be non-deterministic, slow, and cost money on every CI run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, async_playwright
from pydantic import BaseModel

from engram.cache.sqlite import SQLiteCache
from engram.models.config import EngramConfig
from engram.models.resolution import ObservedAction, SelectorCandidate
from engram.page import EngramPage

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "join_page.html"


class _PageInfo(BaseModel):
    title: str
    description: str


class CountingFakeLLM:
    """Records call counts and returns a canned extraction payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls = 0
        self.last_schema: dict[str, Any] | None = None

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
        *,
        prior_failures: list[str] | None = None,
    ) -> list[SelectorCandidate]:
        raise AssertionError("extract() integration test must not invoke action")

    async def resolve_extraction(
        self,
        intent: str,
        preprocessed_dom: str,
        schema: dict[str, Any],
        *,
        prior_failures: list[str] | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        self.last_schema = schema
        return dict(self._payload)

    async def resolve_observation(
        self, preprocessed_dom: str
    ) -> list[ObservedAction]:
        raise AssertionError("extract() integration test must not invoke observation")


def _playwright_available() -> bool:
    """Skip integration if no browser is installed (CI without ``playwright install``)."""
    import os
    import shutil
    import sys

    cache_home = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or str(
        Path.home() / "Library" / "Caches" / "ms-playwright"
        if sys.platform == "darwin"
        else Path.home() / ".cache" / "ms-playwright"
    )
    path = Path(cache_home)
    if not path.exists():
        return False
    return any(path.glob("chromium-*")) or bool(shutil.which("chromium"))


pytestmark = pytest.mark.skipif(
    not _playwright_available(),
    reason="Playwright browsers not installed (run: playwright install chromium)",
)


@pytest.fixture
async def live_page(
    tmp_path: Path,
) -> AsyncIterator[tuple[EngramPage, Page, CountingFakeLLM]]:
    cache = SQLiteCache(tmp_path / "cache.db")
    llm = CountingFakeLLM(
        payload={"title": "Join Session", "description": "Welcome. Click below to join."}
    )
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            engram_page = EngramPage(
                page=page,
                config=EngramConfig(headless=True),
                cache=cache,
                llm=llm,
            )
            await page.goto(f"file://{_FIXTURE.resolve()}")
            yield engram_page, page, llm
            await context.close()
        finally:
            await browser.close()
            await cache.close()


async def test_first_extract_is_cache_miss_and_second_run_is_hit(
    live_page: tuple[EngramPage, Page, CountingFakeLLM],
) -> None:
    engram_page, _, llm = live_page

    out1 = await engram_page.extract("page title and description", _PageInfo)
    assert llm.calls == 1
    assert out1.title == "Join Session"
    assert out1.description == "Welcome. Click below to join."

    # Second call, identical inputs, same DOM hash → cache hit, no LLM call.
    out2 = await engram_page.extract("page title and description", _PageInfo)
    assert llm.calls == 1
    assert out2 == out1


async def test_dom_mutation_triggers_fresh_extraction(
    live_page: tuple[EngramPage, Page, CountingFakeLLM],
) -> None:
    engram_page, raw_page, llm = live_page

    await engram_page.extract("page title and description", _PageInfo)
    assert llm.calls == 1

    # Mutate visible page text — the preprocessor preserves text content, so
    # the new hash differs from the first run.
    await raw_page.evaluate(
        "document.querySelector('p').textContent = 'Welcome back.'"
    )

    await engram_page.extract("page title and description", _PageInfo)
    # Cache key changed → fresh LLM call.
    assert llm.calls == 2
