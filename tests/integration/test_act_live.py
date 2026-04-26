"""Live integration test: real Playwright + real SQLite cache + counting fake LLM.

This test is the stopping criterion for the v1 vertical slice. It proves:

1. A cache miss triggers exactly one LLM call.
2. The second run on the same page hits the cache (zero additional LLM calls).
3. Mutating the DOM causes a cache miss and a fresh LLM call.

The LLM is faked (not mocked at the HTTP layer) because a real LLM call would
be non-deterministic, slow, and cost money — the caching contract is what we
care about here, not the LLM's own quality.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, async_playwright

from engram.cache.sqlite import SQLiteCache
from engram.models.config import EngramConfig
from engram.models.resolution import ObservedAction, SelectorCandidate
from engram.page import EngramPage

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "join_page.html"


class CountingFakeLLM:
    """Test double that records call counts and returns a canned resolution."""

    def __init__(self, candidates: list[SelectorCandidate]) -> None:
        self._candidates = candidates
        self.calls = 0

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
        *,
        prior_failures: list[str] | None = None,
    ) -> list[SelectorCandidate]:
        self.calls += 1
        return list(self._candidates)

    async def resolve_extraction(
        self,
        intent: str,
        preprocessed_dom: str,
        schema: dict[str, Any],
        *,
        prior_failures: list[str] | None = None,
    ) -> dict[str, Any]:
        raise AssertionError("act() integration test must not invoke extraction")

    async def resolve_observation(
        self, preprocessed_dom: str
    ) -> list[ObservedAction]:
        raise AssertionError("act() integration test must not invoke observation")


def _playwright_available() -> bool:
    """Skip integration if no browser is installed (CI without `playwright install`)."""
    # We detect Playwright browsers by checking for the chromium executable in
    # the standard location. Playwright ships its own browsers; if they aren't
    # installed, `launch()` raises a descriptive error.
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
        [
            SelectorCandidate(
                selector="#join-btn",
                selector_type="css",
                action="click",
                action_value=None,
                confidence=0.95,
                reasoning="stable id",
            )
        ]
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


async def test_first_run_is_cache_miss_and_second_run_is_hit(
    live_page: tuple[EngramPage, Page, CountingFakeLLM],
) -> None:
    engram_page, raw_page, llm = live_page

    # First call: expect one LLM call, and the button's click handler to fire.
    await engram_page.act("click the join session button")
    assert llm.calls == 1
    clicks_after_first = await raw_page.evaluate("window.joinClicks")
    assert clicks_after_first == 1

    # Second call on the same page: no new LLM call, but the handler fires again.
    await engram_page.act("click the join session button")
    assert llm.calls == 1
    clicks_after_second = await raw_page.evaluate("window.joinClicks")
    assert clicks_after_second == 2


async def test_dom_mutation_triggers_fresh_resolution(
    live_page: tuple[EngramPage, Page, CountingFakeLLM],
) -> None:
    engram_page, raw_page, llm = live_page

    await engram_page.act("click the join session button")
    assert llm.calls == 1

    # Mutate the DOM: change the button's aria-label, which the preprocessor keeps,
    # so the hash should differ from the first run.
    await raw_page.evaluate(
        "document.getElementById('join-btn').setAttribute('aria-label', 'Join Now')"
    )

    await engram_page.act("click the join session button")
    # The cache key changed → fresh LLM call.
    assert llm.calls == 2
