"""Live integration test: real Playwright + real SQLite cache + counting fake LLM, for observe().

Mirrors :mod:`test_act_live` but covers the ``observe`` primitive. Proves:

1. A cache miss triggers exactly one LLM call.
2. The second observe on the same page hits the cache (zero additional LLM
   calls).
3. Mutating the DOM causes a cache miss and a fresh LLM call.
4. Hallucinated selectors — proposals from the LLM that match zero elements
   on the live page — are filtered out *before* the result is cached, so
   subsequent observations don't have to re-pay validation cost.

The fourth assertion is the one that justifies a live integration test in
addition to the unit tests: the unit-test fakes implement Locator counts via
a hand-rolled stub, but only Playwright knows what the real DOM contains.
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
    """Records call counts and returns a canned list of observed actions."""

    def __init__(self, actions: list[ObservedAction]) -> None:
        self._actions = actions
        self.calls = 0

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
        *,
        prior_failures: list[str] | None = None,
    ) -> list[SelectorCandidate]:
        raise AssertionError("observe() integration test must not invoke action")

    async def resolve_extraction(
        self,
        intent: str,
        preprocessed_dom: str,
        schema: dict[str, Any],
        *,
        prior_failures: list[str] | None = None,
    ) -> dict[str, Any]:
        raise AssertionError("observe() integration test must not invoke extraction")

    async def resolve_observation(
        self, preprocessed_dom: str
    ) -> list[ObservedAction]:
        self.calls += 1
        return list(self._actions)


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


def _action(
    selector: str,
    description: str = "do a thing",
    confidence: float = 0.9,
) -> ObservedAction:
    return ObservedAction(
        description=description,
        selector=selector,
        selector_type="css",
        action="click",
        confidence=confidence,
    )


@pytest.fixture
async def live_page(
    tmp_path: Path,
) -> AsyncIterator[tuple[EngramPage, Page, CountingFakeLLM]]:
    cache = SQLiteCache(tmp_path / "cache.db")
    # The fake LLM proposes both a real selector (#join-btn exists in the
    # fixture) and a hallucinated one (#fake-btn does not), so each test can
    # independently observe the hallucination filter at work.
    llm = CountingFakeLLM(
        actions=[
            _action("#join-btn", description="Click the join session button", confidence=0.95),
            _action("#fake-btn", description="Click a button that does not exist", confidence=0.7),
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


async def test_first_observe_is_cache_miss_and_second_run_is_hit(
    live_page: tuple[EngramPage, Page, CountingFakeLLM],
) -> None:
    engram_page, _, llm = live_page

    actions1 = await engram_page.observe()
    assert llm.calls == 1
    # Hallucinated entry was filtered before caching → only the real one survives.
    assert [a.selector for a in actions1] == ["#join-btn"]

    actions2 = await engram_page.observe()
    assert llm.calls == 1  # cache hit
    assert [a.selector for a in actions2] == ["#join-btn"]


async def test_hallucinated_selectors_are_filtered_against_real_dom(
    live_page: tuple[EngramPage, Page, CountingFakeLLM],
) -> None:
    """Selectors that match zero elements on the live page must not enter the cache.

    Unit tests cover this against a hand-rolled locator stub; this test runs
    the same logic through real Playwright so the assertion is end-to-end.
    """
    engram_page, _, _ = live_page

    actions = await engram_page.observe()
    selectors = [a.selector for a in actions]
    assert "#join-btn" in selectors
    assert "#fake-btn" not in selectors


async def test_dom_mutation_triggers_fresh_observation(
    live_page: tuple[EngramPage, Page, CountingFakeLLM],
) -> None:
    engram_page, raw_page, llm = live_page

    await engram_page.observe()
    assert llm.calls == 1

    # Change a preserved attribute — the preprocessor keeps aria-label, so the
    # hash flips and the next observe call falls through to the LLM.
    await raw_page.evaluate(
        "document.getElementById('join-btn').setAttribute('aria-label', 'Join Now')"
    )

    await engram_page.observe()
    assert llm.calls == 2
