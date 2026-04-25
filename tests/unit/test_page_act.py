"""Tests for the core ``EngramPage.act()`` loop with fakes for Playwright/Cache/LLM."""

from __future__ import annotations

from typing import Any

import pytest

from engram.models.config import EngramConfig
from engram.models.errors import EngramResolutionError
from engram.models.resolution import CacheEntry, SelectorCandidate, SelectorResolution
from engram.page import EngramPage

# --------------------------------------------------------------------------- fakes


class _FakeLocator:
    def __init__(
        self,
        count: int,
        *,
        click_error: Exception | None = None,
        fill_error: Exception | None = None,
    ) -> None:
        self._count = count
        self._click_error = click_error
        self._fill_error = fill_error
        self.clicks = 0
        self.fills: list[str] = []

    @property
    def first(self) -> _FakeLocator:
        return self

    async def count(self) -> int:
        return self._count

    async def click(self) -> None:
        if self._click_error:
            raise self._click_error
        self.clicks += 1

    async def fill(self, value: str) -> None:
        if self._fill_error:
            raise self._fill_error
        self.fills.append(value)


class _FakePage:
    def __init__(self, html: str, selector_counts: dict[str, int]) -> None:
        self._html = html
        self._selector_counts = selector_counts
        self.locators_requested: list[str] = []

    async def content(self) -> str:
        return self._html

    def locator(self, selector: str) -> _FakeLocator:
        self.locators_requested.append(selector)
        # The test page maps *stripped* selectors to match counts; strip engine prefix
        # so tests can assert on the raw selector without caring about "css=" etc.
        for prefix in ("css=", "xpath=", "text=", "role="):
            if selector.startswith(prefix):
                stripped = selector[len(prefix):]
                break
        else:
            stripped = selector
        return _FakeLocator(self._selector_counts.get(stripped, 0))


class _FakeCache:
    def __init__(self, seed: dict[str, CacheEntry] | None = None) -> None:
        self.store: dict[str, CacheEntry] = dict(seed or {})
        self.gets = 0
        self.puts = 0
        self.invalidations = 0

    async def get(self, key: str) -> CacheEntry | None:
        self.gets += 1
        entry = self.store.get(key)
        if entry is None:
            return None
        # Emulate the real cache's hit_count side effect.
        bumped = entry.model_copy(update={"hit_count": entry.hit_count + 1})
        self.store[key] = bumped
        return bumped

    async def put(self, entry: CacheEntry) -> None:
        self.puts += 1
        self.store[entry.key] = entry

    async def invalidate(self, key: str) -> None:
        self.invalidations += 1
        self.store.pop(key, None)

    async def close(self) -> None:
        pass


class _FakeLLM:
    def __init__(self, candidates: list[SelectorCandidate]) -> None:
        self.candidates = candidates
        self.calls = 0
        self.last_intent: str | None = None
        self.last_dom: str | None = None

    async def resolve_action(
        self, intent: str, preprocessed_dom: str
    ) -> list[SelectorCandidate]:
        self.calls += 1
        self.last_intent = intent
        self.last_dom = preprocessed_dom
        return list(self.candidates)


# --------------------------------------------------------------------------- helpers


def _candidate(
    selector: str,
    confidence: float = 0.9,
    action: str = "click",
    action_value: str | None = None,
) -> SelectorCandidate:
    return SelectorCandidate(
        selector=selector,
        selector_type="css",
        action=action,  # type: ignore[arg-type]
        action_value=action_value,
        confidence=confidence,
        reasoning="test",
    )


def _cached_entry(key: str, selector: str) -> CacheEntry:
    cand = _candidate(selector)
    resolution = SelectorResolution(
        intent="click join",
        dom_hash=key,
        candidates=[cand],
        chosen_selector=cand.selector,
        chosen_selector_type=cand.selector_type,
        chosen_action=cand.action,
        chosen_action_value=cand.action_value,
    )
    return CacheEntry(key=key, resolution=resolution)


def _make_page(
    *,
    html: str = "<html><body><button id='join'>Join</button></body></html>",
    selector_counts: dict[str, int] | None = None,
    cache: _FakeCache | None = None,
    llm_candidates: list[SelectorCandidate] | None = None,
) -> tuple[EngramPage, _FakePage, _FakeCache, _FakeLLM]:
    fake_page = _FakePage(html, selector_counts or {"#join": 1})
    fake_cache = cache or _FakeCache()
    fake_llm = _FakeLLM(llm_candidates or [_candidate("#join", confidence=0.95)])
    engram_page = EngramPage(
        page=fake_page,  # type: ignore[arg-type]
        config=EngramConfig(),
        cache=fake_cache,
        llm=fake_llm,
    )
    return engram_page, fake_page, fake_cache, fake_llm


# --------------------------------------------------------------------------- tests


async def test_cache_miss_calls_llm_and_stores_result() -> None:
    page, _, cache, llm = _make_page()
    await page.act("click join")
    assert llm.calls == 1
    assert cache.puts == 1
    # The stored entry should be keyed under the DOM hash and should reference
    # the selector the LLM proposed.
    stored = next(iter(cache.store.values()))
    assert stored.resolution.chosen_selector == "#join"


async def test_cache_hit_skips_llm() -> None:
    from engram.dom.hasher import hash_dom
    from engram.dom.preprocessor import preprocess_dom

    html = "<html><body><button id='join'>Join</button></body></html>"
    preprocessed = preprocess_dom(html)
    key = hash_dom(preprocessed, "click join")

    cache = _FakeCache(seed={key: _cached_entry(key, "#join")})
    page, _, cache, llm = _make_page(html=html, cache=cache)

    await page.act("click join")
    assert llm.calls == 0
    assert cache.gets == 1
    assert cache.puts == 0


async def test_stale_cache_invalidates_and_falls_through_to_llm() -> None:
    from engram.dom.hasher import hash_dom
    from engram.dom.preprocessor import preprocess_dom

    html = "<html><body><button id='join'>Join</button></body></html>"
    preprocessed = preprocess_dom(html)
    key = hash_dom(preprocessed, "click join")

    # Cache says #old-selector but the page only knows #join.
    cache = _FakeCache(seed={key: _cached_entry(key, "#old-selector")})
    page, _, cache, llm = _make_page(
        html=html,
        selector_counts={"#join": 1, "#old-selector": 0},
        cache=cache,
    )

    await page.act("click join")
    assert cache.invalidations == 1
    assert llm.calls == 1
    assert cache.puts == 1  # stored the new resolution
    stored = next(iter(cache.store.values()))
    assert stored.resolution.chosen_selector == "#join"


async def test_first_candidate_fails_second_succeeds() -> None:
    page, _, cache, _llm = _make_page(
        selector_counts={"#join": 1, "#no-such-el": 0},
        llm_candidates=[
            _candidate("#no-such-el", confidence=0.95),
            _candidate("#join", confidence=0.6),
        ],
    )
    await page.act("click join")
    assert cache.puts == 1
    stored = next(iter(cache.store.values()))
    assert stored.resolution.chosen_selector == "#join"


async def test_all_candidates_fail_raises_resolution_error() -> None:
    page, _, _, _ = _make_page(
        selector_counts={},  # nothing matches
        llm_candidates=[
            _candidate("#a", confidence=0.95),
            _candidate("#b", confidence=0.80),
        ],
    )
    with pytest.raises(EngramResolutionError) as excinfo:
        await page.act("click join")
    assert excinfo.value.attempts == 2


async def test_empty_intent_raises() -> None:
    page, _, _, _ = _make_page()
    with pytest.raises(EngramResolutionError):
        await page.act("")
    with pytest.raises(EngramResolutionError):
        await page.act("   ")


async def test_fill_action_sends_value() -> None:
    # Intercept locator creation so we can assert on the fill value.
    html = "<html><body><input id='name'/></body></html>"
    fill_locator = _FakeLocator(count=1)

    class _Page(_FakePage):
        def locator(self, selector: str) -> _FakeLocator:
            self.locators_requested.append(selector)
            return fill_locator

    fake_page: Any = _Page(html, {"#name": 1})
    fake_cache = _FakeCache()
    fake_llm = _FakeLLM(
        [_candidate("#name", action="fill", action_value="Alice")]
    )
    engram_page = EngramPage(
        page=fake_page,
        config=EngramConfig(),
        cache=fake_cache,
        llm=fake_llm,
    )
    await engram_page.act("type Alice in the name field")
    assert fill_locator.fills == ["Alice"]


async def test_low_confidence_candidates_are_rejected() -> None:
    """When every candidate falls below min_candidate_confidence, raise rather than guess."""
    fake_page = _FakePage(
        "<html><body><button id='x'>X</button></body></html>",
        {"#x": 1},
    )
    fake_cache = _FakeCache()
    fake_llm = _FakeLLM(
        [
            _candidate("#x", confidence=0.3),
            _candidate("#x", confidence=0.1),
        ]
    )
    # Default threshold is 0.5 — both candidates below → EngramResolutionError
    engram_page = EngramPage(
        page=fake_page,  # type: ignore[arg-type]
        config=EngramConfig(),
        cache=fake_cache,
        llm=fake_llm,
    )
    with pytest.raises(EngramResolutionError, match="min_candidate_confidence"):
        await engram_page.act("click anything")
    # Must NOT have executed the locator despite it existing on the page.
    assert fake_cache.puts == 0


async def test_threshold_is_configurable() -> None:
    """With a lower threshold, previously-rejected candidates become acceptable."""
    fake_page = _FakePage(
        "<html><body><button id='x'>X</button></body></html>",
        {"#x": 1},
    )
    fake_cache = _FakeCache()
    fake_llm = _FakeLLM([_candidate("#x", confidence=0.3)])
    engram_page = EngramPage(
        page=fake_page,  # type: ignore[arg-type]
        config=EngramConfig(min_candidate_confidence=0.2),
        cache=fake_cache,
        llm=fake_llm,
    )
    await engram_page.act("click anything")
    assert fake_cache.puts == 1


async def test_failed_execute_does_not_poison_cache() -> None:
    """If a validated selector fails at click/fill, no cache entry must be written.

    A stale-but-existent selector (e.g. an overlapped button) passes validation
    (>=1 DOM match) but Playwright's click auto-wait times out. Caching such a
    selector would make every future run pay the action-timeout cost.
    """
    from engram.models.errors import EngramBrowserError

    # Locator that matches 1 element but raises on click — simulates the
    # "element in DOM but not actionable" case.
    failing_locator = _FakeLocator(
        count=1,
        click_error=EngramBrowserError("Playwright click timeout"),
    )

    class _FailingPage(_FakePage):
        def locator(self, selector: str) -> _FakeLocator:
            self.locators_requested.append(selector)
            return failing_locator

    fake_page: Any = _FailingPage("<html><body><button/></body></html>", {})
    fake_cache = _FakeCache()
    fake_llm = _FakeLLM([_candidate("button", confidence=0.9)])
    engram_page = EngramPage(
        page=fake_page,
        config=EngramConfig(),
        cache=fake_cache,
        llm=fake_llm,
    )
    with pytest.raises(EngramBrowserError):
        await engram_page.act("click button")
    # Critical: the cache must be clean so the next run re-resolves rather
    # than hitting a poisoned entry.
    assert fake_cache.puts == 0
    assert fake_cache.store == {}


async def test_candidates_sorted_by_confidence_defensively() -> None:
    """Even if the LLM returns unsorted candidates, the higher-confidence one wins."""
    page, _, cache, _ = _make_page(
        selector_counts={"#low": 1, "#high": 1},
        llm_candidates=[
            _candidate("#low", confidence=0.2),
            _candidate("#high", confidence=0.99),
        ],
    )
    await page.act("click anything")
    stored = next(iter(cache.store.values()))
    assert stored.resolution.chosen_selector == "#high"
