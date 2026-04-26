"""Tests for the ``EngramPage.observe()`` cache loop with fakes."""

from __future__ import annotations

from typing import Any

from engram.dom.hasher import hash_observation
from engram.dom.preprocessor import preprocess_dom
from engram.models.config import EngramConfig
from engram.models.resolution import (
    CacheEntry,
    ObservationResolution,
    ObservedAction,
    SelectorCandidate,
)
from engram.page import EngramPage

# --------------------------------------------------------------------------- fakes


class _FakeLocator:
    def __init__(self, count: int) -> None:
        self._count = count

    @property
    def first(self) -> _FakeLocator:
        return self

    async def count(self) -> int:
        return self._count


class _FakePage:
    def __init__(self, html: str, selector_counts: dict[str, int]) -> None:
        self._html = html
        self._selector_counts = selector_counts

    async def content(self) -> str:
        return self._html

    def locator(self, selector: str) -> _FakeLocator:
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

    async def get(self, key: str) -> CacheEntry | None:
        self.gets += 1
        entry = self.store.get(key)
        if entry is None:
            return None
        bumped = entry.model_copy(update={"hit_count": entry.hit_count + 1})
        self.store[key] = bumped
        return bumped

    async def put(self, entry: CacheEntry) -> None:
        self.puts += 1
        self.store[entry.key] = entry

    async def invalidate(self, key: str) -> None:
        self.store.pop(key, None)

    async def close(self) -> None:
        pass


class _FakeLLM:
    def __init__(self, observation: list[ObservedAction]) -> None:
        self._observation = observation
        self.calls = 0

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
        *,
        prior_failures: list[str] | None = None,
    ) -> list[SelectorCandidate]:
        return []

    async def resolve_extraction(
        self,
        intent: str,
        preprocessed_dom: str,
        schema: dict[str, Any],
        *,
        prior_failures: list[str] | None = None,
    ) -> dict[str, Any]:
        return {}

    async def resolve_observation(
        self, preprocessed_dom: str
    ) -> list[ObservedAction]:
        self.calls += 1
        return list(self._observation)


# --------------------------------------------------------------------------- helpers


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


def _make_page(
    *,
    html: str = "<html><body><button id='join'>Join</button></body></html>",
    selector_counts: dict[str, int] | None = None,
    cache: _FakeCache | None = None,
    observation: list[ObservedAction] | None = None,
) -> tuple[EngramPage, _FakePage, _FakeCache, _FakeLLM]:
    fake_page = _FakePage(html, selector_counts or {"#join": 1})
    fake_cache = cache or _FakeCache()
    # Use an explicit None check — `observation or default` would mask a
    # caller-supplied empty list, which is a legitimate observation.
    fake_llm = _FakeLLM(
        observation if observation is not None else [_action("#join", "Click join", 0.95)]
    )
    engram_page = EngramPage(
        page=fake_page,  # type: ignore[arg-type]
        config=EngramConfig(),
        cache=fake_cache,
        llm=fake_llm,
    )
    return engram_page, fake_page, fake_cache, fake_llm


# --------------------------------------------------------------------------- tests


async def test_cache_miss_calls_llm_and_caches_validated_actions() -> None:
    page, _, cache, llm = _make_page()
    actions = await page.observe()
    assert llm.calls == 1
    assert cache.puts == 1
    assert [a.selector for a in actions] == ["#join"]


async def test_hallucinated_selectors_are_filtered_out() -> None:
    """Selectors the LLM proposes but which match zero elements must be dropped."""
    page, _, cache, _ = _make_page(
        selector_counts={"#real": 1, "#fake": 0},
        observation=[
            _action("#fake", "fake action", 0.9),
            _action("#real", "real action", 0.7),
        ],
    )
    # The selector_counts default doesn't include #real or #fake — override fully.
    actions = await page.observe()
    assert [a.selector for a in actions] == ["#real"]
    # Only the validated subset is cached, so subsequent observations don't
    # have to re-validate.
    stored = next(iter(cache.store.values()))
    assert isinstance(stored.resolution, ObservationResolution)
    assert [a.selector for a in stored.resolution.actions] == ["#real"]


async def test_cache_hit_skips_llm() -> None:
    html = "<html><body><button id='join'>Join</button></body></html>"
    preprocessed = preprocess_dom(html)
    key = hash_observation(preprocessed)

    seeded = CacheEntry(
        key=key,
        resolution=ObservationResolution(
            dom_hash=key,
            actions=[_action("#join", "Click join", 0.95)],
        ),
    )
    cache = _FakeCache(seed={key: seeded})
    page, _, cache, llm = _make_page(html=html, cache=cache)

    actions = await page.observe()
    assert llm.calls == 0
    assert [a.selector for a in actions] == ["#join"]


async def test_returned_actions_sorted_by_confidence() -> None:
    page, _, _, _ = _make_page(
        selector_counts={"#low": 1, "#high": 1},
        observation=[
            _action("#low", "low", 0.3),
            _action("#high", "high", 0.95),
        ],
    )
    actions = await page.observe()
    assert [a.selector for a in actions] == ["#high", "#low"]


async def test_empty_actions_list_is_valid_observation() -> None:
    """A page with nothing to do is a legitimate empty observation, not an error."""
    page, _, cache, llm = _make_page(observation=[])
    actions = await page.observe()
    assert actions == []
    assert llm.calls == 1
    # Empty observation still gets cached so subsequent observations short-circuit.
    assert cache.puts == 1
