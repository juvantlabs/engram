"""Tests for the ``EngramPage.extract()`` cache loop with fakes for Page/Cache/LLM."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from engram.dom.hasher import fingerprint_schema, hash_extraction
from engram.dom.preprocessor import preprocess_dom
from engram.models.config import EngramConfig
from engram.models.errors import EngramExtractionError, EngramLLMError
from engram.models.resolution import (
    CacheEntry,
    ExtractionResolution,
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
    def __init__(self, html: str) -> None:
        self._html = html

    async def content(self) -> str:
        return self._html

    def locator(self, selector: str) -> _FakeLocator:
        del selector  # unused — extract never validates locators
        return _FakeLocator(0)


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
    def __init__(
        self,
        *,
        extraction_payload: dict[str, Any] | None = None,
        extraction_payloads: list[dict[str, Any]] | None = None,
        extraction_error: Exception | None = None,
        action_candidates: list[SelectorCandidate] | None = None,
        observation: list[ObservedAction] | None = None,
    ) -> None:
        # ``extraction_payloads`` (plural) scripts one response per call so
        # retry tests can verify the LLM "recovers" once it sees feedback.
        # ``extraction_payload`` (singular) returns the same payload on every
        # call — used by tests that don't care about retry mechanics.
        if extraction_payloads is not None and extraction_payload is not None:
            raise ValueError("pass either extraction_payload or extraction_payloads, not both")
        self._extraction_payload = extraction_payload
        self._extraction_payloads = extraction_payloads
        self._extraction_error = extraction_error
        self._action_candidates = action_candidates or []
        self._observation = observation or []
        self.extraction_calls = 0
        self.last_intent: str | None = None
        self.last_schema: dict[str, Any] | None = None
        self.prior_failures_seen: list[list[str] | None] = []

    async def resolve_action(
        self,
        intent: str,
        preprocessed_dom: str,
        *,
        prior_failures: list[str] | None = None,
    ) -> list[SelectorCandidate]:
        return list(self._action_candidates)

    async def resolve_extraction(
        self,
        intent: str,
        preprocessed_dom: str,
        schema: dict[str, Any],
        *,
        prior_failures: list[str] | None = None,
    ) -> dict[str, Any]:
        self.extraction_calls += 1
        self.last_intent = intent
        self.last_schema = schema
        self.prior_failures_seen.append(
            list(prior_failures) if prior_failures is not None else None
        )
        if self._extraction_error is not None:
            raise self._extraction_error
        if self._extraction_payloads is not None:
            idx = min(self.extraction_calls - 1, len(self._extraction_payloads) - 1)
            return self._extraction_payloads[idx]
        assert self._extraction_payload is not None
        return self._extraction_payload

    async def resolve_observation(
        self, preprocessed_dom: str
    ) -> list[ObservedAction]:
        return list(self._observation)


# --------------------------------------------------------------------------- helpers


class _Person(BaseModel):
    name: str
    age: int


class _PersonRenamed(BaseModel):
    """Same shape as _Person but a different schema title — should fingerprint differently."""

    full_name: str
    age: int


def _make_page(
    *,
    html: str = "<html><body><p>Alice, 30</p></body></html>",
    cache: _FakeCache | None = None,
    llm: _FakeLLM | None = None,
    config: EngramConfig | None = None,
) -> tuple[EngramPage, _FakePage, _FakeCache, _FakeLLM]:
    fake_page = _FakePage(html)
    fake_cache = cache or _FakeCache()
    fake_llm = llm or _FakeLLM(extraction_payload={"name": "Alice", "age": 30})
    engram_page = EngramPage(
        page=fake_page,  # type: ignore[arg-type]
        config=config or EngramConfig(),
        cache=fake_cache,
        llm=fake_llm,
    )
    return engram_page, fake_page, fake_cache, fake_llm


# --------------------------------------------------------------------------- tests


async def test_cache_miss_calls_llm_and_caches_result() -> None:
    page, _, cache, llm = _make_page()
    out = await page.extract("the person on the page", _Person)
    assert out == _Person(name="Alice", age=30)
    assert llm.extraction_calls == 1
    assert cache.puts == 1
    stored = next(iter(cache.store.values()))
    assert isinstance(stored.resolution, ExtractionResolution)
    assert stored.resolution.intent == "the person on the page"


async def test_cache_hit_skips_llm() -> None:
    html = "<html><body><p>Alice, 30</p></body></html>"
    preprocessed = preprocess_dom(html)
    fingerprint = fingerprint_schema(_Person.model_json_schema())
    key = hash_extraction(preprocessed, "the person", fingerprint)

    seeded = CacheEntry(
        key=key,
        resolution=ExtractionResolution(
            intent="the person",
            dom_hash=key,
            schema_fingerprint=fingerprint,
            data_json=_Person(name="Alice", age=30).model_dump_json(),
        ),
    )
    cache = _FakeCache(seed={key: seeded})
    page, _, cache, llm = _make_page(cache=cache)

    out = await page.extract("the person", _Person)
    assert out == _Person(name="Alice", age=30)
    assert llm.extraction_calls == 0
    assert cache.gets == 1
    assert cache.puts == 0


async def test_changing_schema_changes_cache_key() -> None:
    """Same intent + same DOM with two different schemas must miss both ways."""
    page, _, cache, _ = _make_page(
        llm=_FakeLLM(extraction_payload={"name": "Alice", "age": 30}),
    )
    await page.extract("the person", _Person)

    # Now extract again with a different schema — should be a cache miss even
    # though the intent and DOM are identical.
    page._llm = _FakeLLM(extraction_payload={"full_name": "Alice", "age": 30})
    out2 = await page.extract("the person", _PersonRenamed)
    assert out2 == _PersonRenamed(full_name="Alice", age=30)
    # Two distinct keys in the store.
    assert len(cache.store) == 2


async def test_cached_data_failing_revalidation_triggers_refresh() -> None:
    """If a cached row's stored data no longer matches the schema, re-extract.

    This guards against the edge case where the schema fingerprint is the same
    but the stored JSON was written under an older incompatible shape (or
    hand-edited).
    """
    html = "<html><body><p>Alice, 30</p></body></html>"
    preprocessed = preprocess_dom(html)
    fingerprint = fingerprint_schema(_Person.model_json_schema())
    key = hash_extraction(preprocessed, "the person", fingerprint)

    # Seed with JSON that *won't* satisfy _Person (missing age field).
    bad_entry = CacheEntry(
        key=key,
        resolution=ExtractionResolution(
            intent="the person",
            dom_hash=key,
            schema_fingerprint=fingerprint,
            data_json='{"name": "Alice"}',
        ),
    )
    cache = _FakeCache(seed={key: bad_entry})
    page, _, cache, llm = _make_page(html=html, cache=cache)

    out = await page.extract("the person", _Person)
    assert out == _Person(name="Alice", age=30)
    # Old entry was invalidated and replaced.
    assert cache.invalidations == 1
    assert llm.extraction_calls == 1
    assert cache.puts == 1


async def test_llm_returning_invalid_payload_raises_extraction_error() -> None:
    page, _, _, _ = _make_page(
        llm=_FakeLLM(extraction_payload={"name": "Alice"}),  # missing age
    )
    with pytest.raises(EngramExtractionError, match="schema validation"):
        await page.extract("the person", _Person)


async def test_llm_provider_error_propagates_unwrapped() -> None:
    page, _, _, _ = _make_page(
        llm=_FakeLLM(extraction_error=EngramLLMError("provider down")),
    )
    with pytest.raises(EngramLLMError, match="provider down"):
        await page.extract("the person", _Person)


async def test_empty_intent_raises() -> None:
    page, _, _, _ = _make_page()
    with pytest.raises(EngramExtractionError):
        await page.extract("", _Person)
    with pytest.raises(EngramExtractionError):
        await page.extract("   ", _Person)


# ------------------------------------------------------------ retry behaviour


async def test_retry_recovers_after_first_attempt_fails_validation() -> None:
    """A scripted "first response broken, second response valid" run hits the cache once."""
    page, _, cache, llm = _make_page(
        llm=_FakeLLM(
            extraction_payloads=[
                {"name": "Alice"},  # missing age — fails Pydantic
                {"name": "Alice", "age": 30},  # valid
            ]
        ),
        config=EngramConfig(max_resolution_retries=2),
    )
    out = await page.extract("the person", _Person)
    assert out == _Person(name="Alice", age=30)
    assert llm.extraction_calls == 2
    # The second call must have received the validation-error feedback —
    # otherwise retrying is useless against a deterministic LLM.
    assert llm.prior_failures_seen[0] is None
    assert llm.prior_failures_seen[1] is not None
    assert any("age" in f for f in llm.prior_failures_seen[1])
    # Only the valid payload is what landed in the cache.
    assert cache.puts == 1


async def test_retry_exhausts_budget_then_raises() -> None:
    page, _, _, llm = _make_page(
        llm=_FakeLLM(extraction_payload={"name": "Alice"}),  # always missing age
        config=EngramConfig(max_resolution_retries=2),
    )
    with pytest.raises(EngramExtractionError, match=r"after 3 attempt"):
        await page.extract("the person", _Person)
    assert llm.extraction_calls == 3


async def test_retries_disabled_means_no_re_prompt_for_extract() -> None:
    page, _, _, llm = _make_page(
        llm=_FakeLLM(extraction_payload={"name": "Alice"}),
        config=EngramConfig(max_resolution_retries=0),
    )
    with pytest.raises(EngramExtractionError, match=r"after 1 attempt"):
        await page.extract("the person", _Person)
    assert llm.extraction_calls == 1


async def test_dom_change_yields_fresh_extraction() -> None:
    page, _, cache, llm = _make_page(
        html="<html><body><p>Alice, 30</p></body></html>",
    )
    await page.extract("the person", _Person)
    assert llm.extraction_calls == 1

    # Mutate the HTML behind the fake page.
    page._page = _FakePage("<html><body><p>Bob, 41</p></body></html>")  # type: ignore[assignment]
    page._llm = _FakeLLM(extraction_payload={"name": "Bob", "age": 41})
    out = await page.extract("the person", _Person)
    assert out == _Person(name="Bob", age=41)
    # Two distinct cache entries — the original and the new one.
    assert len(cache.store) == 2
