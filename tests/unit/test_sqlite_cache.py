"""Tests for the SQLite cache backend."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from engram.cache.sqlite import SQLiteCache
from engram.models.errors import EngramCacheError
from engram.models.resolution import CacheEntry, SelectorCandidate, SelectorResolution


def _make_entry(key: str = "k1") -> CacheEntry:
    candidate = SelectorCandidate(
        selector="#join",
        selector_type="css",
        action="click",
        action_value=None,
        confidence=0.9,
        reasoning="stable id",
    )
    resolution = SelectorResolution(
        intent="click join",
        dom_hash=key,
        candidates=[candidate],
        chosen_selector=candidate.selector,
        chosen_selector_type=candidate.selector_type,
        chosen_action=candidate.action,
        chosen_action_value=candidate.action_value,
    )
    return CacheEntry(
        key=key,
        resolution=resolution,
        hit_count=0,
        last_used_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache.db"


async def test_put_then_get_roundtrips(cache_path: Path) -> None:
    cache = SQLiteCache(cache_path)
    try:
        entry = _make_entry("abc")
        await cache.put(entry)
        got = await cache.get("abc")
        assert got is not None
        assert got.key == "abc"
        assert got.resolution.chosen_selector == "#join"
        assert got.resolution.chosen_action == "click"
    finally:
        await cache.close()


async def test_get_missing_returns_none(cache_path: Path) -> None:
    cache = SQLiteCache(cache_path)
    try:
        assert await cache.get("nope") is None
    finally:
        await cache.close()


async def test_hit_count_increments_on_each_get(cache_path: Path) -> None:
    cache = SQLiteCache(cache_path)
    try:
        await cache.put(_make_entry("x"))
        first = await cache.get("x")
        second = await cache.get("x")
        third = await cache.get("x")
        assert first is not None and first.hit_count == 1
        assert second is not None and second.hit_count == 2
        assert third is not None and third.hit_count == 3
    finally:
        await cache.close()


async def test_invalidate_removes_entry(cache_path: Path) -> None:
    cache = SQLiteCache(cache_path)
    try:
        await cache.put(_make_entry("y"))
        assert await cache.get("y") is not None
        await cache.invalidate("y")
        assert await cache.get("y") is None
    finally:
        await cache.close()


async def test_invalidate_missing_is_noop(cache_path: Path) -> None:
    cache = SQLiteCache(cache_path)
    try:
        await cache.invalidate("never-existed")  # should not raise
    finally:
        await cache.close()


async def test_put_is_upsert(cache_path: Path) -> None:
    cache = SQLiteCache(cache_path)
    try:
        await cache.put(_make_entry("z"))
        await cache.get("z")  # bumps hit_count on disk to 1
        # Overwrite with a fresh entry that has hit_count=0.
        await cache.put(_make_entry("z"))
        got = await cache.get("z")
        assert got is not None
        assert got.hit_count == 1  # the fresh 0 + this one get
    finally:
        await cache.close()


async def test_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "cache.db"
    cache = SQLiteCache(nested)
    try:
        await cache.put(_make_entry("k"))
        assert nested.exists()
        assert (await cache.get("k")) is not None
    finally:
        await cache.close()


async def test_corrupted_entry_is_invalidated_and_raises(cache_path: Path, tmp_path: Path) -> None:
    import aiosqlite  # local import to avoid polluting module namespace

    cache = SQLiteCache(cache_path)
    try:
        await cache.put(_make_entry("corrupt"))
        # Manually clobber the stored JSON to simulate disk corruption.
        async with aiosqlite.connect(cache_path) as direct:
            await direct.execute(
                "UPDATE engrams SET resolution = ? WHERE key = ?",
                ("{not-json", "corrupt"),
            )
            await direct.commit()
        with pytest.raises(EngramCacheError):
            await cache.get("corrupt")
        # The entry should have been removed as part of the recovery.
        assert await cache.get("corrupt") is None
    finally:
        await cache.close()
