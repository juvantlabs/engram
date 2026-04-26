"""Tests for the ``build_cache`` factory.

The factory is the single point that decides which backend a given config
gets. The defaults must produce SQLite (no surprise opt-in to a network
service), and explicit Redis configuration must produce Redis — no silent
fallbacks that hide a misconfiguration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engram.cache import build_cache
from engram.cache.redis import RedisCache
from engram.cache.sqlite import SQLiteCache
from engram.models.config import EngramConfig
from engram.models.errors import EngramConfigError


def test_default_config_yields_sqlite_cache(tmp_path: Path) -> None:
    config = EngramConfig(cache_path=tmp_path / "cache.db")
    cache = build_cache(config)
    assert isinstance(cache, SQLiteCache)


def test_explicit_sqlite_yields_sqlite_cache(tmp_path: Path) -> None:
    config = EngramConfig(cache_backend="sqlite", cache_path=tmp_path / "cache.db")
    cache = build_cache(config)
    assert isinstance(cache, SQLiteCache)


def test_redis_with_url_yields_redis_cache() -> None:
    # ``redis://`` is parsed by redis-py without opening a connection, so we
    # can exercise the from_url path without any live server.
    config = EngramConfig(
        cache_backend="redis",
        redis_url="redis://localhost:6379/0",
        redis_key_prefix="test:",
        redis_ttl_seconds=60,
    )
    cache = build_cache(config)
    assert isinstance(cache, RedisCache)


def test_redis_without_url_raises() -> None:
    """A redis backend without a URL is a misconfiguration, not a fallback.

    Silently degrading to SQLite would mean every machine builds its own
    cache, defeating the entire point of opting into Redis — and the user
    would not notice for weeks.
    """
    config = EngramConfig(cache_backend="redis", redis_url=None)
    with pytest.raises(EngramConfigError, match="redis_url"):
        build_cache(config)
