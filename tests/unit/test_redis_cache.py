"""Tests for the Redis-backed cache.

The redis-py async client is too involved to mock by hand for end-to-end
parity, so we drive :class:`RedisCache` against an in-memory fake that
implements the small ``_RedisLike`` Protocol surface the cache uses. The
fake also exposes hooks for simulating Redis errors so we can verify the
cache translates them into :class:`EngramCacheError` consistently.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from redis.exceptions import RedisError

from engram.cache.redis import RedisCache
from engram.models.errors import EngramCacheError
from engram.models.resolution import ActionResolution, CacheEntry, SelectorCandidate

# --------------------------------------------------------------------------- fake


class _FakeRedis:
    """In-memory stand-in for the bits of ``redis.asyncio.Redis`` we use."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, str]] = {}
        self.expirations: dict[str, int] = {}
        self.fail_on: set[str] = set()
        self.aclose_called = False

    def _maybe_fail(self, op: str) -> None:
        if op in self.fail_on:
            raise RedisError(f"simulated {op} failure")

    async def hgetall(self, name: str) -> dict[str, str]:
        self._maybe_fail("hgetall")
        return dict(self.store.get(name, {}))

    async def hset(self, name: str, *, mapping: dict[str, str | int]) -> int:
        self._maybe_fail("hset")
        bucket = self.store.setdefault(name, {})
        added = 0
        for k, v in mapping.items():
            if k not in bucket:
                added += 1
            bucket[k] = str(v)
        return added

    async def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        self._maybe_fail("hincrby")
        bucket = self.store.setdefault(name, {})
        current = int(bucket.get(key, "0"))
        new = current + amount
        bucket[key] = str(new)
        return new

    async def expire(self, name: str, seconds: int) -> bool:
        self._maybe_fail("expire")
        if name not in self.store:
            return False
        self.expirations[name] = seconds
        return True

    async def delete(self, *names: str) -> int:
        self._maybe_fail("delete")
        n = 0
        for name in names:
            if name in self.store:
                del self.store[name]
                self.expirations.pop(name, None)
                n += 1
        return n

    async def aclose(self) -> None:
        self.aclose_called = True


# --------------------------------------------------------------------------- helpers


def _make_entry(key: str = "k1") -> CacheEntry:
    candidate = SelectorCandidate(
        selector="#join",
        selector_type="css",
        action="click",
        action_value=None,
        confidence=0.9,
        reasoning="stable id",
    )
    resolution = ActionResolution(
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


# --------------------------------------------------------------------------- tests


async def test_put_then_get_roundtrips() -> None:
    fake = _FakeRedis()
    cache = RedisCache(fake)
    try:
        await cache.put(_make_entry("abc"))
        got = await cache.get("abc")
        assert got is not None
        assert got.key == "abc"
        assert isinstance(got.resolution, ActionResolution)
        assert got.resolution.chosen_selector == "#join"
        assert got.resolution.chosen_action == "click"
    finally:
        await cache.close()


async def test_get_missing_returns_none() -> None:
    cache = RedisCache(_FakeRedis())
    try:
        assert await cache.get("nope") is None
    finally:
        await cache.close()


async def test_hit_count_increments_on_each_get() -> None:
    fake = _FakeRedis()
    cache = RedisCache(fake)
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


async def test_invalidate_removes_entry() -> None:
    fake = _FakeRedis()
    cache = RedisCache(fake)
    try:
        await cache.put(_make_entry("y"))
        assert await cache.get("y") is not None
        await cache.invalidate("y")
        assert await cache.get("y") is None
    finally:
        await cache.close()


async def test_invalidate_missing_is_noop() -> None:
    cache = RedisCache(_FakeRedis())
    try:
        await cache.invalidate("never-existed")  # should not raise
    finally:
        await cache.close()


async def test_key_prefix_namespaces_keys() -> None:
    """Two caches with different prefixes against the same client are isolated."""
    fake = _FakeRedis()
    cache_a = RedisCache(fake, key_prefix="env-a:")
    cache_b = RedisCache(fake, key_prefix="env-b:")
    try:
        await cache_a.put(_make_entry("shared"))
        # Same logical key, different prefix → distinct entries.
        assert await cache_b.get("shared") is None
        assert "env-a:shared" in fake.store
        assert "env-b:shared" not in fake.store
    finally:
        await cache_a.close()
        await cache_b.close()


async def test_ttl_is_applied_on_put_and_hit() -> None:
    fake = _FakeRedis()
    cache = RedisCache(fake, ttl_seconds=600)
    try:
        await cache.put(_make_entry("ttl-key"))
        assert fake.expirations["engram:ttl-key"] == 600
        # Bump TTL on access — long-tail entries that keep getting hit
        # should not silently expire underneath active workloads.
        fake.expirations.pop("engram:ttl-key")
        await cache.get("ttl-key")
        assert fake.expirations["engram:ttl-key"] == 600
    finally:
        await cache.close()


async def test_corrupted_resolution_is_invalidated_and_raises() -> None:
    fake = _FakeRedis()
    cache = RedisCache(fake)
    try:
        # Bypass the cache and write a malformed JSON blob directly.
        fake.store["engram:corrupt"] = {
            "resolution": "{not-json",
            "hit_count": "0",
            "last_used_at": datetime.now(timezone.utc).isoformat(),
        }
        with pytest.raises(EngramCacheError):
            await cache.get("corrupt")
        # Recovery: corrupt entry is gone, next get is a clean miss.
        assert await cache.get("corrupt") is None
    finally:
        await cache.close()


async def test_missing_resolution_field_is_invalidated_and_raises() -> None:
    fake = _FakeRedis()
    cache = RedisCache(fake)
    try:
        fake.store["engram:partial"] = {"hit_count": "0"}
        with pytest.raises(EngramCacheError, match="missing 'resolution'"):
            await cache.get("partial")
        assert await cache.get("partial") is None
    finally:
        await cache.close()


async def test_redis_errors_are_translated_to_cache_errors() -> None:
    fake = _FakeRedis()
    cache = RedisCache(fake)
    try:
        fake.fail_on = {"hgetall"}
        with pytest.raises(EngramCacheError, match="Cache read failed"):
            await cache.get("anything")

        fake.fail_on = {"hset"}
        with pytest.raises(EngramCacheError, match="Cache write failed"):
            await cache.put(_make_entry("z"))

        fake.fail_on = {"delete"}
        with pytest.raises(EngramCacheError, match="Cache invalidation failed"):
            await cache.invalidate("z")
    finally:
        await cache.close()


async def test_close_is_idempotent() -> None:
    fake = _FakeRedis()
    cache = RedisCache(fake)
    await cache.close()
    await cache.close()  # second call must not raise or re-call aclose
    assert fake.aclose_called is True


async def test_url_credentials_redacted_in_error_messages() -> None:
    """A redis-py error that mentions the URL must not surface the password.

    redis-py commonly embeds the URL in connection errors (auth failures,
    DNS failures, etc). Pasting an :class:`EngramCacheError` into a chat
    or bug tracker would leak the credential without redaction.
    """
    fake = _FakeRedis()
    cache = RedisCache(fake)
    try:
        # Simulate a redis-py exception that names the URL with a password.
        fake.fail_on = {"hgetall"}
        secret = "s3cret-shh"

        # Patch _maybe_fail to raise a richer error message containing the URL.
        original_maybe_fail = fake._maybe_fail

        def loud_fail(op: str) -> None:
            if op in fake.fail_on:
                raise RedisError(
                    f"Cannot connect to redis://alice:{secret}@host:6379/0 — auth"
                )
            original_maybe_fail(op)

        fake._maybe_fail = loud_fail  # type: ignore[method-assign]

        with pytest.raises(EngramCacheError) as excinfo:
            await cache.get("anything")
        msg = str(excinfo.value)
        assert secret not in msg
        assert "alice:***@" in msg
        # Username retained for diagnostics.
        assert "alice" in msg
    finally:
        await cache.close()


def test_redact_credentials_preserves_urls_without_password() -> None:
    """A URL like ``redis://host:6379/0`` (no userinfo) is left alone."""
    from engram.cache.redis import _redact_credentials

    text = "Connection refused: redis://localhost:6379/0"
    assert _redact_credentials(text) == text


def test_redact_credentials_handles_rediss_scheme() -> None:
    from engram.cache.redis import _redact_credentials

    text = "Auth failed: rediss://bob:hunter2@cache.example.com:6380/1"
    redacted = _redact_credentials(text)
    assert "hunter2" not in redacted
    assert "bob:***@" in redacted


# ----------------------------------------------- HMAC-signed cache records


async def test_signed_put_then_signed_get_roundtrips() -> None:
    """A cache configured with a signing key signs writes and verifies reads."""
    fake = _FakeRedis()
    cache = RedisCache(fake, signing_key=b"shared-secret")
    try:
        await cache.put(_make_entry("k"))
        # The signature field must be present on the stored entry.
        assert "signature" in fake.store["engram:k"]
        # Round-trip must verify and return the original.
        got = await cache.get("k")
        assert got is not None
        assert isinstance(got.resolution, ActionResolution)
        assert got.resolution.chosen_selector == "#join"
    finally:
        await cache.close()


async def test_unsigned_put_then_signed_get_is_rejected() -> None:
    """An attacker writing entries without the signing key must not be able to
    poison a signed reader. Engram refuses entries with no signature."""
    fake = _FakeRedis()
    # Writer has no key — simulates an attacker (or legacy entries).
    writer = RedisCache(fake)
    # Reader requires signatures.
    reader = RedisCache(fake, signing_key=b"shared-secret")
    try:
        await writer.put(_make_entry("poisoned"))
        with pytest.raises(EngramCacheError, match="signature verification"):
            await reader.get("poisoned")
        # Failed verification removes the bad entry; second read is a clean miss.
        assert await reader.get("poisoned") is None
    finally:
        await writer.close()
        await reader.close()


async def test_signed_put_then_get_with_wrong_key_is_rejected() -> None:
    """Key rotation hazard: a reader with the wrong key must not accept entries."""
    fake = _FakeRedis()
    writer = RedisCache(fake, signing_key=b"key-A")
    reader = RedisCache(fake, signing_key=b"key-B")
    try:
        await writer.put(_make_entry("rotated"))
        with pytest.raises(EngramCacheError, match="signature verification"):
            await reader.get("rotated")
    finally:
        await writer.close()
        await reader.close()


async def test_tampered_resolution_is_rejected_by_signed_reader() -> None:
    """Modifying the resolution payload after signing breaks verification."""
    fake = _FakeRedis()
    cache = RedisCache(fake, signing_key=b"shared-secret")
    try:
        await cache.put(_make_entry("tampered"))
        # Surreptitiously rewrite the stored resolution to point at a
        # malicious selector — without re-signing.
        bucket = fake.store["engram:tampered"]
        original = bucket["resolution"]
        bucket["resolution"] = original.replace("#join", "#delete-account")
        with pytest.raises(EngramCacheError, match="signature verification"):
            await cache.get("tampered")
        # Tampered entry is gone after the failed verify.
        assert await cache.get("tampered") is None
    finally:
        await cache.close()


async def test_unsigned_reader_ignores_signatures_on_signed_entries() -> None:
    """Backward-compat: a cache without a signing key reads entries normally
    even if the writer signed them. Opting out of signing is the user's
    explicit choice and shouldn't break reads of previously-signed data."""
    fake = _FakeRedis()
    signing_writer = RedisCache(fake, signing_key=b"k")
    plain_reader = RedisCache(fake)
    try:
        await signing_writer.put(_make_entry("legacy"))
        # The signature field is in storage but the reader ignores it.
        assert "signature" in fake.store["engram:legacy"]
        got = await plain_reader.get("legacy")
        assert got is not None
        assert isinstance(got.resolution, ActionResolution)
    finally:
        await signing_writer.close()
        await plain_reader.close()


async def test_factory_passes_signing_key_through() -> None:
    """Verify build_cache wires cache_signing_key into RedisCache.from_url."""
    from pydantic import SecretStr

    from engram.cache import build_cache
    from engram.cache.redis import RedisCache as RedisCacheCls
    from engram.models.config import EngramConfig

    config = EngramConfig(
        cache_backend="redis",
        redis_url="redis://localhost:6379/0",
        cache_signing_key=SecretStr("test-signing-key"),
    )
    cache = build_cache(config)
    assert isinstance(cache, RedisCacheCls)
    # Signing key was plumbed through (private field check is OK here — this
    # test exists exactly to pin the wiring).
    assert cache._signing_key == b"test-signing-key"


async def test_put_overwrites_previous_entry() -> None:
    fake = _FakeRedis()
    cache = RedisCache(fake)
    try:
        await cache.put(_make_entry("z"))
        await cache.get("z")  # hit_count -> 1 on disk
        # Fresh entry resets hit_count to 0; subsequent get bumps to 1.
        await cache.put(_make_entry("z"))
        got = await cache.get("z")
        assert got is not None
        assert got.hit_count == 1
    finally:
        await cache.close()
