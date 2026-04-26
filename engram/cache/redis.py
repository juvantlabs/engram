"""Redis-backed implementation of :class:`AbstractCache`.

For shared multi-machine deployments where a local SQLite file is no longer
sufficient: an Engram cache hit on machine A becomes a hit for every other
machine pointed at the same Redis instance — which is the whole point of
caching LLM-resolved selectors at a team or org level.

Storage layout: each cache entry is a Redis hash at ``{key_prefix}{key}``
with three fields — ``resolution`` (JSON), ``hit_count`` (integer string),
and ``last_used_at`` (ISO timestamp). Hash storage means ``hit_count`` can
be incremented atomically with ``HINCRBY`` instead of read-modify-write,
which matters when many workers share the same cache.

The ``redis`` package is an optional dependency::

    pip install 'engram[redis]'

Importing this module does NOT import ``redis`` itself; only :meth:`from_url`
does. That way users without the extra installed can still import
:mod:`engram.cache` (e.g. when only the SQLite backend is in use) without
crashing at import time.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import re
from datetime import datetime, timezone
from typing import Protocol, cast

from pydantic import TypeAdapter, ValidationError

from engram.models.errors import EngramCacheError, EngramConfigError
from engram.models.resolution import CacheEntry, Resolution

# Matches ``scheme://user:password@host`` userinfo segments and captures the
# scheme+user portion so the substitution can keep the username (useful for
# diagnostics) and only redact the password. Schemes are restricted to ones
# we'd plausibly see in cache config to avoid eating unrelated colons in
# error messages.
_URL_CREDENTIAL_RE = re.compile(
    r"\b((?:rediss?|valkey)://[^:/\s@]+):[^@/\s]+@",
)

# Built once at import time; building per-call costs measurable wall time on
# discriminated unions of this shape.
_RESOLUTION_ADAPTER: TypeAdapter[Resolution] = TypeAdapter(Resolution)


# We translate any exception raised by the underlying client into
# :class:`EngramCacheError`, but to do that without catching unrelated
# bugs we want to catch the redis-py base exception class. The class is
# only available if the ``redis`` extra is installed; if it is not, we
# bind a sentinel that no real client could raise — and ``from_url``
# will have raised :class:`EngramConfigError` long before any method
# would actually need to catch it.
try:
    from redis.exceptions import RedisError as _RedisError
except ImportError:  # pragma: no cover — only true when extra is not installed
    class _RedisError(Exception):  # type: ignore[no-redef]
        """Sentinel; never raised at runtime when the ``redis`` extra is missing."""


def _redact_credentials(text: str) -> str:
    """Replace ``:password@`` with ``:***@`` in any Redis URL in ``text``.

    redis-py's exception messages frequently embed the connection URL in
    plain text — a wrong password becomes ``"Authentication failed: redis://
    alice:s3cret@host"``. Surfacing that string verbatim in an
    :class:`EngramCacheError` (which a user might paste into Slack/Sentry/
    a bug report) leaks the credential. We scrub our wrapped error
    messages defensively without requiring the caller to track which
    strings might contain a URL.

    The username is preserved because it is useful for diagnostics
    ("which user am I authenticated as?") and is not, in practice, the
    sensitive half of the credential pair.
    """
    return _URL_CREDENTIAL_RE.sub(r"\1:***@", text)


class _RedisLike(Protocol):
    """The subset of the redis-py async client surface that :class:`RedisCache` uses.

    Declared as a Protocol so :class:`RedisCache` can be unit-tested with an
    in-memory fake without importing ``redis`` at module load time. This is
    what makes the Redis extra genuinely optional.
    """

    async def hgetall(self, name: str) -> dict[str, str]: ...

    async def hset(self, name: str, *, mapping: dict[str, str | int]) -> int: ...

    async def hincrby(self, name: str, key: str, amount: int = ...) -> int: ...

    async def expire(self, name: str, seconds: int) -> bool: ...

    async def delete(self, *names: str) -> int: ...

    async def aclose(self) -> None: ...


class RedisCache:
    """Distributed selector-resolution cache backed by Redis.

    Construct directly with an injected client (preferred for tests), or via
    :meth:`from_url` for a real Redis connection.
    """

    def __init__(
        self,
        client: _RedisLike,
        *,
        key_prefix: str = "engram:",
        ttl_seconds: int | None = None,
        signing_key: bytes | None = None,
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds
        self._signing_key = signing_key
        self._closed = False

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        key_prefix: str = "engram:",
        ttl_seconds: int | None = None,
        signing_key: bytes | None = None,
    ) -> RedisCache:
        """Build a :class:`RedisCache` from a Redis connection URL.

        :param signing_key: Optional shared secret. When set, every entry
            written is HMAC-SHA256 signed and every entry read is verified;
            unsigned or tampered entries are rejected and removed. Use this
            whenever the Redis instance is reachable from untrusted hosts —
            without it, anyone with Redis write access can inject arbitrary
            selectors that subsequent ``act`` calls will execute.
        :raises EngramConfigError: If the ``redis`` extra is not installed.
        """
        try:
            from redis.asyncio import from_url as redis_from_url
        except ImportError as exc:
            raise EngramConfigError(
                "Redis cache backend requires the 'redis' package. "
                "Install with: pip install 'engram[redis]'"
            ) from exc
        # ``decode_responses=True`` so hgetall returns ``dict[str, str]`` rather
        # than ``dict[bytes, bytes]`` — keeps the protocol clean and avoids
        # decoding noise at every call site.
        client = redis_from_url(url, decode_responses=True)
        # redis-py's async methods are typed ``Awaitable[Any] | Any`` to share
        # one class for sync + async usage. That return type is too wide to
        # satisfy our Protocol via structural matching, so we cast at the
        # boundary; everything inside the cache then stays statically typed.
        return cls(
            cast(_RedisLike, client),
            key_prefix=key_prefix,
            ttl_seconds=ttl_seconds,
            signing_key=signing_key,
        )

    def _sign(self, payload: str) -> str:
        # Asserted by the caller: signing is only invoked when a key is set.
        assert self._signing_key is not None
        return hmac.new(self._signing_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _verify(self, payload: str, expected: str | None) -> bool:
        """Constant-time HMAC verification.

        Returns ``False`` if no signature is present (can't verify what isn't
        there), if it doesn't match, or if the bytes don't decode. Never
        raises — callers translate ``False`` into the recovery path.
        """
        if expected is None:
            return False
        try:
            return hmac.compare_digest(self._sign(payload), expected)
        except (TypeError, ValueError):
            return False

    def _redis_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    async def get(self, key: str) -> CacheEntry | None:
        redis_key = self._redis_key(key)
        try:
            data = await self._client.hgetall(redis_key)
        except _RedisError as exc:
            raise EngramCacheError(
                _redact_credentials(f"Cache read failed: {exc}")
            ) from exc

        if not data:
            return None

        resolution_json = data.get("resolution")
        if resolution_json is None:
            # Hash present but missing the canonical field — treat as corrupt
            # and remove so the next call re-resolves cleanly. Mirrors the
            # SQLite backend's recovery behaviour.
            await self._safe_delete(redis_key)
            raise EngramCacheError(
                f"Corrupted cache entry for key {key!r} "
                "(missing 'resolution' field); entry has been removed"
            )

        if self._signing_key is not None and not self._verify(
            resolution_json, data.get("signature")
        ):
            # Signing was opted into but this entry doesn't verify — either
            # someone wrote it without the key (poisoning), or wrote it with
            # the wrong key (rotation hazard), or tampered after writing.
            # In all cases: refuse, remove, surface a clear error.
            await self._safe_delete(redis_key)
            raise EngramCacheError(
                f"Cache entry for key {key!r} failed signature verification; "
                "entry has been removed"
            )

        try:
            resolution = _RESOLUTION_ADAPTER.validate_json(resolution_json)
        except ValidationError as exc:
            await self._safe_delete(redis_key)
            raise EngramCacheError(
                f"Corrupted cache entry for key {key!r}; entry has been removed"
            ) from exc

        new_last = datetime.now(timezone.utc)
        try:
            new_count = await self._client.hincrby(redis_key, "hit_count", 1)
            await self._client.hset(
                redis_key, mapping={"last_used_at": new_last.isoformat()}
            )
            if self._ttl_seconds is not None:
                await self._client.expire(redis_key, self._ttl_seconds)
        except _RedisError as exc:
            raise EngramCacheError(
                _redact_credentials(f"Cache hit-count update failed: {exc}")
            ) from exc

        return CacheEntry(
            key=key,
            resolution=resolution,
            hit_count=int(new_count),
            last_used_at=new_last,
        )

    async def put(self, entry: CacheEntry) -> None:
        redis_key = self._redis_key(entry.key)
        resolution_json = _RESOLUTION_ADAPTER.dump_json(entry.resolution).decode()
        mapping: dict[str, str | int] = {
            "resolution": resolution_json,
            "hit_count": entry.hit_count,
            "last_used_at": entry.last_used_at.isoformat(),
        }
        # Only the resolution is signed: ``hit_count`` and ``last_used_at``
        # change on every read so there's nothing useful to authenticate, and
        # signing them would force a re-sign on every get.
        if self._signing_key is not None:
            mapping["signature"] = self._sign(resolution_json)
        try:
            await self._client.hset(redis_key, mapping=mapping)
            if self._ttl_seconds is not None:
                await self._client.expire(redis_key, self._ttl_seconds)
        except _RedisError as exc:
            raise EngramCacheError(
                _redact_credentials(f"Cache write failed: {exc}")
            ) from exc

    async def invalidate(self, key: str) -> None:
        try:
            await self._client.delete(self._redis_key(key))
        except _RedisError as exc:
            raise EngramCacheError(
                _redact_credentials(f"Cache invalidation failed: {exc}")
            ) from exc

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Closing a cache should never raise — the caller is past the point
        # where they could meaningfully recover. Swallowing here also matches
        # the SQLiteCache contract.
        with contextlib.suppress(_RedisError):
            await self._client.aclose()

    async def _safe_delete(self, redis_key: str) -> None:
        # Best-effort: we are already raising for the caller, so a delete
        # failure here would only produce a confusing second exception.
        with contextlib.suppress(_RedisError):
            await self._client.delete(redis_key)
