"""Abstract cache interface.

Implementations must be safe to call from a single asyncio event loop. They
do not need to be safe across threads or processes; callers that require that
are responsible for using a backend (e.g. Redis) that provides the guarantee.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from engram.models.resolution import CacheEntry


@runtime_checkable
class AbstractCache(Protocol):
    """Protocol for Engram selector-resolution caches."""

    async def get(self, key: str) -> CacheEntry | None:
        """Return the entry for ``key``, or ``None`` if absent.

        Implementations should update ``last_used_at`` / ``hit_count`` on the
        returned entry when a hit occurs, so that observability and future
        eviction policies have the data they need.
        """
        ...

    async def put(self, entry: CacheEntry) -> None:
        """Insert or replace the entry for ``entry.key``."""
        ...

    async def invalidate(self, key: str) -> None:
        """Remove the entry for ``key`` if present. No-op if absent."""
        ...

    async def close(self) -> None:
        """Release any underlying resources. Safe to call more than once."""
        ...
