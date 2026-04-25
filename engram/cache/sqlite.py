"""SQLite-backed implementation of :class:`AbstractCache`.

Zero external dependencies beyond ``aiosqlite``. The database file is
created on first use if missing, along with any parent directories.

Concurrency model: one connection per cache instance, serialised by
aiosqlite's internal queue. Multiple cache instances pointing at the same
file are safe (SQLite's own file locking handles cross-instance writes),
but for a shared cache across machines you should use a Redis backend
instead — this module is optimised for single-machine local development.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

import aiosqlite

from engram.models.errors import EngramCacheError
from engram.models.resolution import CacheEntry, SelectorResolution

_SCHEMA = """
CREATE TABLE IF NOT EXISTS engrams (
    key            TEXT PRIMARY KEY,
    resolution     TEXT NOT NULL,
    hit_count      INTEGER NOT NULL DEFAULT 0,
    last_used_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_engrams_last_used_at ON engrams (last_used_at);
"""


class SQLiteCache:
    """Local, file-backed cache for selector resolutions."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def _connect(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self._path)
            await self._conn.executescript(_SCHEMA)
            await self._conn.commit()
        except (OSError, aiosqlite.Error) as exc:
            raise EngramCacheError(f"Failed to open cache at {self._path}: {exc}") from exc
        return self._conn

    async def get(self, key: str) -> CacheEntry | None:
        conn = await self._connect()
        try:
            async with conn.execute(
                "SELECT resolution, hit_count, last_used_at FROM engrams WHERE key = ?",
                (key,),
            ) as cursor:
                row = await cursor.fetchone()
        except aiosqlite.Error as exc:
            raise EngramCacheError(f"Cache read failed: {exc}") from exc

        if row is None:
            return None

        resolution_json, hit_count, _last_used_at = row
        try:
            resolution = SelectorResolution.model_validate_json(resolution_json)
        except ValueError as exc:
            # Corrupted row: invalidate so we re-resolve rather than crash.
            await self.invalidate(key)
            raise EngramCacheError(
                f"Corrupted cache entry for key {key!r}; entry has been removed"
            ) from exc

        new_hit_count = int(hit_count) + 1
        new_last_used = datetime.now(timezone.utc)
        try:
            await conn.execute(
                "UPDATE engrams SET hit_count = ?, last_used_at = ? WHERE key = ?",
                (new_hit_count, new_last_used.isoformat(), key),
            )
            await conn.commit()
        except aiosqlite.Error as exc:
            raise EngramCacheError(f"Cache hit-count update failed: {exc}") from exc

        return CacheEntry(
            key=key,
            resolution=resolution,
            hit_count=new_hit_count,
            last_used_at=new_last_used,
        )

    async def put(self, entry: CacheEntry) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                "INSERT OR REPLACE INTO engrams (key, resolution, hit_count, last_used_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    entry.key,
                    entry.resolution.model_dump_json(),
                    entry.hit_count,
                    entry.last_used_at.isoformat(),
                ),
            )
            await conn.commit()
        except aiosqlite.Error as exc:
            raise EngramCacheError(f"Cache write failed: {exc}") from exc

    async def invalidate(self, key: str) -> None:
        conn = await self._connect()
        try:
            await conn.execute("DELETE FROM engrams WHERE key = ?", (key,))
            await conn.commit()
        except aiosqlite.Error as exc:
            raise EngramCacheError(f"Cache invalidation failed: {exc}") from exc

    async def close(self) -> None:
        if self._conn is not None:
            try:
                await self._conn.close()
            finally:
                self._conn = None

    async def __aenter__(self) -> SQLiteCache:
        await self._connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
