"""SQLite-backed implementation of :class:`AbstractCache`.

Zero external dependencies beyond ``aiosqlite``. The database file is
created on first use if missing, along with any parent directories.

Concurrency model: one connection per cache instance, serialised by
aiosqlite's internal queue. Multiple cache instances pointing at the same
file are safe (SQLite's own file locking handles cross-instance writes),
but for a shared cache across machines you should use a Redis backend
instead — this module is optimised for single-machine local development.

Hit-count updates use ``UPDATE ... RETURNING`` (SQLite 3.35+, shipped with
all CPython 3.10 builds we support) so each ``get`` is one atomic statement
rather than the SELECT-then-UPDATE pair that would race across instances.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

import aiosqlite
from pydantic import TypeAdapter, ValidationError

from engram.models.errors import EngramCacheError
from engram.models.resolution import CacheEntry, Resolution

# A single TypeAdapter for the discriminated Resolution union — Pydantic
# dispatches to the correct concrete model (ActionResolution /
# ExtractionResolution / ObservationResolution) based on the ``kind`` tag in
# the stored JSON. Building the adapter once at import time avoids paying its
# (non-trivial) construction cost on every cache read.
_RESOLUTION_ADAPTER: TypeAdapter[Resolution] = TypeAdapter(Resolution)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS engrams (
    key            TEXT PRIMARY KEY,
    resolution     TEXT NOT NULL,
    hit_count      INTEGER NOT NULL DEFAULT 0,
    last_used_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_engrams_last_used_at ON engrams (last_used_at);
"""

# WAL mode allows concurrent readers alongside a single writer (vs the default
# rollback journal which serialises everything). ``busy_timeout`` makes a
# writer wait up to N ms for a lock to clear, so contention manifests as
# latency rather than ``database is locked`` exceptions. These PRAGMAs are
# applied via individual execute() calls (not executescript) because
# changing journal_mode is a no-op inside a transaction, and executescript
# wraps its body in BEGIN/COMMIT.
_PRAGMA_STATEMENTS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=5000",
    "PRAGMA synchronous=NORMAL",
)


class SQLiteCache:
    """Local, file-backed cache for selector resolutions."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._conn: aiosqlite.Connection | None = None
        # Serialises first-time connection so that concurrent ``get`` calls
        # from the same instance don't each try to apply the schema.
        self._connect_lock = asyncio.Lock()

    async def _connect(self) -> aiosqlite.Connection:
        # Fast path: connection already established. Re-checking under the
        # lock below catches the race where two callers reach this point
        # before either has set ``self._conn``.
        if self._conn is not None:
            return self._conn
        async with self._connect_lock:
            if self._conn is not None:
                return self._conn
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # ``isolation_level=None`` puts the connection in autocommit
                # mode — each statement is its own implicit transaction.
                # That lets concurrent ``get``/``put`` calls (which are all
                # single statements) interleave safely without leaving a
                # transaction "in progress" when another coroutine wants to
                # commit. We don't need multi-statement atomicity here: the
                # cache contract is one row in / one row out.
                self._conn = await aiosqlite.connect(
                    self._path, isolation_level=None
                )
                for pragma in _PRAGMA_STATEMENTS:
                    # PRAGMA statements may return a row (e.g. the new journal
                    # mode); consume it so the cursor closes cleanly.
                    async with self._conn.execute(pragma) as cursor:
                        await cursor.fetchall()
                await self._conn.executescript(_SCHEMA)
            except (OSError, aiosqlite.Error) as exc:
                raise EngramCacheError(
                    f"Failed to open cache at {self._path}: {exc}"
                ) from exc
            return self._conn

    async def get(self, key: str) -> CacheEntry | None:
        conn = await self._connect()
        new_last_used = datetime.now(timezone.utc)
        # One atomic UPDATE…RETURNING rather than SELECT-then-UPDATE: prevents
        # the lost-increment race when two SQLiteCache instances point at the
        # same file and call ``get`` on the same key concurrently. SQLite
        # serialises the UPDATE so the second caller sees the bumped counter.
        # No commit() needed — connection is in autocommit mode.
        try:
            async with conn.execute(
                "UPDATE engrams "
                "SET hit_count = hit_count + 1, last_used_at = ? "
                "WHERE key = ? "
                "RETURNING resolution, hit_count",
                (new_last_used.isoformat(), key),
            ) as cursor:
                row = await cursor.fetchone()
        except aiosqlite.Error as exc:
            raise EngramCacheError(f"Cache read failed: {exc}") from exc

        if row is None:
            return None

        resolution_json, hit_count = row
        try:
            resolution = _RESOLUTION_ADAPTER.validate_json(resolution_json)
        except ValidationError as exc:
            # Corrupted row, or a row written by a future version with an
            # unknown ``kind``. Either way: invalidate so the next call
            # re-resolves cleanly rather than crashing forever.
            await self.invalidate(key)
            raise EngramCacheError(
                f"Corrupted cache entry for key {key!r}; entry has been removed"
            ) from exc

        return CacheEntry(
            key=key,
            resolution=resolution,
            hit_count=int(hit_count),
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
                    _RESOLUTION_ADAPTER.dump_json(entry.resolution).decode(),
                    entry.hit_count,
                    entry.last_used_at.isoformat(),
                ),
            )
        except aiosqlite.Error as exc:
            raise EngramCacheError(f"Cache write failed: {exc}") from exc

    async def invalidate(self, key: str) -> None:
        conn = await self._connect()
        try:
            await conn.execute("DELETE FROM engrams WHERE key = ?", (key,))
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
