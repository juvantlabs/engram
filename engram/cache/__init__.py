"""Cache backends for selector resolutions."""

from engram.cache.base import AbstractCache
from engram.cache.redis import RedisCache
from engram.cache.sqlite import SQLiteCache
from engram.models.config import EngramConfig
from engram.models.errors import EngramConfigError


def build_cache(config: EngramConfig) -> AbstractCache:
    """Construct the cache backend selected by ``config``.

    SQLite is the default. Redis is opt-in: the caller must set
    ``cache_backend='redis'`` AND provide ``redis_url`` (typically via
    ``ENGRAM_CACHE_BACKEND=redis`` and ``ENGRAM_REDIS_URL=...``). A Redis
    backend without a URL is a configuration error rather than a silent
    fallback — silently degrading from a shared cache to a per-machine
    one is the kind of failure that hides for weeks before someone
    notices their LLM bill.

    :raises EngramConfigError: If ``cache_backend='redis'`` but
        ``redis_url`` is unset, or if the ``redis`` extra is not
        installed.
    """
    if config.cache_backend == "redis":
        if config.redis_url is None:
            raise EngramConfigError(
                "cache_backend='redis' requires redis_url to be set "
                "(e.g. 'redis://localhost:6379/0', or ENGRAM_REDIS_URL=...)"
            )
        signing_key = (
            config.cache_signing_key.get_secret_value().encode("utf-8")
            if config.cache_signing_key is not None
            else None
        )
        return RedisCache.from_url(
            config.redis_url,
            key_prefix=config.redis_key_prefix,
            ttl_seconds=config.redis_ttl_seconds,
            signing_key=signing_key,
        )
    return SQLiteCache(config.cache_path)


__all__ = ["AbstractCache", "RedisCache", "SQLiteCache", "build_cache"]
