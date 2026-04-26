"""Engram runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from engram.models.errors import EngramConfigError

CacheBackend = Literal["sqlite", "redis"]


class EngramConfig(BaseModel):
    """Runtime configuration for an Engram session.

    All fields have safe defaults suitable for local development; production
    deployments are expected to override them (typically via :meth:`from_env`).
    """

    model_config = {"frozen": True}

    llm_model: str = Field(
        default="anthropic/claude-sonnet-4-5",
        description="LiteLLM model identifier.",
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        description="Optional explicit API key. If None, LiteLLM reads env vars.",
    )
    llm_api_base: str | None = Field(
        default=None,
        description=(
            "Optional API base URL. Required for Azure OpenAI "
            "(e.g. https://<resource>.openai.azure.com/) and any self-hosted "
            "OpenAI-compatible endpoint."
        ),
    )
    llm_api_version: str | None = Field(
        default=None,
        description="Optional API version. Required for Azure OpenAI (e.g. 2024-02-15-preview).",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=30.0, gt=0.0)

    cache_backend: CacheBackend = Field(
        default="sqlite",
        description=(
            "Which cache backend to use. ``sqlite`` (default) is a local file-backed "
            "cache suitable for single-machine development. ``redis`` is for shared "
            "deployments where one resolution discovered on any machine benefits all "
            "of them — set ``redis_url`` whenever this is ``redis``."
        ),
    )
    cache_path: Path = Field(default=Path(".engram/cache.db"))
    redis_url: str | None = Field(
        default=None,
        description=(
            "Redis connection URL (e.g. ``redis://localhost:6379/0``). Required when "
            "``cache_backend='redis'``. Ignored otherwise."
        ),
    )
    redis_key_prefix: str = Field(
        default="engram:",
        description=(
            "Namespace prefix prepended to every Redis key. Lets one Redis instance "
            "host multiple isolated Engram caches (e.g. one per environment)."
        ),
    )
    cache_signing_key: SecretStr | None = Field(
        default=None,
        description=(
            "Optional shared secret for HMAC-signing cache entries. When set, "
            "every entry written to a Redis cache is signed and every entry read "
            "is verified — a tampered or attacker-injected entry is rejected and "
            "removed instead of being executed. Strongly recommended whenever "
            "``cache_backend='redis'`` and the Redis instance is reachable from "
            "untrusted networks. The SQLite backend ignores this setting (single-"
            "machine assumption). Different sessions sharing one Redis instance "
            "must use the same signing key to read each other's entries."
        ),
    )

    redis_ttl_seconds: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional TTL applied to every cached entry in Redis. ``None`` (default) "
            "means entries never expire — invalidation is driven entirely by DOM "
            "hash changes, which is the intended Engram semantics. Set a TTL only "
            "if you also need a hard upper bound on staleness for compliance reasons."
        ),
    )

    max_llm_calls_per_session: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Hard cap on the total number of LLM calls an :class:`EngramBrowser` "
            "session will make across all primitives, before raising "
            ":class:`EngramBudgetExceededError`. ``None`` disables the cap. This "
            "is a defence against runaway flows — a pathological intent combined "
            "with high ``max_resolution_retries`` could otherwise burn through a "
            "token budget unattended. Counts every downstream call, whether it "
            "succeeds or fails."
        ),
    )

    max_resolution_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description=(
            "How many times to re-prompt the LLM with feedback when ``act`` or "
            "``extract`` produce unusable output — e.g. selectors that fail to "
            "match against the live DOM, or extracted data that fails the "
            "user-supplied Pydantic schema. ``0`` disables retries. "
            "Transport-level retries (rate-limits, transient 5xx) are NOT covered "
            "here — those are handled by LiteLLM's own retry settings, since "
            "duplicating retry budgets at two layers would silently multiply call "
            "volume."
        ),
    )
    dom_max_chars: int = Field(default=50_000, gt=0)
    min_candidate_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum LLM-reported confidence for a candidate to be considered for "
            "validation. Candidates below this threshold are filtered out before "
            "Engram tries them against the DOM — preventing the LLM from 'guessing' "
            "a plausible-looking but wrong selector when no good match exists."
        ),
    )

    log_level: str = Field(default="INFO")
    log_intents_verbatim: bool = Field(
        default=False,
        description=(
            "When ``True``, structured-log events emitted by Engram include the "
            "raw user intent string. Defaults to ``False`` because intents are "
            "user-controlled and frequently carry PII or secrets — for example, "
            "``act(\"type my password p4ssw0rd\")``. With the default, only a "
            "12-char SHA-256 prefix of the intent is logged, which is enough to "
            "correlate events without exposing the original text."
        ),
    )

    headless: bool = Field(default=True)

    @classmethod
    def from_env(cls) -> EngramConfig:
        """Build a config from environment variables.

        Recognised variables (all optional; unset or empty → defaults):

        - ``ENGRAM_LLM_MODEL``
        - ``ENGRAM_LLM_API_KEY``
        - ``ENGRAM_LLM_API_BASE`` (Azure OpenAI / self-hosted endpoints)
        - ``ENGRAM_LLM_API_VERSION`` (Azure OpenAI)
        - ``ENGRAM_LLM_TEMPERATURE``
        - ``ENGRAM_LLM_TIMEOUT_SECONDS``
        - ``ENGRAM_CACHE_BACKEND`` (``sqlite`` or ``redis``)
        - ``ENGRAM_CACHE_PATH``
        - ``ENGRAM_REDIS_URL``
        - ``ENGRAM_REDIS_KEY_PREFIX``
        - ``ENGRAM_REDIS_TTL_SECONDS``
        - ``ENGRAM_CACHE_SIGNING_KEY``
        - ``ENGRAM_MAX_RESOLUTION_RETRIES``
        - ``ENGRAM_DOM_MAX_CHARS``
        - ``ENGRAM_MIN_CANDIDATE_CONFIDENCE``
        - ``ENGRAM_LOG_LEVEL``
        - ``ENGRAM_LOG_INTENTS_VERBATIM`` (``1``/``true`` to enable)
        - ``ENGRAM_HEADLESS`` (``0``/``false`` to disable)

        Empty string values are treated as "unset" — this lets a committed
        ``.env`` template contain ``ENGRAM_LLM_API_KEY=`` without silently
        overriding a key the user configured at the provider level.
        """
        overrides: dict[str, Any] = {}

        if v := _str_env("ENGRAM_LLM_MODEL"):
            overrides["llm_model"] = v
        if v := _str_env("ENGRAM_LLM_API_KEY"):
            overrides["llm_api_key"] = SecretStr(v)
        if v := _str_env("ENGRAM_LLM_API_BASE"):
            overrides["llm_api_base"] = v
        if v := _str_env("ENGRAM_LLM_API_VERSION"):
            overrides["llm_api_version"] = v
        if v := _str_env("ENGRAM_LLM_TEMPERATURE"):
            overrides["llm_temperature"] = _parse_float("ENGRAM_LLM_TEMPERATURE", v)
        if v := _str_env("ENGRAM_LLM_TIMEOUT_SECONDS"):
            overrides["llm_timeout_seconds"] = _parse_float("ENGRAM_LLM_TIMEOUT_SECONDS", v)
        if v := _str_env("ENGRAM_CACHE_BACKEND"):
            overrides["cache_backend"] = _parse_cache_backend("ENGRAM_CACHE_BACKEND", v)
        if v := _str_env("ENGRAM_CACHE_PATH"):
            overrides["cache_path"] = Path(v)
        if v := _str_env("ENGRAM_REDIS_URL"):
            overrides["redis_url"] = v
        if v := _str_env("ENGRAM_REDIS_KEY_PREFIX"):
            overrides["redis_key_prefix"] = v
        if v := _str_env("ENGRAM_REDIS_TTL_SECONDS"):
            overrides["redis_ttl_seconds"] = _parse_int("ENGRAM_REDIS_TTL_SECONDS", v)
        if v := _str_env("ENGRAM_CACHE_SIGNING_KEY"):
            overrides["cache_signing_key"] = SecretStr(v)
        if v := _str_env("ENGRAM_MAX_RESOLUTION_RETRIES"):
            overrides["max_resolution_retries"] = _parse_int(
                "ENGRAM_MAX_RESOLUTION_RETRIES", v
            )
        if v := _str_env("ENGRAM_DOM_MAX_CHARS"):
            overrides["dom_max_chars"] = _parse_int("ENGRAM_DOM_MAX_CHARS", v)
        if v := _str_env("ENGRAM_MIN_CANDIDATE_CONFIDENCE"):
            overrides["min_candidate_confidence"] = _parse_float(
                "ENGRAM_MIN_CANDIDATE_CONFIDENCE", v
            )
        if v := _str_env("ENGRAM_LOG_LEVEL"):
            overrides["log_level"] = v
        if v := _str_env("ENGRAM_LOG_INTENTS_VERBATIM"):
            overrides["log_intents_verbatim"] = v.lower() in {"1", "true", "yes"}
        if v := _str_env("ENGRAM_HEADLESS"):
            overrides["headless"] = v.lower() not in {"0", "false", "no"}

        return cls(**overrides)


def _str_env(name: str) -> str | None:
    """Return the environment value for ``name`` stripped, or ``None`` if unset or empty.

    Whitespace-only values are also treated as empty — people accidentally leaving
    a trailing space on a ``.env`` line should get the same behaviour as omitting
    the line entirely.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def _parse_int(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise EngramConfigError(f"Environment variable {name!r} must be an integer, got {raw!r}") from exc


def _parse_float(name: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise EngramConfigError(f"Environment variable {name!r} must be a float, got {raw!r}") from exc


def _parse_cache_backend(name: str, raw: str) -> CacheBackend:
    # Returning the literal directly (rather than casting) lets mypy narrow
    # the result to ``CacheBackend`` without a runtime cast.
    if raw == "sqlite":
        return "sqlite"
    if raw == "redis":
        return "redis"
    raise EngramConfigError(
        f"Environment variable {name!r} must be 'sqlite' or 'redis', got {raw!r}"
    )
