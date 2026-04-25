"""Engram runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from engram.models.errors import EngramConfigError

CacheBackend = Literal["sqlite"]


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

    cache_backend: CacheBackend = Field(default="sqlite")
    cache_path: Path = Field(default=Path(".engram/cache.db"))

    max_llm_retries: int = Field(default=2, ge=0, le=10)
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
        - ``ENGRAM_CACHE_PATH``
        - ``ENGRAM_MAX_LLM_RETRIES``
        - ``ENGRAM_DOM_MAX_CHARS``
        - ``ENGRAM_MIN_CANDIDATE_CONFIDENCE``
        - ``ENGRAM_LOG_LEVEL``
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
        if v := _str_env("ENGRAM_CACHE_PATH"):
            overrides["cache_path"] = Path(v)
        if v := _str_env("ENGRAM_MAX_LLM_RETRIES"):
            overrides["max_llm_retries"] = _parse_int("ENGRAM_MAX_LLM_RETRIES", v)
        if v := _str_env("ENGRAM_DOM_MAX_CHARS"):
            overrides["dom_max_chars"] = _parse_int("ENGRAM_DOM_MAX_CHARS", v)
        if v := _str_env("ENGRAM_MIN_CANDIDATE_CONFIDENCE"):
            overrides["min_candidate_confidence"] = _parse_float(
                "ENGRAM_MIN_CANDIDATE_CONFIDENCE", v
            )
        if v := _str_env("ENGRAM_LOG_LEVEL"):
            overrides["log_level"] = v
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
