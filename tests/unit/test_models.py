"""Tests for model invariants that aren't covered via higher-level tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from engram.models.config import EngramConfig
from engram.models.errors import EngramConfigError
from engram.models.resolution import SelectorCandidate


def test_fill_candidate_requires_action_value() -> None:
    with pytest.raises(ValidationError):
        SelectorCandidate(
            selector="#name",
            selector_type="css",
            action="fill",
            action_value=None,
            confidence=0.8,
            reasoning="name input",
        )


def test_click_candidate_rejects_action_value() -> None:
    with pytest.raises(ValidationError):
        SelectorCandidate(
            selector="#join",
            selector_type="css",
            action="click",
            action_value="oops",
            confidence=0.8,
            reasoning="join button",
        )


def test_confidence_bounds() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            SelectorCandidate(
                selector="#x",
                selector_type="css",
                action="click",
                action_value=None,
                confidence=bad,
                reasoning="r",
            )


def test_config_defaults_are_sensible() -> None:
    c = EngramConfig()
    assert c.cache_backend == "sqlite"
    assert c.max_llm_retries >= 0
    assert c.dom_max_chars > 0


def test_config_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGRAM_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("ENGRAM_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("ENGRAM_MAX_LLM_RETRIES", "5")
    monkeypatch.setenv("ENGRAM_CACHE_PATH", "/tmp/engram-test/cache.db")
    monkeypatch.setenv("ENGRAM_HEADLESS", "false")

    c = EngramConfig.from_env()
    assert c.llm_model == "openai/gpt-4o-mini"
    assert c.llm_api_key is not None
    assert c.llm_api_key.get_secret_value() == "sk-test"
    assert c.max_llm_retries == 5
    assert c.cache_path == Path("/tmp/engram-test/cache.db")
    assert c.headless is False


def test_config_from_env_rejects_bad_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGRAM_MAX_LLM_RETRIES", "not-a-number")
    with pytest.raises(EngramConfigError):
        EngramConfig.from_env()


def test_config_from_env_treats_empty_api_key_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # A committed .env template with `ENGRAM_LLM_API_KEY=` must not clobber the
    # field — otherwise LiteLLM's provider-level env fallback breaks.
    monkeypatch.setenv("ENGRAM_LLM_API_KEY", "")
    c = EngramConfig.from_env()
    assert c.llm_api_key is None


def test_config_from_env_reads_azure_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGRAM_LLM_MODEL", "azure/gpt-4o")
    monkeypatch.setenv("ENGRAM_LLM_API_KEY", "azure-secret")
    monkeypatch.setenv("ENGRAM_LLM_API_BASE", "https://example.openai.azure.com/")
    monkeypatch.setenv("ENGRAM_LLM_API_VERSION", "2024-02-15-preview")
    c = EngramConfig.from_env()
    assert c.llm_model == "azure/gpt-4o"
    assert c.llm_api_base == "https://example.openai.azure.com/"
    assert c.llm_api_version == "2024-02-15-preview"


def test_config_from_env_ignores_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("ENGRAM_"):
            monkeypatch.delenv(key, raising=False)
    # Should not raise and should return defaults.
    c = EngramConfig.from_env()
    assert c.llm_model.startswith("anthropic/") or c.llm_model.startswith("openai/")
