"""Tests for the observability layer — primarily the PII-safe intent logger."""

from __future__ import annotations

import hashlib

from engram.observability import intent_fields


def test_intent_fields_redacts_by_default() -> None:
    fields = intent_fields("type my password p4ssw0rd", verbatim=False)
    assert "intent" not in fields
    assert fields["intent_redacted"] is True
    assert "intent_hash" in fields
    # Hash is a SHA-256 prefix — stable across runs and 12 hex chars long.
    expected = hashlib.sha256(b"type my password p4ssw0rd").hexdigest()[:12]
    assert fields["intent_hash"] == expected


def test_intent_fields_returns_raw_when_verbatim() -> None:
    fields = intent_fields("click join", verbatim=True)
    assert fields == {"intent": "click join"}
    assert "intent_hash" not in fields
    assert "intent_redacted" not in fields


def test_same_intent_produces_same_hash() -> None:
    """Hash must be stable across calls so events for one intent correlate in logs."""
    a = intent_fields("click join", verbatim=False)
    b = intent_fields("click join", verbatim=False)
    assert a["intent_hash"] == b["intent_hash"]


def test_different_intents_produce_different_hashes() -> None:
    a = intent_fields("click join", verbatim=False)
    b = intent_fields("click leave", verbatim=False)
    assert a["intent_hash"] != b["intent_hash"]
