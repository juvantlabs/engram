"""Tests for the DOM/intent hasher."""

from __future__ import annotations

from engram.dom.hasher import hash_action, normalize_intent


def test_same_inputs_same_hash() -> None:
    dom = "<button>Join</button>"
    assert hash_action(dom, "click join") == hash_action(dom, "click join")


def test_intent_is_case_insensitive() -> None:
    dom = "<button>Join</button>"
    assert hash_action(dom, "Click Join") == hash_action(dom, "click join")


def test_intent_whitespace_collapses() -> None:
    dom = "<button>Join</button>"
    assert hash_action(dom, "click   join") == hash_action(dom, "click join")
    assert hash_action(dom, "  click join  ") == hash_action(dom, "click join")


def test_intent_change_changes_hash() -> None:
    dom = "<button>Join</button>"
    assert hash_action(dom, "click join") != hash_action(dom, "click leave")


def test_dom_change_changes_hash() -> None:
    intent = "click join"
    assert hash_action("<button>Join</button>", intent) != hash_action("<button>Leave</button>", intent)


def test_returns_hex_sha256_length() -> None:
    h = hash_action("<p/>", "hi")
    assert len(h) == 64
    int(h, 16)  # must be valid hex


def test_normalize_intent_preserves_inner_words() -> None:
    assert normalize_intent("  CLICK  the\tjoin  button  ") == "click the join button"
