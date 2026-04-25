"""Tests for the DOM/intent hasher."""

from __future__ import annotations

from engram.dom.hasher import hash_dom, normalize_intent


def test_same_inputs_same_hash() -> None:
    dom = "<button>Join</button>"
    assert hash_dom(dom, "click join") == hash_dom(dom, "click join")


def test_intent_is_case_insensitive() -> None:
    dom = "<button>Join</button>"
    assert hash_dom(dom, "Click Join") == hash_dom(dom, "click join")


def test_intent_whitespace_collapses() -> None:
    dom = "<button>Join</button>"
    assert hash_dom(dom, "click   join") == hash_dom(dom, "click join")
    assert hash_dom(dom, "  click join  ") == hash_dom(dom, "click join")


def test_intent_change_changes_hash() -> None:
    dom = "<button>Join</button>"
    assert hash_dom(dom, "click join") != hash_dom(dom, "click leave")


def test_dom_change_changes_hash() -> None:
    intent = "click join"
    assert hash_dom("<button>Join</button>", intent) != hash_dom("<button>Leave</button>", intent)


def test_returns_hex_sha256_length() -> None:
    h = hash_dom("<p/>", "hi")
    assert len(h) == 64
    int(h, 16)  # must be valid hex


def test_normalize_intent_preserves_inner_words() -> None:
    assert normalize_intent("  CLICK  the\tjoin  button  ") == "click the join button"
