"""Tests for DOM preprocessing."""

from __future__ import annotations

import pytest

from engram.dom.preprocessor import preprocess_dom
from engram.models.errors import EngramDOMError


def test_strips_scripts() -> None:
    html = "<html><body><script>alert(1)</script><p>hello</p></body></html>"
    out = preprocess_dom(html)
    assert "<script" not in out
    assert "alert(1)" not in out
    assert "hello" in out


def test_strips_styles() -> None:
    html = "<html><body><style>.x{color:red}</style><p>hi</p></body></html>"
    out = preprocess_dom(html)
    assert "<style" not in out
    assert "color:red" not in out


def test_strips_svg() -> None:
    html = "<html><body><svg><circle/></svg><p>hi</p></body></html>"
    out = preprocess_dom(html)
    assert "<svg" not in out
    assert "<circle" not in out


def test_strips_comments() -> None:
    html = "<html><body><!-- secret --><p>hi</p></body></html>"
    out = preprocess_dom(html)
    assert "secret" not in out


def test_preserves_aria_label() -> None:
    html = '<html><body><button aria-label="Join Session">x</button></body></html>'
    out = preprocess_dom(html)
    assert 'aria-label="Join Session"' in out


def test_preserves_id_and_class() -> None:
    html = '<html><body><div id="root" class="container">x</div></body></html>'
    out = preprocess_dom(html)
    assert 'id="root"' in out
    assert 'class="container"' in out


def test_preserves_data_testid() -> None:
    html = '<html><body><button data-testid="join-btn">x</button></body></html>'
    out = preprocess_dom(html)
    assert 'data-testid="join-btn"' in out


def test_strips_non_semantic_data_attrs() -> None:
    html = '<html><body><div data-analytics="track" data-testid="keep">x</div></body></html>'
    out = preprocess_dom(html)
    assert "data-analytics" not in out
    assert 'data-testid="keep"' in out


def test_strips_onclick_and_other_event_handlers() -> None:
    html = '<html><body><button onclick="doThing()">click</button></body></html>'
    out = preprocess_dom(html)
    assert "onclick" not in out


def test_removes_hidden_attribute() -> None:
    html = "<html><body><p hidden>gone</p><p>kept</p></body></html>"
    out = preprocess_dom(html)
    assert "gone" not in out
    assert "kept" in out


def test_removes_aria_hidden_true() -> None:
    html = '<html><body><p aria-hidden="true">gone</p><p>kept</p></body></html>'
    out = preprocess_dom(html)
    assert "gone" not in out
    assert "kept" in out


def test_removes_display_none() -> None:
    html = '<html><body><p style="display: none">gone</p><p>kept</p></body></html>'
    out = preprocess_dom(html)
    assert "gone" not in out
    assert "kept" in out


def test_raises_on_empty_html() -> None:
    with pytest.raises(EngramDOMError):
        preprocess_dom("")
    with pytest.raises(EngramDOMError):
        preprocess_dom("   ")


def test_raises_when_exceeds_max_chars() -> None:
    html = "<html><body>" + ("<p>" + "x" * 100 + "</p>") * 10 + "</body></html>"
    with pytest.raises(EngramDOMError, match="exceeds configured max"):
        preprocess_dom(html, max_chars=50)


def test_deterministic_output() -> None:
    html = '<html><body><button id="a" aria-label="Join">Click</button></body></html>'
    assert preprocess_dom(html) == preprocess_dom(html)
