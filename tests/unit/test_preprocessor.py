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


# ----------------------------------------------- sensitive form-field scrubbing


def test_strips_value_from_password_input() -> None:
    html = '<html><body><input type="password" value="hunter2"/></body></html>'
    out = preprocess_dom(html)
    assert "hunter2" not in out
    # The element itself should still be visible to the LLM (so it can fill it
    # via act); only the leaked value is scrubbed.
    assert 'type="password"' in out


def test_strips_value_from_hidden_input() -> None:
    """Hidden inputs commonly carry CSRF tokens, session ids, and OAuth state."""
    html = '<html><body><input type="hidden" name="csrf" value="abc123def"/></body></html>'
    out = preprocess_dom(html)
    assert "abc123def" not in out


def test_strips_value_when_name_matches_sensitive_pattern() -> None:
    """Legacy forms sometimes render credentials as ``type=text`` + ``name=password``."""
    html = (
        '<html><body><input type="text" name="password" value="hunter2"/></body></html>'
    )
    out = preprocess_dom(html)
    assert "hunter2" not in out


def test_strips_value_when_id_matches_sensitive_pattern() -> None:
    html = '<html><body><input type="text" id="api-key-input" value="sk-12345"/></body></html>'
    out = preprocess_dom(html)
    assert "sk-12345" not in out


def test_strips_value_when_autocomplete_signals_credential() -> None:
    html = (
        '<html><body><input type="text" autocomplete="current-password" '
        'value="leak"/></body></html>'
    )
    out = preprocess_dom(html)
    assert "leak" not in out


def test_strips_placeholder_from_sensitive_input() -> None:
    """Pre-filled placeholders (e.g. an example token) are also stripped."""
    html = (
        '<html><body><input type="password" placeholder="hunter2-example"/></body></html>'
    )
    out = preprocess_dom(html)
    assert "hunter2-example" not in out


def test_keeps_value_on_non_sensitive_input() -> None:
    """Search/email/text inputs without sensitive cues retain their values —
    the LLM legitimately needs them to decide what action to take."""
    html = (
        '<html><body>'
        '<input type="search" name="query" value="how to use engram"/>'
        '<input type="email" name="contact-email" value="alice@example.com"/>'
        '</body></html>'
    )
    out = preprocess_dom(html)
    assert "how to use engram" in out
    assert "alice@example.com" in out


def test_keeps_other_attributes_on_sensitive_input() -> None:
    """Stripping is targeted: id, name, type, aria-label etc. survive so the
    LLM can still target the field for ``act("type my password")``."""
    html = (
        '<html><body><input type="password" id="login-pass" name="password" '
        'aria-label="Password" value="hunter2"/></body></html>'
    )
    out = preprocess_dom(html)
    assert "hunter2" not in out
    assert 'id="login-pass"' in out
    assert 'name="password"' in out
    assert 'aria-label="Password"' in out


def test_credit_card_autocomplete_tokens_are_sensitive() -> None:
    html = (
        '<html><body>'
        '<input autocomplete="cc-number" value="4111111111111111"/>'
        '<input autocomplete="cc-csc" value="123"/>'
        '</body></html>'
    )
    out = preprocess_dom(html)
    assert "4111111111111111" not in out
    assert "123" not in out
