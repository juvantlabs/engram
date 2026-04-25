"""Reduce raw HTML to a compact, semantic representation suitable for an LLM.

The goals of this module are:

1. Deterministic output — the same input produces the same output byte-for-byte,
   so that the hasher downstream produces stable cache keys.
2. Aggressive reduction — scripts, styles, SVG, comments, tracking attributes,
   and hidden elements are stripped, because they carry no signal for element
   resolution and they dominate the byte count of real SPA pages.
3. Semantic preservation — attributes that users and screen readers rely on
   (``aria-*``, ``role``, ``id``, ``name``, ``placeholder``, ``type``, ``href``,
   ``alt``, ``title``, ``value``, ``data-testid`` and cousins) are kept verbatim.
"""

from __future__ import annotations

import re

from lxml import etree, html

from engram.models.errors import EngramDOMError

# Tags to drop entirely (including their subtrees).
_DROP_TAGS: frozenset[str] = frozenset(
    {"script", "style", "svg", "noscript", "canvas", "template", "iframe", "link", "meta"}
)

# Attributes kept on every element regardless of tag.
_SEMANTIC_ATTRS: frozenset[str] = frozenset(
    {
        "id",
        "class",
        "role",
        "type",
        "name",
        "placeholder",
        "href",
        "alt",
        "title",
        "value",
        "for",
        "disabled",
        "checked",
        "selected",
        "readonly",
        "required",
        "aria-label",
        "aria-labelledby",
        "aria-describedby",
        "aria-hidden",
        "aria-expanded",
        "aria-selected",
        "aria-checked",
        "aria-disabled",
        "aria-pressed",
        "aria-current",
        "aria-controls",
        "aria-live",
        "aria-atomic",
        "aria-haspopup",
    }
)

# Test-hook data-* attributes that are always kept.
_TEST_DATA_ATTRS: frozenset[str] = frozenset({"data-testid", "data-test", "data-cy", "data-qa"})

# Matches inline styles that hide the element.
_HIDDEN_STYLE_RE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\s*(?:;|$)",
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_dom(raw_html: str, max_chars: int = 50_000) -> str:
    """Return a reduced, deterministic HTML representation of ``raw_html``.

    :param raw_html: Source HTML — typically ``await page.content()``.
    :param max_chars: Hard cap on the output length. If exceeded, raises
        :class:`EngramDOMError` rather than silently truncating; silent
        truncation would produce unstable cache keys.
    :raises EngramDOMError: If parsing fails or the reduced output exceeds
        ``max_chars``.
    """
    if not raw_html or not raw_html.strip():
        raise EngramDOMError("Cannot preprocess empty HTML")

    try:
        root = html.fromstring(raw_html)
    except (etree.ParserError, etree.XMLSyntaxError, ValueError) as exc:
        raise EngramDOMError(f"Failed to parse HTML: {exc}") from exc

    _strip_comments(root)
    _drop_unwanted_tags(root)
    _drop_hidden(root)
    _prune_attributes(root)
    _normalize_text(root)

    serialized = html.tostring(root, encoding="unicode", method="html", pretty_print=False)
    collapsed = _WHITESPACE_RE.sub(" ", serialized).strip()

    if len(collapsed) > max_chars:
        raise EngramDOMError(
            f"Preprocessed DOM is {len(collapsed)} chars, exceeds configured max {max_chars}. "
            "Consider scoping the operation to a narrower subtree or raising dom_max_chars."
        )

    return collapsed


def _strip_comments(root: etree._Element) -> None:
    for comment in root.xpath("//comment()"):
        parent = comment.getparent()
        if parent is not None:
            parent.remove(comment)


def _drop_unwanted_tags(root: etree._Element) -> None:
    for tag in _DROP_TAGS:
        for el in root.xpath(f"//{tag}"):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def _drop_hidden(root: etree._Element) -> None:
    for el in list(root.iter()):
        if not isinstance(el.tag, str):
            # Skip ProcessingInstruction / Comment nodes.
            continue
        if el.get("hidden") is not None:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
                continue
        if el.get("aria-hidden", "").lower() == "true":
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
                continue
        style = el.get("style") or ""
        if _HIDDEN_STYLE_RE.search(style):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def _prune_attributes(root: etree._Element) -> None:
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for attr in list(el.attrib.keys()):
            if attr in _SEMANTIC_ATTRS:
                continue
            if attr in _TEST_DATA_ATTRS:
                continue
            del el.attrib[attr]


def _normalize_text(root: etree._Element) -> None:
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el.text:
            el.text = _WHITESPACE_RE.sub(" ", el.text).strip() or None
        if el.tail:
            el.tail = _WHITESPACE_RE.sub(" ", el.tail).strip() or None
