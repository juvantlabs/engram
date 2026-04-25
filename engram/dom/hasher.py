"""Stable hashing of (intent, preprocessed DOM) pairs for cache lookups."""

from __future__ import annotations

import hashlib
import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_intent(intent: str) -> str:
    """Return a canonical form of ``intent`` for hashing.

    Equivalent intents — differing only in letter case or whitespace — must
    produce the same hash so that ``act("Click Join")`` and ``act("click  join")``
    are treated as the same cache key.
    """
    return _WHITESPACE_RE.sub(" ", intent.strip().lower())


def hash_dom(preprocessed_dom: str, intent: str) -> str:
    """Return a hex-encoded SHA-256 of the (normalized intent, DOM) pair."""
    normalized = normalize_intent(intent)
    payload = f"{normalized}\n{preprocessed_dom}".encode()
    return hashlib.sha256(payload).hexdigest()
