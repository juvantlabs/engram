"""Stable hashing of Engram primitive requests for cache lookups.

Each primitive (``act`` / ``extract`` / ``observe``) has its own hash
function so that cache keys can never collide *across* primitives even when
the DOM and intent happen to coincide. The primitive name is folded into
the hashed payload as a leading namespace tag.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_intent(intent: str) -> str:
    """Return a canonical form of ``intent`` for hashing.

    Equivalent intents — differing only in letter case or whitespace — must
    produce the same hash so that ``act("Click Join")`` and ``act("click  join")``
    are treated as the same cache key.
    """
    return _WHITESPACE_RE.sub(" ", intent.strip().lower())


def hash_action(preprocessed_dom: str, intent: str) -> str:
    """Return a hex-encoded SHA-256 of the (action namespace, intent, DOM) tuple.

    Used by ``act``. The leading ``"action\\n"`` tag namespaces the hash so a
    cache key collision across primitives is impossible even if the same DOM
    + intent text were to ever round-trip through the ``extract`` or
    ``observe`` paths (which take additional inputs anyway, but we don't want
    to rely on that for safety).
    """
    normalized = normalize_intent(intent)
    payload = f"action\n{normalized}\n{preprocessed_dom}".encode()
    return hashlib.sha256(payload).hexdigest()


def hash_extraction(
    preprocessed_dom: str,
    intent: str,
    schema_fingerprint: str,
) -> str:
    """Hash for ``extract`` calls.

    The schema fingerprint participates in the key so that the same intent
    against the same DOM with two *different* Pydantic schemas produces two
    *different* cache entries — otherwise a schema change would silently
    return data shaped for the previous schema.
    """
    normalized = normalize_intent(intent)
    payload = (
        f"extract\n{normalized}\n{schema_fingerprint}\n{preprocessed_dom}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def hash_observation(preprocessed_dom: str) -> str:
    """Hash for ``observe`` calls. ``observe`` takes no intent."""
    payload = f"observe\n{preprocessed_dom}".encode()
    return hashlib.sha256(payload).hexdigest()


def fingerprint_schema(json_schema: dict[str, Any]) -> str:
    """Return a short, stable fingerprint of a Pydantic-emitted JSON Schema.

    Two schemas that differ only in key ordering must produce the same
    fingerprint (``sort_keys=True``); two schemas that differ in any
    structural way must produce different fingerprints. We truncate to 16
    hex chars because the fingerprint is mixed into a separate SHA-256 cache
    key downstream — full 256-bit collision resistance there is plenty.
    """
    canonical = json.dumps(json_schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
