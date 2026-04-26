"""Data models for selector resolutions and cache entries.

Engram has three primitive operations — ``act``, ``extract``, ``observe`` —
and each produces a different shape of cached resolution. To let the cache
layer remain a single table keyed by a single hash, the three resolution
types share a discriminator field (``kind``) and are unified through the
:data:`Resolution` discriminated union.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

SelectorType = Literal["css", "xpath", "role", "text"]
ActionType = Literal["click", "fill"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SelectorCandidate(BaseModel):
    """A single candidate selector returned by the LLM.

    Each candidate bundles both the targeting information (what element) and
    the action to perform on it (what to do), because natural-language intents
    like "click the join button" or "type 'Alice' into the name field" carry
    both pieces of information and the LLM is well-placed to decide both.
    """

    model_config = {"frozen": True}

    selector: str = Field(..., min_length=1, description="The selector string.")
    selector_type: SelectorType = Field(..., description="How to interpret the selector.")
    action: ActionType = Field(..., description="The action to perform on the element.")
    action_value: str | None = Field(
        default=None,
        description="Value for fill actions; must be None for click actions.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="LLM-reported confidence.")
    reasoning: str = Field(..., description="Brief LLM explanation for this candidate.")

    @field_validator("action_value")
    @classmethod
    def _validate_action_value(cls, v: str | None, info: object) -> str | None:
        # Pydantic v2 ValidationInfo carries sibling data on .data — kept loose to avoid
        # importing the private type; behaviour is asserted in tests.
        data = getattr(info, "data", {})
        action = data.get("action")
        if action == "fill" and (v is None or v == ""):
            raise ValueError("action_value is required when action is 'fill'")
        if action == "click" and v is not None:
            raise ValueError("action_value must be None when action is 'click'")
        return v


class ActionResolution(BaseModel):
    """A validated resolution of an intent against a specific DOM hash."""

    model_config = {"frozen": True}

    kind: Literal["action"] = "action"
    intent: str
    dom_hash: str
    candidates: list[SelectorCandidate]
    chosen_selector: str
    chosen_selector_type: SelectorType
    chosen_action: ActionType
    chosen_action_value: str | None
    created_at: datetime = Field(default_factory=_utcnow)


class ExtractionResolution(BaseModel):
    """A validated extraction result keyed against a (DOM hash, intent, schema) tuple.

    The actual extracted data is stored as a JSON string rather than a typed
    object because the user-supplied Pydantic schema lives in user code and is
    therefore unknown to the cache layer. On read, callers re-validate
    ``data_json`` against the schema they pass to :meth:`EngramPage.extract`;
    the :attr:`schema_fingerprint` participates in the cache key so a schema
    change yields a fresh cache miss rather than a stale type-mismatched hit.
    """

    model_config = {"frozen": True}

    kind: Literal["extraction"] = "extraction"
    intent: str
    dom_hash: str
    schema_fingerprint: str
    data_json: str
    created_at: datetime = Field(default_factory=_utcnow)


class ObservedAction(BaseModel):
    """A meaningful action available on the current page, as discovered by ``observe``.

    Unlike :class:`SelectorCandidate`, observed actions never carry a
    ``action_value``: ``observe`` is a discovery primitive and the user has
    not yet supplied any text to type. Surfacing a value here would be a
    hallucination.
    """

    model_config = {"frozen": True}

    description: str = Field(..., min_length=1, description="What this action does, in plain English.")
    selector: str = Field(..., min_length=1)
    selector_type: SelectorType
    action: ActionType
    confidence: float = Field(..., ge=0.0, le=1.0)


class ObservationResolution(BaseModel):
    """A validated list of available actions against a specific DOM hash."""

    model_config = {"frozen": True}

    kind: Literal["observation"] = "observation"
    dom_hash: str
    actions: list[ObservedAction]
    created_at: datetime = Field(default_factory=_utcnow)


Resolution = Annotated[
    ActionResolution | ExtractionResolution | ObservationResolution,
    Field(discriminator="kind"),
]


class CacheEntry(BaseModel):
    """A cache record wrapping any :data:`Resolution` with usage metadata."""

    key: str = Field(..., min_length=1)
    resolution: Resolution
    hit_count: int = Field(default=0, ge=0)
    last_used_at: datetime = Field(default_factory=_utcnow)
