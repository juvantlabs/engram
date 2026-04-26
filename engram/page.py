"""The user-facing ``EngramPage`` — an LLM-augmented wrapper around a Playwright page.

Three primitives, one cache loop. ``act`` resolves an intent to a browser
action, ``extract`` resolves an intent to typed data, ``observe`` enumerates
the meaningful actions on the page. All three follow the canonical Engram
loop:

    preprocess → hash → cache.get →
        hit  → validate stored result is still usable → return (or fall through on stale)
        miss → LLM → validate → store → return

Every branch is logged structurally so cache hit-rate, LLM latency, and
resolution failures are observable without wrapping calls in ad-hoc timers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TypeVar

import structlog.contextvars
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Locator, Page
from pydantic import BaseModel, ValidationError

from engram.cache.base import AbstractCache
from engram.dom.hasher import (
    fingerprint_schema,
    hash_action,
    hash_extraction,
    hash_observation,
)
from engram.dom.preprocessor import preprocess_dom
from engram.llm.base import AbstractLLMClient
from engram.models.config import EngramConfig
from engram.models.errors import (
    EngramBrowserError,
    EngramExtractionError,
    EngramResolutionError,
)
from engram.models.resolution import (
    ActionResolution,
    ActionType,
    CacheEntry,
    ExtractionResolution,
    ObservationResolution,
    ObservedAction,
    SelectorCandidate,
    SelectorType,
)
from engram.observability import get_logger, intent_fields

_LOG = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class EngramPage:
    """An Engram-augmented Playwright page.

    Instances are created by :class:`EngramBrowser`; do not construct directly
    unless you are wiring up custom Playwright plumbing in tests.
    """

    def __init__(
        self,
        page: Page,
        config: EngramConfig,
        cache: AbstractCache,
        llm: AbstractLLMClient,
    ) -> None:
        self._page = page
        self._config = config
        self._cache = cache
        self._llm = llm

    @property
    def playwright_page(self) -> Page:
        """Escape hatch: the underlying Playwright ``Page`` for advanced flows."""
        return self._page

    async def goto(self, url: str, *, wait_until: str = "load") -> None:
        """Navigate to ``url``. Thin pass-through to Playwright with typed errors."""
        try:
            await self._page.goto(url, wait_until=wait_until)  # type: ignore[arg-type]
        except PlaywrightError as exc:
            raise EngramBrowserError(f"Navigation to {url!r} failed: {exc}") from exc

    # ------------------------------------------------------------------ act

    async def act(self, intent: str) -> None:
        """Resolve ``intent`` against the current page and execute the resulting action.

        :raises EngramResolutionError: If no LLM-proposed candidate can be
            validated against the page.
        :raises EngramBrowserError: If the validated action fails at the
            Playwright layer (e.g. element detached between validation and click).

        Every log event emitted while this method runs is tagged with a unique
        ``request_id`` (via structlog contextvars) so downstream cache/LLM
        events can be traced back to the originating call. Two parallel
        ``act`` invocations get independent request_ids because asyncio gives
        each task its own contextvars copy.
        """
        if not intent or not intent.strip():
            raise EngramResolutionError(intent=intent, attempts=0, last_error="empty intent")

        # ``bound_contextvars`` is sync, but it works inside an async function
        # because contextvars are simply stored on the current task. The
        # ``with`` exits cleanly even on exception, so partial-failure paths
        # don't leak bindings into a subsequent ``act`` on the same task.
        # Body lives in :meth:`_act_resolve` so the contextvars wrapper stays
        # a tight three-line shim — re-indenting 100+ lines for one extra
        # block scope would be more noise than signal.
        with structlog.contextvars.bound_contextvars(request_id=_new_request_id()):
            await self._act_resolve(intent)

    async def _act_resolve(self, intent: str) -> None:
        preprocessed = await self._preprocess()
        key = hash_action(preprocessed, intent)

        cached = await self._cache.get(key)
        if cached is not None and isinstance(cached.resolution, ActionResolution):
            resolution = cached.resolution
            locator = self._locator_for(
                resolution.chosen_selector,
                resolution.chosen_selector_type,
            )
            if await _validate_locator(locator):
                _LOG.info(
                    "cache.hit",
                    primitive="act",
                    **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
                    key=key,
                    selector=resolution.chosen_selector,
                    hit_count=cached.hit_count,
                )
                await self._execute(
                    locator,
                    resolution.chosen_action,
                    resolution.chosen_action_value,
                )
                return
            _LOG.info(
                "cache.invalidated",
                primitive="act",
                **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
                key=key,
                reason="cached selector no longer matches",
            )
            await self._cache.invalidate(key)

        _LOG.info(
            "cache.miss",
            primitive="act",
            **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
            key=key,
        )

        # Retry loop: a "resolution retry" means re-prompting the LLM with the
        # set of failures from the previous attempt(s) so it can correct itself
        # rather than re-emit the same broken candidates. ``prior_failures`` is
        # built up across iterations and threaded into each LLM call.
        prior_failures: list[str] = []
        last_error: str | None = None
        candidates_attempted = 0
        threshold = self._config.min_candidate_confidence

        for attempt in range(self._config.max_resolution_retries + 1):
            candidates = await self._llm.resolve_action(
                intent=intent,
                preprocessed_dom=preprocessed,
                prior_failures=prior_failures or None,
            )
            # Defensive: enforce confidence ordering in case an impl forgets.
            candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)

            filtered = [c for c in candidates if c.confidence >= threshold]
            if candidates and not filtered:
                # The LLM returned candidates but none cleared the threshold —
                # treat as "no good match" and re-prompt rather than guess.
                top = candidates[0]
                msg = (
                    f"no candidate met min_candidate_confidence={threshold}; "
                    f"top candidate `{top.selector}` had confidence {top.confidence:.2f}"
                )
                _LOG.info(
                    "selector.rejected_low_confidence",
                    **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
                    attempt=attempt + 1,
                    top_selector=top.selector,
                    top_confidence=top.confidence,
                    threshold=threshold,
                )
                prior_failures.append(msg)
                last_error = msg
                continue

            for candidate in filtered:
                candidates_attempted += 1
                locator = self._locator_for(candidate.selector, candidate.selector_type)
                if not await _validate_locator(locator):
                    msg = (
                        f"selector `{candidate.selector}` "
                        f"({candidate.selector_type}) matched 0 elements on the page"
                    )
                    _LOG.info(
                        "selector.failed",
                        **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
                        attempt=attempt + 1,
                        selector=candidate.selector,
                        selector_type=candidate.selector_type,
                        reason="locator matched zero elements",
                    )
                    prior_failures.append(msg)
                    last_error = msg
                    continue

                _LOG.info(
                    "selector.validated",
                    **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
                    attempt=attempt + 1,
                    selector=candidate.selector,
                    selector_type=candidate.selector_type,
                    confidence=candidate.confidence,
                )

                resolution = ActionResolution(
                    intent=intent,
                    dom_hash=key,
                    candidates=list(filtered),
                    chosen_selector=candidate.selector,
                    chosen_selector_type=candidate.selector_type,
                    chosen_action=candidate.action,
                    chosen_action_value=candidate.action_value,
                )
                entry = CacheEntry(
                    key=key,
                    resolution=resolution,
                    hit_count=0,
                    last_used_at=datetime.now(timezone.utc),
                )

                # Execute first, cache second. A selector can pass validation
                # (>=1 match) yet still fail at click/fill (element not actionable,
                # overlapped, mid-animation). Caching before execute would poison
                # the cache with a "valid-but-unclickable" selector and every
                # future run would pay the action-timeout cost to rediscover it's
                # broken.
                await self._execute(locator, candidate.action, candidate.action_value)
                await self._cache.put(entry)
                return
            # Every candidate from this attempt failed validation. The for-else
            # would be more idiomatic here, but ``continue``-from-inner is
            # already needed for the per-candidate skip, so a plain fall-through
            # is clearer than mixing patterns.

        raise EngramResolutionError(
            intent=intent,
            attempts=candidates_attempted,
            last_error=last_error,
        )

    # -------------------------------------------------------------- extract

    async def extract(self, intent: str, schema: type[T]) -> T:
        """Extract data described by ``intent`` from the current page, typed as ``schema``.

        The cache key includes a fingerprint of ``schema``'s JSON Schema, so
        the same intent against the same DOM with two different schemas
        produces two different cache entries.

        :raises EngramExtractionError: If the LLM cannot produce data
            conforming to ``schema``, or if a cached entry's data fails
            validation against the (possibly-evolved) schema.
        :raises EngramLLMError: On provider errors or malformed LLM output.
        """
        if not intent or not intent.strip():
            raise EngramExtractionError(intent=intent, reason="empty intent")

        with structlog.contextvars.bound_contextvars(request_id=_new_request_id()):
            return await self._extract_resolve(intent, schema)

    async def _extract_resolve(self, intent: str, schema: type[T]) -> T:
        preprocessed = await self._preprocess()
        json_schema = schema.model_json_schema()
        fingerprint = fingerprint_schema(json_schema)
        key = hash_extraction(preprocessed, intent, fingerprint)

        cached = await self._cache.get(key)
        if cached is not None and isinstance(cached.resolution, ExtractionResolution):
            try:
                instance = schema.model_validate_json(cached.resolution.data_json)
            except ValidationError as exc:
                # Fingerprint is supposed to prevent this, but defend against:
                # - schema evolutions that produce the same fingerprint hash
                #   (vanishingly unlikely but not impossible)
                # - hand-edited cache rows
                # In either case, invalidate and re-extract rather than crash.
                _LOG.info(
                    "cache.invalidated",
                    primitive="extract",
                    **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
                    key=key,
                    reason=f"cached data no longer matches schema: {exc.error_count()} errors",
                )
                await self._cache.invalidate(key)
            else:
                _LOG.info(
                    "cache.hit",
                    primitive="extract",
                    **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
                    key=key,
                    schema=schema.__name__,
                    hit_count=cached.hit_count,
                )
                return instance

        _LOG.info(
            "cache.miss",
            primitive="extract",
            **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
            key=key,
        )

        # Retry loop: when LLM output fails Pydantic validation we re-prompt
        # with a summary of what was wrong so the LLM can correct itself
        # rather than re-emit the same broken shape.
        prior_failures: list[str] = []
        last_error: str | None = None
        # ``fresh`` is shadowed across the retry loop; using a distinct name
        # from ``instance`` (which is bound on the cache-hit path above) keeps
        # mypy from treating these as the same variable across the function.
        fresh: T | None = None

        for attempt in range(self._config.max_resolution_retries + 1):
            raw = await self._llm.resolve_extraction(
                intent=intent,
                preprocessed_dom=preprocessed,
                schema=json_schema,
                prior_failures=prior_failures or None,
            )
            try:
                fresh = schema.model_validate(raw)
            except ValidationError as exc:
                msg = _summarize_validation_error(exc)
                _LOG.info(
                    "extraction.retry",
                    **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
                    attempt=attempt + 1,
                    reason=msg,
                )
                prior_failures.append(msg)
                last_error = msg
                continue
            break

        if fresh is None:
            raise EngramExtractionError(
                intent=intent,
                reason=(
                    f"LLM output failed schema validation after "
                    f"{self._config.max_resolution_retries + 1} attempt(s): {last_error}"
                ),
            )

        resolution = ExtractionResolution(
            intent=intent,
            dom_hash=key,
            schema_fingerprint=fingerprint,
            data_json=fresh.model_dump_json(),
        )
        entry = CacheEntry(
            key=key,
            resolution=resolution,
            hit_count=0,
            last_used_at=datetime.now(timezone.utc),
        )
        await self._cache.put(entry)
        _LOG.info(
            "extraction.completed",
            **intent_fields(intent, verbatim=self._config.log_intents_verbatim),
            schema=schema.__name__,
        )
        return fresh

    # -------------------------------------------------------------- observe

    async def observe(self) -> list[ObservedAction]:
        """Return a ranked list of meaningful actions available on the current page.

        Hallucinated selectors (those the LLM proposes but which match zero
        elements on the page) are filtered out before the result is cached,
        so subsequent observations on the same DOM hash do not re-pay
        validation cost.
        """
        with structlog.contextvars.bound_contextvars(request_id=_new_request_id()):
            return await self._observe_resolve()

    async def _observe_resolve(self) -> list[ObservedAction]:
        preprocessed = await self._preprocess()
        key = hash_observation(preprocessed)

        cached = await self._cache.get(key)
        if cached is not None and isinstance(cached.resolution, ObservationResolution):
            _LOG.info(
                "cache.hit",
                primitive="observe",
                key=key,
                action_count=len(cached.resolution.actions),
                hit_count=cached.hit_count,
            )
            return list(cached.resolution.actions)

        _LOG.info("cache.miss", primitive="observe", key=key)

        proposed = await self._llm.resolve_observation(preprocessed_dom=preprocessed)

        validated: list[ObservedAction] = []
        for action in proposed:
            locator = self._locator_for(action.selector, action.selector_type)
            if await _validate_locator(locator):
                validated.append(action)
            else:
                _LOG.info(
                    "observation.dropped_hallucination",
                    selector=action.selector,
                    selector_type=action.selector_type,
                    description=action.description,
                )

        validated.sort(key=lambda a: a.confidence, reverse=True)

        resolution = ObservationResolution(dom_hash=key, actions=validated)
        entry = CacheEntry(
            key=key,
            resolution=resolution,
            hit_count=0,
            last_used_at=datetime.now(timezone.utc),
        )
        await self._cache.put(entry)
        _LOG.info(
            "observation.completed",
            proposed=len(proposed),
            validated=len(validated),
        )
        return validated

    # ----------------------------------------------------------- internals

    async def _preprocess(self) -> str:
        try:
            raw_html = await self._page.content()
        except PlaywrightError as exc:
            raise EngramBrowserError(f"Could not read page content: {exc}") from exc
        return preprocess_dom(raw_html, max_chars=self._config.dom_max_chars)

    def _locator_for(self, selector: str, selector_type: SelectorType) -> Locator:
        """Return a Playwright ``Locator`` for the given selector + type.

        Playwright's locator engine accepts explicit engine prefixes (``css=``,
        ``xpath=``, ``text=``, ``role=``); we always use them so the behaviour
        is independent of Playwright's auto-detection heuristics, which can
        change between versions.
        """
        if selector_type == "css":
            return self._page.locator(f"css={selector}")
        if selector_type == "xpath":
            return self._page.locator(f"xpath={selector}")
        if selector_type == "text":
            return self._page.locator(f"text={selector}")
        if selector_type == "role":
            return self._page.locator(f"role={selector}")
        # The Literal type narrows this out, but a runtime guard beats a
        # silent bug if a caller manages to smuggle in a bad string.
        raise EngramBrowserError(f"Unknown selector_type: {selector_type!r}")

    async def _execute(
        self,
        locator: Locator,
        action: ActionType,
        action_value: str | None,
    ) -> None:
        target = locator.first
        try:
            if action == "click":
                await target.click()
            elif action == "fill":
                # Pydantic validation on SelectorCandidate guarantees action_value
                # is set whenever action == "fill"; the assertion here is a
                # belt-and-braces defence for consumers that skip the model.
                assert action_value is not None, "fill action requires action_value"
                await target.fill(action_value)
            else:
                raise EngramBrowserError(f"Unknown action: {action!r}")
        except PlaywrightError as exc:
            raise EngramBrowserError(f"Action {action!r} failed: {exc}") from exc

        _LOG.info(
            "action.executed",
            action=action,
            has_value=action_value is not None,
        )


def _new_request_id() -> str:
    """Return a short, opaque request ID suitable for log correlation.

    12 hex chars from a uuid4 — collision probability across realistic
    session sizes is vanishingly small, and the short form keeps log
    lines compact.
    """
    return f"req_{uuid.uuid4().hex[:12]}"


async def _validate_locator(locator: Locator) -> bool:
    """Return ``True`` iff ``locator`` matches at least one element right now."""
    try:
        count = await locator.count()
    except PlaywrightError:
        return False
    return count > 0


def _summarize_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic ValidationError as a one-line, prompt-friendly summary.

    We cap at the first few errors so the feedback string we feed back into
    the next LLM call stays short — long traces would just dilute the signal
    and waste tokens.
    """
    errors = exc.errors()[:5]
    parts: list[str] = []
    for err in errors:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    summary = "; ".join(parts)
    total = exc.error_count()
    if total > len(errors):
        summary += f" (+{total - len(errors)} more)"
    return f"output failed schema validation: {summary}"


# Re-exported so test code can build SelectorCandidate without reaching deep into models.
__all__ = ["EngramPage", "ObservedAction", "SelectorCandidate"]
