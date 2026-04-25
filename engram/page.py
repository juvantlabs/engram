"""The user-facing ``EngramPage`` — an LLM-augmented wrapper around a Playwright page.

The ``act`` primitive is the whole vertical slice of v1. It implements the
canonical Engram loop:

    preprocess → hash → cache.get →
        hit  → validate stored selector → execute (or fall through on stale)
        miss → LLM → validate candidates → execute best → cache.put

Every branch is logged structurally so that cache hit-rate, LLM latency, and
resolution failures are observable without wrapping calls in ad-hoc timers.
"""

from __future__ import annotations

from datetime import datetime, timezone

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Locator, Page

from engram.cache.base import AbstractCache
from engram.dom.hasher import hash_dom
from engram.dom.preprocessor import preprocess_dom
from engram.llm.base import AbstractLLMClient
from engram.models.config import EngramConfig
from engram.models.errors import (
    EngramBrowserError,
    EngramResolutionError,
)
from engram.models.resolution import (
    ActionType,
    CacheEntry,
    SelectorCandidate,
    SelectorResolution,
    SelectorType,
)
from engram.observability import get_logger

_LOG = get_logger(__name__)


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

    async def act(self, intent: str) -> None:
        """Resolve ``intent`` against the current page and execute the resulting action.

        :raises EngramResolutionError: If no LLM-proposed candidate can be
            validated against the page.
        :raises EngramBrowserError: If the validated action fails at the
            Playwright layer (e.g. element detached between validation and click).
        """
        if not intent or not intent.strip():
            raise EngramResolutionError(intent=intent, attempts=0, last_error="empty intent")

        try:
            raw_html = await self._page.content()
        except PlaywrightError as exc:
            raise EngramBrowserError(f"Could not read page content: {exc}") from exc

        preprocessed = preprocess_dom(raw_html, max_chars=self._config.dom_max_chars)
        key = hash_dom(preprocessed, intent)

        cached = await self._cache.get(key)
        if cached is not None:
            locator = self._locator_for(
                cached.resolution.chosen_selector,
                cached.resolution.chosen_selector_type,
            )
            if await _validate_locator(locator):
                _LOG.info(
                    "cache.hit",
                    intent=intent,
                    key=key,
                    selector=cached.resolution.chosen_selector,
                    hit_count=cached.hit_count,
                )
                await self._execute(
                    locator,
                    cached.resolution.chosen_action,
                    cached.resolution.chosen_action_value,
                )
                return
            _LOG.info(
                "cache.invalidated",
                intent=intent,
                key=key,
                reason="cached selector no longer matches",
            )
            await self._cache.invalidate(key)

        _LOG.info("cache.miss", intent=intent, key=key)

        candidates = await self._llm.resolve_action(intent=intent, preprocessed_dom=preprocessed)
        # Defensive: enforce confidence ordering in case an impl forgets.
        candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)

        threshold = self._config.min_candidate_confidence
        filtered = [c for c in candidates if c.confidence >= threshold]
        if candidates and not filtered:
            # The LLM returned candidates but none cleared the threshold —
            # treat as "no good match" rather than try the most plausible-
            # looking guess anyway.
            top = candidates[0]
            _LOG.info(
                "selector.rejected_low_confidence",
                intent=intent,
                top_selector=top.selector,
                top_confidence=top.confidence,
                threshold=threshold,
            )
            raise EngramResolutionError(
                intent=intent,
                attempts=0,
                last_error=(
                    f"no candidate met min_candidate_confidence={threshold}; "
                    f"top candidate was {top.confidence:.2f}"
                ),
            )
        candidates = filtered

        last_error: str | None = None
        for candidate in candidates:
            locator = self._locator_for(candidate.selector, candidate.selector_type)
            if not await _validate_locator(locator):
                _LOG.info(
                    "selector.failed",
                    intent=intent,
                    selector=candidate.selector,
                    selector_type=candidate.selector_type,
                    reason="locator matched zero elements",
                )
                last_error = "locator matched zero elements"
                continue

            _LOG.info(
                "selector.validated",
                intent=intent,
                selector=candidate.selector,
                selector_type=candidate.selector_type,
                confidence=candidate.confidence,
            )

            resolution = SelectorResolution(
                intent=intent,
                dom_hash=key,
                candidates=list(candidates),
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

        raise EngramResolutionError(
            intent=intent,
            attempts=len(candidates),
            last_error=last_error,
        )

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


async def _validate_locator(locator: Locator) -> bool:
    """Return ``True`` iff ``locator`` matches at least one element right now."""
    try:
        count = await locator.count()
    except PlaywrightError:
        return False
    return count > 0


# Re-exported so test code can build SelectorCandidate without reaching deep into models.
__all__ = ["EngramPage", "SelectorCandidate"]
