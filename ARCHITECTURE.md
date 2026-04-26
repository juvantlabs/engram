# Architecture

This document describes how Engram is built — the module layout, the cache
loop, the data model, and the design decisions that aren't obvious from
reading any one file. It's aimed at contributors and at anyone deciding
whether to depend on Engram for a non-trivial use case.

The user-facing API and quickstart live in [README.md](README.md). The
threat model lives in [SECURITY.md](SECURITY.md). This document is the
internals.

## Goals

Engram lets you write deterministic browser-automation flows in Python
and delegate *element resolution* to an LLM, while caching those
resolutions so the LLM is consulted **once per UI version, not once per
run**. The split is surgical:

- **You own the flow.** Which page to load, which action comes next,
  what data to extract — all of that is regular Python.
- **The LLM owns element resolution.** "The join button" → a stable
  selector, derived once and cached.
- **The cache makes it cheap.** A 1,000-run automation does not pay
  1,000 LLM calls; it pays as many as there are *distinct* DOM versions
  encountered.

## Non-goals

- A fully autonomous agent that decides every step.
- A general-purpose web scraper.
- Anything that requires the user to give up control of the flow.
- A cloud browser platform.

## Module layout

```
engram/
  __init__.py          # public API surface
  page.py              # EngramPage — orchestrates the three primitives
  browser/
    context.py         # EngramBrowser — Playwright lifecycle + DI wiring
  cache/
    base.py            # AbstractCache Protocol
    sqlite.py          # local file-backed implementation
    redis.py           # shared / multi-machine implementation (opt-in HMAC signing)
    __init__.py        # build_cache(config) factory
  llm/
    base.py            # AbstractLLMClient Protocol
    litellm_client.py  # provider-agnostic implementation via LiteLLM
    budgeted.py        # decorator enforcing max_llm_calls_per_session
  dom/
    preprocessor.py    # raw HTML → reduced, deterministic snapshot
    hasher.py          # cache-key construction (per-primitive namespaces)
  models/
    config.py          # EngramConfig (Pydantic, env-loadable)
    resolution.py      # discriminated union of cached resolutions
    errors.py          # exception hierarchy (EngramError + subclasses)
  observability.py     # structlog config + intent_fields PII-safe logger
```

Every component is behind an interface (`AbstractCache`,
`AbstractLLMClient`). Tests mock the interfaces; real deployments wire
real implementations.

## The three primitives

`EngramPage` exposes exactly three methods:

| Method | Returns | Cache key derived from |
|---|---|---|
| `act(intent)` | `None` (executes the action) | `sha256("action\n" + normalized_intent + preprocessed_dom)` |
| `extract(intent, schema)` | `schema` instance | `sha256("extract\n" + normalized_intent + schema_fingerprint + preprocessed_dom)` |
| `observe()` | `list[ObservedAction]` | `sha256("observe\n" + preprocessed_dom)` |

Each cache key is namespaced by a leading primitive tag, so a cache key
collision across primitives is impossible even if the same DOM and
intent text were to appear in two contexts.

The `extract` key folds in a 16-char fingerprint of the user-supplied
Pydantic schema's JSON Schema, sorted-key serialised. Changing the
schema produces a fresh cache miss rather than a stale type-mismatched
hit — this is the kind of bug that's invisible until production.

## The cache loop

Every primitive follows the same canonical flow:

```
preprocess raw HTML → reduce to deterministic snapshot
   │
   └─→ hash(primitive, intent?, schema?, dom) → cache key
        │
        ├── HIT  → re-validate stored result against current page
        │            ├── still valid → return / execute
        │            └── stale       → invalidate, fall through
        │
        └── MISS → LLM call (with prior_failures fed back on retry)
                     │
                     ├── output validates → execute (act) / return data
                     │                       → write cache entry (after success)
                     │
                     └── output unusable  → retry up to max_resolution_retries
                                              with feedback in prompt
```

Two non-obvious decisions in this loop:

1. **Re-validate on cache hit (act).** A cached selector is verified
   against the live DOM before being executed. A page may have changed
   in a way that doesn't shift the preprocessed-DOM hash but does
   eliminate the specific element. We invalidate and re-resolve rather
   than execute against an unrelated element.

2. **Cache *after* a successful execute.** A selector that passes
   structural validation (matches ≥1 element) can still fail at the
   click/fill step (overlapped, mid-animation, transition not settled).
   Caching before execute would poison the cache with a "valid-but-not-
   actionable" selector that every future run would pay timeout cost to
   rediscover.

## Resolution discriminated union

Each primitive produces a different shape of cached resolution. To let
the cache layer remain a single table keyed by a single hash, all three
share a discriminator field (`kind`):

```python
class ActionResolution(BaseModel):
    kind: Literal["action"] = "action"
    intent: str
    dom_hash: str
    candidates: list[SelectorCandidate]
    chosen_selector: str
    chosen_selector_type: SelectorType
    chosen_action: ActionType
    chosen_action_value: str | None

class ExtractionResolution(BaseModel):
    kind: Literal["extraction"] = "extraction"
    intent: str
    dom_hash: str
    schema_fingerprint: str
    data_json: str   # serialized — schema lives in user code

class ObservationResolution(BaseModel):
    kind: Literal["observation"] = "observation"
    dom_hash: str
    actions: list[ObservedAction]

Resolution = Annotated[
    ActionResolution | ExtractionResolution | ObservationResolution,
    Field(discriminator="kind"),
]
```

A `pydantic.TypeAdapter[Resolution]` is built once at import time and
reused on every cache read. Pydantic dispatches to the right concrete
model based on the `kind` tag.

`ExtractionResolution.data_json` stores serialised JSON, not a typed
object, because the user's Pydantic schema lives in user code and is
unknown to the cache layer. On read, the caller revalidates `data_json`
against the schema they pass to `extract`. If validation fails (the
fingerprint guard isn't supposed to let this happen, but defends against
hand-edited rows or future schema-evolution corner cases), the cache
entry is invalidated and the LLM is consulted.

## DOM preprocessing

Raw SPA HTML is too large and too noisy to send to an LLM unmodified.
`engram.dom.preprocessor.preprocess_dom` reduces it deterministically:

- **Drops:** `<script>`, `<style>`, `<svg>`, `<noscript>`, `<canvas>`,
  `<template>`, `<iframe>`, `<link>`, `<meta>`, comments, elements with
  `hidden` / `aria-hidden=true` / inline `display:none` /
  `visibility:hidden`.
- **Keeps:** `id`, `class`, `role`, `type`, `name`, `placeholder`,
  `href`, `alt`, `title`, `value`, `for`, `autocomplete`, common
  `aria-*` attributes, common `data-test*` / `data-cy` /
  `data-qa` attributes (anything else under `data-*` is dropped — they
  rarely carry semantic signal but bloat the prompt).
- **Scrubs sensitive values:** strips `value` and `placeholder` from
  `<input type="password|hidden">`, from inputs whose
  `name`/`id`/`autocomplete` matches a sensitive-name regex
  (`password|secret|token|otp|cvv|ssn|api_key|auth`), and from inputs
  with `autocomplete=current-password|new-password|one-time-code|cc-*`.
  Without this, a legacy SSR form pre-populating a password attribute
  would ship the credential into the LLM prompt verbatim. See
  [SECURITY.md](SECURITY.md) for the full list and the threat model.
- **Normalises whitespace** so two functionally identical pages always
  produce the same hash.
- **Hard size cap:** raises `EngramDOMError` if the result exceeds
  `dom_max_chars` (default 50 KB) rather than silently truncating —
  silent truncation would produce unstable cache keys.

Determinism is the load-bearing property here. Anything non-deterministic
in preprocessing would invalidate the cache on every run.

## Retry loop with feedback

`act` and `extract` re-prompt the LLM with feedback when output is
unusable. Failure modes that trigger a retry:

- **`act`** — the LLM returned candidates but every one of them either
  matched zero elements on the live page or fell below
  `min_candidate_confidence`.
- **`extract`** — the LLM's output failed Pydantic validation against
  the user's schema.

Each retry's failures are accumulated into `prior_failures: list[str]`
and threaded into the next LLM call. The LiteLLM client prepends them
to the user message so the model reads "what went wrong" before "what
to do":

```
Previous attempts at this same task failed. Do not repeat these mistakes:
- selector `#bogus-id` (css) matched 0 elements on the page
- output failed schema validation: age: field required

Intent: ...
<UNTRUSTED_PAGE_CONTENT>
...
</UNTRUSTED_PAGE_CONTENT>
```

This is what makes retries actually correct the model rather than
producing the same broken output 3 times. Without feedback, retrying a
deterministic LLM on the same input is a waste of tokens.

`observe` does NOT retry — an empty action list is a legitimate result
(static error pages, splash screens) rather than a failure.

## LLM client

`AbstractLLMClient` is a `Protocol` with three async methods, one per
primitive. The default implementation, `LiteLLMClient`, talks to any
LiteLLM-supported provider and forces structured output via the
provider's tool-calling interface. Each primitive has its own tool
schema; the schema for `extract` is built dynamically from the
user-supplied Pydantic JSON Schema.

`BudgetedLLMClient` is a decorator that enforces
`max_llm_calls_per_session` — useful as a guardrail against runaway
flows. It counts every downstream call, successful or failed
(otherwise a flaky provider's failures would be invisible to the
budget). When wired in by `EngramBrowser`, it raises
`EngramBudgetExceededError` once the cap is hit.

## Concurrency model

- `EngramPage` is an asyncio coroutine surface. `act` / `extract` /
  `observe` are async; awaiting one returns control to the event loop
  cleanly during DOM reads, LLM calls, and cache I/O.
- `SQLiteCache` uses one `aiosqlite` connection per instance, in
  autocommit mode (`isolation_level=None`). Each cache operation is a
  single SQL statement; SQLite serialises them at the file level via
  WAL + `busy_timeout=5000ms`. A first-time-connect lock prevents
  concurrent `_connect()` calls from racing on schema setup.
- `RedisCache` does atomic hit-count updates via `HINCRBY`. The
  underlying `redis.asyncio` client handles connection pooling.
- `act` and `extract` and `observe` each bind a unique `request_id` via
  `structlog.contextvars.bound_contextvars`. asyncio gives each task
  its own contextvars copy, so two parallel calls don't share IDs.

## Observability

Every state transition emits a `structlog` event:

| Event name | Meaning |
|---|---|
| `cache.hit` / `cache.miss` / `cache.invalidated` | Per-primitive cache outcome |
| `selector.validated` / `selector.failed` / `selector.rejected_low_confidence` | `act`-loop decisions |
| `action.executed` | Click/fill completed |
| `extraction.retry` / `extraction.completed` | `extract`-loop decisions |
| `observation.dropped_hallucination` / `observation.completed` | `observe`-loop decisions |
| `llm.call.action` / `llm.call.extraction` / `llm.call.observation` | LLM round-trips with latency + token counts |

User intents are logged as a 12-char SHA-256 prefix by default
(`intent_hash`), not verbatim — intents are user-controlled input and
frequently carry PII. Set `EngramConfig.log_intents_verbatim=True` only
in trusted environments.

## Library choices and rationale

| Concern | Choice | Why |
|---|---|---|
| Browser automation | Playwright (async) | Best async-native Python API; Chromium / Firefox / WebKit out of the box |
| LLM calls | LiteLLM | Provider-agnostic so swapping Anthropic ↔ OpenAI ↔ Azure is a config change |
| Structured output | Pydantic v2 | Discriminated unions, `TypeAdapter`, fast JSON Schema generation |
| Local cache | `aiosqlite` | Zero-deps async SQLite; UPDATE…RETURNING for atomic hit-count bumps |
| Shared cache | `redis-py` async | Standard Redis client; atomic `HINCRBY` for cross-machine increments |
| HTML parsing | `lxml` | Standards-compliant HTML5 parsing; safe (no XXE) |
| DOM hashing | `hashlib.sha256` | Standard library, fast, stable across runs and Python versions |
| Logging | `structlog` | Structured JSON output; `contextvars` integration for correlation IDs |
| Testing | `pytest` + `pytest-asyncio` | Standard async test stack |
| Type checking | `mypy --strict` | Non-negotiable; catches whole categories of bugs at edit time |

## What's deliberately *not* abstracted

- **`EngramPage.playwright_page` is a public escape hatch.** If you
  need raw Playwright (screenshots, custom waits, network
  interception), you reach in. Wrapping every Playwright capability
  behind Engram methods would balloon the surface area for marginal
  ergonomic wins.
- **There is no plugin system.** Cache backends and LLM clients are
  injected via the constructor. If you need a third backend, write a
  class that satisfies the Protocol and pass it in.
- **There is no browser-provider abstraction.** Playwright already
  abstracts CDP-speaking providers via `chromium.connect_over_cdp(url)`,
  and `EngramPage` accepts any Playwright `Page`. Adding a vendor
  layer (Browserbase / BrowserCat / Steel.dev / Selenium grid) would
  duplicate what Playwright does for free, and would tilt Engram from
  a thin orchestration into a platform-aware library — which is
  explicitly not the goal. The README has a worked example of this
  pattern.
- **There is no automatic retry on transient provider errors.** That's
  LiteLLM's job (`num_retries`, exponential backoff). Engram's
  `max_resolution_retries` only retries on *semantic* failures
  (validation errors), where the prompt itself needs to change.

## Pre-1.0 stability guarantees

The three primitive methods on `EngramPage` and the public-facing
fields of `EngramConfig` are stable for the 0.x line — additive
changes only. Internal modules (anything not re-exported from
`engram/__init__.py`) may change without notice between minor versions
until 1.0.
