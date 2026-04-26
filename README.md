# Engram

**Python-native browser automation with cached LLM-resolved selectors.**

Engram lets you write deterministic Playwright flows in Python and delegate
*element resolution* to an LLM — and crucially, **caches those resolutions so
the LLM is consulted once per UI version, not once per run.**

```python
import asyncio
from engram import EngramBrowser, EngramConfig

async def main() -> None:
    async with EngramBrowser(EngramConfig.from_env()) as browser:
        page = await browser.new_page()
        await page.goto("https://example.com/sessions")
        await page.act("click the join session button")

asyncio.run(main())
```

The first run pays one LLM call to find the button. Every subsequent run on
the same UI hits the local cache for free.

---

## Why?

Existing options force you to choose:

| Tool | Language | Caching | Control |
|---|---|---|---|
| [Stagehand](https://github.com/browserbase/stagehand) | TypeScript | ❌ Re-queries LLM every run | You own the flow |
| [Browser Use](https://github.com/browser-use/browser-use) | Python | ❌ LLM picks every action | LLM owns the flow |
| **Engram** | **Python** | **✅ One LLM call per UI version** | **You own the flow** |

The conceptual insight from Stagehand — *you control the flow, the LLM
resolves elements* — implemented as it should be in Python, with caching as
a first-class architectural primitive.

The name is a neuroscience term for the physical trace a memory leaves in
the brain — the pattern that encodes a learned experience so it can be
recalled later without re-learning it. The first time Engram sees a UI, it
"learns" how to interact with it via LLM. That learning is encoded as an
engram — a cached selector resolution keyed to a hash of the relevant DOM
subtree. Every subsequent run retrieves the engram instead of re-querying.
When the UI changes, the hash changes, the old engram is invalid, and a new
one is formed.

---

## Install

```bash
pip install engram
playwright install chromium
```

For the Redis-backed shared cache (recommended for teams):

```bash
pip install 'engram[redis]'
```

---

## The three primitives

### `act(intent)` — execute a natural-language action

```python
async with EngramBrowser() as browser:
    page = await browser.new_page()
    await page.goto("https://example.com/login")

    # First call: LLM resolves "the username field" → '#username' (~1s, ~$0.001).
    # Cached.
    await page.act("type 'alice' in the username field")

    # First call: LLM resolves "the password field". Cached.
    await page.act("type 'hunter2' in the password field")

    # First call: LLM resolves "the sign in button". Cached.
    await page.act("click the sign in button")

    # Second run of this script: 0 LLM calls. The cache returns the
    # selectors directly and Playwright executes them.
```

`act` raises `EngramResolutionError` if no candidate selector validates
against the page after the configured retry budget. It raises
`EngramBrowserError` if a validated selector then fails at the click/fill
layer (the cache is *not* poisoned in this case — the entry is only
written after a successful execute).

### `extract(intent, schema)` — extract structured data with a Pydantic schema

```python
from pydantic import BaseModel

class Participant(BaseModel):
    name: str
    role: str

class ParticipantList(BaseModel):
    participants: list[Participant]

async with EngramBrowser() as browser:
    page = await browser.new_page()
    await page.goto("https://example.com/meeting/123/participants")

    # The schema fingerprint is part of the cache key, so changing the
    # Pydantic model produces a fresh extraction (no stale type-mismatched
    # results from a previous schema version).
    data = await page.extract(
        "everyone in the participants panel",
        schema=ParticipantList,
    )

    for p in data.participants:
        print(p.name, p.role)
```

`extract` returns a validated instance of your model. If the LLM's output
fails Pydantic validation, the call retries with the validation errors fed
back into the prompt; if it still fails after the retry budget,
`EngramExtractionError` is raised.

### `observe()` — list the meaningful actions on the current page

```python
async with EngramBrowser() as browser:
    page = await browser.new_page()
    await page.goto("https://example.com/dashboard")

    actions = await page.observe()
    for a in actions:
        print(f"[{a.confidence:.2f}] {a.description} → {a.selector}")
    # Example output:
    # [0.95] Click the New Project button → button[data-testid="new-project"]
    # [0.92] Type a query into the search box → #search
    # [0.83] Click the Settings link → a[href="/settings"]
```

`observe` returns a confidence-sorted list of `ObservedAction`. Hallucinated
selectors (those the LLM proposes but which match zero elements on the live
page) are filtered out *before* the result is cached, so subsequent
`observe()` calls on the same DOM hash don't re-pay validation cost.

---

## How the cache works

```
act("click join")
  ├── preprocess DOM (strip scripts, styles, SVG, comments, hidden elements)
  ├── hash sha256(intent + DOM)
  ├── cache lookup
  │   ├── HIT → re-validate selector against current DOM → execute
  │   └── MISS → LLM call → validate candidates → execute → cache
  └── done
```

- **Cache key:** `sha256(primitive_namespace + normalized_intent + preprocessed_dom)`. For `extract`, the schema fingerprint is also folded in.
- **Cache value:** the chosen selector + ranked candidate list + metadata.
- **Invalidation:** automatic when the DOM hash changes (the UI was updated,
  so a new engram is formed) or when a cached selector no longer matches the
  live page (re-resolved on the spot).
- **Backends:** local SQLite (default, zero-config) or shared Redis
  (recommended for teams — community-resolved selectors benefit everyone).

---

## Configuration

Configure via `EngramConfig(...)` directly, or via environment variables
read by `EngramConfig.from_env()`:

| Field | Env var | Default | Notes |
|---|---|---|---|
| `llm_model` | `ENGRAM_LLM_MODEL` | `anthropic/claude-sonnet-4-5` | Any LiteLLM identifier |
| `llm_api_key` | `ENGRAM_LLM_API_KEY` | (provider env vars) | `SecretStr` |
| `llm_temperature` | `ENGRAM_LLM_TEMPERATURE` | `0.0` | |
| `cache_backend` | `ENGRAM_CACHE_BACKEND` | `sqlite` | `sqlite` or `redis` |
| `cache_path` | `ENGRAM_CACHE_PATH` | `.engram/cache.db` | SQLite file path |
| `redis_url` | `ENGRAM_REDIS_URL` | `None` | Required when backend is `redis` |
| `redis_key_prefix` | `ENGRAM_REDIS_KEY_PREFIX` | `engram:` | Multi-tenant namespace |
| `redis_ttl_seconds` | `ENGRAM_REDIS_TTL_SECONDS` | `None` | Optional hard staleness cap |
| `cache_signing_key` | `ENGRAM_CACHE_SIGNING_KEY` | `None` | HMAC-signs Redis entries (see SECURITY.md) |
| `max_resolution_retries` | `ENGRAM_MAX_RESOLUTION_RETRIES` | `2` | Re-prompt with feedback when output is unusable |
| `max_llm_calls_per_session` | (manual) | `None` | Hard session-level call cap |
| `min_candidate_confidence` | `ENGRAM_MIN_CANDIDATE_CONFIDENCE` | `0.5` | Reject candidates below this floor |
| `dom_max_chars` | `ENGRAM_DOM_MAX_CHARS` | `50_000` | Preprocessed DOM size cap |
| `log_intents_verbatim` | `ENGRAM_LOG_INTENTS_VERBATIM` | `False` | When `False`, log a hash instead of the raw intent |
| `headless` | `ENGRAM_HEADLESS` | `True` | Browser visibility |

For Redis on a multi-machine setup:

```python
from pydantic import SecretStr

config = EngramConfig(
    cache_backend="redis",
    redis_url="rediss://cache.internal:6380/0",
    cache_signing_key=SecretStr("a-shared-secret"),  # see SECURITY.md
    max_llm_calls_per_session=200,                   # session budget cap
)
```

---

## Architecture

```
EngramBrowser (Playwright + cache + LLM)
  └── EngramPage
       ├── act(intent)
       ├── extract(intent, schema)
       └── observe()
             │
             ├─→ DOM preprocessing  (engram.dom)
             ├─→ AbstractCache      (engram.cache: SQLite / Redis)
             └─→ AbstractLLMClient  (engram.llm: LiteLLM-backed by default,
                                     wrapped in BudgetedLLMClient when a
                                     session budget is configured)
```

Every component is behind an interface. Tests mock the interfaces; real
deployments wire real implementations. `mypy --strict` runs clean across
the codebase.

---

## Logging

Engram emits structured logs (via [structlog](https://www.structlog.org/))
for every cache hit/miss, LLM call, selector validation, action execution,
and resolution failure. By default, structlog renders human-readable
key-value pairs to stderr; for JSON output (production), call:

```python
from engram import configure_logging
configure_logging(level="INFO", json_output=True)
```

Each `act` / `extract` / `observe` call binds a unique `request_id` to
[`structlog.contextvars`](https://www.structlog.org/en/stable/contextvars.html),
so all downstream events from one user-facing call share a correlation ID.

---

## Security

See [SECURITY.md](SECURITY.md) for the threat model and the residual risks
to be aware of:

- Page content is *untrusted input* to the LLM (prompt-injection risk).
- Pre-filled sensitive form values (passwords, OTPs, credit cards) are
  scrubbed from the preprocessed DOM before it reaches the LLM.
- A shared Redis cache *must* be on a trusted network or signed with
  `cache_signing_key`.
- Intents are logged as a hash by default — flip `log_intents_verbatim` only
  in trusted environments.

---

## Status

`v0.2.0` — alpha. Public API surface is the three primitives + `EngramConfig`.
Anything in `engram._*` (private) may change without notice. `mypy --strict`
+ `ruff` + the full pytest suite pass on every commit.

License: MIT.
