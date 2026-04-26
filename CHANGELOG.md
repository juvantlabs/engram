# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0: minor versions may include backwards-incompatible changes to internal
modules. The three primitives on `EngramPage` and the public-facing fields of
`EngramConfig` are stable for the 0.x line — additive changes only.

## [Unreleased]

## [0.2.0] — 2026-04-26

### Added

- `extract(intent, schema)` — LLM-resolved structured data extraction validated
  against a user-supplied Pydantic schema. The schema's JSON Schema is
  fingerprinted and folded into the cache key, so a schema change yields a
  fresh cache miss rather than a stale type-mismatched hit.
- `observe()` — returns a confidence-sorted `list[ObservedAction]` of the
  meaningful actions on the current page. Hallucinated selectors (those the
  LLM proposes but which match zero elements) are filtered out before the
  result is cached.
- Redis cache backend (`cache_backend="redis"`) for shared multi-machine
  setups. Opt-in via the `redis` extra: `pip install 'engram-browser[redis]'`.
- HMAC-SHA256 signing of Redis cache entries via
  `EngramConfig.cache_signing_key`. Tampered or unsigned entries are rejected
  and removed; recommended whenever the Redis instance is reachable from
  untrusted networks.
- Resolution retries with feedback (`max_resolution_retries`, default `2`):
  when `act` or `extract` produce unusable output, the LLM is re-prompted
  with a summary of what failed, so retries actually correct the model
  rather than re-emitting the same broken response.
- Per-session LLM-call budget (`max_llm_calls_per_session`) enforced by a
  `BudgetedLLMClient` decorator. Raises `EngramBudgetExceededError` when the
  cap is reached.
- Correlation IDs (`request_id`) bound via `structlog.contextvars` at every
  primitive entry — every downstream cache/LLM/action event is tagged so a
  user-facing call can be traced through the logs.
- PII-safe intent logging: intents are logged as a 12-char SHA-256 hash by
  default (`intent_hash`); set `EngramConfig.log_intents_verbatim=True` only
  in trusted environments.
- Prompt-injection mitigations: the preprocessed DOM is wrapped in
  `<UNTRUSTED_PAGE_CONTENT>...</UNTRUSTED_PAGE_CONTENT>` delimiters in every
  LLM call, and each system prompt explicitly instructs the model to ignore
  embedded instructions.
- Sensitive-form-field scrubbing in the DOM preprocessor: `value` and
  `placeholder` are stripped from `<input type="password|hidden">`, from
  inputs whose `name`/`id` matches a credential-name regex, and from inputs
  with `autocomplete=current-password|new-password|one-time-code|cc-*`.
- Redis URL credential redaction: `:password@` segments are masked in any
  error message before reaching `EngramCacheError`.
- Live Playwright integration tests for `extract` and `observe`, mirroring
  the existing `test_act_live` setup.
- `LICENSE` (MIT, copyright Juvant Srls).
- `README.md` with quickstart, worked examples for the three primitives, a
  configuration table, and links to `SECURITY.md` and `ARCHITECTURE.md`.
- `SECURITY.md` documenting the threat model, residual risks, baked-in
  defences, and the vulnerability-reporting channel.
- `ARCHITECTURE.md` describing module layout, the cache loop, the
  Resolution discriminated union, the DOM preprocessor, the retry loop,
  the concurrency model, and library-choice rationale.

### Changed

- SQLite cache hit-count update is now a single atomic
  `UPDATE … RETURNING` statement (autocommit + WAL mode). The previous
  SELECT-then-UPDATE could lose increments under concurrent reads from two
  cache instances pointing at the same file.
- `hash_dom` renamed to `hash_action` and namespaced with a leading
  `"action\n"` tag, matching the shape of `hash_extraction` and
  `hash_observation`.
- `SelectorResolution` renamed to `ActionResolution` so the class name
  matches its `kind="action"` discriminator. `SelectorCandidate` is
  unchanged (it's the candidate type, not the resolution).
- `max_llm_retries` renamed to `max_resolution_retries` with an updated
  docstring clarifying that this is the *semantic* retry budget (re-prompt
  with feedback) and that transport-level retries belong to LiteLLM's own
  retry settings.
- Pyproject `readme` field switched from `CLAUDE.md` (internal brief) to
  `README.md` (user-facing).
- `pyproject.authors` now points at Juvant `dev@juvant.io`.

### Security

- See **Added** above for the full set of security-hardening changes:
  prompt-injection mitigations, sensitive-form-field scrubbing, HMAC-signed
  Redis cache entries, PII-safe intent logging, Redis URL credential
  redaction, and a per-session LLM-call budget.

## [0.1.0] — 2026-04-23

### Added

- Initial release.
- `EngramBrowser` — async context manager wrapping Playwright with
  dependency-injected cache and LLM client.
- `EngramPage.act(intent)` — natural-language browser action with cached
  LLM-resolved selectors. Cache key is `sha256(normalized_intent +
  preprocessed_dom)`.
- `SQLiteCache` — local file-backed implementation of `AbstractCache`.
- `LiteLLMClient` — provider-agnostic LLM client via LiteLLM's tool-calling
  interface; works with Anthropic, OpenAI, Azure, and any LiteLLM-supported
  provider.
- DOM preprocessor that reduces raw SPA HTML to a deterministic, semantic
  snapshot suitable for an LLM prompt.
- Selector validation: every LLM-proposed candidate is validated against
  the live DOM before execution; unusable candidates are skipped.
- `EngramConfig` (Pydantic) loadable from environment variables via
  `EngramConfig.from_env()`.
- Structured logging via `structlog`.
- `mypy --strict` clean across the codebase; `ruff check` clean; full
  `pytest` suite green.

[Unreleased]: https://github.com/juvantlabs/engram/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/juvantlabs/engram/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/juvantlabs/engram/releases/tag/v0.1.0
