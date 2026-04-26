# Contributing to Engram

Thanks for considering a contribution. This document covers how to set up
a dev environment, the checks that must pass before you open a PR, and a
few project conventions worth knowing.

For the architectural picture (module layout, cache loop, Resolution
discriminated union, etc.), read [ARCHITECTURE.md](ARCHITECTURE.md). For
the threat model and residual risks, read [SECURITY.md](SECURITY.md). For
what we're considering shipping next and what we've already decided not
to build, read [ROADMAP.md](ROADMAP.md) — checking there before opening a
big PR can save a lot of work.

## Local setup

Engram targets Python 3.10+.

```bash
git clone https://github.com/juvantlabs/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
playwright install chromium
```

The `dev` extra pulls in pytest, mypy, ruff, and the Redis client (so
the Redis-cache tests run without an extras install). The
`playwright install chromium` step downloads the browser used by the
live integration tests; without it those tests skip with a clear
message.

## The full check

Before opening a PR, all three of these must pass cleanly:

```bash
ruff check engram tests
mypy engram tests
pytest tests
```

Or as a one-liner:

```bash
ruff check engram tests && mypy engram tests && pytest tests
```

This is the same sequence CI runs (or will run, once `.github/workflows/`
lands). `mypy --strict` is non-negotiable on this codebase — it has caught
real bugs and we'd like to keep the streak.

## Test layout

```
tests/
  unit/                          # pure unit tests, no browser, no network
    test_<module>.py             # one file per source module under test
  integration/                   # real Playwright + real SQLite + faked LLM
    test_act_live.py
    test_extract_live.py
    test_observe_live.py
  fixtures/                      # static HTML pages used by integration tests
    join_page.html
```

Integration tests skip automatically when Playwright's browsers are not
installed (the helper checks for a `chromium-*` directory under the
standard cache location). So `pytest tests` is safe to run on a fresh
machine — you just lose integration coverage until you `playwright
install chromium`.

LLM calls in tests are *always* faked. We never hit a real provider in
the test suite — it would be non-deterministic, slow, and cost real
money on every CI run. The pattern: a test-local `_FakeLLM` /
`CountingFakeLLM` class that satisfies the `AbstractLLMClient` Protocol
and returns canned responses. Cache and Playwright stay real (or
faked-but-faithful, in the unit-test case).

## What to put in `engram/` vs. behind a Protocol

The library has three Protocol-defined seams: `AbstractCache`,
`AbstractLLMClient`, and (implicitly via dependency injection)
the Playwright `Page`. New backends — a different cache (e.g. memcached),
a non-LiteLLM client, a remote browser — should be implemented by
subclassing or duck-typing the relevant Protocol and injected via
`EngramBrowser(...)` / `EngramPage(...)`. They do not need to live in
the main package; you can publish them as separate libraries.

Code that *every* user of Engram needs (cache loop, retry behaviour,
DOM preprocessor, error hierarchy) lives in `engram/`.

## Coding conventions

- **Type everything.** `mypy --strict` is enforced; no untyped
  `**kwargs`, no implicit `Any`, no `Dict` / `List` (use `dict` /
  `list`). Pydantic v2 is the default for any structured data crossing
  a boundary.
- **Async by default.** Every I/O method is `async`; no sync wrappers
  hiding event loops.
- **No silent failures.** If a cache entry is corrupt, we raise
  `EngramCacheError` after invalidating it. If a selector can't be
  resolved, we raise `EngramResolutionError` with the failure reasons
  attached. Compare this to silent fallbacks that produce
  works-on-my-machine bugs in production.
- **Determinism in the cache layer.** Anything that affects the cache
  key (DOM preprocessing, intent normalisation, schema fingerprinting)
  must produce byte-stable output across runs. A flaky bit there
  invalidates the whole cache contract.
- **No backwards-compat shims pre-1.0.** If a name is wrong, rename it.
  If a field is unused, delete it. We pre-emptively bumped 0.1.0 → 0.2.0
  for renaming `SelectorResolution` → `ActionResolution` and
  `hash_dom` → `hash_action` rather than carrying aliases forward.

## Commits and PRs

- Commit messages follow the style of the existing log: short
  imperative subject, optional body explaining *why*. Conventional
  Commits prefixes (`feat:`, `fix:`, `chore:`, `docs:`) are used.
- One logical change per PR is ideal but not enforced. If a PR touches
  more than one concern, the commit history within it should make the
  separation clear.
- Tests that exercise the new code path are required for any non-doc
  change. "I tested it locally" is not enough — we need it captured so
  the next person can refactor without fear.

## Reporting security issues

Vulnerability reports go to **security@juvant.io** (with **dev@juvant.io**
in CC). Don't file public issues for security reports. See
[SECURITY.md](SECURITY.md) for the full reporting policy and threat
model.

## Code of conduct

Be respectful. Disagree on the technical merits, not on the person.
Assume good faith. If a maintainer asks you to change something, that
ask comes with the implicit expectation that they're trying to make
the project better — push back if you disagree, but do it on the
merits.

## License

By contributing, you agree your contributions will be licensed under
the project's [MIT License](LICENSE).
