# Roadmap

This document captures **what we're considering**, **what we're explicitly
not building**, and the rationale for each. It's intentionally short — if
it gets long enough that scanning it becomes work, we'll graduate the
rationale-heavy entries to `docs/adr/` (Architecture Decision Records).

The goal is twofold: avoid silently forgetting good ideas, and avoid
relitigating decisions we've already considered carefully.

For shipped work, see [CHANGELOG.md](CHANGELOG.md). For the architectural
picture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## On the list (would accept a PR)

Items here are concrete enough that someone could scope a PR. They're not
prioritised, and a few may never ship if no demand surfaces — file an
issue first if you want to start work, so we can sanity-check fit before
you write code.

### Cache export / import

**Goal.** `engram cache export > cache.json` and `engram cache import < cache.json`,
working symmetrically across the SQLite and Redis backends.

**Why.** A team building an automation can commit a curated cache file to
git alongside the script — fresh clones get a warm cache without
re-running every intent through an LLM. Solves the genuine value of
"shared engrams" (intra-team) without any of the trust / discovery
problems of a public registry.

**Why not yet.** No user has asked. The pattern of "ship a SQLite file
next to your script" already works today; export/import becomes worth
building when somebody wants Redis-backed teams to do the same thing
without standing up Redis themselves.

**Roughly.** ~50 lines plus tests. Compatible with the discriminated
`Resolution` union — same `TypeAdapter` we already use for cache I/O.

### 1.0 stability commitment

The 0.x line allows additive changes only on the public API
(`EngramPage` primitives, `EngramConfig` fields, the `engram.*` re-exports).
We bump to **1.0** when (a) the three primitives have been used in two or
more independent real-world automations beyond the original Collaborate
case, and (b) we've gone six months without renaming or restructuring
anything in the public surface.

Until then, internal modules (anything not re-exported from
`engram/__init__.py`) may change without notice between minor versions.

---

## Considered and not building

Items here are decisions, not deferrals. We've thought through them and
the answer is no for explicit reasons. If you think one is worth
revisiting, please **link to this section** in the issue/PR so we can
respond to the specific argument rather than starting from scratch.

### A public "community engrams" registry

**The pitch.** A public service where users contribute and consume cached
selector resolutions, so automating common SaaS apps (Stripe, GitHub,
Salesforce) becomes free for the next user.

**Why we're not building it.** Five hard sub-problems, each a real product
in its own right:

1. **DOM-hash variability is brutal.** The cache key folds in the entire
   preprocessed DOM. Two users on "the same page" get different hashes
   the moment they're in different A/B buckets, locales, account tiers,
   feature-flag states, or visible-text states (notification banners,
   unread counts). Public hit rate would be much lower than the pitch
   suggests.
2. **Adversarial submissions.** Anyone could publish a "resolution" for
   `act("click join")` on stripe.com pointing at the "delete account"
   button. Our re-validate-on-hit catches "selector matches zero
   elements," not "selector matches the *wrong* element." Defending
   against this requires cryptographic provenance, trusted publishers,
   or active moderation — each its own product.
3. **PII leakage via DOM hashes.** Visible page text frequently contains
   the user's email / plan tier / account ID. Publishing the hash plus
   the resolution leaks structured information about user state.
   Sidestepping this means changing the cache key to a coarser
   site-only fingerprint, which lowers correctness.
4. **Staleness discovery.** When the upstream site ships a UI change,
   every shared engram for that hash is silently invalid. The
   community can't easily learn that. Solving it means active health
   checks against real sites — a hosted service.
5. **Cold start + ongoing hosting.** The registry is useless until many
   users contribute, expensive forever after, and has no obvious
   monetisation path that doesn't tilt Engram into Browserbase's lane.

**What we ship instead.** `cache export/import` (above) covers the 80%
of the value where it actually lands — within a team — without any of
these problems.

**If you still want this:** it's a startup, not a library feature. We'd
be happy to integrate against a third-party registry that solves the
trust/staleness/hash-scope problems via a clean API.

### A Browserbase / cloud-browser-vendor backend

**The pitch.** First-class support for Browserbase, BrowserCat,
Steel.dev, etc. — wrap their SDK so users don't have to.

**Why we're not building it.** Playwright already abstracts CDP-speaking
browser providers via `chromium.connect_over_cdp(url)`, and `EngramPage`
accepts any Playwright `Page`. A vendor-specific layer would duplicate
what Playwright does for free, and would tilt Engram from a thin
orchestration into a platform-aware library — which the
"What we are not building" stance explicitly rules out.

**What we ship instead.** A worked example in [README.md](README.md#remote-browsers-browserbase-browsercat-self-hosted-cdp)
showing the `connect_over_cdp` pattern. The same code works against
Browserbase, BrowserCat, Steel.dev, a self-hosted Selenium grid, or
any future CDP-speaking provider — no Engram-side wiring.

### A fully autonomous "decide-every-step" mode

**The pitch.** Give Engram a goal and let it figure out every action.

**Why we're not building it.** That's the Browser Use design and it has
known weaknesses we deliberately avoided: slow (one LLM round-trip per
action), unpredictable, and uncacheable. Engram's whole proposition is
that *you* own the flow and the LLM owns element resolution. Shifting
control to the LLM at the flow level erases the cache value, the
predictability, and the testability that make the library useful.

---

## How to propose adding something

Open a GitHub issue with the `roadmap` label. Briefly answer:

1. What problem does this solve?
2. Who has the problem? (Including yourself counts.)
3. What's the smallest version that solves it?
4. What does the public API look like?

If the answer to any of those is hand-wavy, the proposal is probably
not ready to ship. That's fine — discussion in the issue is how we get
to ready.
