# Production Prompt Enhancer

An Open WebUI **filter function** that rewrites user prompts into clearer,
more specific requests before they reach the chat model — opt-in per message,
per chat, per user, or instance-wide.

- **File:** [`prompt_enhancer.py`](./prompt_enhancer.py) (single file, v4.11.0)
- **Requires:** Open WebUI ≥ 0.9.1 (verified against 0.11.x)
- **Code review:** [`prompt-enhancer-review.md`](./prompt-enhancer-review.md)

## Install

1. Open WebUI → **Admin Panel → Functions → New Function**.
2. Paste the contents of `prompt_enhancer.py`, save, and enable it.
3. Enable it globally or attach it to specific models (model settings →
   filters).

No other setup is needed. By default nothing changes until someone opts in
with the prefix.

## Quick start

Type `!!` in front of any message to enhance it:

```
!!write a launch plan for our new API
```

The filter sends your prompt (prefix stripped) through an enhancement LLM —
the chat model itself, or a dedicated model via the `model_id` valve — swaps
in the improved version, and keeps your original in
`metadata.original_prompt`. Everything without the prefix passes through
untouched.

## Commands and flags

| Message | Effect |
|---|---|
| `!!<prompt>` | Enhance this message |
| `!!c: <prompt>` / `!!concise: <prompt>` | Concise style for this message |
| `!!s: <prompt>` / `!!standard: <prompt>` | Standard style for this message |
| `!!d: <prompt>` / `!!detailed: <prompt>` | Detailed style for this message |
| `!!show: <prompt>` | Also display the before/after comparison card |
| `!!fresh: <prompt>` | Ignore the cache and re-enhance (also retries a paused enhancer) |
| `!!d,show: <prompt>` | Flags combine with commas |
| `!!on` / `!!off` | Turn automatic enhancement on/off for the current chat |
| `!!help` | Show the command card |
| `!!stats` / `!!stats reset` | Runtime counters / reset (admin) |
| `!!test` | Round-trip the enhancer model, report latency, clear the circuit breaker (admin) |

A header is treated as flags only when *every* token before the colon is a
known flag — `!!Summarize this: my notes` enhances the whole text untouched.

## The control hierarchy

Enhancement can be controlled at four levels; the more specific level wins:

1. **Per message** — the `!!` prefix (and its flags) always means "enhance
   this" in activate mode, "don't enhance this" in bypass mode.
2. **Per chat** — `!!on` / `!!off` override the automatic behavior for
   unprefixed messages in that chat (in-memory, 24h TTL; resets on restart
   and is per worker process).
3. **Per user** — the `prefix_mode` user valve lets an individual run
   `bypass` (always-on) while the instance default stays `activate`, or vice
   versa. Users can also disable the filter entirely, pick a default style,
   and toggle the comparison card for themselves.
4. **Instance** — the admin `prefix_mode` valve: `activate` (opt-in, the
   default) or `bypass` (the classic always-on behavior where `!!` opts out).

## How it works

For each triggered message, the filter:

1. **Skips what shouldn't be enhanced** — trivial greetings (multilingual),
   overly long prompts, mostly-code prompts, admin regex skip patterns,
   skip-listed models, and (configurably) follow-ups, image messages, and
   already-well-structured prompts. Explicit `!!` triggers relax the
   length/follow-up gates and always surface a status explaining a skip.
2. **Detects intent** with regex scoring across 17 built-in categories
   (debugging, coding, creative, security, data science, …; extensible via a
   JSON valve) and injects matching guidance into the enhancement prompt.
3. **Optionally goes Socratic** — when a prompt is too vague to enhance
   ("fix it"), it instead instructs the chat model to ask one clarifying
   question (off by default, `enable_socratic`).
4. **Calls the enhancement LLM** with conversation context, tool IDs,
   date/time, and image-awareness (it is told images exist and must preserve
   references to them), then cleans the output: reasoning-tag stripping
   (balanced, dangling-open, and leading-close), fence/quote unwrapping, and
   preamble-artifact removal.
5. **Guards the result** — configurable max length and expansion-ratio
   limits; any failure falls back to the original prompt, never breaking the
   chat.

### Production machinery

- **Caching** — LRU + TTL, keyed by the full config signature *and* the
  conversation context, image digests, and an hourly date bucket, so a
  cached result is never served under a different configuration or context.
  Optionally shared across users.
- **Request coalescing** — concurrent identical requests share one LLM call.
- **Circuit breaker** — after N consecutive enhancer failures (default 3),
  enhancement pauses (default 120 s) and originals are forwarded instantly
  instead of each message paying the full timeout. Any success, `!!fresh:`,
  `!!test`, or `!!stats reset` clears it.
- **Observability** — `!!stats` shows counters, cache/override sizes, and
  the most recent failure with its age; `!!test` verifies the configured
  model end-to-end.

## Key valves

| Valve | Default | What it does |
|---|---|---|
| `prefix` / `prefix_mode` | `!!` / `activate` | The trigger prefix and whether it opts in or out |
| `model_id` | *(chat model)* | Dedicated enhancement model |
| `enhancement_style` | `standard` | `concise` / `standard` / `detailed` |
| `llm_timeout_seconds` | `30` | Enhancer call timeout (falls back to original) |
| `failure_threshold` / `failure_cooldown_seconds` | `3` / `120` | Circuit breaker tuning (0 disables) |
| `enable_cache` / `cache_ttl_seconds` / `cache_maxsize` | on / 1h / 128 | Enhancement cache |
| `share_cache_across_users` | off | Higher hit rate vs. per-user isolation |
| `context_messages` / `context_snippet_length` | 6 / 300 | Conversation context given to the enhancer |
| `max_enhanced_length` / `max_expansion_ratio` | 4000 / off | Output guards (reject → keep original) |
| `enable_socratic` / `socratic_threshold` | off / 0.6 | Ask-one-question mode for vague prompts |
| `skip_models` | *(empty)* | Model IDs to never enhance (`gpt-4*` prefix syntax) |
| `extra_intent_patterns` | *(empty)* | Custom intents as JSON, merged over built-ins |
| `show_status` / `show_enhanced_prompt` | off / off | Status line / comparison card on every enhancement |
| `store_original_prompt` | on | Stash the raw prompt in `metadata.original_prompt` |

See the valve descriptions in the file for the full list (bypass prefix
handling, skip heuristics, coalescing, custom system prompts, debug logging).

## Deployment notes

- **Multi-worker:** stats, cache, breaker state, and `!!on`/`!!off` overrides
  are per worker process. On a single-worker deployment (Open WebUI's
  default) this is invisible; with multiple workers, treat them as
  best-effort conveniences.
- **Task safety:** background generations (titles, tags, autocomplete) are
  never enhanced; the guard is robust to builds where `TASKS.DEFAULT` is not
  a stable enum member.
- **Failure posture:** every failure path — LLM error, timeout, empty
  output, guard rejection, unexpected exception — degrades to the user's
  original prompt. The filter never blocks a chat.
