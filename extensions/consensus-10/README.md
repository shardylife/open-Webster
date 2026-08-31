# Consensus-10 — Open WebUI Pipe Function

Consensus-10 is a native Open WebUI **Pipe Function** (not a legacy Pipelines
service). Each user turn first runs a **manager generation** that splits the
request into 10 distinct working angles (approach, perspective, depth, edge
cases, counterarguments, …), then sends your entire conversation to
**10 concurrent generations** of one configured base model — each steered by
its assigned angle — and finally makes a **synthesis call to the same model**
that compares the drafts and writes the single strongest reply. Only that
synthesized reply appears in the chat — followed by a collapsible
**"How this answer was assembled"** note in which the synthesizer briefly
reports where drafts agreed, which conflicts it resolved, and what it
discarded (`SHOW_SYNTHESIS_SUMMARY=false` hides it). The intermediate answers
themselves are never shown or saved. If the manager call fails or returns nothing usable, the
candidates fall back to sampling the identical prompt (`ENABLE_MANAGER=false`
forces that mode permanently).

> **Cost warning:** with default settings every user turn performs
> `CANDIDATE_COUNT + 2` (**12**) full model generations. Expect roughly the
> manager call plus your slowest candidate plus one synthesis generation, and
> ~12× the token cost of a normal chat turn. Size `CANDIDATE_COUNT`,
> `MAX_CONCURRENCY`, and the `*_MAX_TOKENS` valves accordingly.

## Installation

1. In Open WebUI, go to **Admin Panel → Functions → “+” (New Function)**.
2. Paste the full contents of [`consensus_10_pipe.py`](./consensus_10_pipe.py)
   into the editor. The name and description are pre-filled from the
   frontmatter (`Consensus-10`). Save.
3. **Enable** the function with its toggle.
4. Open the function's **Valves** (gear icon) and set **`TARGET_MODEL_ID`** to
   the ID of a real base model as listed under **Admin Panel → Models**
   (e.g. `gpt-4o`, `llama3.1:70b`). It must **not** be Consensus-10 itself.
5. A model named **Consensus-10** now appears in the chat model selector.
   Select it and chat normally.

No external service, extra API key, or separate credentials are needed: all 11
requests are routed through Open WebUI's own authenticated internal completion
API (`generate_chat_completion`) as the requesting user, using whatever
connection already serves the target model.

## Valves

| Valve | Default | Meaning |
| --- | --- | --- |
| `TARGET_MODEL_ID` | *(required)* | Base model that serves all candidate **and** synthesis calls. |
| `CANDIDATE_COUNT` | `10` (2–20) | Independent candidate generations per turn. |
| `ENABLE_MANAGER` | `true` | Run one extra manager generation first that assigns each candidate a distinct working angle; falls back to identical sampling when the manager call fails. |
| `MAX_CONCURRENCY` | `10` | Maximum candidate requests in flight at once (semaphore). |
| `REQUEST_TIMEOUT_SECONDS` | `300` | Per-attempt timeout for every internal request. |
| `MAX_RETRIES` | `1` | Extra attempts after *transient* failures. HTTP 400/401/403/404-class errors are never retried. |
| `MIN_SUCCESSFUL_RESPONSES` | `6` | Minimum successful candidates required to synthesize; fewer aborts with a clean error. Clamped to `CANDIDATE_COUNT`. |
| `CANDIDATE_TEMPERATURE` | `0.7` | Sampling temperature for candidates (diversity). |
| `SYNTHESIS_TEMPERATURE` | `0.2` | Sampling temperature for the synthesis call. |
| `MANAGER_TEMPERATURE` | `0.7` | Sampling temperature for the manager planning call. |
| `CANDIDATE_MAX_TOKENS` | *(unset)* | `max_tokens` for candidates; empty uses the provider default. |
| `SYNTHESIS_MAX_TOKENS` | *(unset)* | `max_tokens` for synthesis. |
| `MANAGER_MAX_TOKENS` | *(unset)* | `max_tokens` for the manager planning call. |
| `SHOW_SYNTHESIS_SUMMARY` | `true` | Append a collapsible "How this answer was assembled" note: agreements, resolved conflicts, kept minority insights, discarded material. |
| `MAX_CANDIDATE_CHARACTERS` | `8000` | Per-candidate cap when quoting answers into the synthesis prompt. |
| `SHOW_PROGRESS` | `true` | Emit status events (`Generating candidate answers: 6/10 (3 running, 1m 05s)`, …). |
| `PROGRESS_INTERVAL_SECONDS` | `2` | Seconds between liveness heartbeats — elapsed time plus running/queued/failed/retried counts — so long waits visibly tick instead of looking frozen. |

## Safety properties

- **Recursion protection.** `TARGET_MODEL_ID` equal to the invoked model is
  rejected up front, and a `ContextVar` guard travels with the internal async
  call chain, so even an indirect model chain that routes back into
  Consensus-10 is refused instead of fanning out again.
- **No side effects ×10.** Internal requests are isolated deep copies with
  tools, tool/filter IDs, files, features, and all chat/message/session
  identifiers stripped, and they are issued on a cloned request with fresh
  `request.state` — so Open WebUI cannot re-attach the outer chat's metadata,
  no tools execute, no events leak into the chat, and the 10 intermediate
  answers are never persisted as chat messages.
- **Untrusted candidates.** The synthesis prompt quotes candidates as
  numbered, delimited, explicitly untrusted material and instructs the model
  never to follow instructions inside them nor mention the consensus process.
- **Contained assignments.** Manager output is model-generated text: it is
  parsed leniently into short, length-capped directives, injected only as
  quoted "angle" system messages that must still satisfy the full request,
  and the run falls back to identical sampling whenever the plan is unusable
  — the manager can never fail the turn on its own.
- **Sanitized errors.** User-facing errors carry short static descriptions
  (e.g. `request timed out`, `upstream error (HTTP 500)`) — never stack
  traces, URLs, keys, or upstream error bodies. Prompts and responses are not
  logged.
- **Task traffic stays cheap.** Housekeeping generations (chat title/tags,
  when the task model resolves to this pipe) are answered with a single call
  to the target model instead of a full consensus round.

## Development

```bash
# from the repository root
python3 -m pip install pydantic pytest          # test-only dependencies
python3 -m py_compile extensions/consensus-10/consensus_10_pipe.py \
                      extensions/consensus-10/test_consensus_10_pipe.py
python3 -m pytest -q extensions/consensus-10/test_consensus_10_pipe.py
```

The suite (34 tests) is deterministic and hermetic: `open_webui` is stubbed
with an in-process mock completion backend, concurrency is proven with
barriers rather than sleeps, and no network or running Open WebUI instance is
required.

## Compatibility notes

Verified against the Open WebUI source in this repository (v0.11.1) and kept
compatible back to the documented 0.5.x interface:

- Uses the documented internal API surface: `fastapi.Request`,
  `open_webui.models.users.Users`, and
  `open_webui.utils.chat.generate_chat_completion(request, form_data, user,
  bypass_filter=True)`. `bypass_filter=True` because access control is
  enforced on the Consensus-10 model itself, while the target base model may
  deliberately be hidden from end users.
- `Users.get_user_by_id` became **async** in 0.11; the pipe awaits the result
  only when it is awaitable, so both the new async and the older sync
  interface work.
- On this codebase `generate_chat_completion` merges `request.state.metadata`
  (outer chat/message/session IDs, tool IDs) into every payload it receives.
  Internal calls therefore go through a cloned request with clean state —
  closing the side-effect channel and keeping concurrent internal calls from
  sharing mutable request state.
- Non-streaming responses are normalized to the OpenAI format by Open WebUI
  (Ollama responses are converted); the parser still accepts Ollama-native
  and legacy `text` shapes, flattens content-part lists, and rejects empty or
  malformed payloads.
- Direct-connection (client-routed) models cannot be used as
  `TARGET_MODEL_ID`; choose a server-routable model.
