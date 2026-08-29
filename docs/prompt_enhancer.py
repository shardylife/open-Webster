"""
title: Production Prompt Enhancer
author: bezz
version: 4.9.0
description: Intent-aware LLM-based filter that enhances user prompts for better responses. Enhancement is opt-in per message via an activation prefix (!! by default — only prefixed messages are enhanced, and the prefix is stripped before sending; a legacy 'bypass' mode inverts this to always-on). Inline per-message flags (!!d,show: prompt) override style, force the comparison card, or skip the cache; !!help shows a command card. A failure circuit breaker pauses enhancement after repeated LLM failures so users never pay serial timeouts. Also features per-user overrides, context-aware TTL caching (optionally shared across users), request coalescing, config-driven intents, enhancement style modes, opt-in Socratic mode (asks one clarifying question when a prompt is too vague to enhance), model skip-lists, LLM timeouts, image-aware enhancement, multilingual skip heuristics (EN/ES/FR/DE/PT/IT + CJK), admin !!stats / !!stats reset commands, an optional expansion guard, original-prompt preservation, and smart skip logic (including already-well-structured prompts).
required_open_webui_version: 0.9.1
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import inspect
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence

from pydantic import BaseModel, Field
from fastapi import Request
from open_webui.utils.chat import generate_chat_completion
from open_webui.models.users import Users

logger = logging.getLogger("prompt_enhancer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    # Avoid duplicate lines when the app's root logger (e.g. uvicorn)
    # also has handlers attached.
    logger.propagate = False
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Open WebUI constants (import-guarded: TASKS has moved between builds)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - depends on host build
    from open_webui.constants import TASKS as _TASKS

    _raw_default_task = getattr(_TASKS, "DEFAULT", "default")
    if callable(_raw_default_task):
        # Some builds define DEFAULT as a lambda, which Enum treats as a
        # method rather than a member — str() on it would yield
        # "<function TASKS.<lambda> ...>". Call it to get the real name.
        try:
            _raw_default_task = _raw_default_task()
        except Exception:
            _raw_default_task = "default"
    # str-Enum members stringify as "TASKS.DEFAULT" on some Python versions,
    # so prefer .value when present.
    DEFAULT_TASK: str = str(getattr(_raw_default_task, "value", _raw_default_task))
except Exception:  # pragma: no cover
    DEFAULT_TASK = "default"

# Task names that mean "ordinary chat" across builds. Anything else is a
# background task (title generation, tags, autocomplete, ...) and must never
# be enhanced.
_DEFAULT_TASK_NAMES: frozenset[str] = frozenset(
    {"", "default", "generation", DEFAULT_TASK}
)


def _task_name(task: Any) -> str:
    """Normalize __task__ (str, str-Enum, or None) to a plain string."""
    if task is None:
        return ""
    return str(getattr(task, "value", task))


# ---------------------------------------------------------------------------
# Lightweight runtime stats
# ---------------------------------------------------------------------------


class _Stats:
    """Thread-safe counters plus a monotonic uptime clock.

    Increments are locked because Open WebUI may serve requests from more than
    one thread/event loop, where ``dict[key] += 1`` is not atomic.
    """

    KEYS: tuple[str, ...] = (
        "enhanced",
        "cache_hits",
        "skipped",
        "bypassed",
        "failed",
        "rejected",
        "timeouts",
        "socratic",
        "cooldown",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {key: 0 for key in self.KEYS}
        self._started = time.monotonic()

    def bump(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def uptime_hours(self) -> float:
        return max(0.0, time.monotonic() - self._started) / 3600.0

    def reset(self) -> None:
        with self._lock:
            for key in self._counts:
                self._counts[key] = 0
            self._started = time.monotonic()


_STATS = _Stats()


class _FailureBreaker:
    """Cooldown circuit breaker for the enhancement LLM.

    After N consecutive failures (errors, timeouts, empty output) the breaker
    opens for a cooldown window during which enhancement is skipped instantly
    instead of making every message pay the full LLM timeout again. Any
    successful enhancement closes it. Thread-safe; monotonic clock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consecutive = 0
        self._open_until = 0.0

    def record_failure(self, threshold: int, cooldown_seconds: float) -> None:
        with self._lock:
            self._consecutive += 1
            if (
                threshold > 0
                and cooldown_seconds > 0
                and self._consecutive >= threshold
            ):
                self._open_until = time.monotonic() + cooldown_seconds

    def record_success(self) -> None:
        with self._lock:
            self._consecutive = 0
            self._open_until = 0.0

    def remaining(self) -> float:
        """Seconds the breaker stays open; 0.0 when closed."""
        with self._lock:
            return max(0.0, self._open_until - time.monotonic())

    def reset(self) -> None:
        self.record_success()


_breaker = _FailureBreaker()


def _stats_summary() -> str:
    counts = _STATS.snapshot()
    return (
        f"enhanced: {counts['enhanced']} | cache hits: {counts['cache_hits']} | "
        f"skipped: {counts['skipped']} | bypassed: {counts['bypassed']} | "
        f"failed: {counts['failed']} | rejected: {counts['rejected']} | "
        f"timeouts: {counts['timeouts']} | socratic: {counts['socratic']} | "
        f"cooldown skips: {counts['cooldown']} | "
        f"cache entries: {len(_prompt_cache)} | "
        f"uptime: {_STATS.uptime_hours():.1f}h"
    )


# ---------------------------------------------------------------------------
# LRU + TTL cache for prompt enhancements
# ---------------------------------------------------------------------------


class _PromptCache:
    """LRU cache keyed by (full config signature + context + prompt) with optional TTL.

    TTL uses a monotonic clock so wall-clock adjustments cannot mass-expire
    (or indefinitely extend) cached entries. All state is lock-protected.
    """

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 0.0) -> None:
        self._cache: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
        self._maxsize = max(1, int(maxsize))
        self._ttl = max(0.0, float(ttl_seconds))
        self._lock = threading.Lock()

    def configure(
        self, maxsize: Optional[int] = None, ttl_seconds: Optional[float] = None
    ) -> None:
        with self._lock:
            if maxsize is not None and maxsize > 0:
                self._maxsize = int(maxsize)
                # Evict immediately on shrink instead of waiting for the next put.
                while len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)
            if ttl_seconds is not None and ttl_seconds >= 0:
                self._ttl = float(ttl_seconds)

    @staticmethod
    def _key(signature: str, prompt: str) -> str:
        raw = f"{signature}\x00{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _expired(self, stamp: float) -> bool:
        return self._ttl > 0 and (time.monotonic() - stamp) > self._ttl

    def get(self, signature: str, prompt: str) -> Optional[str]:
        key = self._key(signature, prompt)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, stamp = entry
            if self._expired(stamp):
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return value

    def put(self, signature: str, prompt: str, enhanced: str) -> None:
        key = self._key(signature, prompt)
        with self._lock:
            self._cache[key] = (enhanced, time.monotonic())
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_prompt_cache = _PromptCache(maxsize=128)


# ---------------------------------------------------------------------------
# Request coalescing
# ---------------------------------------------------------------------------

# In-flight enhancement tasks, keyed by the same key the cache uses.
# Concurrent identical requests share a single LLM call instead of N.
_inflight: "dict[str, asyncio.Task[Optional[str]]]" = {}
_inflight_lock = threading.Lock()


def _discard_inflight(key: str, task: "asyncio.Task[Optional[str]]") -> None:
    # Remove only if this exact task is still registered under the key, and
    # retrieve any stored exception so asyncio doesn't log a spurious
    # "exception was never retrieved" warning.
    with _inflight_lock:
        if _inflight.get(key) is task:
            _inflight.pop(key, None)
    if not task.cancelled():
        task.exception()


async def _coalesce(
    key: str, factory: "Callable[[], Awaitable[Optional[str]]]"
) -> Optional[str]:
    """Run `factory` once for a given key; concurrent callers share the result.

    The work runs as an independent task so a single caller's cancellation can
    never cancel the shared work or poison the other waiters. Each caller only
    detaches itself (via asyncio.shield); cleanup is driven by the task's own
    done-callback, so the in-flight entry can't leak even if every caller goes
    away mid-flight.

    A registered task is only reused when it belongs to the current event loop
    and is still pending: awaiting a task from another loop raises, and a
    finished-but-not-yet-reaped task would replay a stale result/exception.
    """
    loop = asyncio.get_running_loop()
    with _inflight_lock:
        task = _inflight.get(key)
        if task is not None and (task.done() or task.get_loop() is not loop):
            task = None
        if task is None:
            task = loop.create_task(factory())
            _inflight[key] = task
            task.add_done_callback(lambda t: _discard_inflight(key, t))
    return await asyncio.shield(task)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_THINK_TAGS = "think|thinking|reason|reasoning|thought"

# Balanced reasoning blocks (tags may carry attributes).
_THINKING_RE = re.compile(
    rf"<(?P<tag>{_THINK_TAGS})\b[^>]*>.*?</(?P=tag)\s*>"
    r"|"
    r"<?\|begin_of_thought\|>?.*?<?\|end_of_thought\|>?",
    re.DOTALL | re.IGNORECASE,
)

# A lone opening reasoning tag with no matching close (truncated reasoning):
# strip from the tag to end-of-string so raw chain-of-thought never leaks.
_DANGLING_OPEN_THINK_RE = re.compile(
    rf"<(?:{_THINK_TAGS})\b[^>]*>.*\Z" r"|" r"<?\|begin_of_thought\|>?.*\Z",
    re.DOTALL | re.IGNORECASE,
)

# Some providers strip the OPENING tag and emit only the closing one. Anything
# before that close tag is reasoning, not prompt text.
_CLOSE_THINK_PROBE_RE = re.compile(
    rf"</(?:{_THINK_TAGS})\s*>|\|end_of_thought\|", re.IGNORECASE
)
_LEADING_CLOSE_THINK_RE = re.compile(
    rf"\A.*?</(?:{_THINK_TAGS})\s*>" r"|" r"\A.*?<?\|end_of_thought\|>?",
    re.DOTALL | re.IGNORECASE,
)

# The model wrapped its ENTIRE output in a single code fence. Tolerates CRLF,
# single-line fences, and dotted fence infos ("c++", "py3.11").
_FULL_FENCE_RE = re.compile(
    r"\A```[A-Za-z0-9_+.#-]*[ \t]*(?:\r?\n)?(.*?)(?:\r?\n)?```\s*\Z", re.DOTALL
)

_ARTIFACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(?:enhanced\s+prompt|refined\s+prompt|improved\s+prompt|"
        r"here(?:'s| is) (?:the |your )?(?:enhanced|refined|improved) "
        r"(?:prompt|version))[\s:]*",
        re.IGNORECASE,
    ),
    # The politeness word alone is NOT enough to strip: require the
    # conversational tail (punctuation or a "here's ..." continuation) so a
    # prompt legitimately starting with "Surely ..." or "Absolutely ..."
    # is never mangled.
    re.compile(
        r"^\s*(?:sure|certainly|of course|absolutely)\b"
        r"(?:[!,.]\s*|\s+(?=here\b))"
        r"(?:here(?:'s| is)\s*)?"
        r"(?:the (?:enhanced|refined) (?:prompt|version))?[\s:]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*\*\*(?:enhanced|refined|improved) prompt:?\*\*\s*",
        re.IGNORECASE,
    ),
)

# Matching quote pairs the model may wrap the whole prompt in.
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("“", "”"),
    ("«", "»"),
)


def _clean_llm_output(text: str) -> str:
    """Strip reasoning traces, wrapper fences, and preamble artifacts."""
    if not text:
        return ""

    cleaned = _THINKING_RE.sub("", text)

    # Unbalanced CLOSE tag first (opener stripped upstream), then an
    # unbalanced OPEN tag (truncated generation).
    if _CLOSE_THINK_PROBE_RE.search(cleaned):
        cleaned = _LEADING_CLOSE_THINK_RE.sub("", cleaned, count=1)
    cleaned = _DANGLING_OPEN_THINK_RE.sub("", cleaned).strip()

    # Unwrap a whole-output code fence, but only when the inside has no fences
    # of its own — otherwise the fence is likely legitimate prompt content.
    fence = _FULL_FENCE_RE.match(cleaned)
    if fence:
        inner = fence.group(1).strip()
        if inner and "```" not in inner:
            cleaned = inner

    for pattern in _ARTIFACT_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()

    for opener, closer in _QUOTE_PAIRS:
        if len(cleaned) > 2 and cleaned.startswith(opener) and cleaned.endswith(closer):
            inner = cleaned[1:-1].strip()
            if len(inner) > 10:
                cleaned = inner
            break

    return cleaned


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

# Heuristics below are best-effort multilingual (EN/ES/FR/DE/PT/IT plus a few
# CJK markers). Unsupported languages fail safe: messages are simply treated
# as standalone prompts and enhanced normally.
_FOLLOWUP_RE = re.compile(
    r"^(?:"
    # English
    r"now |also |instead |change |modify |update |add |remove |make it |"
    r"try |use |switch |but |and |then |what about |how about |can you also |"
    # Spanish
    r"ahora |tambi[ée]n |adem[áa]s |en vez |cambia |modifica |actualiza |"
    r"agrega |a[ñn]ade |quita |elimina |hazlo |prueba |pero |y |entonces |"
    r"qu[ée] tal |"
    # French
    r"maintenant |aussi |plut[ôo]t |modifie |mets [àa] jour |ajoute |"
    r"supprime |enl[èe]ve |essaie |utilise |mais |et |alors |et si |"
    # German
    r"jetzt |auch |stattdessen |[äa]ndere |aktualisiere |f[üu]ge |entferne |"
    r"mach es |versuche |benutze |verwende |aber |und |dann |was ist mit |"
    # Portuguese
    r"agora |em vez |muda |atualiza |adiciona |remove |tenta |usa |mas |e |"
    r"ent[ãa]o |que tal |"
    # Italian
    r"ora |adesso |anche |invece |aggiorna |aggiungi |rimuovi |prova |ma |"
    r"poi |allora "
    r")",
    re.IGNORECASE,
)

_DEICTIC_RE = re.compile(
    r"\b(it|its|that|this|these|those|them|they|the same|instead|again|"
    r"the above|previous|former|latter|"
    # Spanish
    r"eso|esto|aquello|ese|esa|esos|esas|lo mismo|la misma|de nuevo|"
    r"otra vez|lo anterior|"
    # French
    r"[çc]a|cela|ceci|celui|celle|ceux|le m[êe]me|la m[êe]me|encore|"
    r"[àa] nouveau|ci-dessus|"
    # German
    r"dies|diese[rs]?|dasselbe|das gleiche|nochmal|wieder|oben|"
    # Portuguese
    r"isso|isto|aquilo|o mesmo|a mesma|de novo|novamente|acima|"
    # Italian
    r"questo|quello|lo stesso|la stessa|di nuovo|ancora|sopra" r")\b"
    # CJK deictic markers (no word boundaries in CJK scripts)
    r"|[它这那]|これ|それ|あれ|ここ|そこ",
    re.IGNORECASE,
)

_FOLLOWUP_MAX_WORDS = 40
_DEICTIC_MAX_WORDS = 8


def _has_prior_assistant(messages: Sequence[dict]) -> bool:
    return any(m.get("role") == "assistant" for m in messages[:-1])


def _is_followup(messages: Sequence[dict], user_message: str) -> bool:
    if len(messages) < 3 or not _has_prior_assistant(messages):
        return False
    word_count = len(user_message.split())
    if word_count > _FOLLOWUP_MAX_WORDS:
        return False
    if _FOLLOWUP_RE.match(user_message):
        return True
    # Short message that leans on prior context (deictic reference) — not just
    # any short message, to avoid treating new short questions as follow-ups.
    return word_count <= _DEICTIC_MAX_WORDS and bool(_DEICTIC_RE.search(user_message))


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\s*#{1,3}\s", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*]\s", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)
_ROLE_FRAMING_RE = re.compile(r"\b(you are|act as|role:)\b", re.IGNORECASE)

_WELL_STRUCTURED_MIN_INDICATORS = 3
_WELL_STRUCTURED_MIN_WORDS = 150


def _is_well_structured(text: str) -> bool:
    """True when the prompt already reads like a deliberately engineered prompt."""
    indicators = 0
    if _HEADING_RE.search(text):
        indicators += 1
    if len(_BULLET_RE.findall(text)) >= 3:
        indicators += 1
    if len(_NUMBERED_RE.findall(text)) >= 3:
        indicators += 1
    if "```" in text:
        indicators += 1
    if len(text.split()) > _WELL_STRUCTURED_MIN_WORDS:
        indicators += 1
    if _ROLE_FRAMING_RE.search(text):
        indicators += 1
    return indicators >= _WELL_STRUCTURED_MIN_INDICATORS


# ---------------------------------------------------------------------------
# Code-only detection — skip prompts that are mostly code
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
# A final fence the model never closed ("```python\n...<EOF>").
_UNCLOSED_FENCE_RE = re.compile(r"```[^`]*\Z", re.DOTALL)

_CODE_ONLY_MAX_PROSE_CHARS = 40
_CODE_ONLY_MIN_RATIO = 0.85


def _is_code_only(text: str) -> bool:
    stripped = text.strip()
    total_len = len(stripped)
    if total_len == 0:
        return False

    spans = [m.span() for m in _CODE_BLOCK_RE.finditer(stripped)]
    remainder_start = spans[-1][1] if spans else 0
    unclosed = _UNCLOSED_FENCE_RE.search(stripped, remainder_start)
    if unclosed:
        spans.append(unclosed.span())
    if not spans:
        return False

    # Clamp so overlapping/whitespace effects can't push the ratio past 1.0.
    code_len = min(total_len, sum(end - start for start, end in spans))
    non_code = total_len - code_len
    return (
        non_code < _CODE_ONLY_MAX_PROSE_CHARS
        and (code_len / total_len) > _CODE_ONLY_MIN_RATIO
    )


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

INTENT_DEFS: dict[str, dict[str, Any]] = {
    "debugging": {
        "priority": 95,
        "patterns": [
            r"\b(error|exception|traceback|stack ?trace|bug|crash|panic|segfault)\b",
            r"\b(not working|doesn'?t work|fails?|failing|broken)\b",
            r"\b(debug|debugging|troubleshoot)\b",
        ],
        "hint": (
            "This is a debugging request. The enhanced prompt should ask the AI to:\n"
            "  - Identify the root cause from the error/symptoms described\n"
            "  - Provide the corrected code or exact fix steps\n"
            "  - Explain why the fix works\n"
            "  - Suggest how to prevent similar issues"
        ),
    },
    "coding": {
        "priority": 90,
        "patterns": [
            r"\b(code|program|function|class|method|script)\b",
            r"\b(python|javascript|typescript|java|rust|go(?:lang)?|ruby|php|sql)\b",
            r"\bc\+\+",
            r"\b(refactor|optimize|compile|unit test|pytest|jest)\b",
            r"\b(api|endpoint|rest|graphql)\b",
            r"```[\s\S]*?```",
        ],
        "hint": (
            "This is a coding request. The enhanced prompt should specify:\n"
            "  - Target language/framework if not already stated\n"
            "  - Expected input/output behavior\n"
            "  - Error handling and edge case requirements\n"
            "  - Whether tests, type hints, or documentation are expected\n"
            "  - Code should be in fenced blocks with language tags"
        ),
    },
    "creative": {
        "priority": 75,
        "patterns": [
            # Allows up to two adjectives between the article and noun, so
            # phrasing like "write me a long detailed poem" matches.
            r"\b(write|compose|craft|draft)\s+(?:(?:a|an|me a|me an|some|us a)\s+)?"
            r"(?:[\w-]+\s+){0,2}"
            r"(story|poem|song|novel|chapter|scene|dialogue|script|essay|blog post)\b",
            r"\b(fiction|fantasy|sci[- ]?fi|short story|narrative)\b",
            # Negative lookahead prevents "setting up docker" style false hits.
            r"\b(character|plot|prose|verse)\b|\bsetting\b(?!\s+up)",
        ],
        "hint": (
            "This is a creative writing request. The enhanced prompt should specify:\n"
            "  - Tone and mood (dark, whimsical, serious, etc.)\n"
            "  - Approximate length or structure\n"
            "  - Point of view and tense preferences\n"
            "  - Any thematic elements to emphasize\n"
            "  - Do NOT over-constrain — leave room for creative expression"
        ),
    },
    "analysis": {
        "priority": 70,
        "patterns": [
            r"\b(analy[sz]e|analysis|evaluate|assess|examine|investigate|review)\b",
            r"\b(root cause|implications?|interpret|significance)\b",
        ],
        "hint": (
            "This is an analysis request. The enhanced prompt should ask for:\n"
            "  - Structured breakdown with clear sections\n"
            "  - Evidence-based reasoning with cited sources where possible\n"
            "  - Consideration of counterarguments or alternative interpretations\n"
            "  - A clear, confidence-rated conclusion"
        ),
    },
    "explanation": {
        "priority": 65,
        "patterns": [
            r"\b(explain|describe|clarify|elaborate)\b",
            r"\b(what (is|are|does)|how (does|do|is|are))\b",
            r"\b(teach me|help me understand|walk me through|eli5)\b",
        ],
        "hint": (
            "This is an explanation request. The enhanced prompt should ask for:\n"
            "  - An intuitive overview before diving into details\n"
            "  - Concrete examples and analogies\n"
            "  - Definitions of jargon on first use\n"
            "  - Common misconceptions addressed"
        ),
    },
    "comparison": {
        "priority": 78,
        "patterns": [
            r"\b(compare|comparison|contrast|versus)\b",
            r"\bvs\b\.?",
            r"\b(difference|differences|similarities?) (between|in)\b",
            r"\b(pros? and cons?|trade[- ]offs?)\b",
        ],
        "hint": (
            "This is a comparison request. The enhanced prompt should ask for:\n"
            "  - A structured comparison table on key dimensions\n"
            "  - Specific use-case scenarios for each option\n"
            "  - A clear recommendation with justification"
        ),
    },
    "planning": {
        "priority": 72,
        "patterns": [
            r"\b(plan|roadmap|schedule|timeline|milestones?)\b",
            r"\b(break ?down|step[- ]by[- ]step|steps to)\b",
            r"\b(project|strategy|outline|organi[sz]e)\b",
        ],
        "hint": (
            "This is a planning request. The enhanced prompt should ask for:\n"
            "  - Numbered, sequenced action items\n"
            "  - Effort estimates and dependencies\n"
            "  - Risks, blockers, and milestones\n"
            "  - Clear scope boundaries"
        ),
    },
    "summarization": {
        "priority": 85,
        "patterns": [
            r"\b(summari[sz]e|summary|tl;?dr|brief overview|executive summary)\b",
            r"\b(key points?|main points?|takeaways?|recap)\b",
        ],
        "hint": (
            "This is a summarization request. The enhanced prompt should ask for:\n"
            "  - A one-sentence TL;DR followed by key points\n"
            "  - Faithfulness to the original — no added opinions\n"
            "  - Proportionate length — shorter is usually better"
        ),
    },
    "research": {
        "priority": 68,
        "patterns": [
            r"\b(research|find (info|information|sources?))\b",
            r"\b(sources?|citations?|references?|studies|papers?)\b",
            r"\b(latest|recent|current|state[- ]of[- ]the[- ]art)\b",
        ],
        "hint": (
            "This is a research request. The enhanced prompt should ask for:\n"
            "  - Specific, citable sources where possible\n"
            "  - Clear distinction between established facts and emerging findings\n"
            "  - Explicit flagging of uncertain or unverifiable claims\n"
            "  - Recency-aware information"
        ),
    },
    "brainstorming": {
        "priority": 60,
        "patterns": [
            r"\b(brainstorm|ideate|ideas? for|suggestions? for|come up with)\b",
            r"\b(list (of )?ideas?|options?|alternatives?)\b",
        ],
        "hint": (
            "This is a brainstorming request. The enhanced prompt should ask for:\n"
            "  - A diverse range of ideas (safe to bold)\n"
            "  - Brief rationale for each idea\n"
            "  - A ranked shortlist of the strongest options"
        ),
    },
    "translation": {
        "priority": 88,
        "patterns": [
            r"\btranslate\b",
            r"\b(from|to|into|in) (english|spanish|french|german|italian|portuguese|"
            r"russian|chinese|japanese|korean|arabic|hindi)\b",
        ],
        "hint": (
            "This is a translation request. The enhanced prompt should specify:\n"
            "  - Natural, fluent phrasing over literal word-for-word\n"
            "  - Appropriate register (formal/informal)\n"
            "  - Translator notes for idioms or culturally-bound terms"
        ),
    },
    "problem_solving": {
        "priority": 80,
        "patterns": [
            r"\b(solve|solution|fix|resolve|figure out)\b",
            r"\b(problem|issue|challenge|obstacle)\b",
            r"\b(stuck|can'?t|cannot|unable to)\b",
        ],
        "hint": (
            "This is a problem-solving request. The enhanced prompt should ask for:\n"
            "  - Restated problem and constraints\n"
            "  - Multiple solution options with trade-offs\n"
            "  - A recommended approach with implementation steps\n"
            "  - Verification and rollback strategies"
        ),
    },
    "data_science": {
        "priority": 82,
        "patterns": [
            r"\b(data ?set|dataframe|pandas|numpy|scipy|sklearn|scikit[- ]learn)\b",
            r"\b(machine learning|deep learning|neural net(work)?|model training)\b",
            r"\b(regression|classification|clustering|feature engineering)\b",
            r"\b(visualization|matplotlib|seaborn|plotly)\b",
            r"\b(csv|parquet|json[l]?)\b.*\b(load|read|parse|import)\b",
        ],
        "hint": (
            "This is a data science request. The enhanced prompt should specify:\n"
            "  - Data shape, types, and size if relevant\n"
            "  - Target metric or success criterion\n"
            "  - Whether exploratory analysis or production-ready code is expected\n"
            "  - Visualization or reporting requirements\n"
            "  - Library preferences (pandas, polars, etc.)"
        ),
    },
    "math": {
        "priority": 76,
        "patterns": [
            r"\b(calculate|compute|derive|prove|equation|formula)\b",
            r"\b(integral|derivative|matrix|vector|probability|statistics)\b",
            r"\b(algebra|calculus|geometry|trigonometry|linear algebra)\b",
            r"\b(theorem|proof|conjecture|lemma)\b",
        ],
        "hint": (
            "This is a math request. The enhanced prompt should ask for:\n"
            "  - Step-by-step working shown clearly\n"
            "  - LaTeX formatting for equations where appropriate\n"
            "  - Verification of the answer with a sanity check\n"
            "  - Intuitive explanation alongside formal derivation"
        ),
    },
    "devops": {
        "priority": 74,
        "patterns": [
            r"\b(docker|kubernetes|k8s|helm|terraform|ansible|ci/?cd)\b",
            r"\b(deploy|deployment|infrastructure|pipeline|container)\b",
            r"\b(aws|azure|gcp|cloud|serverless|lambda)\b",
            r"\b(nginx|apache|load ?balancer|reverse ?proxy)\b",
            r"\b(monitoring|observability|prometheus|grafana|logs?)\b",
        ],
        "hint": (
            "This is a DevOps/infrastructure request. The enhanced prompt should specify:\n"
            "  - Target platform and environment constraints\n"
            "  - Security and access control requirements\n"
            "  - Scalability and high-availability needs\n"
            "  - Rollback and disaster recovery considerations\n"
            "  - Configuration as code where applicable"
        ),
    },
    "security": {
        "priority": 86,
        "patterns": [
            r"\b(security|vulnerabilit(y|ies)|exploit|attack|threat)\b",
            r"\b(authentication|authorization|oauth|jwt|rbac)\b",
            r"\b(encrypt(ion)?|hash(ing)?|ssl|tls|certificate)\b",
            r"\b(xss|csrf|injection|owasp|penetration|pentest)\b",
        ],
        "hint": (
            "This is a security request. The enhanced prompt should ask for:\n"
            "  - Threat model and attack vectors considered\n"
            "  - Defense-in-depth approach\n"
            "  - Compliance standards if applicable (SOC2, GDPR, etc.)\n"
            "  - Concrete remediation steps, not just theory\n"
            "  - Code examples with secure defaults"
        ),
    },
    "design": {
        "priority": 66,
        "patterns": [
            r"\b(ui|ux|design|wireframe|mockup|prototype|figma)\b",
            r"\b(layout|color ?scheme|typography|responsive|mobile[- ]first)\b",
            r"\b(accessibility|a11y|wcag|user experience|usability)\b",
        ],
        "hint": (
            "This is a design request. The enhanced prompt should specify:\n"
            "  - Target audience and platform (web, mobile, desktop)\n"
            "  - Accessibility requirements\n"
            "  - Brand guidelines or visual constraints\n"
            "  - Interaction patterns and user flow considerations"
        ),
    },
}

_DEFAULT_INTENT_PRIORITY = 50
_MAX_ACTIVE_INTENTS = 3


def _compile_intent_defs(defs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    compiled: dict[str, dict[str, Any]] = {}
    for name, cfg in defs.items():
        patterns: list[re.Pattern[str]] = []
        for raw_pattern in cfg.get("patterns", []) or []:
            try:
                patterns.append(re.compile(str(raw_pattern), re.IGNORECASE))
            except (re.error, TypeError) as exc:
                # Keep the intent alive on the remaining valid patterns
                # instead of silently dropping the whole definition.
                logger.warning(
                    "Intent %r: dropping invalid pattern %r (%s)",
                    name,
                    raw_pattern,
                    exc,
                )
        if not patterns:
            continue
        try:
            priority = float(cfg.get("priority", _DEFAULT_INTENT_PRIORITY))
        except (TypeError, ValueError):
            priority = float(_DEFAULT_INTENT_PRIORITY)
        compiled[name] = {
            "priority": priority,
            "patterns": tuple(patterns),
            "hint": str(cfg.get("hint", "") or ""),
        }
    return compiled


COMPILED_INTENTS: dict[str, dict[str, Any]] = _compile_intent_defs(INTENT_DEFS)


# Memoized compilation of admin-supplied custom intents (keyed by raw string).
_MEMO_LIMIT = 8
_custom_intent_cache: "dict[str, dict[str, dict[str, Any]]]" = {}
_custom_intent_lock = threading.Lock()


def _parse_extra_intents(raw: str) -> dict[str, dict[str, Any]]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    with _custom_intent_lock:
        cached = _custom_intent_cache.get(raw)
    if cached is not None:
        return cached

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("extra_intent_patterns must be a JSON object")
        compiled = _compile_intent_defs(
            {
                str(name): {
                    "priority": cfg.get("priority", _DEFAULT_INTENT_PRIORITY),
                    "patterns": cfg.get("patterns", []),
                    "hint": cfg.get("hint", ""),
                }
                for name, cfg in data.items()
                if isinstance(cfg, dict)
            }
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Invalid extra_intent_patterns, ignoring: %s", exc)
        compiled = {}

    with _custom_intent_lock:
        # Bound the memo cache: admins editing the valve repeatedly would
        # otherwise grow this dict without limit.
        if len(_custom_intent_cache) >= _MEMO_LIMIT:
            _custom_intent_cache.clear()
        _custom_intent_cache[raw] = compiled
    return compiled


TRIVIAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*(hi|hello|hey|yo|sup|howdy|greetings|hiya)( there)?[\s!.?]*$",
        r"^\s*(thanks?|thank you|thx|ty|cheers)[\s!.?]*$",
        r"^\s*(ok|okay|k|cool|nice|great|awesome|got it|alright)[\s!.?]*$",
        r"^\s*(bye|goodbye|cya|see ya|later)[\s!.?]*$",
        r"^\s*(yes|no|yep|nope|yeah|nah|sure|maybe)[\s!.?]*$",
        r"^\s*(test|ping|hello world)[\s!.?]*$",
        # Greetings — ES/FR/DE/PT/IT/RU/zh/ja/ko
        r"^\s*(hola|buenas|bonjour|salut|coucou|hallo|servus|moin|ciao|"
        r"ol[áa]|oi|привет|здравствуйте|你好|您好|こんにちは|やあ|안녕|안녕하세요)"
        r"[\s!.?!?。]*$",
        # Thanks
        r"^\s*(gracias|muchas gracias|merci|merci beaucoup|danke|danke sch[öo]n|"
        r"obrigad[oa]|grazie|grazie mille|спасибо|谢谢|多谢|ありがとう|"
        r"ありがとうございます|감사합니다)[\s!.?!?。]*$",
        # Acknowledgements / yes / no
        r"^\s*(s[íi]|claro|vale|oui|non|d'accord|ja|nein|sim|n[ãa]o|certo|"
        r"да|нет|хорошо|是|不是|好的|はい|いいえ|了解|네|아니요)[\s!.?!?。]*$",
        # Goodbyes
        r"^\s*(adi[óo]s|hasta luego|au revoir|[àa] bient[ôo]t|tsch[üu]ss|"
        r"auf wiedersehen|tchau|at[ée] logo|arrivederci|пока|до свидания|"
        r"再见|さようなら|またね|안녕히 가세요)[\s!.?!?。]*$",
    )
)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_STYLE_INSTRUCTIONS: dict[str, str] = {
    "concise": (
        "\nStyle: CONCISE — enhance the prompt minimally. Add only the most "
        "critical missing details. Keep the result close to the original length. "
        "Prefer tightening over expanding."
    ),
    "standard": "",
    "detailed": (
        "\nStyle: DETAILED — produce a thorough, comprehensive enhanced prompt. "
        "Expand the original into a fully specified request that a capable model "
        "could execute with minimal ambiguity.\n"
        "\n"
        "Add the following wherever they strengthen the prompt:\n"
        "- Context & background: state the implied domain, audience, and purpose so "
        "the model isn't guessing at intent.\n"
        "- Constraints: scope boundaries, things to include and explicitly exclude, "
        "tone, length, and any technical or factual limits.\n"
        "- Format requirements: the expected output structure (prose, table, JSON, "
        "code block, sections), ordering, and any required fields or headings.\n"
        "- Edge cases: ambiguous inputs, empty or malformed data, boundary values, "
        "and how the model should handle them rather than failing silently.\n"
        "- Quality criteria: what 'good' looks like — accuracy, completeness, "
        "internal consistency, and any standards the output should be judged against.\n"
        "- Examples: where they clarify expectations, include a brief input→output "
        "example, but keep it illustrative rather than exhaustive.\n"
        "\n"
        "The result can be significantly longer than the original, up to 3-4x "
        "(this supersedes the base guidance to stay within 2-3x), but "
        "every addition must add value — never pad with filler, restate the obvious, "
        "or invent requirements the user clearly did not intend. Preserve the "
        "original's core intent, voice, and any specific terminology; you are "
        "sharpening and completing the request, not replacing it. If the original is "
        "already detailed, refine and tighten rather than inflate. Do not answer the "
        "prompt or add meta-commentary — output only the enhanced prompt itself."
    ),
}

VALID_STYLES: tuple[str, ...] = ("concise", "standard", "detailed")
DEFAULT_STYLE = "standard"

BASE_SYSTEM_PROMPT = """\
You are an expert prompt engineer. Your task is to enhance the given prompt \
by making it more detailed, specific, and effective while preserving the \
user's original intent and voice.

Guidelines:
- Return ONLY the enhanced prompt. No headers, no "Enhanced Prompt:", no \
  wrapper text, no introductory phrases.
- Preserve the original language (English, Spanish, etc.).
- Make the prompt more specific and actionable without changing what the user \
  is asking for.
- Add details about expected format, depth, constraints, and quality where \
  the original is vague.
- Keep the enhanced prompt concise — improve clarity, don't add bloat. The \
  result should be at most 2-3x the original length.
- Preserve code blocks, URLs, and technical terms exactly as written.
- Do not add requirements the user didn't imply.

Examples:

Original: "Explain how DNS works"
Enhanced: "Explain how DNS works, starting with a plain-language overview of what happens when a user types a URL into their browser. Cover the role of recursive resolvers, root servers, TLD servers, and authoritative nameservers. Include a concrete example tracing the resolution of a real domain name. Define any technical terms on first use."

Original: "Write a Python script to rename files"
Enhanced: "Write a Python script that batch-renames files in a given directory. Accept a source directory path and a naming pattern (e.g., prefix + sequential number) as command-line arguments. Handle edge cases: empty directories, permission errors, and filename collisions. Use pathlib and argparse. Include type hints and a brief usage example in a docstring."

Original: "Compare React and Vue"
Enhanced: "Compare React and Vue.js for building a mid-sized single-page application. Cover: learning curve, ecosystem maturity, performance characteristics, state management approaches, TypeScript support, and community/job market. Use a comparison table for key dimensions, then provide a scenario-based recommendation for different team profiles."

IMPORTANT: Return ONLY the enhanced prompt text. Nothing else.\
"""

FOLLOWUP_SYSTEM_PROMPT = """\
You are an expert prompt engineer. The user is sending a follow-up message in \
an ongoing conversation. Your task is to enhance this follow-up while keeping \
it contextual — do NOT try to make it a standalone prompt.

Guidelines:
- Return ONLY the enhanced follow-up. No headers, no wrapper text.
- Keep the conversational tone — this is a continuation, not a new request.
- Add specificity about what "it", "that", "this" refer to when clear from context.
- If the user is asking for a modification, clarify what aspects to change and \
  what to preserve.
- Keep it brief — follow-ups should stay concise.
- Do not repeat information from earlier in the conversation.

Example:
Context: User asked for a Python CSV parser, assistant provided one.
Original follow-up: "now add error handling"
Enhanced follow-up: "Add robust error handling to the CSV parser: handle missing files (FileNotFoundError), malformed rows (skip and log them), and encoding issues (try UTF-8, fall back to latin-1). Add type hints to any new functions."

IMPORTANT: Return ONLY the enhanced follow-up text. Nothing else.\
"""

WELL_STRUCTURED_SYSTEM_PROMPT = """\
You are an expert prompt engineer. The user has written a detailed, \
well-structured prompt. It needs only light refinement — do NOT restructure \
or significantly expand it.

Guidelines:
- Return ONLY the refined prompt. No headers, no wrapper text.
- Make minimal, high-impact improvements only: fill obvious gaps, sharpen \
  vague requirements, fix ambiguities.
- Preserve the user's structure, formatting, and voice exactly.
- Do NOT add sections, change the organization, or significantly increase \
  the length.
- If the prompt is already excellent, return it nearly unchanged.

IMPORTANT: Return ONLY the refined prompt text. Nothing else.\
"""


# ---------------------------------------------------------------------------
# Enhancement context
# ---------------------------------------------------------------------------


@dataclass
class EnhancementContext:
    """Single source of truth for everything that affects the enhanced output.

    Also used to derive the cache signature, so the cache can never serve a
    result produced under a different configuration. The output guards
    (max_enhanced_length / max_expansion_ratio) are part of the signature so
    tightening them invalidates cached results that would no longer pass.
    """

    style: str = DEFAULT_STYLE
    intents: list[str] = field(default_factory=list)
    is_followup: bool = False
    is_well_structured: bool = False
    custom_system_prompt: str = ""
    additional_instructions: str = ""
    model: str = ""
    temperature: float = 0.7
    user_id: str = ""
    max_enhanced_length: int = 0
    max_expansion_ratio: float = 0.0

    def signature(self) -> str:
        payload = {
            "v": 8,
            "style": self.style,
            "intents": sorted(self.intents),
            "followup": self.is_followup,
            "structured": self.is_well_structured,
            "custom": self.custom_system_prompt.strip(),
            "extra": self.additional_instructions.strip(),
            "model": self.model,
            "temp": round(float(self.temperature), 3),
            "uid": self.user_id,
            "max_len": int(self.max_enhanced_length),
            "max_ratio": round(float(self.max_expansion_ratio), 3),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Intent scoring
# ---------------------------------------------------------------------------

_SHORT_PROMPT_CHARS = 30


def _detect_intents_scored(
    text: str,
    threshold: float = 0.55,
    intents: Optional[dict[str, dict[str, Any]]] = None,
) -> tuple[list[str], float]:
    """Detect intents and also report the best raw confidence seen.

    The max confidence (even when below threshold) lets Socratic mode
    distinguish "clearly about nothing we recognize" from "almost matched
    a known intent", which suppresses unnecessary clarifying questions.
    """
    catalog = COMPILED_INTENTS if intents is None else intents
    if not catalog or not text:
        return [], 0.0

    results: list[tuple[str, float]] = []
    total_words = max(1, len(text.split()))
    # Graduated ramp for short prompts: a constant floor here used to cap
    # confidence below the default threshold, silently disabling intent
    # detection for short prompts like "fix this bug".
    length_factor = (
        (0.7 + 0.3 * (len(text) / _SHORT_PROMPT_CHARS))
        if len(text) < _SHORT_PROMPT_CHARS
        else 1.0
    )

    for intent, cfg in catalog.items():
        total_hits = 0
        unique_hits = 0
        for pattern in cfg["patterns"]:
            found = pattern.findall(text)
            if found:
                unique_hits += 1
                total_hits += len(found)
        if total_hits == 0:
            continue

        sqrt_norm = min(1.0, (total_hits * 5) / (total_words**0.5))
        base = min(1.0, 0.35 + 0.15 * unique_hits + 0.05 * min(total_hits, 5))
        base = (base + sqrt_norm) / 2.0
        priority = cfg["priority"] / 100.0
        confidence = min(1.0, base * 0.7 + priority * 0.3) * length_factor
        results.append((intent, round(confidence, 3)))

    if not results:
        return [], 0.0

    results.sort(key=lambda item: (-item[1], item[0]))
    max_conf = results[0][1]
    selected = [name for name, conf in results if conf >= threshold]
    return selected[:_MAX_ACTIVE_INTENTS], max_conf


def _detect_intents(
    text: str,
    threshold: float = 0.55,
    intents: Optional[dict[str, dict[str, Any]]] = None,
) -> list[str]:
    return _detect_intents_scored(text, threshold, intents)[0]


def _is_trivial(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(p.match(stripped) for p in TRIVIAL_PATTERNS)


# ---------------------------------------------------------------------------
# Socratic mode — detect vague prompts and ask, don't guess
# ---------------------------------------------------------------------------

# Trailing politeness/filler that shouldn't count as a real object after a
# deictic ("fix it please" is still objectless).
_POLITENESS_RE = re.compile(
    r"\b(please|pls|plz|thanks|thank you|thx|asap|now|quickly|today|"
    r"for me|real quick)\b",
    re.IGNORECASE,
)

# Underspecified verb + terminal deictic: "fix it", "improve this", "clean
# that up". The negative lookahead keeps "fix this bug" (deictic followed by
# a real object) from matching. Run against politeness-stripped text.
_VAGUE_VERB_RE = re.compile(
    r"\b(fix|improve|change|update|optimi[sz]e|refactor|rewrite|review|debug|"
    r"summari[sz]e|translate|clean|sort|check|handle|make)\s+"
    r"(it|this|that|these|those|them)(?:\s+(?:up|out))?\b(?!\s+\w)",
    re.IGNORECASE,
)

# Generic asks that carry almost no task information on their own.
_GENERIC_ASK_RE = re.compile(
    r"\b(any ideas|any thoughts|what should i do|what do you think|"
    r"can you help( me)?|help me out|make it better|thoughts\??|"
    r"what would you do|where do i start|how do i start)\b",
    re.IGNORECASE,
)

# Bare deictic reference anywhere in the prompt.
_DEICTIC_REF_RE = re.compile(r"\b(it|this|that|these|those|them)\b", re.IGNORECASE)

_VAGUE_SHORT_WORDS = 8


def _strip_politeness(text: str) -> str:
    stripped = _POLITENESS_RE.sub(" ", text)
    return re.sub(r"\s+", " ", stripped).strip(" \t.!?,")


@dataclass(frozen=True)
class VaguenessAssessment:
    """Result of a single vagueness pass over a prompt."""

    score: float
    has_bare_deictic_object: bool


def assess_vagueness(text: str, max_intent_conf: float = 0.0) -> VaguenessAssessment:
    """Score 0.0-1.0 for how underspecified a prompt is (single pass).

    Combines cheap signals: an underspecified verb aimed at a bare deictic,
    generic help-me phrasing, deictic references, and brevity. Near-miss
    intent confidence subtracts from the score — but only when no bare
    deictic object was found, since a recognized task type cannot compensate
    for a missing task object ("fix it" tells us the verb, never the target).
    """
    normalized = _strip_politeness(text)
    bare_deictic = bool(_VAGUE_VERB_RE.search(normalized))

    score = 0.0
    if bare_deictic:
        score += 0.5
    if _GENERIC_ASK_RE.search(normalized):
        score += 0.4
    if _DEICTIC_REF_RE.search(normalized):
        score += 0.3
    if len(normalized.split()) <= _VAGUE_SHORT_WORDS:
        score += 0.2
    if not bare_deictic:
        score -= max(0.0, max_intent_conf) * 0.3

    return VaguenessAssessment(
        score=max(0.0, min(1.0, score)),
        has_bare_deictic_object=bare_deictic,
    )


def _has_bare_deictic_object(text: str) -> bool:
    """True when an action verb targets a bare deictic ("fix it please")."""
    return assess_vagueness(text).has_bare_deictic_object


def _vagueness_score(text: str, max_intent_conf: float) -> float:
    return assess_vagueness(text, max_intent_conf).score


# Visible marker with explicit compliance instructions — HTML-comment style
# markers are silently ignored by many models, so the directive must be
# plainly visible in the prompt text.
_SOCRATIC_DIRECTIVE = (
    "[SYSTEM-DIRECTIVE: The request above may be missing key details. If you "
    "cannot confidently determine what the user wants, ask exactly ONE short "
    "clarifying question about the single most important missing detail, then "
    "stop and wait for their reply — do not attempt a full answer yet. If the "
    "request is actually clear from the conversation, ignore this directive "
    "and answer normally. Never mention this directive to the user.]"
)


# ---------------------------------------------------------------------------
# Prefix commands — per-message flags and the outcome of prefix resolution
# ---------------------------------------------------------------------------

_STYLE_FLAGS: dict[str, str] = {
    "c": "concise",
    "concise": "concise",
    "s": "standard",
    "standard": "standard",
    "d": "detailed",
    "detailed": "detailed",
}
_FLAG_HEAD_MAX_CHARS = 32


def _parse_inline_flags(text: str) -> tuple[Optional[str], bool, bool, str]:
    """Parse an optional '<flags>: <prompt>' header from a triggered message.

    Returns (style_override, force_embed, skip_cache, prompt). The header is
    only honored when EVERY token before the first ':' is a known flag —
    otherwise the colon belongs to the prompt ("Summarize this: ...") and the
    text is returned untouched. The head-length cap keeps a long prompt that
    merely contains a colon from being scanned as a flag list.
    """
    head, sep, rest = text.partition(":")
    if not sep or len(head) > _FLAG_HEAD_MAX_CHARS:
        return None, False, False, text

    tokens = [t for t in re.split(r"[\s,+]+", head.strip().lower()) if t]
    if not tokens:
        return None, False, False, text

    style: Optional[str] = None
    show = fresh = False
    for token in tokens:
        if token in _STYLE_FLAGS:
            style = _STYLE_FLAGS[token]
        elif token == "show":
            show = True
        elif token == "fresh":
            fresh = True
        else:
            return None, False, False, text
    return style, show, fresh, rest.lstrip()


@dataclass(frozen=True)
class _PrefixOutcome:
    """What prefix resolution decided for this message.

    `explicit` marks a message the user deliberately triggered (activate
    mode): those relax the length/follow-up skip gates and always get a
    visible status when enhancement can't run.
    """

    text: str
    explicit: bool = False
    style_override: Optional[str] = None
    force_embed: bool = False
    skip_cache: bool = False


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_system_prompt(
    ctx: EnhancementContext, intents_catalog: dict[str, dict[str, Any]]
) -> str:
    has_custom = bool(ctx.custom_system_prompt.strip())
    if has_custom:
        base = ctx.custom_system_prompt.strip()
    elif ctx.is_followup:
        base = FOLLOWUP_SYSTEM_PROMPT
    elif ctx.is_well_structured:
        base = WELL_STRUCTURED_SYSTEM_PROMPT
    else:
        base = BASE_SYSTEM_PROMPT

    parts: list[str] = [base]

    if not has_custom:
        style_instruction = _STYLE_INSTRUCTIONS.get(ctx.style, "")
        if style_instruction:
            parts.append(style_instruction)

        hints = [
            (intents_catalog.get(intent) or {}).get("hint", "")
            for intent in ctx.intents
        ]
        hints = [hint for hint in hints if hint]
        if hints:
            parts.append("Intent-specific guidance:\n" + "\n\n".join(hints))

    if ctx.additional_instructions.strip():
        parts.append(
            "Additional instructions from the user:\n"
            + ctx.additional_instructions.strip()
        )

    return "\n\n".join(parts)


def _message_text(content: Any) -> str:
    """Flatten a message content field (str or multimodal parts) to plain text.

    Joins ALL text parts (unlike Open WebUI's get_last_user_message, which
    returns only the first) so a multi-text-part message is enhanced — and
    later replaced — without silently losing content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _build_context_block(
    messages: Sequence[dict], max_messages: int, snippet_length: int
) -> str:
    """Render recent conversation turns as a deterministic text block.

    Used both inside the enhancement prompt and as part of the cache key, so
    a cached result is never served under a different conversation context.
    """
    if max_messages <= 0 or len(messages) < 2:
        return ""

    # Filter to conversational roles FIRST, then take the tail. Slicing
    # before filtering let system/tool messages consume the max_messages
    # budget, silently shrinking the real conversation context.
    convo = [m for m in messages[:-1] if m.get("role") in ("user", "assistant")]
    limit = max(4, int(snippet_length))

    lines: list[str] = []
    for msg in convo[-max_messages:]:
        role = str(msg.get("role", "user")).upper()
        snippet = _message_text(msg.get("content", "")).strip()
        if not snippet:
            continue
        if len(snippet) > limit:
            snippet = snippet[: limit - 3] + "..."
        lines.append(f"{role}: {snippet}")
    return "\n".join(lines)


def _image_signature(messages: Sequence[dict]) -> tuple[int, str]:
    """Count image attachments on the last user message and digest their refs.

    The digest feeds the cache key so the same text with a different image
    never reuses a cached enhancement. Only the last user message is checked —
    that is the message being enhanced.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            return 0, ""
        digest = hashlib.sha256()
        count = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", "")
                if isinstance(url, dict):
                    url = url.get("url", "")
                digest.update(str(url).encode("utf-8", "ignore"))
                count += 1
        return count, (digest.hexdigest()[:16] if count else "")
    return 0, ""


def _build_user_prompt(
    user_message: str,
    context_block: str,
    tool_ids: Optional[Sequence[str]] = None,
    include_datetime: bool = True,
    image_count: int = 0,
) -> str:
    parts: list[str] = []

    if context_block:
        parts.append('Conversation context:\n"""\n' + context_block + '\n"""')

    meta_bits: list[str] = []
    if include_datetime:
        # Timezone-aware so the enhancer isn't guessing at a naive timestamp.
        now = dt.datetime.now().astimezone()
        meta_bits.append(f"Current date: {now.strftime('%Y-%m-%d %H:%M %Z')}".strip())
    if tool_ids:
        meta_bits.append(f"Available tools: {', '.join(tool_ids)}")
    if image_count:
        meta_bits.append(
            f"The user attached {image_count} image(s) to this message. You "
            "cannot see them. Keep every reference to the attached image(s) "
            "(e.g. 'this image', 'the screenshot') as a reference — do not "
            "invent, assume, or describe image content, and do not rewrite "
            "the prompt as if no image were attached."
        )
    if meta_bits:
        parts.append("\n".join(meta_bits))

    parts.append(f'Prompt to enhance:\n"""{user_message}"""')
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Model / message plumbing
# ---------------------------------------------------------------------------


def _resolve_model(
    valves_model_id: Optional[str], model_info: Optional[dict], body: dict
) -> str:
    if valves_model_id:
        return valves_model_id.strip()
    if model_info:
        base = model_info.get("base_model_id") or ""
        if base:
            return str(base)
        info = model_info.get("info")
        if isinstance(info, dict) and info.get("id"):
            return str(info["id"])
        if model_info.get("id"):
            return str(model_info["id"])
    return str(body.get("model") or "")


def _model_skipped(model_id: str, patterns_str: str) -> bool:
    """True if model_id matches the skip-list (exact, or prefix via trailing *)."""
    if not model_id:
        return False
    for token in re.split(r"[,\n]", patterns_str or ""):
        token = token.strip()
        if not token or token.startswith("#"):
            continue
        if token.endswith("*"):
            if model_id.startswith(token[:-1]):
                return True
        elif model_id == token:
            return True
    return False


def _set_last_user_message_text(messages: list[dict], new_text: str) -> None:
    """Replace the text of the last user message, preserving non-text parts.

    Extra text parts are dropped rather than blanked: their content was
    already folded into new_text by _message_text, and leftover empty text
    parts are rejected outright by some providers.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")

        if isinstance(content, list):
            rebuilt: list[Any] = []
            replaced = False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    if replaced:
                        continue  # drop trailing fragments of the original prompt
                    part = {**part, "text": new_text}
                    replaced = True
                rebuilt.append(part)
            if not replaced:
                # No text part — prepend one so image/other parts survive.
                rebuilt.insert(0, {"type": "text", "text": new_text})
            message["content"] = rebuilt
        else:
            message["content"] = new_text
        return


def _stash_original_prompt(body: dict, original: str) -> None:
    """Store the untouched prompt without assuming body['metadata'] is a dict.

    setdefault() would return an existing None instead of replacing it, so a
    caller that passes {"metadata": None} would blow up on item assignment.
    """
    meta = body.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        body["metadata"] = meta
    meta["original_prompt"] = original


# Compiled custom skip patterns, memoized so invalid regexes are reported
# once per configuration instead of on every request.
_skip_pattern_cache: "dict[str, tuple[re.Pattern[str], ...]]" = {}
_skip_pattern_lock = threading.Lock()


def _compile_skip_patterns(raw: str) -> tuple[re.Pattern[str], ...]:
    raw = raw or ""
    with _skip_pattern_lock:
        cached = _skip_pattern_cache.get(raw)
    if cached is not None:
        return cached

    compiled: list[re.Pattern[str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            compiled.append(re.compile(line, re.IGNORECASE))
        except re.error as exc:
            logger.warning("Ignoring invalid custom skip pattern %r (%s)", line, exc)

    result = tuple(compiled)
    with _skip_pattern_lock:
        if len(_skip_pattern_cache) >= _MEMO_LIMIT:
            _skip_pattern_cache.clear()
        _skip_pattern_cache[raw] = result
    return result


def _matches_custom_skip(text: str, patterns_str: str) -> bool:
    if not (patterns_str or "").strip():
        return False
    return any(pattern.search(text) for pattern in _compile_skip_patterns(patterns_str))


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _extract_content(response: Any) -> Optional[str]:
    """Defensively pull assistant content from a chat-completion response.

    Returns None for streaming responses, error payloads, or any unexpected
    shape so callers fall back to the original prompt instead of raising.
    """
    data: Any = response
    if data is None:
        return None

    if not isinstance(data, dict):
        # pydantic models / objects exposing a dict conversion.
        for attr in ("model_dump", "dict"):
            converter = getattr(data, attr, None)
            if callable(converter):
                try:
                    data = converter()
                except Exception:  # noqa: BLE001 - degrade, don't crash
                    return None
                break

    if not isinstance(data, dict):
        # Starlette JSONResponse-style objects.
        raw_body = getattr(response, "body", None)
        if isinstance(raw_body, (bytes, bytearray)):
            try:
                data = json.loads(raw_body)
            except (ValueError, TypeError):
                return None

    if not isinstance(data, dict):
        return None

    if data.get("error"):
        logger.warning(
            "Enhancement LLM returned an error payload: %s", str(data["error"])[:300]
        )
        return None

    try:
        choices = data.get("choices")
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return _message_text(content)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return None


# Sentinel returned by _call_llm on timeout so the retry wrapper can
# distinguish "timed out" (do not retry) from "failed" (retry once).
_TIMED_OUT = object()

# Display-only clamp so an 8k-character prompt can't produce a giant embed.
_EMBED_TEXT_LIMIT = 4000


def _clamp_for_display(text: str, limit: int = _EMBED_TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# HTML embed builders
# ---------------------------------------------------------------------------


def _chip(label: str) -> str:
    return (
        '<span style="display:inline-block;padding:2px 9px;border-radius:999px;'
        "background:rgba(99,102,241,0.14);border:1px solid rgba(99,102,241,0.30);"
        'font-size:11px;font-weight:600;color:inherit;opacity:0.85;">'
        f"{_escape_html(label)}</span>"
    )


def _stats_embed_html() -> str:
    counts = _STATS.snapshot()
    rows: tuple[tuple[str, Any], ...] = (
        ("Enhanced", counts["enhanced"]),
        ("Cache hits", counts["cache_hits"]),
        ("Skipped", counts["skipped"]),
        ("Bypassed", counts["bypassed"]),
        ("Failed", counts["failed"]),
        ("Rejected (guards)", counts["rejected"]),
        ("LLM timeouts", counts["timeouts"]),
        ("Socratic questions", counts["socratic"]),
        ("Cooldown skips", counts["cooldown"]),
        ("Cache entries", len(_prompt_cache)),
        ("Uptime (hours)", f"{_STATS.uptime_hours():.1f}"),
    )
    cells = "".join(
        '<div style="padding:8px 16px;display:flex;justify-content:space-between;'
        'border-bottom:1px solid rgba(128,128,128,0.12);">'
        f'<span style="opacity:0.65;">{_escape_html(label)}</span>'
        f"<strong>{_escape_html(str(value))}</strong></div>"
        for label, value in rows
    )
    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
        "border:1px solid rgba(128,128,128,0.25);border-radius:12px;overflow:hidden;"
        'margin:8px 0;font-size:13px;color:inherit;">'
        '<div style="background:rgba(128,128,128,0.08);padding:8px 16px;'
        "border-bottom:1px solid rgba(128,128,128,0.2);font-size:11px;font-weight:600;"
        'opacity:0.7;text-transform:uppercase;letter-spacing:0.5px;">'
        "Prompt Enhancer — Stats</div>"
        f"{cells}</div>"
    )


def _help_embed_html(prefix: str, mode: str, style: str, is_admin: bool) -> str:
    rows: list[tuple[str, str]] = []
    if mode == "activate":
        rows.append((f"{prefix}<prompt>", "Enhance this message (prefix stripped before sending)"))
        rows.extend(
            [
                (f"{prefix}c: <prompt>", "Concise style for this message (or 'concise:')"),
                (f"{prefix}s: <prompt>", "Standard style for this message (or 'standard:')"),
                (f"{prefix}d: <prompt>", "Detailed style for this message (or 'detailed:')"),
                (f"{prefix}show: <prompt>", "Also display the before/after comparison card"),
                (f"{prefix}fresh: <prompt>", "Ignore any cached result and re-enhance"),
                (f"{prefix}d,show: <prompt>", "Flags combine with commas"),
            ]
        )
    else:
        rows.append(("<prompt>", "Every message is enhanced automatically"))
        rows.append((f"{prefix}<prompt>", "Send this message WITHOUT enhancement"))
    rows.append((f"{prefix}help", "Show this help"))
    if is_admin:
        rows.append((f"{prefix}stats / {prefix}stats reset", "Runtime counters / reset (admin)"))

    cells = "".join(
        '<div style="padding:8px 16px;display:flex;justify-content:space-between;'
        'gap:16px;border-bottom:1px solid rgba(128,128,128,0.12);">'
        '<code style="font-size:12px;white-space:nowrap;">'
        f"{_escape_html(cmd)}</code>"
        f'<span style="opacity:0.7;text-align:right;">{_escape_html(desc)}</span></div>'
        for cmd, desc in rows
    )
    footer = (
        '<div style="padding:8px 16px;font-size:11px;opacity:0.6;">'
        f"Mode: {_escape_html(mode)} · default style: {_escape_html(style)}</div>"
    )
    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
        "border:1px solid rgba(128,128,128,0.25);border-radius:12px;overflow:hidden;"
        'margin:8px 0;font-size:13px;color:inherit;">'
        '<div style="background:rgba(128,128,128,0.08);padding:8px 16px;'
        "border-bottom:1px solid rgba(128,128,128,0.2);font-size:11px;font-weight:600;"
        'opacity:0.7;text-transform:uppercase;letter-spacing:0.5px;">'
        "Prompt Enhancer — Commands</div>"
        f"{cells}{footer}</div>"
    )


def _build_comparison_embed(
    *,
    original: str,
    enhanced: str,
    intents: Sequence[str],
    followup: bool,
    well_structured: bool,
    style: str,
    cached: bool,
    elapsed_ms: Optional[int],
    model: str,
) -> str:
    intent_label = f' — {_escape_html(", ".join(intents))}' if intents else ""
    mode_label = (
        " (follow-up)" if followup else " (light refinement)" if well_structured else ""
    )
    style_label = f" [{_escape_html(style)}]" if style != DEFAULT_STYLE else ""
    cache_label = " (cached)" if cached else ""

    # --- metrics footer chips (measured on the full, unclamped text) ---
    o_chars, e_chars = len(original), len(enhanced)
    o_words, e_words = len(original.split()), len(enhanced.split())
    ratio = e_chars / o_chars if o_chars else 0.0
    if ratio >= 1.0:
        growth = f"{ratio:.1f}× longer"
    elif ratio > 0:
        growth = f"{(1 - ratio) * 100:.0f}% shorter"
    else:
        growth = ""

    # Strip only the provider path (e.g. "openai/gpt-4.1" -> "gpt-4.1").
    # Do NOT split on ".", or model names with dots get mangled
    # ("gpt-4.1" -> "1", "qwen2.5:14b" -> "5:14b").
    short_model = model.rsplit("/", 1)[-1] if model else ""

    chips = [
        _chip(f"{o_chars} → {e_chars} chars"),
        _chip(f"{o_words} → {e_words} words"),
    ]
    if growth:
        chips.append(_chip(growth))
    if elapsed_ms is not None:
        chips.append(_chip(f"{elapsed_ms} ms"))
    elif cached:
        chips.append(_chip("from cache"))
    if intents:
        chips.append(_chip("🎯 " + ", ".join(intents)))
    if short_model:
        chips.append(_chip("⚙ " + short_model))

    footer_html = (
        '<div style="padding:9px 16px;background:rgba(128,128,128,0.05);'
        "border-top:1px solid rgba(128,128,128,0.18);display:flex;flex-wrap:wrap;"
        'gap:6px;align-items:center;">' + "".join(chips) + "</div>"
    )

    # Fixed accent colors (indigo header, emerald "enhanced") that read on
    # both light and dark themes; body text stays color:inherit so it
    # adapts. Backgrounds are translucent so the card never looks like a
    # dark slab on a dark theme.
    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
        "border:1px solid rgba(139,92,246,0.35);border-radius:14px;overflow:hidden;"
        "margin:10px 0;font-size:13px;line-height:1.55;color:inherit;"
        'box-shadow:0 1px 3px rgba(0,0,0,0.12);">'
        # --- header: indigo→violet gradient, white text ---
        '<div style="background:linear-gradient(90deg,#6366f1,#8b5cf6);'
        "padding:9px 16px;font-size:11px;font-weight:700;color:#ffffff;"
        "text-transform:uppercase;letter-spacing:0.6px;display:flex;"
        'align-items:center;gap:6px;">'
        "<span>✨</span><span>"
        f"Prompt Enhanced{intent_label}{mode_label}{style_label}{cache_label}</span></div>"
        # --- original: muted, slightly tinted ---
        '<div style="padding:11px 16px;background:rgba(128,128,128,0.05);'
        'border-bottom:1px solid rgba(128,128,128,0.18);">'
        '<div style="font-size:10px;font-weight:700;color:#9ca3af;'
        'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:5px;">Original</div>'
        '<div style="opacity:0.8;font-style:italic;white-space:pre-wrap;">'
        f"{_escape_html(_clamp_for_display(original))}</div></div>"
        # --- enhanced: emerald tint + left accent bar, full-strength text ---
        '<div style="padding:11px 16px;background:rgba(16,185,129,0.10);'
        'border-left:3px solid #10b981;">'
        '<div style="font-size:10px;font-weight:700;color:#10b981;'
        'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:5px;">Enhanced</div>'
        f'<div style="white-space:pre-wrap;">'
        f"{_escape_html(_clamp_for_display(enhanced))}</div></div>"
        + footer_html
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for filter ordering in Open WebUI.",
        )
        enabled: bool = Field(
            default=True,
            description="Master on/off switch.",
        )
        model_id: Optional[str] = Field(
            default=None,
            description="Model for enhancement. Leave empty to use the chat model.",
        )
        show_status: bool = Field(
            default=False,
            description="Show status indicators during enhancement.",
        )
        show_enhanced_prompt: bool = Field(
            default=False,
            description=(
                "Display a comparison of the original and enhanced prompt in chat. "
                "Do not use with custom pipes."
            ),
        )

        # --- Skip controls ---
        min_prompt_length: int = Field(
            default=12,
            description="Skip prompts shorter than this (characters, whitespace-trimmed).",
        )
        max_prompt_length: int = Field(
            default=8000,
            description="Skip prompts longer than this (characters). 0 = no limit.",
        )
        skip_followups: bool = Field(
            default=False,
            description="Skip enhancement entirely for follow-up messages in a conversation.",
        )
        skip_well_structured: bool = Field(
            default=False,
            description=(
                "Skip the LLM call entirely when the prompt is already detailed "
                "and well-structured (headings, lists, role framing, etc.). Saves "
                "latency and cost on prompts that need no help. When off, such "
                "prompts get a light-refinement pass instead."
            ),
        )
        skip_code_only: bool = Field(
            default=True,
            description="Skip enhancement for prompts that are predominantly code blocks.",
        )
        custom_skip_patterns: str = Field(
            default="",
            description=(
                "Additional regex patterns (one per line) to skip enhancement. "
                "Lines starting with # are ignored. Invalid patterns are logged "
                "once and ignored."
            ),
        )
        skip_models: str = Field(
            default="",
            description=(
                "Model IDs to never enhance (comma or newline separated). "
                "A trailing * makes the entry a prefix match, e.g. 'gpt-4*'. "
                "Note: a lone '*' matches (and therefore skips) every model."
            ),
        )
        skip_with_images: bool = Field(
            default=False,
            description=(
                "Skip enhancement when the message has image attachments. "
                "When off, the enhancer is told images are attached and "
                "instructed to preserve references to them."
            ),
        )
        prefix: str = Field(
            default="!!",
            description=(
                "Per-message prefix. In 'activate' mode (the default), ONLY "
                "messages starting with this prefix are enhanced, and the prefix "
                "is stripped before sending. In 'bypass' mode, every message is "
                "enhanced EXCEPT those starting with the prefix. Empty disables "
                "prefix handling ('activate' mode then never enhances; 'bypass' "
                "mode always enhances). Admins can send '<prefix>stats' to view "
                "runtime counters, or '<prefix>stats reset' to zero the counters "
                "and clear the cache."
            ),
        )
        prefix_mode: Literal["activate", "bypass"] = Field(
            default="activate",
            description=(
                "'activate': enhancement is opt-in — the prefix turns it ON for "
                "that message. 'bypass': enhancement runs on every message and "
                "the prefix turns it OFF (the pre-4.8 behavior)."
            ),
        )
        enable_inline_commands: bool = Field(
            default=True,
            description=(
                "Allow per-message flags after the prefix in activate mode "
                "('!!d,show: prompt' → detailed style + comparison card; flags: "
                "c/concise, s/standard, d/detailed, show, fresh) and the "
                "'<prefix>help' command card. The admin stats command is always "
                "available regardless of this setting."
            ),
        )
        failure_threshold: int = Field(
            default=3,
            ge=0,
            description=(
                "Consecutive enhancement failures (LLM errors, timeouts, empty "
                "output) before the enhancer pauses itself. 0 disables the "
                "circuit breaker."
            ),
        )
        failure_cooldown_seconds: float = Field(
            default=120.0,
            ge=0.0,
            description=(
                "How long the enhancer stays paused (forwarding original prompts "
                "instantly) after tripping failure_threshold. Any successful "
                "enhancement closes the breaker early; '<prefix>stats reset' "
                "also clears it."
            ),
        )

        # --- Enhancement tuning ---
        enhancement_style: Literal["concise", "standard", "detailed"] = Field(
            default="standard",
            description=(
                "Enhancement depth: 'concise' (minimal changes), "
                "'standard' (balanced), 'detailed' (thorough expansion)."
            ),
        )
        temperature: float = Field(
            default=0.7,
            ge=0.0,
            le=2.0,
            description=(
                "LLM temperature for the enhancement call. "
                "Lower = more consistent, higher = more creative."
            ),
        )
        llm_timeout_seconds: float = Field(
            default=30.0,
            ge=0.0,
            description=(
                "Max seconds to wait for the enhancement LLM call before falling "
                "back to the original prompt. 0 = no timeout."
            ),
        )
        intent_threshold: float = Field(
            default=0.55,
            ge=0.0,
            le=1.0,
            description="Minimum confidence to apply intent-specific hints.",
        )
        enable_intent_detection: bool = Field(
            default=True,
            description="Use regex intent detection for domain-specific hints.",
        )
        enable_socratic: bool = Field(
            default=False,
            description=(
                "When a prompt is too vague to enhance (no intent detected and "
                "high vagueness score), skip enhancement and instead instruct "
                "the primary model to ask ONE clarifying question before "
                "answering. Requires enable_intent_detection."
            ),
        )
        socratic_threshold: float = Field(
            default=0.6,
            ge=0.0,
            le=1.0,
            description=(
                "Vagueness score (0-1) at or above which Socratic mode fires. "
                "Lower = asks clarifying questions more often."
            ),
        )
        extra_intent_patterns: str = Field(
            default="",
            description=(
                "JSON object of custom intents merged over the built-ins. "
                'Format: {"name": {"priority": 70, "patterns": ["regex", ...], '
                '"hint": "guidance text"}}.'
            ),
        )
        include_tool_context: bool = Field(
            default=True,
            description="Pass available tool IDs to the enhancer for context.",
        )
        include_datetime: bool = Field(
            default=True,
            description="Include current date/time in enhancer context.",
        )
        context_messages: int = Field(
            default=6,
            ge=0,
            le=20,
            description=(
                "How many prior conversation messages to give the enhancer. "
                "0 disables context."
            ),
        )
        context_snippet_length: int = Field(
            default=300,
            ge=50,
            description="Max characters per prior message included as context.",
        )
        max_enhanced_length: int = Field(
            default=4000,
            description=(
                "Maximum character length for the enhanced prompt. "
                "If exceeded, the original prompt is used instead. 0 = no limit."
            ),
        )
        max_expansion_ratio: float = Field(
            default=0.0,
            ge=0.0,
            description=(
                "Reject enhancements longer than this multiple of the original "
                "prompt and fall back to the original. Off by default (0) because "
                "creative/short prompts legitimately expand a lot; max_enhanced_length "
                "already bounds absolute size. Set e.g. 6.0 only if you see bloat."
            ),
        )
        enable_cache: bool = Field(
            default=True,
            description=(
                "Cache enhanced prompts to avoid duplicate LLM calls for identical inputs."
            ),
        )
        share_cache_across_users: bool = Field(
            default=False,
            description=(
                "Share cached enhancements across all users instead of keying the "
                "cache per-user. Greatly raises hit rate on common questions, but "
                "the same input always yields the same enhanced prompt regardless "
                "of who sent it. Leave off if per-user output must stay isolated."
            ),
        )
        cache_ttl_seconds: int = Field(
            default=3600,
            ge=0,
            description="Seconds before a cached enhancement expires. 0 = never expire.",
        )
        cache_maxsize: int = Field(
            default=128,
            ge=1,
            description="Maximum number of cached enhancements (LRU eviction).",
        )
        enable_coalescing: bool = Field(
            default=True,
            description=(
                "Merge concurrent identical enhancement requests into a single "
                "LLM call (works even with caching disabled)."
            ),
        )
        retry_on_failure: bool = Field(
            default=True,
            description="Retry once on transient LLM failure before falling back to original.",
        )

        # --- Prompt customization ---
        store_original_prompt: bool = Field(
            default=True,
            description=(
                "Stash the user's untouched original prompt in "
                "body['metadata']['original_prompt'] before swapping in the "
                "enhanced version, so the raw text is never silently lost."
            ),
        )
        custom_system_prompt: str = Field(
            default="",
            description="Fully replace the default enhancement system prompt.",
        )
        additional_instructions: str = Field(
            default="",
            description=(
                "Extra instructions appended to the system prompt. "
                "Use this to steer enhancement style without replacing the whole prompt. "
                "Example: 'Always ask the AI to show its reasoning step by step.'"
            ),
        )

        debug: bool = Field(
            default=False,
            description="Verbose debug logging.",
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable prompt enhancement for your messages.",
        )
        # Plain str (validated in code against VALID_STYLES) instead of a
        # Literal: an invalid stored value like "" would fail UserValves
        # construction outright, and Open WebUI then drops the WHOLE user
        # valves object — including an 'enabled: False' opt-out.
        enhancement_style: Optional[str] = Field(
            default=None,
            description=(
                "Override the enhancement style for your messages: "
                "'concise', 'standard', or 'detailed'. Leave empty to use admin default."
            ),
        )
        show_enhanced_prompt: Optional[bool] = Field(
            default=None,
            description=(
                "Override whether to show the enhanced prompt comparison. "
                "Leave empty for admin default."
            ),
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # Event emission (safe when no emitter is provided)
    # ------------------------------------------------------------------

    async def _emit_status(
        self,
        emitter: Optional[Callable[[Any], Awaitable[None]]],
        description: str,
        done: bool,
        *,
        force: bool = False,
    ) -> None:
        # `force` surfaces the status even when show_status is off — used when
        # the user explicitly triggered enhancement and it could not run, so
        # the trigger never fails silently.
        if emitter is None or not (force or self.valves.show_status):
            return
        try:
            await emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the request
            logger.debug("Status emit failed", exc_info=True)

    async def _emit_html(
        self,
        emitter: Optional[Callable[[Any], Awaitable[None]]],
        html: str,
    ) -> None:
        if emitter is None:
            return
        try:
            await emitter({"type": "embeds", "data": {"embeds": [html]}})
        except Exception:  # noqa: BLE001
            logger.debug("Embed emit failed", exc_info=True)

    async def _emit_embed(
        self,
        emitter: Optional[Callable[[Any], Awaitable[None]]],
        *,
        original: str,
        enhanced: str,
        intents: Sequence[str],
        followup: bool,
        well_structured: bool,
        style: str,
        cached: bool = False,
        elapsed_ms: Optional[int] = None,
        model: str = "",
    ) -> None:
        if emitter is None:
            return
        await self._emit_html(
            emitter,
            _build_comparison_embed(
                original=original,
                enhanced=enhanced,
                intents=list(intents),
                followup=followup,
                well_structured=well_structured,
                style=style,
                cached=cached,
                elapsed_ms=elapsed_ms,
                model=model,
            ),
        )

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        request: Optional[Request],
        payload: dict,
        user: Any,
    ) -> Any:
        """Call the enhancement LLM once.

        Returns the extracted text on success, ``None`` on ordinary failure,
        or the module-level ``_TIMED_OUT`` sentinel on timeout so the retry
        wrapper can avoid doubling the worst-case wait.
        """
        if request is None:
            logger.warning("No __request__ available — skipping enhancement call")
            return None

        timeout = float(self.valves.llm_timeout_seconds)
        try:
            coro = generate_chat_completion(
                request, payload, user=user, bypass_filter=True
            )
            if timeout > 0:
                response = await asyncio.wait_for(coro, timeout=timeout)
            else:
                response = await coro
        except asyncio.TimeoutError:
            _STATS.bump("timeouts")
            logger.warning("Enhancement LLM call timed out after %.1fs", timeout)
            return _TIMED_OUT
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - degrade to original prompt
            logger.warning("Enhancement LLM call failed: %s", exc)
            return None
        return _extract_content(response)

    async def _call_llm_with_retry(
        self,
        request: Optional[Request],
        payload: dict,
        user: Any,
    ) -> Optional[str]:
        """Call the LLM with a single retry on ordinary failure.

        Timeouts are deliberately NOT retried: a retry after a full timeout
        would double the user-facing worst-case latency for a prompt that is
        likely to time out again anyway. Likewise, a missing user object is
        not transient — the second attempt would fail identically, so the
        retry (and its 1s backoff) is skipped.
        """
        result = await self._call_llm(request, payload, user)
        if result is _TIMED_OUT:
            return None
        if result is not None:
            return result
        if not self.valves.retry_on_failure or user is None:
            return None

        await asyncio.sleep(1.0)
        result = await self._call_llm(request, payload, user)
        return None if result is _TIMED_OUT else result

    # ------------------------------------------------------------------
    # Valve resolution / skip logic
    # ------------------------------------------------------------------

    def _resolve_user_overrides(self, user_valves: Any) -> tuple[str, bool]:
        style: str = self.valves.enhancement_style
        show_embed: bool = self.valves.show_enhanced_prompt

        uv_style: Any = None
        uv_show: Any = None
        if isinstance(user_valves, self.UserValves):
            uv_style = user_valves.enhancement_style
            uv_show = user_valves.show_enhanced_prompt
        elif isinstance(user_valves, dict):
            uv_style = user_valves.get("enhancement_style")
            uv_show = user_valves.get("show_enhanced_prompt")

        if isinstance(uv_style, str):
            uv_style = uv_style.strip().lower()
        if uv_style in VALID_STYLES:
            style = str(uv_style)
        if isinstance(uv_show, bool):
            show_embed = uv_show
        if style not in VALID_STYLES:
            style = DEFAULT_STYLE
        return style, show_embed

    @staticmethod
    def _user_enabled(user_valves: Any) -> bool:
        if isinstance(user_valves, dict):
            return bool(user_valves.get("enabled", True))
        return bool(getattr(user_valves, "enabled", True))

    def _should_skip(
        self, user_message: str, *, followup: bool, explicit: bool = False
    ) -> Optional[str]:
        """Return a human-readable skip reason, or None to proceed.

        An explicitly triggered message (activate-mode prefix) relaxes the
        gates that exist only to avoid enhancing things nobody asked about:
        the minimum length and the follow-up opt-out. Cost and quality guards
        (max length, code-only, trivial, admin skip patterns) still apply.
        """
        stripped = user_message.strip()
        if _is_trivial(stripped):
            return "trivial"

        length = len(stripped)
        if not explicit and length < self.valves.min_prompt_length:
            return "too short"
        max_length = self.valves.max_prompt_length
        if max_length > 0 and length > max_length:
            return "too long"

        if self.valves.skip_code_only and _is_code_only(user_message):
            return "code-only"
        if _matches_custom_skip(user_message, self.valves.custom_skip_patterns):
            return "custom skip pattern"
        if followup and self.valves.skip_followups and not explicit:
            return "follow-up"
        return None

    # ------------------------------------------------------------------
    # inlet sub-steps
    # ------------------------------------------------------------------

    async def _resolve_prefix(
        self,
        body: dict,
        messages: list[dict],
        user_message: str,
        user_info: Optional[dict],
        emitter: Optional[Callable[[Any], Awaitable[None]]],
    ) -> Optional[_PrefixOutcome]:
        """Apply the per-message prefix, inline flags, and utility commands.

        Returns a _PrefixOutcome describing how to enhance this message, or
        None when enhancement should not run:

        - 'activate' mode: the prefix opts a message IN. Prefixed messages
          have the prefix (and any inline flags) stripped — and are forwarded
          stripped even if a later skip-check or LLM failure prevents
          enhancement; everything else passes through untouched, unenhanced.
        - 'bypass' mode (legacy): every message is enhanced except prefixed
          ones, which are stripped and forwarded as-is.

        The stats and help commands work in both modes and raise to abort the
        request so they never reach the model.
        """
        prefix = self.valves.prefix.strip()
        activate = self.valves.prefix_mode == "activate"
        if not prefix:
            # No prefix configured: nothing can trigger opt-in enhancement;
            # legacy mode simply enhances everything.
            return None if activate else _PrefixOutcome(text=user_message)

        leading = user_message.lstrip()
        if not leading.startswith(prefix):
            return None if activate else _PrefixOutcome(text=user_message)

        remainder = leading[len(prefix) :].lstrip()
        command = " ".join(remainder.lower().split())
        is_admin = bool(user_info) and user_info.get("role") == "admin"

        if command in ("stats", "stats reset"):
            if not is_admin:
                # Abort with a notice instead of forwarding the literal word
                # "stats" to the model.
                raise RuntimeError(
                    f"Prompt Enhancer: '{prefix}stats' is admin-only. "
                    f"Send '{prefix}help' for available commands."
                )
            if command == "stats reset":
                _STATS.reset()
                _prompt_cache.clear()
                _breaker.reset()
                label = "Prompt Enhancer stats reset"
            else:
                label = "Prompt Enhancer stats"
            await self._emit_html(emitter, _stats_embed_html())
            # Abort the request so the command never reaches the model;
            # the exception text doubles as a plain-text stats readout.
            raise RuntimeError(f"{label} — {_stats_summary()}")

        if command == "help" and self.valves.enable_inline_commands:
            user_valves = user_info.get("valves") if user_info else None
            style, _ = self._resolve_user_overrides(user_valves)
            await self._emit_html(
                emitter,
                _help_embed_html(prefix, self.valves.prefix_mode, style, is_admin),
            )
            # Plain-text fallback for clients that don't render the embed.
            raise RuntimeError(
                f"Prompt Enhancer commands: '{prefix}<prompt>' enhance | "
                f"'{prefix}c:/s:/d: <prompt>' style | '{prefix}show: <prompt>' "
                f"comparison card | '{prefix}fresh: <prompt>' skip cache | "
                f"'{prefix}stats' counters (admin). Flags combine: "
                f"'{prefix}d,show: <prompt>'."
            )

        if not remainder:
            # Bare prefix with nothing behind it: leave the message untouched
            # rather than forwarding an empty message some providers reject.
            return None

        if not activate:
            _set_last_user_message_text(messages, remainder)
            body["messages"] = messages
            _STATS.bump("bypassed")
            logger.debug("Bypass prefix used — enhancement skipped")
            return None

        style_override: Optional[str] = None
        force_embed = skip_cache = False
        if self.valves.enable_inline_commands:
            style_override, force_embed, skip_cache, remainder = _parse_inline_flags(
                remainder
            )
            if not remainder:
                # Flags with no prompt behind them ("!!d,show:") — nothing to
                # enhance; leave the message untouched.
                return None

        _set_last_user_message_text(messages, remainder)
        body["messages"] = messages
        logger.debug(
            "Activation prefix used — enhancing (style=%s, show=%s, fresh=%s)",
            style_override or "default",
            force_embed,
            skip_cache,
        )
        return _PrefixOutcome(
            text=remainder,
            explicit=True,
            style_override=style_override,
            force_embed=force_embed,
            skip_cache=skip_cache,
        )

    async def _apply_socratic(
        self,
        body: dict,
        messages: list[dict],
        user_message: str,
        *,
        followup: bool,
        active_intents: Sequence[str],
        max_intent_conf: float,
        emitter: Optional[Callable[[Any], Awaitable[None]]],
    ) -> bool:
        """Optionally replace enhancement with a one-question clarification ask.

        Only for fresh (non-followup) messages: in a followup, deictics like
        "it" are usually resolved by the visible conversation, so a clarifying
        question would be annoying rather than helpful. No enhancer LLM call is
        made on this path, so it costs nothing.
        """
        if not (self.valves.enable_socratic and self.valves.enable_intent_detection):
            return False
        if followup:
            return False

        assessment = assess_vagueness(user_message, max_intent_conf)
        # A bare-deictic object ("fix it") fires even when an intent was
        # detected: the intent identifies the task type, but the object is
        # still missing.
        if active_intents and not assessment.has_bare_deictic_object:
            return False
        if assessment.score < self.valves.socratic_threshold:
            return False

        _STATS.bump("socratic")
        logger.debug(
            "Socratic mode fired (vagueness=%.2f >= %.2f)",
            assessment.score,
            self.valves.socratic_threshold,
        )
        if self.valves.store_original_prompt:
            _stash_original_prompt(body, user_message)
        _set_last_user_message_text(
            messages, f"{user_message}\n\n{_SOCRATIC_DIRECTIVE}"
        )
        body["messages"] = messages
        await self._emit_status(
            emitter, "Request looks ambiguous — asking for details", True
        )
        return True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __task__: Optional[str] = None,
        __request__: Optional[Request] = None,
    ) -> dict:
        if not self.valves.enabled:
            return body
        if not isinstance(body, dict):
            return body

        logger.setLevel(logging.DEBUG if self.valves.debug else logging.INFO)

        user_valves = __user__.get("valves") if __user__ else None
        if user_valves is not None and not self._user_enabled(user_valves):
            return body

        # Background tasks (title/tags/autocomplete generation, ...) must
        # never be enhanced. Compared against a set of known default-task
        # names because TASKS.DEFAULT is not a stable enum member on every
        # build (some define it as a lambda).
        task = _task_name(__task__)
        if task not in _DEFAULT_TASK_NAMES:
            return body

        messages = body.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(messages[-1], dict)
            or messages[-1].get("role") != "user"
        ):
            return body

        # Read the last user message with our own flattener: it joins ALL
        # text parts, where Open WebUI's helper returns only the first —
        # which would enhance partial text and then drop the rest on write.
        user_message = _message_text(messages[-1].get("content"))
        if not user_message.strip():
            return body

        # --- Per-message prefix (activate/bypass), flags, utility commands ---
        prefixed = await self._resolve_prefix(
            body, messages, user_message, __user__, __event_emitter__
        )
        if prefixed is None:
            return body
        user_message = prefixed.text
        explicit = prefixed.explicit

        followup = _is_followup(messages, user_message)

        skip_reason = self._should_skip(
            user_message, followup=followup, explicit=explicit
        )
        if skip_reason:
            _STATS.bump("skipped")
            logger.debug("Skipped: %s", skip_reason)
            await self._emit_status(
                __event_emitter__,
                f"Enhancement skipped ({skip_reason}) — sending original.",
                True,
                force=explicit,
            )
            return body

        image_count, image_digest = _image_signature(messages)
        if image_count and self.valves.skip_with_images:
            _STATS.bump("skipped")
            logger.debug("Skipped: message has %d image attachment(s)", image_count)
            await self._emit_status(
                __event_emitter__,
                "Enhancement skipped (image attachments) — sending original.",
                True,
                force=explicit,
            )
            return body

        well_structured = _is_well_structured(user_message)
        # An explicit trigger overrides the well-structured skip: the user
        # asked, so give the light-refinement pass instead of doing nothing.
        if well_structured and self.valves.skip_well_structured and not explicit:
            _STATS.bump("skipped")
            logger.debug("Skipped: prompt already well-structured")
            return body

        # Resolve the model BEFORE intent detection so skip-listed models never
        # pay for the (relatively expensive) regex scan.
        model_to_use = _resolve_model(self.valves.model_id, __model__, body)
        if not model_to_use:
            _STATS.bump("skipped")
            logger.debug("Skipped: no model could be resolved")
            await self._emit_status(
                __event_emitter__,
                "Enhancement skipped (no model resolved) — sending original.",
                True,
                force=explicit,
            )
            return body
        if _model_skipped(model_to_use, self.valves.skip_models):
            _STATS.bump("skipped")
            logger.debug("Skipped: model %s is on the skip-list", model_to_use)
            await self._emit_status(
                __event_emitter__,
                "Enhancement skipped (model on skip-list) — sending original.",
                True,
                force=explicit,
            )
            return body

        style, show_embed = self._resolve_user_overrides(user_valves)
        if prefixed.style_override:
            style = prefixed.style_override
        if prefixed.force_embed:
            show_embed = True

        # --- Intent detection (built-ins + admin custom intents) ---
        intents_catalog = dict(COMPILED_INTENTS)
        intents_catalog.update(_parse_extra_intents(self.valves.extra_intent_patterns))
        active_intents: list[str] = []
        max_intent_conf = 0.0
        if self.valves.enable_intent_detection:
            active_intents, max_intent_conf = _detect_intents_scored(
                user_message, self.valves.intent_threshold, intents_catalog
            )

        # --- Socratic mode: too vague to enhance -> ask, don't guess ---
        if await self._apply_socratic(
            body,
            messages,
            user_message,
            followup=followup,
            active_intents=active_intents,
            max_intent_conf=max_intent_conf,
            emitter=__event_emitter__,
        ):
            return body

        ctx = EnhancementContext(
            style=style,
            intents=active_intents,
            is_followup=followup,
            is_well_structured=well_structured,
            custom_system_prompt=self.valves.custom_system_prompt,
            additional_instructions=self.valves.additional_instructions,
            model=model_to_use,
            temperature=self.valves.temperature,
            # Blank user_id when sharing so the cache signature is identical
            # across users; otherwise partition the cache per-user.
            user_id=(
                ""
                if self.valves.share_cache_across_users
                else (str(__user__.get("id", "")) if __user__ else "")
            ),
            max_enhanced_length=self.valves.max_enhanced_length,
            max_expansion_ratio=self.valves.max_expansion_ratio,
        )
        signature = ctx.signature()

        raw_tools = body.get("tool_ids") if self.valves.include_tool_context else None
        tool_ids: Optional[list[str]] = (
            [str(t) for t in raw_tools]
            if isinstance(raw_tools, (list, tuple))
            else None
        )
        context_block = _build_context_block(
            messages, self.valves.context_messages, self.valves.context_snippet_length
        )
        cache_prompt_key = self._compute_cache_key(
            user_message=user_message,
            context_block=context_block,
            tool_ids=tool_ids,
            image_count=image_count,
            image_digest=image_digest,
        )

        # Keep the shared cache configured to the current admin valves.
        _prompt_cache.configure(
            maxsize=self.valves.cache_maxsize,
            ttl_seconds=float(self.valves.cache_ttl_seconds),
        )

        # --- Cache check ('fresh' flag bypasses the read, not the write) ---
        if self.valves.enable_cache and not prefixed.skip_cache:
            cached = _prompt_cache.get(signature, cache_prompt_key)
            if cached is not None:
                _STATS.bump("cache_hits")
                logger.debug("Cache hit for prompt (len=%d)", len(user_message))
                if self.valves.store_original_prompt:
                    _stash_original_prompt(body, user_message)
                _set_last_user_message_text(messages, cached)
                body["messages"] = messages
                await self._emit_status(
                    __event_emitter__, "Prompt enhanced (cached).", True
                )
                if show_embed:
                    await self._emit_embed(
                        __event_emitter__,
                        original=user_message,
                        enhanced=cached,
                        intents=active_intents,
                        followup=followup,
                        well_structured=well_structured,
                        style=style,
                        cached=True,
                        model=model_to_use,
                    )
                return body

        return await self._run_enhancement(
            body=body,
            messages=messages,
            user_message=user_message,
            ctx=ctx,
            signature=signature,
            cache_prompt_key=cache_prompt_key,
            intents_catalog=intents_catalog,
            context_block=context_block,
            tool_ids=tool_ids,
            image_count=image_count,
            model_to_use=model_to_use,
            style=style,
            show_embed=show_embed,
            followup=followup,
            well_structured=well_structured,
            active_intents=active_intents,
            explicit=explicit,
            user_info=__user__,
            request=__request__,
            emitter=__event_emitter__,
        )

    # ------------------------------------------------------------------
    # Enhancement execution
    # ------------------------------------------------------------------

    def _compute_cache_key(
        self,
        *,
        user_message: str,
        context_block: str,
        tool_ids: Optional[Sequence[str]],
        image_count: int,
        image_digest: str,
    ) -> str:
        """Build the prompt-side cache key.

        Everything that feeds the enhancement prompt must also feed the cache
        key — otherwise a result produced in one conversation could be served
        in another where "it"/"that" mean something different.
        """
        tools_key = ",".join(tool_ids) if tool_ids else ""
        images_key = f"{image_count}:{image_digest}" if image_count else ""
        # When the current time is baked into the enhanced prompt, key the
        # cache by the hour so a TTL of 0 (never expire) can't serve a stale
        # timestamp indefinitely.
        date_key = (
            dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H")
            if self.valves.include_datetime
            else ""
        )
        return "\x00".join(
            (context_block, tools_key, images_key, date_key, user_message)
        )

    async def _run_enhancement(
        self,
        *,
        body: dict,
        messages: list[dict],
        user_message: str,
        ctx: EnhancementContext,
        signature: str,
        cache_prompt_key: str,
        intents_catalog: dict[str, dict[str, Any]],
        context_block: str,
        tool_ids: Optional[Sequence[str]],
        image_count: int,
        model_to_use: str,
        style: str,
        show_embed: bool,
        followup: bool,
        well_structured: bool,
        active_intents: Sequence[str],
        explicit: bool,
        user_info: Optional[dict],
        request: Optional[Request],
        emitter: Optional[Callable[[Any], Awaitable[None]]],
    ) -> dict:
        # Circuit breaker: after repeated LLM failures, forward originals
        # instantly instead of making every message pay the full timeout.
        cooldown_left = _breaker.remaining()
        if cooldown_left > 0:
            _STATS.bump("cooldown")
            logger.debug(
                "Enhancer paused (breaker open, %.0fs left) — using original",
                cooldown_left,
            )
            await self._emit_status(
                emitter,
                "Enhancer paused after repeated failures "
                f"(~{int(cooldown_left) + 1}s left) — using original prompt.",
                True,
                force=explicit,
            )
            return body

        system_prompt = _build_system_prompt(ctx, intents_catalog)

        logger.debug(
            "Enhancing | intents=%s | followup=%s | structured=%s | style=%s | len=%d",
            list(active_intents),
            followup,
            well_structured,
            style,
            len(user_message),
        )

        await self._emit_status(emitter, "Enhancing prompt...", False)

        user = await self._lookup_user(user_info)

        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        user_message=user_message,
                        context_block=context_block,
                        tool_ids=tool_ids,
                        include_datetime=self.valves.include_datetime,
                        image_count=image_count,
                    ),
                },
            ],
            "stream": False,
            "temperature": self.valves.temperature,
        }

        started = time.perf_counter()

        # Captures WHY an enhancement was dropped so the fallback status can
        # name the reason instead of a generic "skipped", and whether it was a
        # transport failure or a policy rejection (separate counters). Mutable
        # holder so the nested coroutine can write to it. Note: under request
        # coalescing only the leader's closure runs, so follower requests see
        # the generic default reason rather than the specific one.
        outcome: dict[str, Any] = {"why": "no usable output", "rejected": False}

        async def _produce() -> Optional[str]:
            # Breaker recording lives here (leader-only under coalescing) so N
            # coalesced followers of one failed call can't trip it N times.
            # Guard rejections (too long / over-expanded) never count: the LLM
            # itself is healthy on those.
            raw = await self._call_llm_with_retry(request, payload, user)
            if raw is None:
                outcome["why"] = "enhancer model failed or timed out"
                _breaker.record_failure(
                    self.valves.failure_threshold,
                    float(self.valves.failure_cooldown_seconds),
                )
                return None

            cleaned = _clean_llm_output(raw)
            if not cleaned.strip():
                outcome["why"] = "enhancer returned empty output (possible refusal)"
                _breaker.record_failure(
                    self.valves.failure_threshold,
                    float(self.valves.failure_cooldown_seconds),
                )
                return None
            _breaker.record_success()

            max_length = ctx.max_enhanced_length
            if max_length > 0 and len(cleaned) > max_length:
                outcome["rejected"] = True
                outcome["why"] = (
                    f"enhanced prompt too long ({len(cleaned)} > {max_length} chars)"
                )
                logger.debug("%s — keeping original", outcome["why"])
                return None

            ratio_limit = ctx.max_expansion_ratio
            original_length = max(1, len(user_message))
            if ratio_limit > 0 and len(cleaned) > ratio_limit * original_length:
                outcome["rejected"] = True
                outcome["why"] = (
                    f"expansion {len(cleaned) / original_length:.1f}× exceeded "
                    f"the {ratio_limit:.1f}× limit"
                )
                logger.debug("%s — keeping original", outcome["why"])
                return None

            if self.valves.enable_cache:
                _prompt_cache.put(signature, cache_prompt_key, cleaned)
            return cleaned

        coalesce_key = hashlib.sha256(
            f"{signature}\x00{cache_prompt_key}".encode("utf-8")
        ).hexdigest()

        try:
            if self.valves.enable_coalescing:
                enhanced = await _coalesce(coalesce_key, _produce)
            else:
                enhanced = await _produce()

            if not enhanced:
                _STATS.bump("rejected" if outcome["rejected"] else "failed")
                logger.info(
                    "Enhancement skipped (%s) — using original prompt", outcome["why"]
                )
                await self._emit_status(
                    emitter,
                    f"Enhancement skipped ({outcome['why']}) — using original prompt.",
                    True,
                    force=explicit,
                )
                return body

            if self.valves.store_original_prompt:
                _stash_original_prompt(body, user_message)
            _set_last_user_message_text(messages, enhanced)
            body["messages"] = messages

            _STATS.bump("enhanced")
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            logger.debug(
                "Enhanced (%d chars, %dms): %s | stats=%s",
                len(enhanced),
                elapsed_ms,
                enhanced[:200],
                _STATS.snapshot(),
            )

            intent_tag = f" [{', '.join(active_intents)}]" if active_intents else ""
            await self._emit_status(
                emitter, f"Prompt enhanced ({elapsed_ms}ms){intent_tag}.", True
            )

            if show_embed:
                await self._emit_embed(
                    emitter,
                    original=user_message,
                    enhanced=enhanced,
                    intents=active_intents,
                    followup=followup,
                    well_structured=well_structured,
                    style=style,
                    elapsed_ms=elapsed_ms,
                    model=model_to_use,
                )

        except asyncio.CancelledError:
            # This request was cancelled (client disconnect / shutdown). Do not
            # swallow it — let it propagate so the request unwinds cleanly.
            raise
        except Exception as exc:  # noqa: BLE001 - never break the user's chat
            _STATS.bump("failed")
            logger.exception("Enhancement failed: %s", exc)
            await self._emit_status(
                emitter,
                "Enhancement error — using original prompt.",
                True,
                force=explicit,
            )

        return body

    @staticmethod
    async def _lookup_user(user_info: Optional[dict]) -> Any:
        """Resolve the Open WebUI user object needed by generate_chat_completion.

        Users.get_user_by_id is synchronous in some Open WebUI builds and
        async in others, so guard for either case instead of assuming one
        signature and crashing the filter.
        """
        if not user_info or not user_info.get("id"):
            return None
        try:
            result = Users.get_user_by_id(user_info["id"])
            return await result if inspect.isawaitable(result) else result
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash
            logger.warning("User lookup failed (%s); continuing without user", exc)
            return None

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __request__: Optional[Request] = None,
    ) -> dict:
        return body
