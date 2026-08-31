"""Deterministic tests for the Consensus-10 Open WebUI pipe.

Hermetic: `open_webui` (and `fastapi`, when absent) are replaced with stub
modules before the pipe is imported, and every model completion is served by
an in-process mock backend. No network, no timing races: concurrency claims
are proven with barriers, not sleeps.

Run with:  python -m pytest -q extensions/consensus-10/test_consensus_10_pipe.py
"""

import asyncio
import copy
import importlib.util
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

# ----------------------------------------------------------------------
# Mock completion backend
# ----------------------------------------------------------------------

DRAFT_MARK = "--- BEGIN DRAFT ANSWER"
MANAGER_MARK = None  # populated with the pipe module's marker after import


def ok_response(text):
    """A minimal OpenAI-format completion payload."""
    return {"choices": [{"message": {"content": text}}]}


def default_plan():
    """The default manager reply: plenty of numbered assignment lines."""
    return "\n".join(f"{i}. default-angle-{i}" for i in range(1, 21))


class Backend:
    """Records every internal completion call and serves configurable replies.

    Set ``impl`` to an ``async def impl(call) -> dict`` to control behavior;
    by default the manager gets a numbered assignment list, candidates get
    "candidate-<n>", and synthesis gets "SYNTH-FINAL".
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = []
        self.total = 0
        self.manager_calls = 0
        self.candidate_calls = 0
        self.synthesis_calls = 0
        self.active = 0
        self.max_active = 0
        self.started = 0
        self.cancelled_count = 0
        self.impl = None
        self.user = SimpleNamespace(id="u1", role="user", email="t@example.com")

    def lookup_user(self, user_id):
        return self.user if user_id == "u1" else None

    @staticmethod
    def is_synthesis(form_data):
        return any(
            isinstance(m, dict)
            and isinstance(m.get("content"), str)
            and DRAFT_MARK in m["content"]
            for m in form_data.get("messages", [])
        )

    @staticmethod
    def is_manager(form_data):
        return bool(MANAGER_MARK) and any(
            isinstance(m, dict)
            and isinstance(m.get("content"), str)
            and MANAGER_MARK in m["content"]
            for m in form_data.get("messages", [])
        )

    async def __call__(self, request, form_data, user, bypass_filter=False):
        self.total += 1
        synthesis = self.is_synthesis(form_data)
        manager = not synthesis and self.is_manager(form_data)
        if synthesis:
            self.synthesis_calls += 1
            candidate_no = None
        elif manager:
            self.manager_calls += 1
            candidate_no = None
        else:
            self.candidate_calls += 1
            candidate_no = self.candidate_calls
        call = SimpleNamespace(
            no=self.total,
            candidate_no=candidate_no,
            is_synthesis=synthesis,
            is_manager=manager,
            form=copy.deepcopy(form_data),
            user=user,
            bypass_filter=bypass_filter,
            request=request,
        )
        self.calls.append(call)
        self.started += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.impl is not None:
                return await self.impl(call)
            if synthesis:
                return ok_response("SYNTH-FINAL")
            if manager:
                return ok_response(default_plan())
            return ok_response(f"candidate-{candidate_no}")
        except asyncio.CancelledError:
            self.cancelled_count += 1
            raise
        finally:
            self.active -= 1


BACKEND = Backend()

# ----------------------------------------------------------------------
# Stub modules installed before the pipe module is imported
# ----------------------------------------------------------------------


def _install_stubs():
    owui = types.ModuleType("open_webui")
    utils = types.ModuleType("open_webui.utils")
    chat = types.ModuleType("open_webui.utils.chat")
    models_pkg = types.ModuleType("open_webui.models")
    users_mod = types.ModuleType("open_webui.models.users")

    async def generate_chat_completion(
        request, form_data, user, bypass_filter=False, bypass_system_prompt=False
    ):
        return await BACKEND(request, form_data, user, bypass_filter=bypass_filter)

    chat.generate_chat_completion = generate_chat_completion

    class Users:  # async like Open WebUI >= 0.11
        @staticmethod
        async def get_user_by_id(user_id):
            return BACKEND.lookup_user(user_id)

    users_mod.Users = Users
    owui.utils = utils
    owui.models = models_pkg
    utils.chat = chat
    models_pkg.users = users_mod
    sys.modules["open_webui"] = owui
    sys.modules["open_webui.utils"] = utils
    sys.modules["open_webui.utils.chat"] = chat
    sys.modules["open_webui.models"] = models_pkg
    sys.modules["open_webui.models.users"] = users_mod

    try:
        import fastapi  # noqa: F401
    except ImportError:
        fastapi_stub = types.ModuleType("fastapi")

        class Request:  # annotation-only stand-in
            def __init__(self, scope=None):
                self.scope = scope or {}

        fastapi_stub.Request = Request
        sys.modules["fastapi"] = fastapi_stub


_install_stubs()

_PIPE_PATH = Path(__file__).resolve().parent / "consensus_10_pipe.py"
_spec = importlib.util.spec_from_file_location("consensus_10_pipe", _PIPE_PATH)
consensus_10_pipe = importlib.util.module_from_spec(_spec)
sys.modules["consensus_10_pipe"] = consensus_10_pipe
_spec.loader.exec_module(consensus_10_pipe)
Pipe = consensus_10_pipe.Pipe
MANAGER_MARK = consensus_10_pipe._MANAGER_MARK
SUMMARY_MARKER = consensus_10_pipe._SUMMARY_MARKER

# ----------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------


class FakeRequest:
    """Scope-backed request double; the pipe clones it via type(req)(scope)."""

    def __init__(self, scope):
        self.scope = scope

    @property
    def app(self):
        return self.scope["app"]


def make_app():
    return SimpleNamespace(
        state=SimpleNamespace(
            MODELS={"base-model": {"id": "base-model", "owned_by": "openai"}}
        )
    )


def make_request():
    return FakeRequest(
        {
            "type": "http",
            "app": make_app(),
            "state": {
                "metadata": {
                    "chat_id": "chat-1",
                    "message_id": "m-1",
                    "session_id": "s-1",
                    "tool_ids": ["danger"],
                }
            },
        }
    )


def make_body():
    """A realistic chat body full of fields that must NOT reach the backend."""
    return {
        "model": "consensus-10-model",
        "stream": True,
        "temperature": 0.9,
        "top_p": 0.5,
        "messages": [
            {"role": "system", "content": "You are terse.", "id": "sys-1"},
            {"role": "user", "content": "What is 2+2?", "id": "m-0", "models": ["x"]},
            {"role": "tool", "content": "tool junk", "tool_call_id": "t1"},
            {"role": "assistant", "content": None},
        ],
        "tools": [{"type": "function", "function": {"name": "dangerous"}}],
        "tool_ids": ["danger"],
        "files": [{"id": "f1"}],
        "features": {"web_search": True},
        "metadata": {"chat_id": "chat-1", "message_id": "m-1", "session_id": "s-1"},
        "chat_id": "chat-1",
        "id": "m-1",
        "session_id": "s-1",
        "stream_options": {"include_usage": True},
        "background_tasks": {"title_generation": True},
        "filter_ids": ["some-filter"],
    }


def make_pipe(**valve_overrides):
    pipe = Pipe()
    pipe.RETRY_BACKOFF_SECONDS = 0.0  # keep retry tests instant
    values = {"TARGET_MODEL_ID": "base-model"}
    values.update(valve_overrides)
    pipe.valves = Pipe.Valves(**values)
    return pipe


async def call_pipe(pipe, body=None, emitter=None, task=None, request=None):
    return await pipe.pipe(
        body=body if body is not None else make_body(),
        __user__={"id": "u1", "role": "user", "name": "T", "email": "t@example.com"},
        __request__=request if request is not None else make_request(),
        __event_emitter__=emitter,
        __task__=task,
    )


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fresh_backend():
    BACKEND.reset()
    yield


# ----------------------------------------------------------------------
# 1-2. Call counts and model pinning
# ----------------------------------------------------------------------


def test_default_flow_manager_10_candidates_synthesis():
    async def scenario():
        pipe = make_pipe()
        result = await call_pipe(pipe)
        assert result == "SYNTH-FINAL"
        assert BACKEND.total == 12
        assert BACKEND.manager_calls == 1
        assert BACKEND.candidate_calls == 10
        assert BACKEND.synthesis_calls == 1
        assert BACKEND.calls[0].is_manager  # the manager plans first
        assert BACKEND.calls[-1].is_synthesis  # synthesis is the final call

    run(scenario())


def test_all_calls_use_target_model():
    async def scenario():
        pipe = make_pipe()
        await call_pipe(pipe)
        assert len(BACKEND.calls) == 12  # manager included
        for call in BACKEND.calls:
            assert call.form["model"] == "base-model"
            assert call.form["stream"] is False
            assert call.bypass_filter is True

    run(scenario())


# ----------------------------------------------------------------------
# 3. Concurrency
# ----------------------------------------------------------------------


def test_candidates_run_concurrently():
    async def scenario():
        # Every candidate blocks until all six are in flight simultaneously;
        # a sequential implementation would dead-end on the 5s barrier.
        gate = asyncio.Event()

        async def impl(call):
            if call.is_synthesis:
                return ok_response("SYNTH-FINAL")
            if BACKEND.active >= 6:
                gate.set()
            await asyncio.wait_for(gate.wait(), timeout=5)
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(
            CANDIDATE_COUNT=6, MIN_SUCCESSFUL_RESPONSES=6, ENABLE_MANAGER=False
        )
        result = await call_pipe(pipe)
        assert result == "SYNTH-FINAL"
        assert BACKEND.max_active == 6

    run(scenario())


def test_max_concurrency_respected():
    async def scenario():
        async def impl(call):
            if call.is_synthesis:
                return ok_response("SYNTH-FINAL")
            await asyncio.sleep(0.02)
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(CANDIDATE_COUNT=9, MAX_CONCURRENCY=3, ENABLE_MANAGER=False)
        await call_pipe(pipe)
        assert BACKEND.candidate_calls == 9
        assert BACKEND.max_active <= 3  # the hard cap
        assert BACKEND.max_active >= 2  # and still actually parallel

    run(scenario())


# ----------------------------------------------------------------------
# 4. Input immutability
# ----------------------------------------------------------------------


def test_original_body_not_mutated():
    async def scenario():
        body = make_body()
        snapshot = copy.deepcopy(body)
        await call_pipe(make_pipe(), body=body)
        assert body == snapshot

    run(scenario())


# ----------------------------------------------------------------------
# 5. Recursion protection
# ----------------------------------------------------------------------


def test_direct_recursion_rejected():
    async def scenario():
        pipe = make_pipe(TARGET_MODEL_ID="consensus-10-model")
        with pytest.raises(Exception, match="itself"):
            await call_pipe(pipe)  # body["model"] == "consensus-10-model"
        assert BACKEND.total == 0

    run(scenario())


def test_reentrancy_blocked_for_internal_calls():
    async def scenario():
        pipe = make_pipe(
            CANDIDATE_COUNT=3, MIN_SUCCESSFUL_RESPONSES=1, ENABLE_MANAGER=False
        )
        nested = {}

        async def impl(call):
            if call.candidate_no == 1 and "error" not in nested:
                # Simulate a model chain that routes an internal request back
                # into this pipe: it must refuse before fanning out again.
                try:
                    await call_pipe(pipe)
                except Exception as exc:
                    nested["error"] = str(exc)
            return ok_response(
                "SYNTH-FINAL" if call.is_synthesis else f"candidate-{call.candidate_no}"
            )

        BACKEND.impl = impl
        result = await call_pipe(pipe)
        assert result == "SYNTH-FINAL"
        assert "recursive" in nested["error"].lower()
        assert BACKEND.total == 4  # 3 candidates + 1 synthesis, nothing nested

    run(scenario())


# ----------------------------------------------------------------------
# 6. Sanitization of internal requests
# ----------------------------------------------------------------------


def test_sanitization_strips_side_effect_fields():
    async def scenario():
        pipe = make_pipe(
            CANDIDATE_COUNT=2,
            MIN_SUCCESSFUL_RESPONSES=2,
            CANDIDATE_MAX_TOKENS=222,
            SYNTHESIS_MAX_TOKENS=333,
            ENABLE_MANAGER=False,  # assignment injection has its own test
        )
        await call_pipe(pipe)

        stripped = consensus_10_pipe._UNSAFE_BODY_KEYS
        candidates = [c for c in BACKEND.calls if not c.is_synthesis]
        synthesis = [c for c in BACKEND.calls if c.is_synthesis]
        assert len(candidates) == 2 and len(synthesis) == 1

        for call in candidates:
            for key in stripped:
                assert key not in call.form, f"{key} leaked into a candidate call"
            assert call.form["stream"] is False
            assert call.form["temperature"] == 0.7
            assert call.form["max_tokens"] == 222
            assert call.form["top_p"] == 0.5  # benign sampling prefs survive
            # whole conversation, reduced to plain role/content pairs
            assert call.form["messages"] == [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "What is 2+2?"},
            ]

        synth = synthesis[0]
        for key in stripped:
            assert key not in synth.form
        assert synth.form["temperature"] == 0.2
        assert synth.form["max_tokens"] == 333
        # synthesis system prompt + original conversation + drafts message
        assert len(synth.form["messages"]) == 4
        assert synth.form["messages"][0]["role"] == "system"
        assert "untrusted" in synth.form["messages"][0]["content"]
        assert synth.form["messages"][1:3] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        assert DRAFT_MARK in synth.form["messages"][3]["content"]

    run(scenario())


def test_internal_calls_get_isolated_request_state():
    async def scenario():
        # Open WebUI merges request.state.metadata into every completion call;
        # the pipe must hand internal calls a clone with clean state.
        request = make_request()
        await call_pipe(make_pipe(CANDIDATE_COUNT=2), request=request)
        assert BACKEND.total == 4  # manager + 2 candidates + synthesis
        for call in BACKEND.calls:
            assert isinstance(call.request, FakeRequest)
            assert call.request is not request
            assert call.request.scope["state"] == {}
        assert request.scope["state"]["metadata"]["chat_id"] == "chat-1"

    run(scenario())


# ----------------------------------------------------------------------
# 7-8. Failure tolerance and the minimum-success threshold
# ----------------------------------------------------------------------


def test_partial_failures_tolerated_above_threshold():
    async def scenario():
        async def impl(call):
            if call.is_synthesis:
                return ok_response("SYNTH-FINAL")
            if call.candidate_no <= 3:
                raise RuntimeError("transient sk-SECRET")
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        events = []

        async def emitter(event):
            events.append(event)

        pipe = make_pipe(MAX_RETRIES=0, ENABLE_MANAGER=False)  # 10 candidates, min 6
        result = await call_pipe(pipe, emitter=emitter)
        assert result == "SYNTH-FINAL"
        assert BACKEND.candidate_calls == 10
        assert BACKEND.synthesis_calls == 1
        drafts = BACKEND.calls[-1].form["messages"][-1]["content"]
        assert drafts.count(DRAFT_MARK) == 7
        final_line = next(
            e["data"]["description"]
            for e in events
            if e["data"]["description"].startswith("Generating candidate answers: 10/10")
        )
        assert "3 failed" in final_line  # failures are visible in the status

    run(scenario())


def test_too_many_failures_produce_sanitized_error():
    async def scenario():
        async def impl(call):
            if call.candidate_no is not None and call.candidate_no <= 6:
                raise RuntimeError("boom sk-SECRET http://10.0.0.5:9999/v1")
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(MAX_RETRIES=0, ENABLE_MANAGER=False)  # 4 successes < min 6
        with pytest.raises(Exception) as excinfo:
            await call_pipe(pipe)
        message = str(excinfo.value)
        assert "4 of 10" in message
        assert "sk-SECRET" not in message
        assert "10.0.0.5" not in message
        assert "Traceback" not in message
        assert BACKEND.synthesis_calls == 0  # no synthesis without quorum

    run(scenario())


# ----------------------------------------------------------------------
# 9. Timeouts and retries
# ----------------------------------------------------------------------


def test_per_call_timeout_enforced():
    async def scenario():
        async def impl(call):
            await asyncio.sleep(60)
            return ok_response("never")

        BACKEND.impl = impl
        pipe = make_pipe(
            CANDIDATE_COUNT=3,
            REQUEST_TIMEOUT_SECONDS=0.05,
            MAX_RETRIES=1,
            ENABLE_MANAGER=False,
        )
        started = time.monotonic()
        with pytest.raises(Exception, match="timed out"):
            await call_pipe(pipe)
        assert time.monotonic() - started < 5
        assert BACKEND.started == 6  # 3 candidates x (1 attempt + 1 retry)
        assert BACKEND.cancelled_count == 6  # timed-out attempts were cancelled
        assert BACKEND.active == 0

    run(scenario())


def test_transient_failures_are_retried():
    async def scenario():
        gate = asyncio.Event()

        async def impl(call):
            if call.is_synthesis:
                return ok_response("SYNTH-FINAL")
            if call.candidate_no <= 4:
                # Hold every first attempt until all four are in flight, then
                # fail them together: the four retries must then all succeed.
                if BACKEND.candidate_calls >= 4:
                    gate.set()
                await asyncio.wait_for(gate.wait(), timeout=5)
                raise RuntimeError("transient upstream hiccup")
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        events = []

        async def emitter(event):
            events.append(event)

        pipe = make_pipe(
            CANDIDATE_COUNT=4,
            MAX_CONCURRENCY=4,
            MIN_SUCCESSFUL_RESPONSES=4,
            MAX_RETRIES=1,
            ENABLE_MANAGER=False,
        )
        result = await call_pipe(pipe, emitter=emitter)
        assert result == "SYNTH-FINAL"
        assert BACKEND.candidate_calls == 8  # 4 failures + 4 successful retries
        assert BACKEND.total == 9
        final_line = next(
            e["data"]["description"]
            for e in events
            if e["data"]["description"].startswith("Generating candidate answers: 4/4")
        )
        assert "4 retried" in final_line  # retries are visible in the status

    run(scenario())


def test_permanent_errors_are_not_retried():
    async def scenario():
        async def impl(call):
            error = RuntimeError("bad request sk-SECRET")
            error.status_code = 400
            raise error

        BACKEND.impl = impl
        pipe = make_pipe(CANDIDATE_COUNT=3, MAX_RETRIES=2, ENABLE_MANAGER=False)
        with pytest.raises(Exception) as excinfo:
            await call_pipe(pipe)
        assert BACKEND.candidate_calls == 3  # one attempt each, no retries
        assert "HTTP 400" in str(excinfo.value)
        assert "sk-SECRET" not in str(excinfo.value)

    run(scenario())


# ----------------------------------------------------------------------
# 10-12. Synthesis input and output
# ----------------------------------------------------------------------


def test_only_successful_answers_reach_synthesis():
    async def scenario():
        async def impl(call):
            if call.is_synthesis:
                return ok_response("SYNTH-FINAL")
            if call.candidate_no in (2, 5):
                raise RuntimeError("candidate exploded")
            if call.candidate_no == 3:
                return ok_response("")  # empty answers are rejected too
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(
            CANDIDATE_COUNT=6,
            MIN_SUCCESSFUL_RESPONSES=1,
            MAX_RETRIES=0,
            ENABLE_MANAGER=False,
        )
        await call_pipe(pipe)
        drafts = BACKEND.calls[-1].form["messages"][-1]["content"]
        assert drafts.count(DRAFT_MARK) == 3
        for expected in ("candidate-1", "candidate-4", "candidate-6"):
            assert expected in drafts
        assert "1 OF 3" in drafts and "3 OF 3" in drafts
        assert "candidate exploded" not in drafts

    run(scenario())


def test_final_answer_is_synthesis_not_a_candidate():
    async def scenario():
        result = await call_pipe(make_pipe())
        # no marker in the default reply -> passthrough, no details block
        assert result == "SYNTH-FINAL"
        candidate_texts = {
            f"candidate-{c.candidate_no}" for c in BACKEND.calls if not c.is_synthesis
        }
        assert result not in candidate_texts

    run(scenario())


# ----------------------------------------------------------------------
# Synthesis summary
# ----------------------------------------------------------------------


def test_synthesis_summary_rendered_as_collapsible_section():
    async def scenario():
        async def impl(call):
            if call.is_synthesis:
                return ok_response(
                    "FINAL-ANSWER\n\n"
                    f"{SUMMARY_MARKER}\n"
                    "- compared 2 drafts, both agreed on the result\n"
                    "- kept draft 2's edge-case note"
                )
            if call.is_manager:
                return ok_response(default_plan())
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        result = await call_pipe(make_pipe(CANDIDATE_COUNT=2))
        assert result.startswith("FINAL-ANSWER")
        assert SUMMARY_MARKER not in result  # marker is consumed, not shown
        assert "<details>" in result and "</details>" in result
        assert "How this answer was assembled" in result
        assert "- kept draft 2's edge-case note" in result
        # the summary instruction was actually sent to the synthesis call
        synthesis_system = BACKEND.calls[-1].form["messages"][0]["content"]
        assert SUMMARY_MARKER in synthesis_system

    run(scenario())


def test_synthesis_summary_can_be_disabled():
    async def scenario():
        async def impl(call):
            if call.is_synthesis:
                return ok_response(f"FINAL\n\n{SUMMARY_MARKER}\n- stray summary")
            if call.is_manager:
                return ok_response(default_plan())
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(CANDIDATE_COUNT=2, SHOW_SYNTHESIS_SUMMARY=False)
        result = await call_pipe(pipe)
        assert result == "FINAL"  # stray marker section stripped, no details
        synthesis_system = BACKEND.calls[-1].form["messages"][0]["content"]
        assert SUMMARY_MARKER not in synthesis_system  # never asked for one

    run(scenario())


def test_degenerate_summary_only_output_returned_raw():
    async def scenario():
        async def impl(call):
            if call.is_synthesis:
                # marker first would leave an empty visible answer; the pipe
                # must fall back to the raw text instead of returning nothing
                return ok_response(f"{SUMMARY_MARKER}\n- summary with no answer")
            if call.is_manager:
                return ok_response(default_plan())
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        result = await call_pipe(make_pipe(CANDIDATE_COUNT=2))
        assert result == f"{SUMMARY_MARKER}\n- summary with no answer"

    run(scenario())


# ----------------------------------------------------------------------
# 13. Cancellation
# ----------------------------------------------------------------------


def test_cancellation_cleans_up_child_tasks():
    async def scenario():
        forever = asyncio.Event()

        async def impl(call):
            await forever.wait()
            return ok_response("never")

        BACKEND.impl = impl
        # An emitter is passed so the heartbeat ticker task exists too: the
        # leftover-task assertion below then also proves the ticker is reaped.
        pipe = make_pipe(
            CANDIDATE_COUNT=6,
            MAX_CONCURRENCY=6,
            PROGRESS_INTERVAL_SECONDS=0.5,
            ENABLE_MANAGER=False,
        )

        async def emitter(event):
            pass

        runner = asyncio.create_task(call_pipe(pipe, emitter=emitter))

        deadline = time.monotonic() + 5
        while BACKEND.started < 6:
            assert time.monotonic() < deadline, "candidates never started"
            await asyncio.sleep(0.01)

        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
        assert BACKEND.cancelled_count == 6  # every in-flight call was cancelled
        assert BACKEND.active == 0
        leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert leftovers == []

        # The recursion guard was reset on the way out: a new run still works.
        BACKEND.impl = None
        assert await call_pipe(make_pipe(CANDIDATE_COUNT=2)) == "SYNTH-FINAL"

    run(scenario())


# ----------------------------------------------------------------------
# Progress events
# ----------------------------------------------------------------------


def test_progress_events_report_counts_but_no_content():
    async def scenario():
        events = []

        async def emitter(event):
            events.append(event)

        await call_pipe(make_pipe(), emitter=emitter)
        assert all(e["type"] == "status" for e in events)
        descriptions = [e["data"]["description"] for e in events]
        assert descriptions[0].startswith("Planning 10 distinct assignments")
        assert any(
            d.startswith("Generating candidate answers: 0/10") for d in descriptions
        )
        assert any(
            d.startswith("Generating candidate answers: 10/10") for d in descriptions
        )
        assert "Synthesizing 10 successful answers" in descriptions
        assert events[-1]["data"]["done"] is True
        assert events[-1]["data"]["description"].startswith("Consensus complete")
        for description in descriptions:
            assert "candidate-" not in description
            assert "SYNTH-FINAL" not in description
            assert "default-angle" not in description  # assignments stay private

    run(scenario())


def test_heartbeat_ticks_while_waiting_on_the_model():
    async def scenario():
        events = []

        async def emitter(event):
            events.append(event)

        candidate_gate = asyncio.Event()
        synthesis_gate = asyncio.Event()

        async def impl(call):
            if call.is_synthesis:
                await synthesis_gate.wait()
                return ok_response("SYNTH-FINAL")
            await candidate_gate.wait()
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(
            CANDIDATE_COUNT=3,
            MIN_SUCCESSFUL_RESPONSES=3,
            PROGRESS_INTERVAL_SECONDS=0.5,
            ENABLE_MANAGER=False,
        )
        runner = asyncio.create_task(call_pipe(pipe, emitter=emitter))

        def candidate_ticks():
            return [
                d
                for d in (e["data"]["description"] for e in events)
                if d.startswith("Generating candidate answers: 0/3 (")
                and "3 running" in d
            ]

        # With zero candidates finished, heartbeats alone must keep the status
        # moving: liveness ticks carrying running counts and elapsed time.
        deadline = time.monotonic() + 10
        while len(candidate_ticks()) < 2:
            assert time.monotonic() < deadline, "no heartbeats while candidates ran"
            await asyncio.sleep(0.02)
        assert all(d.rstrip(")").endswith("s") for d in candidate_ticks())

        candidate_gate.set()

        def synthesis_ticks():
            return [
                d
                for d in (e["data"]["description"] for e in events)
                if d.startswith("Synthesizing 3 successful answers (")
            ]

        deadline = time.monotonic() + 10
        while not synthesis_ticks():
            assert time.monotonic() < deadline, "no heartbeat during synthesis"
            await asyncio.sleep(0.02)

        synthesis_gate.set()
        assert await runner == "SYNTH-FINAL"
        assert events[-1]["data"]["done"] is True

    run(scenario())


def test_progress_events_can_be_disabled():
    async def scenario():
        events = []

        async def emitter(event):
            events.append(event)

        await call_pipe(make_pipe(SHOW_PROGRESS=False), emitter=emitter)
        assert events == []

    run(scenario())


# ----------------------------------------------------------------------
# Manager delegation
# ----------------------------------------------------------------------


def test_manager_plans_first_and_assigns_distinct_angles():
    async def scenario():
        async def impl(call):
            if call.is_manager:
                return ok_response("1. alpha\n2. beta\n3. gamma\n4. delta")
            if call.is_synthesis:
                return ok_response("SYNTH-FINAL")
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(
            CANDIDATE_COUNT=4, MIN_SUCCESSFUL_RESPONSES=4, MANAGER_TEMPERATURE=0.9
        )
        result = await call_pipe(pipe)
        assert result == "SYNTH-FINAL"
        assert BACKEND.total == 6
        assert BACKEND.calls[0].is_manager

        manager_form = BACKEND.calls[0].form
        assert manager_form["model"] == "base-model"
        assert manager_form["temperature"] == 0.9
        assert manager_form["messages"][0]["role"] == "system"
        assert MANAGER_MARK in manager_form["messages"][0]["content"]
        # the conversation sits between the manager prompt and its task line
        assert manager_form["messages"][1:3] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        assert manager_form["messages"][-1]["role"] == "user"
        assert "numbered list" in manager_form["messages"][-1]["content"]

        directives = []
        for call in BACKEND.calls[1:5]:
            assert not call.is_manager and not call.is_synthesis
            last = call.form["messages"][-1]
            assert last["role"] == "system"
            assert "assigned angle" in last["content"]
            directives.append(last["content"].split('"')[1])
        # every candidate got its own distinct assignment
        assert sorted(directives) == ["alpha", "beta", "delta", "gamma"]

        drafts = BACKEND.calls[-1].form["messages"][-1]["content"]
        for directive in ("alpha", "beta", "gamma", "delta"):
            assert f'assigned angle: "{directive}"' in drafts

    run(scenario())


def test_manager_failure_falls_back_to_identical_sampling():
    async def scenario():
        events = []

        async def emitter(event):
            events.append(event)

        async def impl(call):
            if call.is_manager:
                raise RuntimeError("manager exploded sk-SECRET")
            if call.is_synthesis:
                return ok_response("SYNTH-FINAL")
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(CANDIDATE_COUNT=4, MIN_SUCCESSFUL_RESPONSES=4, MAX_RETRIES=0)
        result = await call_pipe(pipe, emitter=emitter)
        assert result == "SYNTH-FINAL"
        assert BACKEND.manager_calls == 1
        assert BACKEND.candidate_calls == 4
        for call in BACKEND.calls:
            if not call.is_manager and not call.is_synthesis:
                assert not any(
                    "assigned angle" in str(m.get("content"))
                    for m in call.form["messages"]
                )
        descriptions = [e["data"]["description"] for e in events]
        assert any("sample the same prompt" in d for d in descriptions)
        assert not any("sk-SECRET" in d for d in descriptions)

    run(scenario())


def test_manager_unparseable_plan_falls_back():
    async def scenario():
        async def impl(call):
            if call.is_manager:
                return ok_response("Just answer as well as you can, no list needed.")
            if call.is_synthesis:
                return ok_response("SYNTH-FINAL")
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(CANDIDATE_COUNT=4, MIN_SUCCESSFUL_RESPONSES=4)
        result = await call_pipe(pipe)
        assert result == "SYNTH-FINAL"
        assert BACKEND.total == 6  # manager attempt + 4 candidates + synthesis
        for call in BACKEND.calls[1:5]:
            assert not any(
                "assigned angle" in str(m.get("content"))
                for m in call.form["messages"]
            )

    run(scenario())


def test_short_plan_is_cycled_across_candidates():
    async def scenario():
        async def impl(call):
            if call.is_manager:
                return ok_response("1. alpha\n2. beta")
            if call.is_synthesis:
                return ok_response("SYNTH-FINAL")
            return ok_response(f"candidate-{call.candidate_no}")

        BACKEND.impl = impl
        pipe = make_pipe(CANDIDATE_COUNT=4, MIN_SUCCESSFUL_RESPONSES=4)
        await call_pipe(pipe)
        directives = [
            call.form["messages"][-1]["content"].split('"')[1]
            for call in BACKEND.calls[1:5]
        ]
        assert sorted(directives) == ["alpha", "alpha", "beta", "beta"]

    run(scenario())


def test_manager_can_be_disabled():
    async def scenario():
        pipe = make_pipe(CANDIDATE_COUNT=2, ENABLE_MANAGER=False)
        result = await call_pipe(pipe)
        assert result == "SYNTH-FINAL"
        assert BACKEND.total == 3  # candidates + synthesis only
        assert BACKEND.manager_calls == 0
        for call in BACKEND.calls:
            assert not any(
                "assigned angle" in str(m.get("content"))
                for m in call.form["messages"]
            )

    run(scenario())


# ----------------------------------------------------------------------
# Configuration behavior
# ----------------------------------------------------------------------


def test_internal_task_uses_a_single_call():
    async def scenario():
        # Title/tag generation must not trigger an 11-generation consensus.
        result = await call_pipe(make_pipe(), task="title_generation")
        assert result == "candidate-1"
        assert BACKEND.total == 1
        assert BACKEND.manager_calls == 0  # no planning for housekeeping tasks
        assert BACKEND.calls[0].form["model"] == "base-model"
        assert BACKEND.calls[0].form["stream"] is False

    run(scenario())


def test_min_successes_clamped_to_candidate_count():
    async def scenario():
        pipe = make_pipe(CANDIDATE_COUNT=4, MIN_SUCCESSFUL_RESPONSES=20)
        assert await call_pipe(pipe) == "SYNTH-FINAL"  # 4 successes suffice
        assert BACKEND.candidate_calls == 4

    run(scenario())


def test_valve_bounds_are_enforced():
    for overrides in (
        {"CANDIDATE_COUNT": 1},
        {"CANDIDATE_COUNT": 25},
        {"REQUEST_TIMEOUT_SECONDS": 0},
        {"MAX_RETRIES": -1},
        {"CANDIDATE_TEMPERATURE": 3.0},
        {"PROGRESS_INTERVAL_SECONDS": 0.1},
        {"MANAGER_TEMPERATURE": 3.0},
    ):
        with pytest.raises(ValidationError):
            Pipe.Valves(TARGET_MODEL_ID="base-model", **overrides)


def test_missing_target_model_fails_fast():
    async def scenario():
        pipe = make_pipe(TARGET_MODEL_ID="ghost-model")
        with pytest.raises(Exception, match="not found"):
            await call_pipe(pipe)
        assert BACKEND.total == 0

    run(scenario())


def test_unconfigured_target_model_fails_fast():
    async def scenario():
        pipe = make_pipe(TARGET_MODEL_ID="")
        with pytest.raises(Exception, match="TARGET_MODEL_ID"):
            await call_pipe(pipe)
        assert BACKEND.total == 0

    run(scenario())


def test_sync_users_interface_still_supported():
    async def scenario():
        # Open WebUI < 0.11 exposes a synchronous Users.get_user_by_id.
        class SyncUsers:
            @staticmethod
            def get_user_by_id(user_id):
                return BACKEND.lookup_user(user_id)

        original = consensus_10_pipe.Users
        consensus_10_pipe.Users = SyncUsers
        try:
            assert await call_pipe(make_pipe(CANDIDATE_COUNT=2)) == "SYNTH-FINAL"
        finally:
            consensus_10_pipe.Users = original

    run(scenario())
