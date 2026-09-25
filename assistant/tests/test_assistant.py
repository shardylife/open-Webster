import numpy as np
import pytest


def test_config_parsing(assistant):
    assert assistant.API_KEY == "sk-test"
    assert assistant.MODEL_ID == "test/model"


def test_prompt_wavs_generated(assistant):
    import os
    assert os.path.exists(assistant.CHIME)
    assert os.path.exists(assistant.BEEP)


def test_clean_transcript(assistant):
    assert assistant.clean_transcript("  [BLANK_AUDIO] play  (music)  jazz \n") == "play jazz"


def _clocked_frames(assistant, frames):
    clock = [0.0]
    seq = iter(frames)

    def now():
        return clock[0]

    def next_frame():
        clock[0] += assistant.CHUNK / assistant.RATE
        return next(seq, None)

    return next_frame, now


def test_record_stops_after_silence(assistant):
    loud = (np.ones(assistant.CHUNK) * 2000).astype(np.int16)
    quiet = np.zeros(assistant.CHUNK, dtype=np.int16)
    next_frame, now = _clocked_frames(assistant, [loud] * 3 + [quiet] * 200)
    pcm = assistant.record_until_silence(
        next_frame, 350.0, now, initial_pcm=np.array([1, 2, 3], dtype=np.int16))
    assert pcm[:3].tolist() == [1, 2, 3]
    secs = (pcm.size - 3) / assistant.RATE
    assert assistant.MIN_COMMAND_SECS < secs < 3.5


def test_record_caps_at_max_length(assistant):
    loud = (np.ones(assistant.CHUNK) * 2000).astype(np.int16)
    next_frame, now = _clocked_frames(assistant, [loud] * 10000)
    pcm = assistant.record_until_silence(next_frame, 350.0, now)
    assert abs(pcm.size / assistant.RATE - assistant.MAX_COMMAND_SECS) < 0.1


@pytest.fixture
def fake_mpc(assistant, monkeypatch):
    calls = []

    def mpc(*args):
        calls.append(args)
        if args == ("volume",):
            return "volume: 42%"
        if args[0] == "listall":
            return "RADIO/Triple J.pls\nRADIO/Jazz FM.pls"
        return ""

    monkeypatch.setattr(assistant, "mpc", mpc)
    return calls


def test_tools(assistant, fake_mpc):
    assert assistant.run_tool("volume", {"level": 250}) == "volume now 42%"
    assert ("volume", "100") in fake_mpc
    assert assistant.run_tool("music", {"action": "previous"}).startswith("ok, previous")
    assert assistant.run_tool("play_radio", {"query": "triple"}) == "playing RADIO/Triple J.pls"
    assert ("add", "RADIO/Triple J.pls") in fake_mpc
    assert assistant.run_tool("nope", {}) == "unknown tool nope"


def test_tool_args_must_be_object(assistant):
    assert assistant._parse_tool_args({"function": {"arguments": "[1,2]"}}) == {}
    assert assistant._parse_tool_args({"function": {"arguments": "{bad"}}) == {}


def test_timer_relabel_cancels_old(assistant):
    assert assistant.tool_set_timer("2", "tea") == "timer 'tea' set for 2 minutes"
    old = assistant.TIMERS["tea"].timer
    assistant.tool_set_timer(3, "tea")
    assert old.finished.is_set()
    assert assistant.TIMERS["tea"].timer is not old
    assert assistant.tool_set_timer(-1) == "timer needs a positive number of minutes"
    assert assistant.tool_set_timer("abc") == "timer needs a number of minutes"
    assert assistant.tool_cancel_timers() == "cancelled 1 timers"
    assert not assistant.TIMERS


def test_llm_errors_surface_eagerly(assistant):
    def failing_post(*a, **k):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        assistant.llm_events("hello", post=failing_post)
    texts = [e["text"] for e in assistant.reply_events("x") if e["type"] == "text_delta"]
    assert texts == [assistant.MSG_NOT_HEARD]


class Resp:
    def __init__(self, body=None, lines=None):
        self.body, self.lines = body, lines or []

    def raise_for_status(self):
        pass

    def json(self):
        return self.body

    def iter_lines(self, decode_unicode=False):
        return iter(self.lines)


def test_tools_then_stream(assistant, fake_mpc):
    posts = []

    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        posts.append(json)
        if stream:
            return Resp(lines=[
                b'data: {"choices":[{"delta":{"content":"Paused. "}}]}', b"",
                b'data: {"choices":[{"delta":{"content":"Enjoy the quiet."}}]}',
                b"data: [DONE]"])
        if len(posts) == 1:
            call = {"id": "c1", "function": {"name": "music", "arguments": '{"action":"pause"}'}}
        else:
            call = {"id": "c2", "function": {"name": "respond", "arguments": "{}"}}
        return Resp({"choices": [{"message": {"role": "assistant", "tool_calls": [call]}}]})

    events = list(assistant.llm_events("pause the music", post=fake_post))
    assert events == [
        {"type": "text_delta", "text": "Paused. "},
        {"type": "text_delta", "text": "Enjoy the quiet."},
        {"type": "done"},
    ]
    assert ("pause",) in fake_mpc
    assert posts[0]["tool_choice"] == "required"
    assert posts[-1]["stream"] is True
    assert posts[-1]["messages"][3]["role"] == "tool"


class Session:
    def __init__(self):
        self.queued = []
        self.barged = False

    def start_playback(self, allow_barge_in):
        pass

    def queue_pcm(self, chunks):
        self.queued.append(b"".join(chunks))
        return not self.barged

    def wait_playback(self):
        return True

    def barge_in_preroll(self):
        return np.zeros(4, dtype=np.int16)


def test_speak_streaming_phrases_and_barge_in(assistant, monkeypatch):
    monkeypatch.setattr(assistant, "iter_openrouter_speech",
                        lambda phrase, post=None: iter([phrase.encode()]))
    session = Session()
    assert assistant.speak_streaming(assistant.static_events("One. Two three."), session) == (True, None)
    assert session.queued == [b"One.", b"Two three."]
    session.barged = True
    ok, preroll = assistant.speak_streaming(assistant.static_events("Hi there."), session)
    assert ok is False
    assert preroll.size == 4


def test_speak_streaming_falls_back_to_piper(assistant, monkeypatch):
    def boom(*a, **k):
        raise ValueError("tts down")

    monkeypatch.setattr(assistant, "iter_openrouter_speech", boom)
    monkeypatch.setattr(assistant, "piper_session_pcm", lambda text: b"PIPER")
    session = Session()
    assert assistant.speak_streaming(assistant.static_events("Hello."), session) == (True, None)
    assert session.queued == [b"PIPER"]
