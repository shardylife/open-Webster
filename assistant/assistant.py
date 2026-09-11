#!/usr/bin/env python3
"""Voice assistant for the moOde studio streamer.

Pipeline: Yeti mic -> openWakeWord (hey mycroft) -> record until silence
-> whisper.cpp STT -> OpenRouter (tool-calling LLM) -> TTS -> monitors.

Two audio back ends:

* streaming  - ``audio_session.AudioSession`` owns mic and speaker, so
               replies stream phrase-by-phrase and the user can barge in.
* legacy     - ``arecord`` / ``aplay`` per interaction (no barge-in).

Music ducking: MPD sources pause/resume around interactions. If another
renderer (Spotify/AirPlay) holds the device, playback retries briefly then
gives up.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import requests
from openwakeword.model import Model

try:
    from audio_session import AudioSession, PhraseChunker, StreamingWavePcmDecoder
except ImportError:  # streaming back end is optional
    AudioSession = PhraseChunker = StreamingWavePcmDecoder = None  # type: ignore

# ---------- configuration ----------

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "assistant")
CONFIG_PATH = os.path.join(BASE, "config.env")


def load_env(path: str = CONFIG_PATH) -> dict:
    """Parse a simple KEY=VALUE file (comments, ``export`` and quotes allowed)."""
    cfg = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, value = line.split("=", 1)
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                cfg[key.strip()] = value
    except FileNotFoundError:
        raise SystemExit(f"config file not found: {path}")
    return cfg


CFG = load_env()
try:
    API_KEY = CFG["OPENROUTER_API_KEY"]
except KeyError:
    raise SystemExit(f"OPENROUTER_API_KEY missing from {CONFIG_PATH}")


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(CFG.get(key, default))
    except ValueError:
        return default


def _cfg_bool(key: str, default: bool) -> bool:
    return CFG.get(key, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


MODEL_ID = CFG.get("MODEL", "anthropic/claude-haiku-4.5")
TTS_MODEL = CFG.get("TTS_MODEL", "deepgram/aura-2")
TTS_VOICE = CFG.get("TTS_VOICE", "aura-2-hyperion-en")
MIC_DEVICE = CFG.get("MIC_DEVICE", "plughw:3,0")
OUT_DEVICE = CFG.get("OUT_DEVICE", "_audioout")
STREAMING_ENABLED = _cfg_bool("STREAMING_AUDIO_ENABLED", True)

WHISPER = os.path.join(BASE, "whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.join(BASE, "whisper.cpp/models/ggml-base.en.bin")
WHISPER_THREADS = 3
PIPER = os.path.join(BASE, "piper/piper")
VOICE = os.path.join(BASE, "voices/alan.onnx")
CURRENTSONG_PATH = "/var/local/www/currentsong.txt"

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"
HTTP_TIMEOUT = 30
STREAM_TIMEOUT = (5, 60)

RATE = 16000                     # mic sample rate
CHUNK = 1280                     # 80 ms @ 16 kHz
SESSION_RATE = 24000             # PCM rate the streaming session plays
OUT_RATE = 48000                 # what the moOde EQ chain wants from aplay
WAKE_THRESHOLD = _cfg_float("WAKE_THRESHOLD", 0.65)
WAKE_PHRASE = "hey mycroft"
WAKE_MODEL_GLOB = "hey_mycroft*.onnx"
WAKE_VAD_THRESHOLD = 0.5
WAKE_COOLDOWN_SECS = 1.0
SILENCE_RMS = _cfg_float("SILENCE_RMS", 350.0)
SILENCE_SECS = 1.2
MIN_COMMAND_SECS = 2.0
MAX_COMMAND_SECS = 10.0
MIN_TRANSCRIPT_CHARS = 2
BEEP_GAP_SECS = 0.12
MAX_TOOL_ROUNDS = 4

MSG_NOT_HEARD = "Sorry, I didn't catch that."
MSG_LLM_DOWN = "Sorry, I couldn't reach the language model."
MSG_TOO_MANY_STEPS = "Sorry, that took too many steps."
MSG_RESPONSE_FAILED = "Sorry, I couldn't complete that response."


class AssistantState(Enum):
    STANDBY = auto()
    RECORDING = auto()
    THINKING = auto()
    SPEAKING = auto()
    BARGE_RECORDING = auto()


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def set_state(state: AssistantState, suffix: str = "") -> None:
    log(f"state={state.name}{suffix}")


# ---------- shell helpers ----------

def run(cmd: list, timeout: float = 15, **kwargs) -> str:
    """Run a command without a shell; return stripped stdout ('' on failure)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
    except (subprocess.TimeoutExpired, OSError) as exc:
        log(f"command failed {cmd[0]}: {type(exc).__name__}")
        return ""
    return r.stdout.strip()


def mpc(*args: str) -> str:
    return run(["mpc", *args])


def mpd_playing() -> bool:
    return "[playing]" in mpc("status")


# ---------- audio out (legacy ALSA path) ----------

# Only one thing may talk at a time: an interaction, or a timer announcement.
OUTPUT_LOCK = threading.RLock()
# The streaming session, when active, so timers can speak through it.
ACTIVE_SESSION = None


def _kill_stale_dsp() -> None:
    """A camilladsp instance leaked by an interrupted playback holds the DAC
    forever. Only safe to clear when MPD isn't playing (renderers spawn their
    own instance which we must not touch while music is running)."""
    if not mpd_playing():
        run(["pkill", "-x", "camilladsp"])
        time.sleep(0.5)


def play_wav(path: str, tries: int = 3) -> bool:
    for _ in range(tries):
        try:
            r = subprocess.run(["aplay", "-D", OUT_DEVICE, path],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                return True
            log(f"aplay failed: {r.stderr.strip()[:120]}")
        except subprocess.TimeoutExpired:
            log("aplay timed out")
        _kill_stale_dsp()
        time.sleep(0.6)
    return False


def _envelope_tone(sample_rate: int, freq: float, duration: float, amp: float,
                   attack: float, release: float) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    env = np.clip(t / attack, 0, 1) * np.clip((duration - t) / release, 0, 1)
    return amp * np.sin(2 * np.pi * freq * t) * env


def _to_pcm16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16)


def _write_stereo_wav(path: str, mono_pcm: np.ndarray, sample_rate: int) -> None:
    stereo = np.column_stack([mono_pcm, mono_pcm]).ravel()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(stereo.tobytes())


def chime_samples(sample_rate: int) -> np.ndarray:
    t = np.arange(int(sample_rate * 0.28)) / sample_rate
    tone = 0.25 * np.sin(2 * np.pi * 880 * t) * np.exp(-t * 9)
    tone += 0.18 * np.sin(2 * np.pi * 1318.5 * t) * np.exp(-t * 7)
    return tone


def beep_samples(sample_rate: int) -> np.ndarray:
    return _envelope_tone(sample_rate, 1000, 0.12, 0.22, attack=0.008, release=0.025)


def ensure_wav(path: str, samples_for: Callable[[int], np.ndarray]) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_stereo_wav(path, _to_pcm16(samples_for(OUT_RATE)), OUT_RATE)
    return path


CHIME = ensure_wav(os.path.join(BASE, "chime.wav"), chime_samples)
BEEP = ensure_wav(os.path.join(BASE, "beep.wav"), beep_samples)
SESSION_BEEP = _to_pcm16(beep_samples(SESSION_RATE)).tobytes()
SESSION_CHIME = _to_pcm16(chime_samples(SESSION_RATE)).tobytes()


def play_beeps(count: int, path: str = BEEP) -> None:
    for index in range(count):
        if index:
            time.sleep(BEEP_GAP_SECS)
        play_wav(path)


def _repeat_pcm(pcm: bytes, count: int) -> list:
    gap = b"\x00\x00" * int(SESSION_RATE * BEEP_GAP_SECS)
    chunks = []
    for index in range(count):
        if index:
            chunks.append(gap)
        chunks.append(pcm)
    return chunks


def play_session_beeps(session, count: int, pcm: bytes = SESSION_BEEP) -> None:
    session.start_playback(allow_barge_in=False)
    session.queue_pcm(_repeat_pcm(pcm, count))
    session.wait_playback()


# ---------- TTS ----------

def _tts_request(text: str, response_format: str, stream: bool, post=requests.post):
    response = post(
        OPENROUTER_TTS_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": TTS_MODEL,
            "input": text,
            "voice": TTS_VOICE,
            "response_format": response_format,
        },
        stream=stream,
        timeout=STREAM_TIMEOUT,
    )
    response.raise_for_status()
    if not response.headers.get("Content-Type", "").lower().startswith("audio/"):
        raise ValueError("OpenRouter TTS returned a non-audio response")
    return response


def download_openrouter_speech(text: str, path: str, post=requests.post) -> None:
    response = _tts_request(text, "mp3", stream=False, post=post)
    if not response.content:
        raise ValueError("OpenRouter TTS returned empty audio")
    with open(path, "wb") as output:
        output.write(response.content)


def iter_openrouter_speech(text: str, post=requests.post) -> Iterator[bytes]:
    """Yield raw 24 kHz mono S16LE from OpenRouter's streamed WAV response."""
    response = _tts_request(text, "pcm", stream=True, post=post)
    decoder = StreamingWavePcmDecoder()
    received = False
    try:
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            received = True
            yield from decoder.feed(chunk)
        if not received:
            raise ValueError("OpenRouter TTS returned empty audio")
        yield from decoder.feed(b"", final=True)
    finally:
        close = getattr(response, "close", None)
        if close:
            close()


class _TempFiles:
    """Create temp paths and unlink them all on exit."""

    def __init__(self, *suffixes: str):
        self.paths = []
        for suffix in suffixes:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                self.paths.append(f.name)

    def __enter__(self):
        return self.paths

    def __exit__(self, *exc):
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def speak_openrouter(text: str) -> bool:
    with _TempFiles(".mp3", ".wav") as (raw, conv):
        try:
            download_openrouter_speech(text, raw)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                 "-ar", str(OUT_RATE), "-ac", "2", conv],
                capture_output=True, timeout=30, check=True,
            )
        except Exception as exc:
            log(f"OpenRouter TTS failed ({type(exc).__name__}); using Piper")
            return False
        return play_wav(conv)


def speak_piper(text: str) -> bool:
    with _TempFiles(".wav", ".wav") as (raw, conv):
        try:
            subprocess.run([PIPER, "-m", VOICE, "-f", raw],
                           input=text.encode(), capture_output=True, timeout=60, check=True)
            # EQ chain wants stereo 48k; piper emits mono 22.05k
            subprocess.run(["sox", raw, "-r", str(OUT_RATE), "-c", "2", conv],
                           capture_output=True, timeout=30, check=True)
        except Exception as exc:
            log(f"Piper TTS failed ({type(exc).__name__})")
            return False
        return play_wav(conv)


def piper_session_pcm(text: str) -> bytes:
    """Synthesize with Piper and resample to the session's mono 24 kHz PCM."""
    piper = subprocess.run([PIPER, "-m", VOICE, "--output-raw"],
                           input=text.encode(), capture_output=True, timeout=60, check=True)
    sox = subprocess.run(
        ["sox", "-t", "raw", "-r", "22050", "-e", "signed", "-b", "16", "-c", "1", "-",
         "-t", "raw", "-r", str(SESSION_RATE), "-e", "signed", "-b", "16", "-c", "1", "-"],
        input=piper.stdout, capture_output=True, timeout=30, check=True,
    )
    return sox.stdout


def speak(text: str) -> bool:
    """Speak through aplay (legacy path). Returns True if audio played."""
    if not text:
        return False
    log(f"speak: {text[:120]}")
    with OUTPUT_LOCK:
        return speak_openrouter(text) or speak_piper(text)


# ---------- music ducking ----------

class Duck:
    """Pause MPD for the duration of the block, resuming only if it was playing."""

    def __enter__(self):
        self.was_playing = mpd_playing()
        if self.was_playing:
            mpc("pause")
        return self

    def __exit__(self, *exc):
        if self.was_playing:
            mpc("play")


# ---------- tools ----------

@dataclass
class TimerEntry:
    timer: threading.Timer
    due_at: float


TIMERS: dict = {}
TIMER_LOCK = threading.Lock()
TIMER_SEQ = itertools.count(1)


def announce(text: str, chimes: int = 0) -> None:
    """Speak an unsolicited message (timer) via whichever back end is active."""
    with OUTPUT_LOCK, Duck():
        session = ACTIVE_SESSION
        if session is not None:
            try:
                if chimes:
                    play_session_beeps(session, chimes, SESSION_CHIME)
                speak_streaming(static_events(text), session)
                return
            except Exception as exc:
                log(f"session announce failed ({type(exc).__name__}); using aplay")
        for _ in range(chimes):
            play_wav(CHIME)
        speak(text)


def timer_fire(name: str) -> None:
    with TIMER_LOCK:
        TIMERS.pop(name, None)
    announce(f"Timer {name} is done.", chimes=2)


def tool_set_timer(minutes, label: Optional[str] = None) -> str:
    try:
        minutes = float(minutes)
    except (TypeError, ValueError):
        return "timer needs a number of minutes"
    if minutes <= 0:
        return "timer needs a positive number of minutes"
    seconds = minutes * 60
    with TIMER_LOCK:
        name = (label or "").strip() or str(next(TIMER_SEQ))
        existing = TIMERS.pop(name, None)
        if existing:
            existing.timer.cancel()
        t = threading.Timer(seconds, timer_fire, args=[name])
        t.daemon = True
        TIMERS[name] = TimerEntry(t, time.time() + seconds)
        t.start()
    return f"timer '{name}' set for {minutes:g} minutes"


def tool_cancel_timers() -> str:
    with TIMER_LOCK:
        n = len(TIMERS)
        for entry in TIMERS.values():
            entry.timer.cancel()
        TIMERS.clear()
    return f"cancelled {n} timers"


MUSIC_ACTIONS = {
    "play": "play", "pause": "pause", "stop": "stop",
    "next": "next", "previous": "prev",
}


def tool_music(action: str) -> str:
    if action not in MUSIC_ACTIONS:
        return f"unknown action {action}"
    mpc(MUSIC_ACTIONS[action])
    return f"ok, {action}: {mpc('current') or 'nothing queued'}"


def current_volume() -> str:
    match = re.search(r"volume:\s*(\S+)", mpc("volume"))
    return match.group(1) if match else "unknown"


def tool_volume(direction: Optional[str] = None, level=None) -> str:
    if level is not None:
        try:
            mpc("volume", str(max(0, min(100, int(level)))))
        except (TypeError, ValueError):
            return "volume level must be a whole number from 0 to 100"
    elif direction == "up":
        mpc("volume", "+10")
    elif direction == "down":
        mpc("volume", "-10")
    return f"volume now {current_volume()}"


def find_radio_station(query: str) -> Optional[str]:
    needle = query.lower()
    for candidates, extra in ((mpc("listall", "RADIO"), None), (mpc("listall"), "radio")):
        for line in candidates.splitlines():
            lower = line.lower()
            if needle in lower and (extra is None or extra in lower):
                return line
    return None


def tool_play_radio(query: str) -> str:
    query = query.strip()
    if not query:
        return "no station name given"
    station = find_radio_station(query)
    if not station:
        return f"no radio station matching '{query}'"
    mpc("clear")
    mpc("add", station)
    mpc("play")
    return f"playing {station}"


def tool_now_playing() -> str:
    cur = mpc("current")
    if cur:
        return f"MPD playing: {cur}"
    try:
        with open(CURRENTSONG_PATH) as f:
            info = " ".join(line.strip() for line in itertools.islice(f, 4) if line.strip())
    except OSError:
        info = ""
    return "MPD idle. currentsong info: " + (info or "nothing playing")


def _tool(name: str, description: str, properties: Optional[dict] = None,
          required: Optional[list] = None) -> dict:
    params = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": params}}


TOOLS_SPEC = [
    _tool("music",
          "Control music playback on the studio system (MPD sources: radio, local files, "
          "queued tracks). Cannot control Spotify Connect or AirPlay streams.",
          {"action": {"type": "string", "enum": list(MUSIC_ACTIONS)}}, ["action"]),
    _tool("volume", "Adjust playback volume (MPD sources).",
          {"direction": {"type": "string", "enum": ["up", "down"]},
           "level": {"type": "integer", "minimum": 0, "maximum": 100}}),
    _tool("play_radio",
          "Find and play an internet radio station from the moOde library by name, "
          "e.g. 'triple j', 'jazz'.",
          {"query": {"type": "string"}}, ["query"]),
    _tool("now_playing", "Report what is currently playing."),
    _tool("set_timer", "Set a countdown timer that chimes and announces when done.",
          {"minutes": {"type": "number"}, "label": {"type": "string"}}, ["minutes"]),
    _tool("cancel_timers", "Cancel all running timers."),
]

RESPOND_TOOL = _tool(
    "respond",
    "Choose this when no device action is needed and the user only needs a spoken answer.",
)

TOOL_HANDLERS = {
    "music": lambda a: tool_music(a.get("action", "")),
    "volume": lambda a: tool_volume(a.get("direction"), a.get("level")),
    "play_radio": lambda a: tool_play_radio(a.get("query", "")),
    "now_playing": lambda a: tool_now_playing(),
    "set_timer": lambda a: tool_set_timer(a.get("minutes", 5), a.get("label")),
    "cancel_timers": lambda a: tool_cancel_timers(),
}


def run_tool(name: str, args: dict) -> str:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"unknown tool {name}"
    try:
        return handler(args)
    except Exception as e:
        return f"tool error: {e}"


# ---------- LLM ----------

SYSTEM_PROMPT = (
    "You are the voice assistant built into a studio music streamer (Raspberry Pi 5, "
    "moOde audio, Focusrite interface, studio monitors) in Jordan's home studio in Australia. "
    "You hear transcribed speech and answer by text-to-speech, so answer briefly - one or two "
    "spoken-style sentences, no markdown, no lists, no emoji. Numbers written out naturally. "
    "Use tools for music control and timers. Spotify and AirPlay are controlled from the phone, "
    "not by you - if asked, say so. If the transcription looks garbled, ask to repeat."
)


def _initial_messages(text: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def _chat(payload: dict, post=requests.post, stream: bool = False):
    response = post(
        OPENROUTER_CHAT_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL_ID, **payload},
        stream=stream,
        timeout=STREAM_TIMEOUT if stream else HTTP_TIMEOUT,
    )
    response.raise_for_status()
    return response


def _parse_tool_args(call: dict) -> dict:
    try:
        args = json.loads(call.get("function", {}).get("arguments") or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def _apply_tool_calls(messages: list, calls: Iterable[dict]) -> None:
    for call in calls:
        name = call.get("function", {}).get("name", "")
        args = _parse_tool_args(call)
        result = run_tool(name, args)
        log(f"tool {name}({args}) -> {result[:80]}")
        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})


def ask_llm(text: str, post=requests.post) -> str:
    """Blocking tool loop that returns the final spoken reply (legacy path)."""
    messages = _initial_messages(text)
    for _ in range(MAX_TOOL_ROUNDS):
        response = _chat({"messages": messages, "tools": TOOLS_SPEC, "max_tokens": 400}, post)
        msg = response.json()["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            return msg.get("content") or ""
        _apply_tool_calls(messages, calls)
    return MSG_TOO_MANY_STEPS


def route_tools(text: str, post=requests.post) -> Optional[list]:
    """Run device actions first. Returns the conversation to answer from, or
    None when the model would not stop calling tools."""
    messages = _initial_messages(text)
    for _ in range(MAX_TOOL_ROUNDS):
        response = _chat({
            "messages": messages,
            "tools": TOOLS_SPEC + [RESPOND_TOOL],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tokens": 300,
        }, post)
        message = response.json()["choices"][0]["message"]
        actionable = [
            call for call in message.get("tool_calls") or []
            if call.get("function", {}).get("name") != "respond"
        ]
        if not actionable:
            return messages
        messages.append(message)
        _apply_tool_calls(messages, actionable)
    return None


def static_events(text: str) -> Iterator[dict]:
    yield {"type": "text_delta", "text": text}
    yield {"type": "done"}


def stream_text_events(messages: list, post=requests.post) -> Iterator[dict]:
    """Stream a tool-free spoken answer as text_delta events."""
    response = _chat({"messages": messages, "max_tokens": 400, "stream": True}, post, stream=True)
    try:
        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                yield {"type": "text_delta", "text": content}
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
    yield {"type": "done"}


def llm_events(text: str, post=requests.post) -> Iterator[dict]:
    """Route tools now (raising on failure), then return the streamed answer.

    This is deliberately not a generator: tool routing happens eagerly so a
    network error surfaces to the caller instead of inside the TTS loop.
    """
    messages = route_tools(text, post)
    if messages is None:
        return static_events(MSG_TOO_MANY_STEPS)
    return stream_text_events(messages, post)


# ---------- streaming speech ----------

def speak_streaming(text_events: Iterable[dict], session):
    """Synthesize safe phrases in order.

    Returns (completed, barge_in_preroll): completed is False when the user
    interrupted, in which case the pre-roll PCM captured during the barge-in
    is returned so the next recording can start from it.
    """
    chunker = PhraseChunker()
    session.start_playback(allow_barge_in=True)

    def play_phrase(phrase: str) -> bool:
        if not phrase:
            return True
        log(f"speak: {phrase[:120]}")
        try:
            return session.queue_pcm(iter_openrouter_speech(phrase))
        except Exception as exc:
            log(f"streaming TTS failed ({type(exc).__name__}); using Piper")
            try:
                return session.queue_pcm([piper_session_pcm(phrase)])
            except Exception as piper_exc:
                raise RuntimeError("both streaming and fallback TTS failed") from piper_exc

    finalized = False
    for event in text_events:
        event_type = event.get("type")
        if event_type == "text_delta":
            phrases = chunker.feed(event.get("text", ""))
        elif event_type == "done":
            phrases = chunker.feed("", final=True)
            finalized = True
        else:
            continue
        for phrase in phrases:
            if not play_phrase(phrase):
                return False, session.barge_in_preroll()

    if not finalized:
        for phrase in chunker.feed("", final=True):
            if not play_phrase(phrase):
                return False, session.barge_in_preroll()
    if not session.wait_playback():
        return False, session.barge_in_preroll()
    return True, None


# ---------- STT ----------

def transcribe(pcm16: np.ndarray) -> str:
    with _TempFiles(".wav") as (path,):
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(np.asarray(pcm16, dtype=np.int16).tobytes())
        try:
            r = subprocess.run(
                [WHISPER, "-m", WHISPER_MODEL, "-f", path, "-nt", "-np",
                 "-t", str(WHISPER_THREADS)],
                capture_output=True, text=True, timeout=60)
        except (subprocess.TimeoutExpired, OSError) as exc:
            log(f"whisper failed: {type(exc).__name__}")
            return ""
    if r.returncode != 0:
        log(f"whisper exited {r.returncode}: {r.stderr.strip()[:120]}")
        return ""
    return clean_transcript(r.stdout)


def clean_transcript(raw: str) -> str:
    text = re.sub(r"\[.*?\]|\(.*?\)", "", raw)
    return re.sub(r"\s+", " ", text).strip()


# ---------- recording ----------

def frame_rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


def record_until_silence(next_frame: Callable[[], Optional[np.ndarray]],
                         threshold: float,
                         now: Callable[[], float] = time.monotonic,
                         initial_pcm=None) -> np.ndarray:
    """Pull frames until the speaker goes quiet or the command times out."""
    frames = []
    if initial_pcm is not None and np.asarray(initial_pcm).size:
        frames.append(np.asarray(initial_pcm, dtype=np.int16).copy())
    silent = 0.0
    started = now()
    while True:
        elapsed = now() - started
        if elapsed >= MAX_COMMAND_SECS:
            break
        frame = next_frame()
        if frame is None or frame.size == 0:
            break
        frames.append(frame)
        if frame_rms(frame) < threshold:
            silent += frame.size / RATE
            if silent >= SILENCE_SECS and elapsed > MIN_COMMAND_SECS:
                break
        else:
            silent = 0.0
    return np.concatenate(frames) if frames else np.zeros(1, dtype=np.int16)


def _read_pcm_frame(stdout) -> Optional[np.ndarray]:
    data = stdout.read(CHUNK * 2)
    if len(data) < CHUNK * 2:
        return None
    return np.frombuffer(data, dtype=np.int16)


def record_command(proc, now: Callable[[], float] = time.monotonic) -> np.ndarray:
    play_beeps(1)
    # Discard audio captured while the ready beep was playing.
    proc.stdout.read(RATE * 2)
    pcm = record_until_silence(lambda: _read_pcm_frame(proc.stdout), SILENCE_RMS, now)
    play_beeps(2)
    return pcm


def record_session_command(session, initial_pcm=None, start_beep: bool = True,
                           now: Callable[[], float] = time.monotonic) -> np.ndarray:
    session.discard_frames()
    if start_beep:
        play_session_beeps(session, 1)
        session.discard_frames()
    pcm = record_until_silence(
        lambda: session.read_frame(timeout=2.0),
        session.detector.threshold, now, initial_pcm)
    play_session_beeps(session, 2)
    return pcm


# ---------- interaction ----------

def reply_events(text: str) -> Iterator[dict]:
    """Turn a transcript into spoken-text events, never raising."""
    if len(text) < MIN_TRANSCRIPT_CHARS:
        return static_events(MSG_NOT_HEARD)
    try:
        return llm_events(text)
    except Exception as exc:
        log(f"llm error: {type(exc).__name__}: {exc}")
        return static_events(MSG_LLM_DOWN)


def handle_streaming_interaction(session) -> None:
    initial_pcm = None
    while True:
        set_state(AssistantState.RECORDING if initial_pcm is None
                  else AssistantState.BARGE_RECORDING)
        pcm = record_session_command(session, initial_pcm=initial_pcm,
                                     start_beep=initial_pcm is None)
        set_state(AssistantState.THINKING)
        text = transcribe(pcm)
        log(f"heard: {text!r}")
        events = reply_events(text)
        set_state(AssistantState.SPEAKING)
        try:
            completed, initial_pcm = speak_streaming(events, session)
        except Exception as exc:
            log(f"streaming response failed: {type(exc).__name__}: {exc}")
            try:
                speak_streaming(static_events(MSG_RESPONSE_FAILED), session)
            except Exception:
                speak(MSG_RESPONSE_FAILED)
            return
        if completed:
            return


def handle_legacy_interaction(proc) -> None:
    set_state(AssistantState.RECORDING)
    pcm = record_command(proc)
    set_state(AssistantState.THINKING)
    text = transcribe(pcm)
    log(f"heard: {text!r}")
    if len(text) < MIN_TRANSCRIPT_CHARS:
        reply = MSG_NOT_HEARD
    else:
        try:
            reply = ask_llm(text)
        except Exception as exc:
            log(f"llm error: {type(exc).__name__}: {exc}")
            reply = MSG_LLM_DOWN
    set_state(AssistantState.SPEAKING)
    speak(reply)


# ---------- wake word ----------

def find_wake_model(model_dir: Optional[str] = None) -> str:
    if model_dir is None:
        import openwakeword as openwakeword_pkg
        model_dir = os.path.join(
            os.path.dirname(openwakeword_pkg.__file__), "resources", "models")
    matches = sorted(glob.glob(os.path.join(model_dir, WAKE_MODEL_GLOB)))
    if not matches:
        raise FileNotFoundError(
            f"wake model {WAKE_MODEL_GLOB!r} not found in {model_dir}")
    return matches[0]


def load_wake_model() -> Model:
    return Model(wakeword_model_paths=[find_wake_model()], vad_threshold=WAKE_VAD_THRESHOLD)


def reset_wake_model(model) -> None:
    reset = getattr(model, "reset", None)
    if reset:
        reset()


class WakeDetector:
    """Wraps openWakeWord with a post-interaction cooldown."""

    def __init__(self, model, threshold: float = WAKE_THRESHOLD):
        self.model = model
        self.threshold = threshold
        self.cooldown_until = 0.0

    def triggered(self, frame: np.ndarray) -> bool:
        score = max(self.model.predict(frame).values())
        if score < self.threshold or time.monotonic() < self.cooldown_until:
            return False
        log(f"wake ({score:.2f})")
        reset_wake_model(self.model)
        return True

    def finished_interaction(self) -> None:
        reset_wake_model(self.model)
        self.cooldown_until = time.monotonic() + WAKE_COOLDOWN_SECS


# ---------- main loops ----------

def mic_stream() -> subprocess.Popen:
    return subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(RATE),
         "-c", "1", "-t", "raw", "-q"],
        stdout=subprocess.PIPE)


def legacy_main() -> None:
    log(f"assistant starting; model={MODEL_ID}; streaming_audio=off")
    wake = WakeDetector(load_wake_model())
    proc = mic_stream()
    set_state(AssistantState.STANDBY, f"; listening for {WAKE_PHRASE!r}")
    while True:
        frame = _read_pcm_frame(proc.stdout)
        if frame is None:
            log("mic stream ended; restarting arecord")
            try:
                proc.kill()
            except Exception:
                pass
            time.sleep(1)
            proc = mic_stream()
            continue
        if not wake.triggered(frame):
            continue
        try:
            with OUTPUT_LOCK, Duck():
                handle_legacy_interaction(proc)
        except Exception as exc:
            log(f"interaction failed: {type(exc).__name__}: {exc}")
        wake.finished_interaction()
        set_state(AssistantState.STANDBY, f"; listening for {WAKE_PHRASE!r}")


def start_streaming_session():
    """Bring up the streaming back end, or raise if it can't be used."""
    if AudioSession is None:
        raise RuntimeError("audio_session module not available")
    session = AudioSession(MIC_DEVICE, OUT_DEVICE)
    session.start()
    return session


def streaming_main(session, wake: WakeDetector) -> None:
    global ACTIVE_SESSION
    log(f"assistant starting; model={MODEL_ID}; streaming_audio=on")
    ACTIVE_SESSION = session
    set_state(AssistantState.STANDBY, f"; listening for {WAKE_PHRASE!r}")
    try:
        while True:
            frame = session.read_frame(timeout=None)
            if frame is None or not wake.triggered(frame):
                continue
            try:
                with OUTPUT_LOCK, Duck():
                    handle_streaming_interaction(session)
            except Exception as exc:
                log(f"interaction failed: {type(exc).__name__}: {exc}")
            wake.finished_interaction()
            session.discard_frames()
            set_state(AssistantState.STANDBY, f"; listening for {WAKE_PHRASE!r}")
    finally:
        ACTIVE_SESSION = None
        session.close()


def main() -> None:
    if STREAMING_ENABLED:
        try:
            session = start_streaming_session()
            wake = WakeDetector(load_wake_model())
        except Exception as exc:
            log(f"streaming audio unavailable ({type(exc).__name__}: {exc}); using ALSA fallback")
        else:
            return streaming_main(session, wake)
    return legacy_main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopping")
