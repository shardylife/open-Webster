# moOde voice assistant

Wake-word voice assistant for a Raspberry Pi 5 running moOde audio.

Pipeline: mic → openWakeWord ("hey mycroft") → record until silence →
whisper.cpp → OpenRouter tool-calling LLM → TTS (OpenRouter, Piper fallback) → monitors.

## Layout on the Pi

```
~/assistant/
  assistant.py          this script
  audio_session.py      streaming mic/speaker session (barge-in support)
  config.env            see config.env.example
  whisper.cpp/          built whisper-cli + models/ggml-base.en.bin
  piper/piper           Piper binary
  voices/alan.onnx      Piper voice
```

System tools used: `mpc`, `aplay`/`arecord`, `ffmpeg`, `sox`, `pkill`.

## Run

```
python3 assistant.py
```

Set `STREAMING_AUDIO_ENABLED=0` in `config.env` to force the plain ALSA
path (no barge-in). The streaming path is also skipped automatically if
`audio_session` cannot start.

## Tests

The tests stub the wake-word model and audio session, so they run anywhere:

```
pip install numpy requests pytest
pytest assistant/tests
```
