"""OpenAI-compatible chat backend helpers."""

from __future__ import annotations

import base64
import io
import wave

import numpy as np

from realtime_interpreter.backends.openai_chat import (
    _chat_completions_url,
    _encode_wav_base64,
    _extract_chat_content,
)


def test_chat_completions_url_from_base_url() -> None:
    assert (
        _chat_completions_url("http://localhost:11434/v1")
        == "http://localhost:11434/v1/chat/completions"
    )


def test_chat_completions_url_accepts_full_endpoint() -> None:
    endpoint = "http://localhost:11434/v1/chat/completions"
    assert _chat_completions_url(endpoint) == endpoint


def test_encode_wav_base64_outputs_pcm16_wav() -> None:
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    raw = base64.b64decode(_encode_wav_base64(audio, sample_rate=16000))

    with wave.open(io.BytesIO(raw), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 3
        frames = wav.readframes(3)

    pcm = np.frombuffer(frames, dtype="<i2")
    assert pcm.tolist() == [0, 16383, -16383]


def test_extract_chat_content_string() -> None:
    payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "SRC: hi\nTGT: やあ"}}
        ]
    }
    assert _extract_chat_content(payload) == "SRC: hi\nTGT: やあ"


def test_extract_chat_content_parts() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}, {"text": " world"}],
                }
            }
        ]
    }
    assert _extract_chat_content(payload) == "hello world"


def test_extract_chat_content_malformed() -> None:
    assert _extract_chat_content({}) == ""
