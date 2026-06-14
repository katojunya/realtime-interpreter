"""OpenAI-compatible chat backend helpers."""

from __future__ import annotations

import base64
import io
import wave

import numpy as np

from realtime_interpreter.backends.openai_chat import (
    OPENAI_CHAT_AUDIO_PROMPT,
    _chat_completions_url,
    _encode_wav_base64,
    _extract_chat_content,
    _normalize_for_dedup,
    _RepetitionGuard,
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


def test_normalize_for_dedup() -> None:
    # 小文字化・空白圧縮・前後の記号除去で同一視される
    assert _normalize_for_dedup("  I'm going TO go. ") == "i'm going to go"
    assert _normalize_for_dedup("Oh no!") == "oh no"
    assert _normalize_for_dedup("「はい」") == "はい"


def test_repetition_guard_drops_verbatim_repeat() -> None:
    guard = _RepetitionGuard()
    phrase = "I'm going to give you a little bit of background on that"
    assert guard.is_repeat(phrase) is False  # 初出は通す
    assert guard.is_repeat(phrase) is True  # 2回目以降は幻覚として弾く
    assert guard.is_repeat(phrase + ".") is True  # 記号差は同一視


def test_repetition_guard_allows_distinct_segments() -> None:
    guard = _RepetitionGuard()
    assert guard.is_repeat("This is the first real sentence.") is False
    assert guard.is_repeat("And this is a different real sentence.") is False


def test_repetition_guard_ignores_short_phrases() -> None:
    # 短い相づち等は逐語一致でも誤除去しない (min_chars 未満)
    guard = _RepetitionGuard()
    assert guard.is_repeat("Thanks") is False
    assert guard.is_repeat("Thanks") is False


def test_repetition_guard_session_wide_far_apart() -> None:
    # 既定 (history=None) は離れて再出現した定型句も弾く
    guard = _RepetitionGuard()
    boiler = "I'm going to give you a quick overview of our new product today"
    assert guard.is_repeat(boiler) is False  # 初出
    for i in range(30):
        guard.is_repeat(f"real distinct sentence number {i} here")
    assert guard.is_repeat(boiler) is True  # 30 件後の再出現でも弾く


def test_repetition_guard_history_window() -> None:
    # history を超えて離れた再出現は許容される
    guard = _RepetitionGuard(history=2, min_chars=4)
    assert guard.is_repeat("alpha repeats here") is False
    assert guard.is_repeat("beta filler text") is False
    assert guard.is_repeat("gamma filler text") is False  # alpha が窓から押し出される
    assert guard.is_repeat("alpha repeats here") is False  # 再び許容


def test_prompt_has_anti_hallucination_rules() -> None:
    prompt = OPENAI_CHAT_AUDIO_PROMPT
    assert "non-speech" in prompt
    assert "NEVER output filler" in prompt
