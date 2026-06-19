"""Detailed status display and level meter helper tests."""

from __future__ import annotations

import numpy as np
from rich.text import Text

from realtime_interpreter.backends.base import BackendState, TranslationBackend
from realtime_interpreter.audio import build_level_meter, format_status, _compute_dbfs
from realtime_interpreter.backends.mlx_local import LocalMLXBackend
from realtime_interpreter.backends.openai_chat import OpenAIChatBackend, OpenAIChatAudioTranslator
from realtime_interpreter.backends.openai_realtime import OpenAIRealtimeBackend
from realtime_interpreter.backends.gemini_realtime import GeminiRealtimeBackend
from realtime_interpreter.translator import GemmaAudioTranslator


def test_backend_state_enum() -> None:
    assert BackendState.LOADING_MODEL.value == "loading_model"
    assert BackendState.CONNECTING.value == "connecting"
    assert BackendState.LISTENING.value == "listening"
    assert BackendState.SPEAKING.value == "speaking"
    assert BackendState.TRANSLATING.value == "translating"
    assert BackendState.RECONNECTING.value == "reconnecting"


def test_level_meter_logic() -> None:
    # メーターは幅依存を避けるため ASCII (#/-) を使う。
    # Test min DB bounds
    meter_min = build_level_meter(-60.0, num_blocks=10)
    assert meter_min == "-" * 10

    # Test max DB bounds
    meter_max = build_level_meter(0.0, num_blocks=10)
    assert meter_max == "#" * 10

    # Test intermediate
    meter_mid = build_level_meter(-24.0, num_blocks=10)
    assert "#" in meter_mid
    assert "-" in meter_mid
    assert len(meter_mid) == 10  # 全環境で幅1 ASCII = 10 文字


def test_compute_dbfs() -> None:
    # Empty chunk
    assert _compute_dbfs(np.zeros(0)) == -90.0
    # Zero amplitude
    assert _compute_dbfs(np.zeros(10)) == -90.0
    # Full scale sine peak
    assert _compute_dbfs(np.array([1.0, -1.0, 0.0])) == 0.0


def test_format_status_text() -> None:
    # format_status は「左スロット」= 音声入力レベルのみ (簡素化)。
    # バックエンド名や Listening/Capturing の語は含めない (通信ステータス側で表現)。
    listening_text = format_status(
        backend_name="Test Backend",
        in_segment=False,
        db=-30.0,
    )
    assert isinstance(listening_text, Text)
    raw_listening = listening_text.plain
    assert "Test Backend" not in raw_listening
    assert "Listening" not in raw_listening
    assert "-30.0dB" in raw_listening
    assert "[" in raw_listening and "]" in raw_listening  # メーター
    # 非発話時は (cur/max) を出さない
    assert "/" not in raw_listening

    # 発話区間中は (cur/max) を付与
    speaking_text = format_status(
        backend_name="Test Backend",
        in_segment=True,
        db=-15.0,
        current_duration=2.5,
        max_duration=8.0,
    )
    raw_speaking = speaking_text.plain
    assert "Capturing" not in raw_speaking
    assert "-15.0dB" in raw_speaking
    assert "2.5s / 8.0s" in raw_speaking


def test_renderer_status_composition() -> None:
    """左(音声) | 右(通信) の 1 行合成. 片方のみなら区切りなし."""
    from realtime_interpreter.renderer import StreamingRenderer

    r = StreamingRenderer()
    assert r._compose_status() is None  # 両方空 → 行なし

    r._audio_status = "[■■□□□□□□□□] -20.0dB"
    assert r._compose_status().plain == "[■■□□□□□□□□] -20.0dB"  # 左のみ

    r._comm_status = Text("● Listening (X)...")
    combined = r._compose_status().plain
    assert "[■■□□□□□□□□] -20.0dB" in combined
    assert "|" in combined
    assert "Listening (X)" in combined

    r._audio_status = ""
    assert r._compose_status().plain == "● Listening (X)..."  # 右のみ


class DummyStream:
    def stop(self) -> None:
        pass
    def close(self) -> None:
        pass


class DummySD:
    def query_devices(self) -> list[dict]:
        return [{"name": "dummy", "max_input_channels": 2}]


def test_backends_implement_status_callbacks() -> None:
    # We verify that backends conform to the TranslationBackend protocol
    # and properly implement set_status_callback.
    
    # MLX
    translator_mlx = GemmaAudioTranslator(model="e2b")
    backend_mlx = LocalMLXBackend(
        sd_module=DummySD(),  # type: ignore[arg-type]
        translator=translator_mlx,
        device_name="dummy",
        end_silence_ms=800,
        max_segment_seconds=8.0,
    )
    assert isinstance(backend_mlx, TranslationBackend)
    audio_mlx, comm_mlx = [], []
    backend_mlx.set_status_callback(audio_mlx.append, comm_mlx.append)
    # capture(メーター)は audio_cb、通信ステータスは comm_cb に振り分け
    assert backend_mlx._capture.status_callback == audio_mlx.append
    assert backend_mlx._comm_cb == comm_mlx.append

    # OpenAI Chat
    translator_chat = OpenAIChatAudioTranslator(api_key="dummy")
    backend_chat = OpenAIChatBackend(
        sd_module=DummySD(),  # type: ignore[arg-type]
        device_name="dummy",
        translator=translator_chat,
    )
    assert isinstance(backend_chat, TranslationBackend)
    audio_chat, comm_chat = [], []
    backend_chat.set_status_callback(audio_chat.append, comm_chat.append)
    assert backend_chat._capture.status_callback == audio_chat.append
    assert backend_chat._comm_cb == comm_chat.append

    # OpenAI Realtime
    backend_ort = OpenAIRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy",
    )
    assert isinstance(backend_ort, TranslationBackend)
    audio_ort, comm_ort = [], []
    backend_ort.set_status_callback(audio_ort.append, comm_ort.append)
    assert backend_ort._audio_cb == audio_ort.append
    assert backend_ort._comm_cb == comm_ort.append

    # Gemini Realtime
    backend_gemini = GeminiRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy",
    )
    assert isinstance(backend_gemini, TranslationBackend)
    audio_gem, comm_gem = [], []
    backend_gemini.set_status_callback(audio_gem.append, comm_gem.append)
    assert backend_gemini._audio_cb == audio_gem.append
    assert backend_gemini._comm_cb == comm_gem.append
