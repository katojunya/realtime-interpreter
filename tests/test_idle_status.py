"""openai-chat / mlx バックエンドの「推論完了→待機(Listening)」表示のテスト.

VAD ベースの request/response バックエンドは、translate() 完了後に右スロットを
Listening に戻す (それまでは Translating が出っぱなしだった)。
"""

from __future__ import annotations

import numpy as np

from realtime_interpreter.backends.openai_chat import (
    OpenAIChatBackend,
    OpenAIChatAudioTranslator,
)
from realtime_interpreter.backends.mlx_local import LocalMLXBackend
from realtime_interpreter.translator import GemmaAudioTranslator


class DummySD:
    def query_devices(self) -> list[dict]:
        return [{"name": "dummy", "max_input_channels": 2}]


class _FakeSeg:
    start_offset_seconds = 0.0
    duration_seconds = 1.0
    audio = np.zeros(16, dtype=np.float32)


class _Result:
    source = "hello"
    target = "こんにちは"


def _plain(v) -> str:
    return v.plain if hasattr(v, "plain") else str(v)


# ---------------- openai-chat ----------------


def _make_chat_backend() -> OpenAIChatBackend:
    return OpenAIChatBackend(
        sd_module=DummySD(),  # type: ignore[arg-type]
        device_name="dummy",
        translator=OpenAIChatAudioTranslator(api_key="dummy"),
    )


def test_chat_comm_strings() -> None:
    b = _make_chat_backend()
    assert b._comm_translating().plain == "> Translating (OpenAI Chat API Request)... [Waiting API]"
    assert b._comm_listening().plain == "> Listening (OpenAI Chat API)..."


def test_chat_returns_to_listening_after_translate() -> None:
    b = _make_chat_backend()
    comm: list[object] = []
    b.set_status_callback(lambda _x: None, comm.append)
    b._capture.segments = lambda: iter([_FakeSeg()])  # type: ignore[assignment]
    b.translator.translate = lambda audio: _Result()  # type: ignore[assignment]

    segs = list(b.stream_segments())
    assert len(segs) == 1
    plains = [_plain(c) for c in comm]
    assert any("Translating" in p for p in plains)  # 推論中
    assert plains[-1] == "> Listening (OpenAI Chat API)..."  # 完了後は待機


# ---------------- mlx ----------------


def _make_mlx_backend() -> LocalMLXBackend:
    return LocalMLXBackend(
        sd_module=DummySD(),  # type: ignore[arg-type]
        translator=GemmaAudioTranslator(model="e2b"),
        device_name="dummy",
        end_silence_ms=800,
        max_segment_seconds=8.0,
    )


def test_mlx_comm_strings() -> None:
    b = _make_mlx_backend()
    assert b._comm_listening().startswith("> Listening (")
    assert b._comm_listening().endswith(" Local GPU)...")
    assert "Local GPU Inference)..." in b._comm_translating()


def test_mlx_returns_to_listening_after_translate() -> None:
    b = _make_mlx_backend()
    comm: list[object] = []
    b.set_status_callback(lambda _x: None, comm.append)
    b._capture.segments = lambda: iter([_FakeSeg()])  # type: ignore[assignment]
    b.translator.translate = lambda audio: _Result()  # type: ignore[assignment]

    segs = list(b.stream_segments())
    assert len(segs) == 1
    plains = [_plain(c) for c in comm]
    assert any("Translating" in p for p in plains)
    assert plains[-1] == b._comm_listening()
    assert plains[-1].startswith("> Listening (")


def test_returns_to_listening_even_when_translate_fails() -> None:
    b = _make_chat_backend()
    comm: list[object] = []
    b.set_status_callback(lambda _x: None, comm.append)
    b._capture.segments = lambda: iter([_FakeSeg()])  # type: ignore[assignment]

    def _boom(audio):
        raise RuntimeError("api down")

    b.translator.translate = _boom  # type: ignore[assignment]
    segs = list(b.stream_segments())
    assert segs == []  # 例外時は yield されない
    assert _plain(comm[-1]) == "> Listening (OpenAI Chat API)..."  # それでも待機へ戻す
