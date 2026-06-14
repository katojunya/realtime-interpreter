"""WebSocketStreamingBackend の共有機構(emit/debounce/reconnect)のテスト.

実 WebSocket・実オーディオには触れず、フックをモックしたサブクラスで検証する。
"""

from __future__ import annotations

import time

from realtime_interpreter.backends._ws_streaming_base import (
    WebSocketStreamingBackend,
    _PendingTurn,
)


class _FakeWS:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeBackend(WebSocketStreamingBackend):
    SAMPLE_RATE = 16000
    PROACTIVE_RECONNECT_SECONDS = None
    LOG_NAME = "Fake"
    THREAD_PREFIX = "fake"

    def __init__(self, **kw) -> None:
        defaults = dict(
            sd_module=None,
            device_name="dev",
            api_key="key",
            model="m",
            turn_debounce_ms=800,
            max_segment_seconds=8.0,
            source_lang="en",
            target_lang="ja",
            loopback=False,
        )
        defaults.update(kw)
        super().__init__(**defaults)
        self.opened = 0
        self.config_resume: list[bool] = []

    def _open_websocket(self) -> None:
        self._ws = _FakeWS()
        self.opened += 1

    def _send_session_config(self, resume: bool = False) -> None:
        self.config_resume.append(resume)

    def _send_audio_chunk(self, audio) -> None:  # pragma: no cover - unused here
        pass

    def _handle_event(self, event) -> None:  # pragma: no cover - unused here
        pass


def _drain(b: _FakeBackend):
    out = []
    while not b._segment_queue.empty():
        out.append(b._segment_queue.get_nowait())
    return out


def test_emit_pending_final() -> None:
    b = _FakeBackend()
    now = time.monotonic()
    b._pending_turn = _PendingTurn(0.0, now, now, ["hello"], ["やあ"])
    with b._pending_lock:
        b._emit_pending_locked()
    segs = _drain(b)
    assert len(segs) == 1
    assert segs[0].source == "hello" and segs[0].target == "やあ"
    assert segs[0].is_partial is False
    assert b._pending_turn is None


def test_emit_skips_empty() -> None:
    b = _FakeBackend()
    now = time.monotonic()
    b._pending_turn = _PendingTurn(0.0, now, now, [], [])
    with b._pending_lock:
        b._emit_pending_locked()
    assert _drain(b) == []


def test_append_accumulates_and_emits_partial() -> None:
    b = _FakeBackend()
    b._append_input("Hello ")
    b._append_output("やあ ")
    assert b._pending_turn is not None
    assert b._pending_turn.source() == "Hello"
    partials = [s for s in _drain(b) if s.is_partial]
    assert partials  # 暫定セグメントが出ている


def test_clean_hook_default_passthrough() -> None:
    b = _FakeBackend()
    assert b._clean("  spaced  ") == "  spaced  "


def test_reconnect_success() -> None:
    b = _FakeBackend()
    assert b._reconnect() is True
    assert b.opened == 1
    assert b.config_resume == [True]  # 再接続時は resume=True
    assert not b._reconnecting.is_set()  # finally で解除


def test_reconnect_aborts_when_closing() -> None:
    b = _FakeBackend()
    b._closing.set()
    assert b._reconnect() is False
    assert b.opened == 0


def test_proactive_reconnect_disabled_when_none() -> None:
    b = _FakeBackend()  # PROACTIVE_RECONNECT_SECONDS = None
    b._ws = _FakeWS()
    b._connected_at = time.monotonic() - 10_000
    b._maybe_proactive_reconnect()
    assert b._ws.closed is False
    assert not b._reconnecting.is_set()


class _ProactiveBackend(_FakeBackend):
    PROACTIVE_RECONNECT_SECONDS = 1.0


def test_proactive_reconnect_fires_past_threshold() -> None:
    b = _ProactiveBackend()
    b._ws = _FakeWS()
    b._connected_at = time.monotonic() - 5.0  # 閾値 1.0 を超過
    b._maybe_proactive_reconnect()
    assert b._ws.closed is True  # 自分から ws を閉じた
    assert b._reconnecting.is_set()
