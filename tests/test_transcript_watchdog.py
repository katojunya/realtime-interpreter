"""入力転写ウォッチドッグ (沈黙検知 → session.update 再送による蘇生) のテスト.

ja 連続発話で入力転写ストリームがサーバ側で沈黙する事象への対策。
「翻訳 delta は届いているのに入力転写 delta が一定時間来ない」ときだけ
session.update を再送する。
"""

from __future__ import annotations

import json
import time

from realtime_interpreter.backends.openai_realtime import (
    INPUT_TRANSCRIPT_STALL_SECONDS,
    OpenAIRealtimeBackend,
)


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, payload: str) -> None:
        self.sent.append(payload)


def _backend() -> OpenAIRealtimeBackend:
    b = OpenAIRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy_key",
        loopback=False,
    )
    b._ws = _FakeWS()
    b._capture_start_monotonic = time.monotonic()
    b._send_session_config()  # 本番同様に計測基準をここで開始
    return b


def test_fresh_input_delta_suppresses_kick() -> None:
    """入力転写が直近に届いていれば再送は予約されない."""
    b = _backend()
    b._handle_event({"type": "session.input_transcript.delta", "delta": "こんにちは"})
    b._handle_event({"type": "session.output_transcript.delta", "delta": "hello"})
    assert not b._transcript_kick_pending


def test_stalled_input_schedules_kick_and_send_loop_resends_config() -> None:
    """沈黙検知 → 送信スレッド側で session.update が再送される."""
    b = _backend()
    b._last_input_delta_at = time.monotonic() - (INPUT_TRANSCRIPT_STALL_SECONDS + 1)
    b._handle_event({"type": "session.output_transcript.delta", "delta": "hello"})
    assert b._transcript_kick_pending

    sent_before = len(b._ws.sent)
    b._maybe_kick_transcription()
    assert not b._transcript_kick_pending  # 成功でクリア
    assert len(b._ws.sent) == sent_before + 1
    msg = json.loads(b._ws.sent[-1])
    assert msg["type"] == "session.update"
    # 転写設定込みの完全な config が再送される (これが蘇生のトリガ)
    assert msg["session"]["audio"]["input"]["transcription"]["model"]


def test_silence_without_translation_does_not_kick() -> None:
    """無音 (翻訳 delta も来ない) では検知自体が走らず再送されない."""
    b = _backend()
    b._last_input_delta_at = time.monotonic() - (INPUT_TRANSCRIPT_STALL_SECONDS + 1)
    # output delta を受けない限り _check_transcript_stall は呼ばれない
    b._maybe_kick_transcription()
    assert len([m for m in b._ws.sent if "session.update" in m]) == 1  # 初回 config のみ


def test_kick_respects_cooldown() -> None:
    """直前に再送済みならクールダウン中は再予約しない."""
    b = _backend()
    b._last_input_delta_at = time.monotonic() - (INPUT_TRANSCRIPT_STALL_SECONDS + 1)
    b._last_transcript_kick_at = time.monotonic()
    b._handle_event({"type": "session.output_transcript.delta", "delta": "hello"})
    assert not b._transcript_kick_pending


def test_session_config_resets_watchdog() -> None:
    """config 送信 (接続/再接続/蘇生) で計測が仕切り直される."""
    b = _backend()
    b._last_input_delta_at = 0.0
    b._transcript_kick_pending = True
    b._send_session_config()
    assert not b._transcript_kick_pending
    assert time.monotonic() - b._last_input_delta_at < 5.0


def test_no_kick_while_reconnecting() -> None:
    """再接続中は送らない (再接続経路が config を送り直す)."""
    b = _backend()
    b._transcript_kick_pending = True
    b._reconnecting.set()
    sent_before = len(b._ws.sent)
    b._maybe_kick_transcription()
    assert len(b._ws.sent) == sent_before
    assert b._transcript_kick_pending  # 再接続完了後の config 送信でクリアされる
