"""Gemini-specific backend and summarizer tests."""

from __future__ import annotations

import json
import numpy as np
import pytest

from realtime_interpreter.backends.gemini_realtime import (
    GeminiRealtimeBackend,
    _PendingTurn,
    _gemini_target_language_code,
)
from realtime_interpreter.summarizer import GeminiRESTSummarizer


class _FakeWS:
    """送信ペイロードを記録するだけの WebSocket スタブ."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, payload: str) -> None:
        self.sent.append(payload)


def test_gemini_target_language_code_maps_chinese_variants() -> None:
    assert _gemini_target_language_code("zh") == "zh-Hans"
    assert _gemini_target_language_code("zh-CN") == "zh-Hans"
    assert _gemini_target_language_code("zh-Hans") == "zh-Hans"
    assert _gemini_target_language_code("zh-TW") == "zh-Hant"
    assert _gemini_target_language_code("zh-hk") == "zh-Hant"
    assert _gemini_target_language_code("zh-Hant") == "zh-Hant"
    # 非中国語はそのまま
    assert _gemini_target_language_code("ja") == "ja"
    assert _gemini_target_language_code("pt-BR") == "pt-BR"


def test_gemini_setup_sends_mapped_target_language_code() -> None:
    # --to zh-Hant / zh-tw 等が translationConfig.targetLanguageCode に正しく反映される。
    for target, expected in [("zh-Hant", "zh-Hant"), ("zh-tw", "zh-Hant"), ("zh", "zh-Hans")]:
        backend = GeminiRealtimeBackend(
            sd_module=None,
            device_name="dummy",
            api_key="dummy_key",
            model="models/gemini-3.5-live-translate-preview",
            loopback=False,
            target_lang=target,
        )
        backend._ws = _FakeWS()
        backend._send_session_config()
        setup = json.loads(backend._ws.sent[-1])["setup"]
        tlc = setup["generationConfig"]["translationConfig"]["targetLanguageCode"]
        assert tlc == expected, f"{target} -> {tlc}, expected {expected}"


def test_gemini_pending_turn_accumulation() -> None:
    turn = _PendingTurn(
        start_offset_seconds=1.0,
        started_at=100.0,
        last_activity_at=100.0,
    )
    assert not turn.has_content()

    turn.source_parts.append("Hello ")
    turn.source_parts.append("world")
    assert turn.source() == "Hello world"
    assert turn.has_content()

    turn.target_parts.append("こんにちは")
    assert turn.target() == "こんにちは"


def test_gemini_backend_event_handling_transcription() -> None:
    # We can test _handle_event by stubbing out dependencies.
    backend = GeminiRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy_key",
        model="models/gemini-3.5-live-translate-preview",
        loopback=False,
    )

    # Trigger user transcription event
    event = {
        "serverContent": {
            "inputAudioTranscription": {
                "text": "Hello world",
                "finished": True
            }
        }
    }
    backend._handle_event(event)

    assert backend._pending_turn is not None
    assert backend._pending_turn.source() == "Hello world"


def test_gemini_backend_event_handling_model_turn() -> None:
    backend = GeminiRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy_key",
        model="models/gemini-3.5-live-translate-preview",
        loopback=False,
    )

    # Trigger model turn translation event
    event = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"text": "こんにちは"}
                ]
            }
        }
    }
    backend._handle_event(event)

    assert backend._pending_turn is not None
    assert backend._pending_turn.target() == "こんにちは"


def test_gemini_backend_event_handling_turn_complete() -> None:
    backend = GeminiRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy_key",
        model="models/gemini-3.5-live-translate-preview",
        loopback=False,
    )

    event_src = {
        "serverContent": {
            "inputAudioTranscription": {
                "text": "Hello",
                "finished": True
            }
        }
    }
    event_tgt = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"text": "こんにちは"}
                ]
            }
        }
    }
    event_complete = {
        "serverContent": {
            "turnComplete": True
        }
    }

    backend._handle_event(event_src)
    backend._handle_event(event_tgt)
    backend._handle_event(event_complete)

    # The pending turn is not committed immediately on turnComplete (due to debounce design).
    # We explicitly emit it to complete the segment.
    assert backend._pending_turn is not None
    with backend._pending_lock:
        backend._emit_pending_locked()

    assert backend._pending_turn is None
    assert backend._segment_queue.qsize() == 3  # 2 partials + 1 final commit

    # Verify segments in queue
    partial1 = backend._segment_queue.get()
    assert partial1.is_partial
    assert partial1.source == "Hello"

    partial2 = backend._segment_queue.get()
    assert partial2.is_partial
    assert partial2.target == "こんにちは"

    final = backend._segment_queue.get()
    assert not final.is_partial
    assert final.source == "Hello"
    assert final.target == "こんにちは"


def test_gemini_backend_event_handling_output_audio_transcription() -> None:
    backend = GeminiRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy_key",
        model="models/gemini-3.5-live-translate-preview",
        loopback=False,
    )

    event = {
        "serverContent": {
            "outputAudioTranscription": {
                "text": "こんにちは世界",
                "finished": True
            }
        }
    }
    backend._handle_event(event)

    assert backend._pending_turn is not None
    assert backend._pending_turn.target() == "こんにちは世界"


def test_gemini_backend_event_handling_modern_transcription() -> None:
    backend = GeminiRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy_key",
        model="models/gemini-3.5-live-translate-preview",
        loopback=False,
    )

    # 1. Test inputTranscription (user speech transcript)
    event_in = {
        "serverContent": {
            "inputTranscription": {
                "text": "Hello world",
                "finished": True
            }
        }
    }
    backend._handle_event(event_in)
    assert backend._pending_turn is not None
    assert backend._pending_turn.source() == "Hello world"

    # 2. Test inputTranscription incremental append
    event_in_append = {
        "serverContent": {
            "inputTranscription": {
                "text": " this is a test",
                "finished": True
            }
        }
    }
    backend._handle_event(event_in_append)
    # The new text should be appended, resulting in "Hello world this is a test"
    assert backend._pending_turn.source() == "Hello world this is a test"

    # 3. Test outputTranscription (model translation transcript)
    event_out = {
        "serverContent": {
            "outputTranscription": {
                "text": "こんにちは世界",
                "finished": True
            }
        }
    }
    backend._handle_event(event_out)
    assert backend._pending_turn.target() == "こんにちは世界"


def test_gemini_debounce_default_is_1200ms() -> None:
    """gemini の debounce 既定は 1200ms (翻訳バースト間ギャップ対策; 実ログ計測で決定).

    CLI 定数とバックエンドのコンストラクタ既定が一致していることも保証する。
    openai-realtime の 800ms とは独立 (混同して揃えられたら退行)。
    """
    import inspect

    from realtime_interpreter.main import DEFAULT_GEMINI_DEBOUNCE_MS
    from realtime_interpreter.backends.openai_realtime import TURN_DEBOUNCE_MS

    assert DEFAULT_GEMINI_DEBOUNCE_MS == 1200
    sig = inspect.signature(GeminiRealtimeBackend.__init__)
    assert sig.parameters["turn_debounce_ms"].default == 1200
    # openai-realtime は人間のポーズ統計基準のまま (別バックエンド別既定)
    assert TURN_DEBOUNCE_MS == 800

    # argparse 経由の実値も 1200 (add_argument にリテラルが直書きされたら検出)
    import sys

    from realtime_interpreter import main as m

    argv_backup = sys.argv
    try:
        sys.argv = ["realtime-interpreter", "--backend", "gemini-realtime"]
        args = m._parse_args()
        assert args.gemini_debounce_ms == 1200
    finally:
        sys.argv = argv_backup


def test_gemini_backend_debounce_commit() -> None:
    import time
    backend = GeminiRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy_key",
        model="models/gemini-3.5-live-translate-preview",
        turn_debounce_ms=50,  # 50ms for quick test
        max_segment_seconds=0.2, # 200ms
        loopback=False,
    )

    # Starts background threads when entering, but we can call it manually/simulate
    # Let's mock capture start
    backend._capture_start_monotonic = time.monotonic()

    # 1. Test debounce commit
    event = {
        "serverContent": {
            "inputTranscription": {
                "text": "Hello",
                "finished": True
            }
        }
    }
    backend._handle_event(event)
    assert backend._pending_turn is not None
    assert backend._pending_turn.source() == "Hello"

    # Start the emit loop manually or simulate its check
    # Let's simulate a quiet period exceeding 50ms
    time.sleep(0.06)
    with backend._pending_lock:
        now = time.monotonic()
        quiet_for = now - backend._pending_turn.last_activity_at
        debounce_hit = quiet_for >= backend._turn_debounce_seconds
        assert debounce_hit
        backend._emit_pending_locked()

    # The pending turn should be committed and cleared
    assert backend._pending_turn is None
    assert backend._segment_queue.qsize() == 2 # 1 partial + 1 final commit


