"""音量メーターのティッカー (translate() ブロッキング中も止まらない) のテスト.

実デバイス不要。_BaseSpeechSegmentCapture を直接使い、status ティッカーが
segments()/translate() と独立に周期送出することを確認する。
"""

from __future__ import annotations

import time

from rich.text import Text

from realtime_interpreter.audio import _BaseSpeechSegmentCapture


def _make() -> _BaseSpeechSegmentCapture:
    return _BaseSpeechSegmentCapture(
        sample_rate=16000, end_silence_ms=800, max_segment_seconds=8.0
    )


def test_emit_meter_once_uses_current_level() -> None:
    cap = _make()
    got: list = []
    cap.status_callback = got.append
    cap.current_level_db = -22.5
    cap._emit_meter_once()
    assert len(got) == 1
    assert isinstance(got[0], Text)
    assert "-22.5dB" in got[0].plain


def test_emit_meter_once_noop_without_callback() -> None:
    cap = _make()
    cap.current_level_db = -10.0
    cap._emit_meter_once()  # callback 無し → 例外なく no-op


def test_ticker_emits_without_segments_loop() -> None:
    # segments()/translate() を一切回さなくても、ティッカーが周期的にメーターを送出する
    # = 翻訳ブロッキング中でもメーターが動き続けることの担保。
    cap = _make()
    got: list = []
    cap.status_callback = got.append
    cap.current_level_db = -30.0
    cap._start_status_thread()
    try:
        time.sleep(0.25)  # 0.1s 間隔で ~2-3 回
    finally:
        cap._stop_status_thread()
    assert len(got) >= 2
    # 停止後は増えない
    n = len(got)
    time.sleep(0.15)
    assert len(got) == n
