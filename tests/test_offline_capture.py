"""OfflineSegmentCapture の VAD 状態機械テスト.

実 Silero は使わず、VAD をスタブに差し替えて決定論的に検証する。
波形は「先頭サンプルが正なら発話」と判定するスタブに合わせて作る。
"""

from __future__ import annotations

import numpy as np

from realtime_interpreter.offline_capture import OfflineSegmentCapture


class _StubVAD:
    """window の先頭サンプル > 0 を発話とみなすスタブ VAD."""

    def __init__(self, window: int) -> None:
        self.window_size_samples = window

    def process(self, raw: bytes) -> float:
        arr = np.frombuffer(raw, dtype=np.float32)
        return 1.0 if arr.size and arr[0] > 0.0 else 0.0


WIN = 512
SR = 16000


def _make_capture(end_silence_ms=800, max_segment_seconds=8.0, min_segment_seconds=0.5):
    cap = OfflineSegmentCapture(
        sample_rate=SR,
        end_silence_ms=end_silence_ms,
        max_segment_seconds=max_segment_seconds,
        min_segment_seconds=min_segment_seconds,
    )
    # 実 Silero をスタブへ差し替え (ロジックだけ決定論的に検証)
    cap._vad = _StubVAD(WIN)
    cap._vad_window = WIN
    return cap


def _windows(pattern: str) -> np.ndarray:
    """'S'=発話(先頭+1.0), '.'=無音(全0) のウィンドウ列を波形にする."""
    out = []
    for ch in pattern:
        w = np.zeros(WIN, dtype=np.float32)
        if ch == "S":
            w[0] = 1.0
        out.append(w)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def test_no_speech_yields_nothing():
    cap = _make_capture()
    segs = list(cap.segments_from(_windows("." * 50)))
    assert segs == []


def test_single_segment_ends_on_silence():
    # end_silence=800ms → 25 windows (512/16000=32ms, 800/32=25)
    # 十分長い発話 (>=min 0.5s=~16win) のあと 25 無音で確定
    cap = _make_capture()
    audio = _windows("." * 5 + "S" * 40 + "." * 30)
    segs = list(cap.segments_from(audio))
    assert len(segs) == 1
    seg = segs[0]
    # offset は発話開始ウィンドウ位置 = 5*512/16000
    assert abs(seg.start_offset_seconds - (5 * WIN / SR)) < 1e-6
    # 末尾無音はトリムされるので duration は ~発話長 + 終端無音閾値前まで
    assert seg.duration_seconds > 0.5


def test_two_segments_separated_by_silence():
    cap = _make_capture()
    audio = _windows("S" * 40 + "." * 30 + "S" * 40 + "." * 30)
    segs = list(cap.segments_from(audio))
    assert len(segs) == 2
    assert segs[0].start_offset_seconds < segs[1].start_offset_seconds


def test_short_blip_below_min_is_discarded():
    # end_silence を min(0.5s) より短くすると、blip(発話3win)+終端無音の合計が
    # min 未満になり破棄される。既定の end_silence=800ms だと無音分だけで min を
    # 超えてしまい破棄されない(=無音主体セグメントになる)のが実挙動。
    cap = _make_capture(end_silence_ms=100)  # 100ms ≒ 4 window
    audio = _windows("." * 5 + "S" * 3 + "." * 10)
    segs = list(cap.segments_from(audio))
    assert segs == []


def test_max_segment_force_cut():
    # 連続発話 (無音なし) で max_segment=0.5s に達したら強制カット
    cap = _make_capture(max_segment_seconds=0.5)
    # 0.5s = ~16 window。長い連続発話を入れる
    audio = _windows("S" * 60)
    segs = list(cap.segments_from(audio))
    assert len(segs) >= 2  # 強制カットで複数に割れる
    # 強制カットは閾値超過の最初のウィンドウで起きるため、最大 1 window 分超過しうる
    assert max(s.duration_seconds for s in segs) <= 0.5 + WIN / SR + 1e-3


def test_trailing_segment_flushed_at_eof():
    # 末尾が発話のまま終端 (無音で閉じない) → EOF で flush される
    cap = _make_capture()
    audio = _windows("." * 5 + "S" * 40)
    segs = list(cap.segments_from(audio))
    assert len(segs) == 1
    assert segs[0].duration_seconds > 0.5
