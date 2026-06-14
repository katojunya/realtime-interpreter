"""ファイル/配列駆動のオフライン VAD セグメンタ.

ライブ用の `_BaseSpeechSegmentCapture` と **同一の VAD 状態機械**(同じ閾値・
ウィンドウ・無音終端・最大長・末尾無音トリム)を、メモリ上の固定波形に対して
**決定論的**に走らせる。タイムスタンプは壁時計ではなく **音声内の位置** から計算する。

狙い: 同じ音声を 1 度だけセグメント化し、得た同一チャンク列を複数モデルへ与えて
**公平にオフライン比較**する(ライブ実行は run ごとに区切りがズレて比較を汚す)。
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from realtime_interpreter.audio import (
    END_SILENCE_MS,
    MAX_SEGMENT_SECONDS,
    MIN_SEGMENT_SECONDS,
    SAMPLE_RATE,
    VAD_THRESHOLD,
    SpeechSegment,
    _BaseSpeechSegmentCapture,
)


class OfflineSegmentCapture(_BaseSpeechSegmentCapture):
    """固定波形に対して VAD セグメント検出を行うオフライン版.

    `segments_from(audio)` は、ライブの `segments()` と同じ判定ロジックを
    波形配列に適用し、`SpeechSegment` を yield する。offset は
    `(ウィンドウ先頭サンプル位置 / sample_rate)` = 音声内の実時刻。
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        end_silence_ms: int = END_SILENCE_MS,
        max_segment_seconds: float = MAX_SEGMENT_SECONDS,
        min_segment_seconds: float = MIN_SEGMENT_SECONDS,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            end_silence_ms=end_silence_ms,
            max_segment_seconds=max_segment_seconds,
            min_segment_seconds=min_segment_seconds,
        )
        # 現在処理中ウィンドウの先頭サンプル位置 (offset 算出に使う)
        self._pos_samples = 0

    def _now_offset(self) -> float:
        # 壁時計の代わりに音声内位置を返す (基底の _start_segment から呼ばれる)。
        return self._pos_samples / self.sample_rate

    def segments_from(self, audio: np.ndarray) -> Iterator[SpeechSegment]:
        """波形全体を VAD ウィンドウ単位で走査し、確定セグメントを yield する."""
        buf = np.ascontiguousarray(audio, dtype=np.float32)
        win = self._vad_window
        n = buf.size
        i = 0
        while i + win <= n:
            window = buf[i : i + win]
            self._pos_samples = i  # このウィンドウの先頭位置

            prob = self._vad.process(window.tobytes())
            is_speech = prob >= VAD_THRESHOLD

            if not self._in_segment:
                if is_speech:
                    self._start_segment(window)
                i += win
                continue

            # in-segment: 発話/無音問わず累積 (ライブと同一)
            self._segment_chunks.append(window.copy())
            self._segment_total_samples += win
            if is_speech:
                self._segment_silence_samples = 0
            else:
                self._segment_silence_samples += win

            # 終了条件1: 無音閾値到達
            if self._segment_silence_samples >= self._end_silence_samples:
                if self._segment_total_samples >= self._min_segment_samples:
                    seg = self._make_segment(trim_trailing_silence=True)
                    self._reset_segment()
                    i += win
                    yield seg
                else:
                    self._reset_segment()
                    i += win
                continue

            # 終了条件2: 最大長到達
            if self._segment_total_samples >= self._max_segment_samples:
                seg = self._make_segment(trim_trailing_silence=False)
                self._reset_segment()
                i += win
                yield seg
                continue

            i += win

        # EOF: 進行中セグメントが残っていれば最後に確定する (有限入力のため)。
        if self._in_segment and self._segment_total_samples >= self._min_segment_samples:
            seg = self._make_segment(trim_trailing_silence=True)
            self._reset_segment()
            yield seg
