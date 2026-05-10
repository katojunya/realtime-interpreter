"""音声キャプチャ + VAD ベースの発話セグメント検出.

BlackHole 2ch から音声を取得し、VAD で発話セグメントを検出する。
セグメントが完結したタイミングで「発話開始時刻 + 全体の音声」を yield する。

仕様:
- VAD で発話開始を検出するとバッファリング開始
- END_SILENCE_MS の無音が続くか、MAX_SEGMENT_SECONDS に達したら yield
- yield されたら次のセグメント検出に移行 (中間スナップショットは出さない)
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Iterator

import numpy as np
from silero_vad_lite import SileroVAD

logger = logging.getLogger(__name__)

DEVICE_NAME = "BlackHole 2ch"
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.3

# 発話終了とみなす無音の長さ (ms)
END_SILENCE_MS = 500
# セグメントの最大長 (これを超えたら強制 finalize)
MAX_SEGMENT_SECONDS = 15.0
# これより短いセグメントはノイズとして捨てる
MIN_SEGMENT_SECONDS = 0.5


@dataclass
class SpeechSegment:
    """確定した発話セグメント.

    audio: モノラル float32 @ 16kHz の波形 (発話開始から無音検知前まで, 末尾無音はトリム)
    start_offset_seconds: capture コンテキスト開始からの経過秒
    duration_seconds: audio の長さ (秒)
    """

    audio: np.ndarray
    start_offset_seconds: float
    duration_seconds: float


def find_device(name: str, sd_module: ModuleType) -> int:
    """デバイス名で入力デバイスを検索しインデックスを返す."""
    devices = sd_module.query_devices()
    for index, device in enumerate(devices):
        if name in device["name"] and device["max_input_channels"] > 0:
            return index
    available = [d["name"] for d in devices]
    raise RuntimeError(f"Device '{name}' not found. Available: {available}")


class SpeechSegmentCapture:
    """VAD ベースの発話セグメント検出.

    `segments()` は generator として、セグメントが完結するたびに `SpeechSegment` を yield する。
    発話中・無音中はブロッキングする (yield しない)。
    """

    def __init__(
        self,
        sd_module: ModuleType,
        device_name: str = DEVICE_NAME,
        sample_rate: int = SAMPLE_RATE,
        end_silence_ms: int = END_SILENCE_MS,
        max_segment_seconds: float = MAX_SEGMENT_SECONDS,
        min_segment_seconds: float = MIN_SEGMENT_SECONDS,
    ) -> None:
        self._sd = sd_module
        self._device_index = find_device(device_name, sd_module)
        self.sample_rate = sample_rate
        self._end_silence_samples = int(sample_rate * end_silence_ms / 1000)
        self._max_segment_samples = int(sample_rate * max_segment_seconds)
        self._min_segment_samples = int(sample_rate * min_segment_seconds)

        self._raw_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._stream = None

        self._vad = SileroVAD(sample_rate)
        self._vad_window = self._vad.window_size_samples

        self._capture_start_monotonic: float | None = None

        # セグメント状態
        self._in_segment = False
        self._segment_chunks: list[np.ndarray] = []
        self._segment_total_samples = 0
        self._segment_silence_samples = 0
        self._segment_start_offset = 0.0

    def __enter__(self) -> SpeechSegmentCapture:
        self._capture_start_monotonic = time.monotonic()
        self._stream = self._sd.InputStream(
            device=self._device_index,
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if status:
            logger.warning("Audio status: %s", status)
        mono = np.mean(indata, axis=1).astype(np.float32)
        self._raw_queue.put(mono)

    def _drain_queue(self) -> None:
        chunks: list[np.ndarray] = [self._buffer]
        while True:
            try:
                chunks.append(self._raw_queue.get_nowait())
            except queue.Empty:
                break
        self._buffer = np.concatenate(chunks)

    def _now_offset(self) -> float:
        if self._capture_start_monotonic is None:
            return 0.0
        return time.monotonic() - self._capture_start_monotonic

    def _start_segment(self, first_window: np.ndarray) -> None:
        self._in_segment = True
        self._segment_chunks = [first_window.copy()]
        self._segment_total_samples = len(first_window)
        self._segment_silence_samples = 0
        # VAD ウィンドウ単位の処理なので開始時刻は近似. ms オーダーの誤差は無視.
        self._segment_start_offset = self._now_offset()

    def _reset_segment(self) -> None:
        self._in_segment = False
        self._segment_chunks = []
        self._segment_total_samples = 0
        self._segment_silence_samples = 0
        self._segment_start_offset = 0.0

    def _make_segment(self, trim_trailing_silence: bool) -> SpeechSegment:
        audio = np.concatenate(self._segment_chunks)
        if trim_trailing_silence and self._segment_silence_samples > 0:
            keep = max(0, len(audio) - self._segment_silence_samples)
            if keep >= self._min_segment_samples:
                audio = audio[:keep]
        return SpeechSegment(
            audio=audio,
            start_offset_seconds=self._segment_start_offset,
            duration_seconds=len(audio) / self.sample_rate,
        )

    def segments(self, poll_interval: float = 0.05) -> Iterator[SpeechSegment]:
        """発話セグメントが完結するたびに yield する."""
        while True:
            self._drain_queue()
            offset = 0
            while len(self._buffer) - offset >= self._vad_window:
                window = self._buffer[offset : offset + self._vad_window]
                offset += self._vad_window

                prob = self._vad.process(window.tobytes())
                is_speech = prob >= VAD_THRESHOLD

                if not self._in_segment:
                    if is_speech:
                        self._start_segment(window)
                    continue

                # in-segment: 発話 / 無音問わず累積
                self._segment_chunks.append(window.copy())
                self._segment_total_samples += self._vad_window
                if is_speech:
                    self._segment_silence_samples = 0
                else:
                    self._segment_silence_samples += self._vad_window

                # 終了条件1: 無音閾値到達
                if self._segment_silence_samples >= self._end_silence_samples:
                    if self._segment_total_samples >= self._min_segment_samples:
                        seg = self._make_segment(trim_trailing_silence=True)
                        self._reset_segment()
                        self._buffer = self._buffer[offset:]
                        yield seg
                        offset = 0
                    else:
                        self._reset_segment()
                    continue

                # 終了条件2: 最大長到達
                if self._segment_total_samples >= self._max_segment_samples:
                    seg = self._make_segment(trim_trailing_silence=False)
                    self._reset_segment()
                    self._buffer = self._buffer[offset:]
                    yield seg
                    offset = 0
                    continue

            self._buffer = self._buffer[offset:]
            time.sleep(poll_interval)
