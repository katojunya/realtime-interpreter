"""訳文の読み上げ音声を再生する出力コンポーネント.

realtime 系バックエンド (gemini-realtime / openai-realtime) はモデルが生成した
翻訳音声 (PCM16) をストリームで返す。本モジュールはそれを指定の出力デバイスへ
低遅延に再生する。`audio.py` のキャプチャの対称形:
- macOS/Linux: sounddevice OutputStream (`SoundDeviceAudioPlayer`)
- Windows: PyAudioWPatch 出力ストリーム (`WindowsAudioPlayer`)

入力 PCM はソースのサンプルレート (例 24kHz) で届くため、出力デバイスの既定レートへ
`_resample_linear` でリサンプルしてからリングバッファに積み、コールバックで書き出す。

注意: 読み上げ音声がキャプチャ入力へループバック再入力されないよう、出力デバイスは
キャプチャ対象と別の物理デバイスにすること (同一でないかの検証は main.py 側で行う)。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from types import ModuleType, TracebackType

import numpy as np

from realtime_interpreter.audio import _resample_linear

logger = logging.getLogger(__name__)

# 出力ストリームの 1 ブロックあたりフレーム数 (低遅延と安定のバランス)。
PLAYBACK_BLOCK_FRAMES = 1024


def _to_float32_mono(pcm: bytes | np.ndarray, channels: int = 1) -> np.ndarray:
    """PCM (int16 bytes / 任意の ndarray) をモノラル float32 [-1, 1] に正規化する."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(pcm), dtype="<i2").astype(np.float32) / 32768.0
    else:
        arr = np.asarray(pcm)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        else:
            arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1).astype(np.float32)
    elif channels > 1 and arr.ndim == 1 and arr.size % channels == 0:
        arr = np.mean(arr.reshape(-1, channels), axis=1).astype(np.float32)
    return arr


class _BaseAudioPlayer:
    """リサンプル + リングバッファ管理の共通部分.

    `enqueue()` は任意スレッドから呼ばれ、`_pull()` は出力コールバックスレッドから
    呼ばれる。両者を 1 つのロックで直列化する。バッファが枯渇したフレームは無音
    (ゼロ) で埋める (アンダーランで例外を出さない)。
    """

    def __init__(self, device_rate: int) -> None:
        self._device_rate = device_rate
        self._lock = threading.Lock()
        self._pending: deque[np.ndarray] = deque()
        self._leftover = np.zeros(0, dtype=np.float32)

    def enqueue(self, pcm: bytes | np.ndarray, src_rate: int) -> None:
        """翻訳音声チャンクを再生キューへ追加する (src_rate からデバイスレートへ変換)."""
        mono = _to_float32_mono(pcm)
        if mono.size == 0:
            return
        resampled = _resample_linear(mono, src_rate, self._device_rate)
        if resampled.size == 0:
            return
        with self._lock:
            self._pending.append(resampled)

    def _pull(self, frames: int) -> np.ndarray:
        """次の `frames` サンプル分の float32 モノラルを返す (不足分はゼロ埋め)."""
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames:
                if self._leftover.size == 0:
                    if not self._pending:
                        break  # アンダーラン: 残りは無音
                    self._leftover = self._pending.popleft()
                take = min(frames - filled, self._leftover.size)
                out[filled : filled + take] = self._leftover[:take]
                self._leftover = self._leftover[take:]
                filled += take
        return out


class SoundDeviceAudioPlayer(_BaseAudioPlayer):
    """sounddevice OutputStream による再生 (macOS/Linux)."""

    def __init__(self, device_index: int, device_rate: int, sd_module: ModuleType) -> None:
        super().__init__(device_rate)
        self._sd = sd_module
        self._device_index = device_index
        self._stream = None

    def __enter__(self) -> "SoundDeviceAudioPlayer":
        self._stream = self._sd.OutputStream(
            device=self._device_index,
            samplerate=self._device_rate,
            channels=1,
            dtype="float32",
            blocksize=PLAYBACK_BLOCK_FRAMES,
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
            self._stream = None

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if status:
            logger.debug("playback status: %s", status)
        outdata[:, 0] = self._pull(frames)


class WindowsAudioPlayer(_BaseAudioPlayer):
    """PyAudioWPatch 出力ストリームによる再生 (Windows)."""

    def __init__(self, device_index: int, device_rate: int) -> None:
        super().__init__(device_rate)
        self._device_index = device_index
        self._pa = None
        self._stream = None

    def __enter__(self) -> "WindowsAudioPlayer":
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()

        def _pa_callback(in_data, frame_count, time_info, status):
            mono = self._pull(frame_count)
            pcm16 = np.clip(mono, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype("<i2")
            return (pcm16.tobytes(), pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._device_rate,
            frames_per_buffer=PLAYBACK_BLOCK_FRAMES,
            output=True,
            output_device_index=self._device_index,
            stream_callback=_pa_callback,
        )
        self._stream.start_stream()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        finally:
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
