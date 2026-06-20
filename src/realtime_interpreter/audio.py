"""音声キャプチャ + VAD ベースの発話セグメント検出.

macOS では BlackHole 2ch 等の入力デバイスから、Windows では WASAPI loopback
から音声を取得し、VAD で発話セグメントを検出する。
セグメントが完結したタイミングで「発話開始時刻 + 全体の音声」を yield する。

仕様:
- VAD で発話開始を検出するとバッファリング開始
- END_SILENCE_MS の無音が続くか、MAX_SEGMENT_SECONDS に達したら yield
- yield されたら次のセグメント検出に移行 (中間スナップショットは出さない)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Iterator, Callable

from rich.text import Text

import numpy as np

logger = logging.getLogger(__name__)

DEVICE_NAME = "BlackHole 2ch"
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.3

# 発話終了とみなす無音の長さ (ms)
END_SILENCE_MS = 800
# セグメントの最大長 (これを超えたら強制 finalize)
MAX_SEGMENT_SECONDS = 8.0
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


def find_output_device(name: str, sd_module: ModuleType) -> int:
    """デバイス名で出力 (再生) デバイスを検索しインデックスを返す.

    `find_device` の対称形だが、こちらは `max_output_channels > 0` で絞り込む
    (読み上げ音声の再生先デバイス選択に使う)。
    """
    devices = sd_module.query_devices()
    for index, device in enumerate(devices):
        if name in device["name"] and device["max_output_channels"] > 0:
            return index
    available = [d["name"] for d in devices if d["max_output_channels"] > 0]
    raise RuntimeError(f"Output device '{name}' not found. Available: {available}")


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """ステレオ (N, channels) または 1 次元音声をモノラル float32 に変換."""
    if samples.ndim == 2:
        return np.mean(samples, axis=1).astype(np.float32)
    return samples.astype(np.float32)


def _resample_linear(mono: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """線形補間でサンプルレート変換する (依存追加なし)."""
    if src_rate == dst_rate or mono.size == 0:
        return mono.astype(np.float32, copy=False)
    n_out = int(round(mono.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(mono.size, dtype=np.float64)
    x_new = np.linspace(0.0, mono.size - 1, n_out)
    return np.interp(x_new, x_old, mono).astype(np.float32)


def _compute_dbfs(mono: np.ndarray) -> float:
    if mono.size == 0:
        return -90.0
    peak = float(np.max(np.abs(mono)))
    if peak <= 0:
        return -90.0
    return 20.0 * np.log10(max(peak, 1e-10))


def build_level_meter(db: float, num_blocks: int = 10) -> str:
    # ASCII 文字 (#/-) を使う。■/□ など East Asian Width=Ambiguous の文字は
    # ターミナル/フォントによって幅 1/2 が揺れ、メーターが dB 表示に重なるため。
    # ASCII は常に幅 1 で全環境で整列する。
    min_db = -45.0
    max_db = -3.0
    if db <= min_db:
        filled = 0
    elif db >= max_db:
        filled = num_blocks
    else:
        filled = int((db - min_db) / (max_db - min_db) * num_blocks)
    return "#" * filled + "-" * (num_blocks - filled)


def format_status(
    backend_name: str,
    in_segment: bool,
    db: float,
    current_duration: float = 0.0,
    max_duration: float = 0.0,
) -> Text:
    """音声入力レベルの簡素表示 (ステータス行の「左」スロット用).

    `[meter] dB` を返し、発話区間中 (in_segment) のみ ` (cur/max)` を付与する。
    バックエンド名や状態 (Capturing/Listening 等) は通信ステータス側 (右スロット)
    で表現するため、ここには含めない。`backend_name` は呼び出し互換のため残置。
    """
    meter = build_level_meter(db)
    # dB は右詰め固定幅 (符号込み幅5 + "dB" = 7) にして桁数差でのガタつきを防ぐ。
    # 符号は数値に密着させ、不足分は左に空白 (例 "-15.3dB" / " -1.3dB" / " +0.5dB")。
    # dBFS は約 -90 で下限 (それ以下は -inf 分岐) のため 6 桁 (-100.0) は発生しない。
    db_str = f"{db:+5.1f}dB" if db > -90.0 else "-inf dB"

    text = Text(f"[{meter}] ", style="green")
    text.append(db_str, style="yellow")
    if in_segment:
        text.append(f" ({current_duration:.1f}s / {max_duration:.1f}s)", style="dim")
    return text


def _find_loopback_device(pa, name: str | None):
    """PyAudioWPatch で WASAPI loopback デバイスを解決する (Windows).

    - name=None or 既定デバイス名(DEVICE_NAME) のとき: 既定出力に対応する loopback
    - name 指定時: loopback デバイス名の部分一致
    """
    import pyaudiowpatch as pyaudio

    wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    loopbacks = list(pa.get_loopback_device_info_generator())
    if not loopbacks:
        raise RuntimeError(
            "No WASAPI loopback devices found. This environment cannot capture "
            "system audio via loopback."
        )

    use_default = (not name) or (name == DEVICE_NAME)
    if use_default:
        default_out = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        for lb in loopbacks:
            if default_out["name"] in lb["name"]:
                return lb
        return loopbacks[0]

    for lb in loopbacks:
        if name in lb["name"]:
            return lb
    available = [lb["name"] for lb in loopbacks]
    raise RuntimeError(
        f"Loopback device matching {name!r} not found. Available: {available}"
    )


class _BaseSpeechSegmentCapture:
    """VAD ベースの発話セグメント検出.

    `segments()` は generator として、セグメントが完結するたびに `SpeechSegment` を yield する。
    発話中・無音中はブロッキングする (yield しない)。
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        end_silence_ms: int = END_SILENCE_MS,
        max_segment_seconds: float = MAX_SEGMENT_SECONDS,
        min_segment_seconds: float = MIN_SEGMENT_SECONDS,
    ) -> None:
        self.sample_rate = sample_rate
        self._end_silence_samples = int(sample_rate * end_silence_ms / 1000)
        self._max_segment_samples = int(sample_rate * max_segment_seconds)
        self._min_segment_samples = int(sample_rate * min_segment_seconds)

        self._raw_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._stream = None

        # silero-vad-lite はローカル VAD バックエンド (mlx/openai-chat) の依存。
        # モジュール先頭ではなくここで遅延 import し、VAD を使わない openai realtime
        # バックエンドでは未インストール環境でも audio.py を import できるようにする。
        from silero_vad_lite import SileroVAD

        self._vad = SileroVAD(sample_rate)
        self._vad_window = self._vad.window_size_samples

        self._capture_start_monotonic: float | None = None

        self.current_level_db = -90.0
        self.status_callback: Callable[[str | Text], None] | None = None
        self.backend_name = "Local"

        # メーター送出用ティッカー. segments()/translate() のブロッキングに依らず
        # 0.1s 毎にレベルを送出し続けるための専用スレッド。
        self._status_thread: threading.Thread | None = None
        self._status_stop = threading.Event()

        # セグメント状態
        self._in_segment = False
        self._segment_chunks: list[np.ndarray] = []
        self._segment_total_samples = 0
        self._segment_silence_samples = 0
        self._segment_start_offset = 0.0

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

    def _emit_meter_once(self) -> None:
        """現在の音量レベルをメーターとして 1 回送出する.

        値 (current_level_db / _in_segment / _segment_total_samples) は音声コールバックや
        segments() が更新する。ここでは読むだけなので、translate() ブロッキング中でも
        ティッカースレッドから安全に呼べる。
        """
        if self.status_callback is None:
            return
        cur_dur = self._segment_total_samples / self.sample_rate
        max_dur = self._max_segment_samples / self.sample_rate
        self.status_callback(
            format_status(
                backend_name=self.backend_name,
                in_segment=self._in_segment,
                db=self.current_level_db,
                current_duration=cur_dur,
                max_duration=max_dur,
            )
        )

    def _status_loop(self) -> None:
        while not self._status_stop.is_set():
            self._emit_meter_once()
            self._status_stop.wait(0.1)

    def _start_status_thread(self) -> None:
        """メーター送出ティッカーを開始する (サブクラスの __enter__ から呼ぶ)."""
        self._status_stop.clear()
        self._status_thread = threading.Thread(
            target=self._status_loop, name="capture-meter", daemon=True
        )
        self._status_thread.start()

    def _stop_status_thread(self) -> None:
        """メーター送出ティッカーを停止する (サブクラスの __exit__ から呼ぶ)."""
        self._status_stop.set()
        if self._status_thread is not None:
            self._status_thread.join(timeout=1.0)
            self._status_thread = None

    def segments(self, poll_interval: float = 0.05) -> Iterator[SpeechSegment]:
        """発話セグメントが完結するたびに yield する.

        メーター送出は _status_loop ティッカースレッドが担うため、ここでは行わない
        (translate() ブロッキング中もメーターを動かし続けるため)。
        """
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


class SpeechSegmentCapture(_BaseSpeechSegmentCapture):
    """sounddevice 入力デバイスから音声を取得する VAD セグメントキャプチャ."""

    def __init__(
        self,
        sd_module: ModuleType,
        device_name: str = DEVICE_NAME,
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
        self._sd = sd_module
        self._device_index = find_device(device_name, sd_module)

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
        self._start_status_thread()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._stop_status_thread()
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
        mono = _to_mono(indata)
        self.current_level_db = _compute_dbfs(mono)
        self._raw_queue.put(mono)


class WindowsLoopbackSpeechSegmentCapture(_BaseSpeechSegmentCapture):
    """Windows の WASAPI loopback からシステム音声を取得する VAD キャプチャ."""

    def __init__(
        self,
        device_name: str = DEVICE_NAME,
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
        self._device_name = device_name
        self._pa = None
        self._capture_rate = sample_rate

    def __enter__(self) -> WindowsLoopbackSpeechSegmentCapture:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as e:
            raise RuntimeError(
                f"PyAudioWPatch is required for Windows loopback capture but is not "
                f"installed ({e}). Run `uv sync` on Windows to install it."
            )

        self._capture_start_monotonic = time.monotonic()
        self._pa = pyaudio.PyAudio()
        device = _find_loopback_device(self._pa, self._device_name)
        self._capture_rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])
        logger.info(
            "WASAPI loopback device: [%s] %s (channels=%d, rate=%d -> %d)",
            device["index"], device["name"], channels,
            self._capture_rate, self.sample_rate,
        )

        def _pa_callback(in_data, frame_count, time_info, status):
            try:
                arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    arr = arr.reshape(-1, channels)
                mono = _to_mono(arr)
                mono = _resample_linear(mono, self._capture_rate, self.sample_rate)
                self.current_level_db = _compute_dbfs(mono)
                if mono.size:
                    self._raw_queue.put(mono)
            except Exception:
                logger.exception("loopback callback failed")
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=self._capture_rate,
            frames_per_buffer=1024,
            input=True,
            input_device_index=device["index"],
            stream_callback=_pa_callback,
        )
        self._stream.start_stream()
        self._start_status_thread()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._stop_status_thread()
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        finally:
            if self._pa is not None:
                self._pa.terminate()
