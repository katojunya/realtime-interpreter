"""Gemini Multimodal Live API バックエンド.

WebSocket 経由で 16kHz PCM16 音声をストリームし、source 言語の転写（inputAudioTranscription）と
target 言語訳（modelTurn）の delta を受け取る。
ターン終了はサーバーから送られる `turnComplete: true` イベントで確定する。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from types import ModuleType, TracebackType
from typing import Iterator

import numpy as np

from realtime_interpreter.audio import DEVICE_NAME
from realtime_interpreter.backends.base import TranslatedSegment
from realtime_interpreter.i18n import DEFAULT_SOURCE, DEFAULT_TARGET, language_name

logger = logging.getLogger(__name__)

GEMINI_SAMPLE_RATE = 16000
GEMINI_REALTIME_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"


def _is_windows() -> bool:
    return sys.platform == "win32"


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """ステレオ (N, channels) または 1 次元音声をモノラル float32 に変換."""
    if samples.ndim == 2:
        return np.mean(samples, axis=1).astype(np.float32)
    return samples.astype(np.float32)


def _resample_linear(mono: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """線形補間でサンプルレート変換する."""
    if src_rate == dst_rate or mono.size == 0:
        return mono.astype(np.float32, copy=False)
    n_out = int(round(mono.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(mono.size, dtype=np.float64)
    x_new = np.linspace(0.0, mono.size - 1, n_out)
    return np.interp(x_new, x_old, mono).astype(np.float32)


def _find_input_device(name: str, sd_module: ModuleType) -> int:
    devices = sd_module.query_devices()
    for index, device in enumerate(devices):
        if name in device["name"] and device["max_input_channels"] > 0:
            return index
    available = [d["name"] for d in devices]
    raise RuntimeError(f"Device '{name}' not found. Available: {available}")


def _find_loopback_device(pa, name: str | None):
    import pyaudiowpatch as pyaudio

    wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    loopbacks = list(pa.get_loopback_device_info_generator())
    if not loopbacks:
        raise RuntimeError("No WASAPI loopback devices found.")

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
    raise RuntimeError(f"Loopback device matching {name!r} not found. Available: {available}")


@dataclass
class _PendingTurn:
    """delta を蓄積中のターン."""

    start_offset_seconds: float
    started_at: float
    last_activity_at: float
    source_parts: list[str] = field(default_factory=list)
    target_parts: list[str] = field(default_factory=list)

    def source(self) -> str:
        return "".join(self.source_parts).strip()

    def target(self) -> str:
        return "".join(self.target_parts).strip()

    def has_content(self) -> bool:
        return bool(self.source() or self.target())


class GeminiRealtimeBackend:
    """Gemini Multimodal Live API を使った WebSocket ベースの翻訳バックエンド."""

    def __init__(
        self,
        sd_module: ModuleType | None,
        device_name: str,
        api_key: str | None = None,
        model: str = "models/gemini-3.5-live-translate-preview",
        turn_debounce_ms: int = 800,
        max_segment_seconds: float = 8.0,
        source_lang: str = DEFAULT_SOURCE,
        target_lang: str = DEFAULT_TARGET,
        loopback: bool | None = None,
    ) -> None:
        self._sd = sd_module
        self._device_name = device_name
        self._device_index: int | None = None
        self._loopback = _is_windows() if loopback is None else loopback
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("No Gemini API key found. Set GEMINI_API_KEY environment variable.")
        self._model = model
        self._turn_debounce_seconds = turn_debounce_ms / 1000.0
        self._max_segment_seconds = max_segment_seconds
        self.source_lang = source_lang
        self.target_lang = target_lang

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._segment_queue: queue.Queue[TranslatedSegment] = queue.Queue()
        self._stop_event = threading.Event()

        self._pending_lock = threading.Lock()
        self._pending_turn: _PendingTurn | None = None

        self._stream = None
        self._ws = None  # type: ignore[assignment]
        self._send_thread: threading.Thread | None = None
        self._recv_thread: threading.Thread | None = None
        self._emit_thread: threading.Thread | None = None

        self._pa = None
        self._capture_rate: int = GEMINI_SAMPLE_RATE
        self._capture_start_monotonic: float | None = None

        # 音声レベルメータ (デバッグ用)
        self._level_max: float = 0.0
        self._level_samples: int = 0
        self._level_last_log_at: float = 0.0

    def __enter__(self) -> GeminiRealtimeBackend:
        self._capture_start_monotonic = time.monotonic()
        self._open_websocket()
        self._send_session_config()

        if self._loopback:
            self._open_loopback_stream_windows()
        else:
            self._device_index = _find_input_device(self._device_name, self._sd)
            self._open_audio_stream()

        self._start_threads()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._stop_event.set()
        try:
            if self._stream is not None:
                if self._loopback:
                    self._stream.stop_stream()
                    self._stream.close()
                else:
                    self._stream.stop()
                    self._stream.close()
        except Exception:
            logger.debug("audio stream close failed", exc_info=True)
        try:
            if self._pa is not None:
                self._pa.terminate()
        except Exception:
            logger.debug("pyaudio terminate failed", exc_info=True)
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            logger.debug("ws close failed", exc_info=True)
        for t in (self._send_thread, self._recv_thread, self._emit_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)

        # 終了時に未確定のターンが残っていれば emit
        with self._pending_lock:
            if self._pending_turn is not None and self._pending_turn.has_content():
                self._emit_pending_locked()

    def stream_segments(self) -> Iterator[TranslatedSegment]:
        while not self._stop_event.is_set():
            try:
                seg = self._segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            yield seg

    def _open_websocket(self) -> None:
        import websocket

        url = f"{GEMINI_REALTIME_URL}?key={self._api_key}"
        self._ws = websocket.WebSocket()
        self._ws.connect(url)
        logger.info("connected to Gemini Live API: %s", GEMINI_REALTIME_URL)

    def _send_session_config(self) -> None:
        is_translation_model = "live-translate" in self._model.lower()

        if is_translation_model:
            msg = {
                "setup": {
                    "model": self._model,
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "translationConfig": {
                            "targetLanguageCode": self.target_lang,
                            "echoTargetLanguage": False
                        }
                    },
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {}
                }
            }
        else:
            msg = {
                "setup": {
                    "model": self._model,
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "temperature": 0.0,
                    },
                    "systemInstruction": {
                        "parts": [
                            {
                                "text": (
                                    f"You are a professional simultaneous interpreter from {language_name(self.source_lang)} to {language_name(self.target_lang)}. "
                                    f"Translate the incoming audio stream to {language_name(self.target_lang)}. "
                                    f"Output only the translation, with no extra text or commentary."
                                )
                            }
                        ]
                    },
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {}
                }
            }

        self._ws.send(json.dumps(msg))
        logger.debug("setup sent: %s", json.dumps(msg))

    def _open_audio_stream(self) -> None:
        self._capture_rate = GEMINI_SAMPLE_RATE
        self._stream = self._sd.InputStream(
            device=self._device_index,
            samplerate=GEMINI_SAMPLE_RATE,
            channels=2,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def _open_loopback_stream_windows(self) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as e:
            raise RuntimeError(
                f"PyAudioWPatch is required for Windows loopback capture but is not "
                f"installed ({e}). Run `uv sync` on Windows to install it."
            )

        self._pa = pyaudio.PyAudio()
        device = _find_loopback_device(self._pa, self._device_name)
        self._capture_rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])
        logger.info(
            "WASAPI loopback device: [%s] %s (channels=%d, rate=%d -> %d)",
            device["index"], device["name"], channels,
            self._capture_rate, GEMINI_SAMPLE_RATE,
        )

        def _pa_callback(in_data, frame_count, time_info, status):
            try:
                arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    arr = arr.reshape(-1, channels)
                mono = _to_mono(arr)
                mono = _resample_linear(mono, self._capture_rate, GEMINI_SAMPLE_RATE)
                if mono.size:
                    self._audio_queue.put(mono)
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

    def _start_threads(self) -> None:
        self._send_thread = threading.Thread(
            target=self._send_loop, name="gemini-send", daemon=True
        )
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name="gemini-recv", daemon=True
        )
        self._emit_thread = threading.Thread(
            target=self._emit_loop, name="gemini-emit", daemon=True
        )
        self._send_thread.start()
        self._recv_thread.start()
        self._emit_thread.start()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if status:
            logger.warning("audio status: %s", status)
        mono = np.mean(indata, axis=1).astype(np.float32)
        self._audio_queue.put(mono)

    def _send_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                self._maybe_log_level()
                continue
            self._update_level(chunk)
            try:
                self._send_audio_chunk(chunk)
            except Exception:
                logger.exception("audio send failed")
                self._stop_event.set()
                return
            self._maybe_log_level()

    def _update_level(self, chunk: np.ndarray) -> None:
        if len(chunk) == 0:
            return
        peak = float(np.max(np.abs(chunk)))
        if peak > self._level_max:
            self._level_max = peak
        self._level_samples += len(chunk)

    def _maybe_log_level(self) -> None:
        now = time.monotonic()
        if self._level_last_log_at == 0.0:
            self._level_last_log_at = now
            return
        if now - self._level_last_log_at < 1.0:
            return
        sent_seconds = self._level_samples / GEMINI_SAMPLE_RATE
        if self._level_max > 0:
            dbfs = 20.0 * np.log10(max(self._level_max, 1e-10))
            level_str = f"peak={dbfs:+.1f} dBFS"
        else:
            level_str = "peak=-inf dBFS (silence)"
        logger.info(
            "audio: %s, sent=%.2fs of audio in last %.1fs",
            level_str,
            sent_seconds,
            now - self._level_last_log_at,
        )
        self._level_max = 0.0
        self._level_samples = 0
        self._level_last_log_at = now

    def _send_audio_chunk(self, audio: np.ndarray) -> None:
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
        msg = {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "mimeType": "audio/pcm;rate=16000",
                        "data": b64
                    }
                ]
            }
        }
        self._ws.send(json.dumps(msg))

    def _emit_loop(self) -> None:
        """100ms ごとに pending turn を見て、debounce 経過 or 最大長到達で emit.

        Commit 条件 (どちらか満たせば発火):
        1. delta が `_turn_debounce_seconds` 来ない → 発話の切れ目
        2. ターン累積時間が `_max_segment_seconds` 超過 → 連続発話の強制カット
        """
        while not self._stop_event.is_set():
            time.sleep(0.1)
            with self._pending_lock:
                if self._pending_turn is None:
                    continue
                if not self._pending_turn.has_content():
                    continue
                now = time.monotonic()
                quiet_for = now - self._pending_turn.last_activity_at
                elapsed = now - self._pending_turn.started_at

                debounce_hit = quiet_for >= self._turn_debounce_seconds
                max_hit = (
                    self._max_segment_seconds > 0
                    and elapsed >= self._max_segment_seconds
                )

                if not (debounce_hit or max_hit):
                    continue

                if max_hit and not debounce_hit:
                    logger.info(
                        "force-commit at max_segment_seconds (%.1fs accumulated)",
                        elapsed,
                    )
                self._emit_pending_locked()

    def _recv_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self._ws.recv()
            except Exception:
                if not self._stop_event.is_set():
                    logger.exception("ws recv failed")
                self._stop_event.set()
                return
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self._handle_event(event)

    def _handle_event(self, event: dict) -> None:
        error = event.get("error")
        if error:
            logger.error("Gemini Live API error: %s", error)
            self._stop_event.set()
            return

        if "setupComplete" in event:
            logger.info("Gemini Live API setup complete.")
            return

        usage_metadata = event.get("usageMetadata")
        if usage_metadata:
            logger.debug("Gemini usage metadata: %s", usage_metadata)

        session_resumption = event.get("sessionResumptionUpdate")
        if session_resumption:
            logger.debug("Gemini session resumption update: %s", session_resumption)

        server_content = event.get("serverContent")
        is_known_event = bool(usage_metadata or session_resumption or "setupComplete" in event)

        if not server_content:
            if not is_known_event:
                logger.debug("Received unhandled Gemini event: %s", event)
            return

        # 1. User Input transcription
        # Try both inputTranscription (modern API) and inputAudioTranscription (fallback/older API)
        input_transcription = server_content.get("inputTranscription") or server_content.get("inputAudioTranscription")
        if input_transcription:
            text = input_transcription.get("text", "")
            if text:
                self._append_input(text)

        # 2. Model generated content (translation)
        # Try both outputTranscription (modern API) and outputAudioTranscription (fallback/older API)
        output_transcription = server_content.get("outputTranscription") or server_content.get("outputAudioTranscription")
        if output_transcription:
            text = output_transcription.get("text", "")
            if text:
                self._append_output(text)

        model_turn = server_content.get("modelTurn")
        if model_turn:
            parts = model_turn.get("parts", [])
            for part in parts:
                text_part = part.get("text", "")
                if text_part:
                    self._append_output(text_part)

        # 3. Turn complete
        if server_content.get("turnComplete"):
            with self._pending_lock:
                if self._pending_turn is not None:
                    self._emit_pending_locked()

    def _ensure_pending_locked(self) -> None:
        if self._pending_turn is None:
            now = time.monotonic()
            offset = now - (self._capture_start_monotonic or now)
            self._pending_turn = _PendingTurn(
                start_offset_seconds=offset,
                started_at=now,
                last_activity_at=now,
            )

    def _append_input(self, text: str) -> None:
        with self._pending_lock:
            self._ensure_pending_locked()
            assert self._pending_turn is not None
            self._pending_turn.source_parts.append(text)
            self._pending_turn.last_activity_at = time.monotonic()
            self._emit_partial_locked()

    def _append_output(self, delta: str) -> None:
        with self._pending_lock:
            self._ensure_pending_locked()
            assert self._pending_turn is not None
            self._pending_turn.target_parts.append(delta)
            self._pending_turn.last_activity_at = time.monotonic()
            self._emit_partial_locked()

    def _emit_partial_locked(self) -> None:
        if self._pending_turn is None:
            return
        turn = self._pending_turn
        src = turn.source()
        tgt = turn.target()
        if not src and not tgt:
            return
        duration = max(0.0, turn.last_activity_at - turn.started_at)
        seg = TranslatedSegment(
            start_offset_seconds=turn.start_offset_seconds,
            duration_seconds=duration,
            source=src,
            target=tgt,
            is_partial=True,
        )
        self._segment_queue.put(seg)

    def _emit_pending_locked(self) -> None:
        if self._pending_turn is None:
            return
        turn = self._pending_turn
        self._pending_turn = None
        src = turn.source()
        tgt = turn.target()
        if not src and not tgt:
            return
        duration = max(0.0, turn.last_activity_at - turn.started_at)
        seg = TranslatedSegment(
            start_offset_seconds=turn.start_offset_seconds,
            duration_seconds=duration,
            source=src,
            target=tgt,
            is_partial=False,
        )
        self._segment_queue.put(seg)
