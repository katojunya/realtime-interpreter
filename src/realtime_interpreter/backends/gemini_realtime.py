"""Gemini Multimodal Live API バックエンド.

WebSocket 経由で 16kHz PCM16 音声をストリームし、source 言語の転写（inputAudioTranscription）と
target 言語訳（modelTurn）の delta を受け取る。
セグメントの確定は emit_loop の debounce（一定時間 delta が来ない＝発話の切れ目）と
max_segment（連続発話の強制カット上限）で行う。サーバの `turnComplete` は短い翻訳単位
ごとに頻発し過剰分割を招くため、確定トリガとしては使わない（連続ターンを束ねる）。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from types import ModuleType, TracebackType
from typing import Iterator

import numpy as np

from realtime_interpreter.audio import (
    DEVICE_NAME,
    find_device,
    find_loopback_device,
    is_windows,
    resample_linear,
    to_mono,
)
from realtime_interpreter.backends.base import TranslatedSegment
from realtime_interpreter.i18n import DEFAULT_SOURCE, DEFAULT_TARGET, language_name

logger = logging.getLogger(__name__)

GEMINI_SAMPLE_RATE = 16000
GEMINI_REALTIME_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"


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
        self._loopback = is_windows() if loopback is None else loopback
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
        # _stop_event: セッションを恒久終了する (ユーザ Ctrl+C or 復旧不能エラー)。
        # _closing:    ユーザ起因の終了のみ (サーバ切断と区別するため)。
        # 接続が ~10 分上限で切れた場合は両方未セットのまま再接続を試みる。
        self._stop_event = threading.Event()
        self._closing = threading.Event()

        # Session Resumption: 切断をまたいでセッションを継続するための handle。
        # サーバが sessionResumptionUpdate.newHandle で送ってくる。
        self._resumption_handle: str | None = None
        self._ws_lock = threading.Lock()        # ws の差し替えを保護
        self._reconnecting = threading.Event()  # 再接続中は送信を止める

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
            self._device_index = find_device(self._device_name, self._sd)
            self._open_audio_stream()

        self._start_threads()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # ユーザ起因の終了. _closing を先に立てて「サーバ切断ではない」と区別する。
        self._closing.set()
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

    def _send_session_config(self, resume: bool = False) -> None:
        is_translation_model = "live-translate" in self._model.lower()

        if is_translation_model:
            setup: dict = {
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
        else:
            setup = {
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

        # Session Resumption を有効化. 初回は空 ({}) でハンドル発行を要求し、
        # 再接続時は保存済みハンドルを渡して同一セッションを継続する。
        if resume and self._resumption_handle:
            setup["sessionResumption"] = {"handle": self._resumption_handle}
        else:
            setup["sessionResumption"] = {}

        # Context Window Compression を有効化 (sliding window, 既定パラメータ).
        # session resumption が「~10分の接続切断」を乗り越えるのに対し、こちらは
        # 「15分のセッション時間 / 128k トークン上限」を回避してセッションを無制限に延長する。
        # 両者は独立機構で、長時間 (数時間〜) 運用には両方必要。
        # 短時間セッションでは sliding window は発動しないため常時有効でも無害。
        setup["contextWindowCompression"] = {"slidingWindow": {}}

        msg = {"setup": setup}
        self._ws.send(json.dumps(msg))
        # ハンドルはログに出さない (機密に近いため). resume フラグのみ記録。
        logger.debug("setup sent (resume=%s)", resume)

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
        device = find_loopback_device(self._pa, self._device_name)
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
                mono = to_mono(arr)
                mono = resample_linear(mono, self._capture_rate, GEMINI_SAMPLE_RATE)
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
            # 再接続中は送信しない (まだ ws が無効). 古い音声は捨ててバックログを防ぐ。
            if self._reconnecting.is_set():
                continue
            self._update_level(chunk)
            try:
                self._send_audio_chunk(chunk)
            except Exception:
                # 切断由来の失敗で send_loop を殺さない。recv_loop が再接続を駆動するので、
                # ここでは少し待って次チャンクへ進む (再接続完了後に送信再開)。
                if self._closing.is_set() or self._stop_event.is_set():
                    return
                logger.debug("audio send failed (will resume after reconnect)")
                time.sleep(0.1)
                continue
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
        # 公式 live-translate ドキュメントの現行形式 `realtimeInput.audio` を使う。
        # (旧形式 `realtimeInput.mediaChunks[]` はレガシーのため移行済み)
        msg = {
            "realtimeInput": {
                "audio": {
                    "data": b64,
                    "mimeType": f"audio/pcm;rate={GEMINI_SAMPLE_RATE}",
                }
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
                # ユーザ起因の終了なら静かに抜ける。
                if self._closing.is_set() or self._stop_event.is_set():
                    return
                # サーバ切断 (~10分の接続上限等). 進行中ターンを確定してから再接続する。
                logger.info("Gemini connection lost; attempting session resumption...")
                self._flush_pending_on_disconnect()
                if self._reconnect():
                    continue  # 新しい ws で受信再開
                # 再接続に失敗 → 恒久終了
                logger.error("Gemini reconnection failed; stopping.")
                self._stop_event.set()
                return
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self._handle_event(event)

    def _flush_pending_on_disconnect(self) -> None:
        """切断時に進行中ターンを確定して取りこぼしを防ぐ."""
        with self._pending_lock:
            if self._pending_turn is not None and self._pending_turn.has_content():
                self._emit_pending_locked()

    def _reconnect(self, max_attempts: int = 5) -> bool:
        """保存済み resumption handle で再接続する.

        成功で True. closing/stop が立ったら False を返して諦める。
        指数バックオフ (0.5, 1, 2, 4, 8 秒上限) でリトライ。
        """
        self._reconnecting.set()
        try:
            delay = 0.5
            for attempt in range(1, max_attempts + 1):
                if self._closing.is_set() or self._stop_event.is_set():
                    return False
                try:
                    with self._ws_lock:
                        try:
                            if self._ws is not None:
                                self._ws.close()
                        except Exception:
                            pass
                        self._open_websocket()
                        self._send_session_config(resume=True)
                    logger.info(
                        "Gemini reconnected (attempt %d, resumed=%s)",
                        attempt,
                        self._resumption_handle is not None,
                    )
                    return True
                except Exception:
                    logger.warning(
                        "Gemini reconnect attempt %d/%d failed; retrying in %.1fs",
                        attempt, max_attempts, delay,
                    )
                    if self._stop_event.wait(timeout=delay):
                        return False
                    delay = min(delay * 2, 8.0)
            return False
        finally:
            self._reconnecting.clear()

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
            # 再接続に使う handle を保存. resumable=False のときは更新しない。
            new_handle = session_resumption.get("newHandle")
            if new_handle:
                self._resumption_handle = new_handle
            logger.debug(
                "session resumption update (resumable=%s, handle=%s)",
                session_resumption.get("resumable"),
                "set" if new_handle else "none",
            )

        # goAway: サーバが接続をまもなく切ることの予告. 残り時間が含まれる。
        # 当面は記録のみ (実際の切断を recv エラーで検知して再接続する)。
        go_away = event.get("goAway")
        if go_away:
            logger.info("Gemini goAway received (time left: %s)", go_away.get("timeLeft"))

        server_content = event.get("serverContent")
        is_known_event = bool(
            usage_metadata or session_resumption or go_away or "setupComplete" in event
        )

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
        # NOTE: ここで即コミットしない。live-translate は短い翻訳単位ごとに
        # turnComplete を頻発させるため、即コミットするとセグメントが過剰分割される
        # (1〜数語の細切れが多発)。代わりに確定は emit_loop の debounce(無音検出)
        # と max_segment(連続発話の上限)だけに委ね、連続するターンを 1 セグメントに
        # 束ねる。これにより openai-realtime と同等の読みやすい粒度になる。
        # turnComplete は「ひと続きの翻訳が一段落した」マーカーに過ぎず、その後 deltas が
        # 来なければ debounce が、来続ければ max_segment が確定させる。

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
