"""WebSocket ストリーミング翻訳バックエンドの共通基底.

`openai_realtime` と `gemini_realtime` は、音声キャプチャ・送信/受信/emit スレッド・
debounce による発話確定・音声レベルメータ・切断時の自動再接続といった機構をほぼ
共有する。それらをこの基底クラスに集約し、各バックエンドはプロトコル固有の差分のみを
オーバーライドする(`audio.py` の `_BaseSpeechSegmentCapture` と同じ構成)。

サブクラスが定義するもの:
- クラス属性 `SAMPLE_RATE` / `PROACTIVE_RECONNECT_SECONDS` / `LOG_NAME` / `THREAD_PREFIX`
- フック `_open_websocket()` / `_send_session_config(resume)` / `_send_audio_chunk(audio)` /
  `_handle_event(event)`
- 任意フック `_clean(text)`(確定/暫定セグメントのテキスト整形; 既定は素通し)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from types import ModuleType, TracebackType
from typing import Iterator

import numpy as np

from realtime_interpreter.audio import (
    find_device,
    find_loopback_device,
    is_windows,
    resample_linear,
    to_mono,
)
from realtime_interpreter.backends.base import TranslatedSegment

logger = logging.getLogger(__name__)


@dataclass
class _PendingTurn:
    """delta を蓄積中のターン. debounce で確定 → emit する."""

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


class WebSocketStreamingBackend:
    """WebSocket でストリーミング翻訳する外部 API バックエンドの共通実装.

    スレッド構成:
        - 音声コールバック: 音声を `_audio_queue` に入れる
        - 送信スレッド: queue から取り出して `_send_audio_chunk` で送信
        - 受信スレッド: WebSocket から読み、`_handle_event` で delta を累積
        - emit スレッド: debounce / max_segment で確定したターンを `_segment_queue` に push
        - メインスレッド: `stream_segments()` で `_segment_queue` を消費
    """

    # --- サブクラスで上書きする ---
    SAMPLE_RATE: int = 16000
    # プロアクティブ再接続の閾値(秒). None なら無効(切断検知後の reactive 再接続のみ)。
    PROACTIVE_RECONNECT_SECONDS: float | None = None
    LOG_NAME: str = "ws"
    THREAD_PREFIX: str = "ws"

    def __init__(
        self,
        sd_module: ModuleType | None,
        device_name: str,
        api_key: str,
        model: str,
        turn_debounce_ms: int,
        max_segment_seconds: float,
        source_lang: str,
        target_lang: str,
        loopback: bool | None,
    ) -> None:
        self._sd = sd_module
        self._device_name = device_name
        self._device_index: int | None = None
        # loopback=None なら Windows のみ自動有効. 明示指定があればそれに従う。
        self._loopback = is_windows() if loopback is None else loopback
        self._api_key = api_key
        self._model = model
        self._turn_debounce_seconds = turn_debounce_ms / 1000.0
        self._max_segment_seconds = max_segment_seconds
        self.source_lang = source_lang
        self.target_lang = target_lang

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._segment_queue: queue.Queue[TranslatedSegment] = queue.Queue()
        # _stop_event: セッションを恒久終了する (ユーザ Ctrl+C or 復旧不能エラー)。
        # _closing:    ユーザ起因の終了のみ (サーバ切断と区別するため)。
        self._stop_event = threading.Event()
        self._closing = threading.Event()
        self._reconnecting = threading.Event()  # 再接続中は送信を止める
        self._ws_lock = threading.Lock()         # ws の差し替えを保護
        self._connected_at: float = 0.0          # 現接続を確立した monotonic 時刻

        # 音声レベルメータ (デバッグ目的)
        self._level_max: float = 0.0
        self._level_samples: int = 0
        self._level_last_log_at: float = 0.0

        # ターン状態 (recv / emit スレッドから触るのでロック)
        self._pending_lock = threading.Lock()
        self._pending_turn: _PendingTurn | None = None

        self._stream = None
        self._ws = None  # type: ignore[assignment]
        self._send_thread: threading.Thread | None = None
        self._recv_thread: threading.Thread | None = None
        self._emit_thread: threading.Thread | None = None

        self._pa = None  # PyAudio インスタンス (Windows loopback 用)
        self._capture_rate: int = self.SAMPLE_RATE  # 実キャプチャレート (Win は native)
        self._capture_start_monotonic: float | None = None

    # ---------------- context manager ----------------

    def __enter__(self) -> "WebSocketStreamingBackend":
        self._capture_start_monotonic = time.monotonic()
        self._open_websocket()
        self._connected_at = time.monotonic()
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
            with self._ws_lock:
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

    # ---------------- public iterator ----------------

    def stream_segments(self) -> Iterator[TranslatedSegment]:
        while not self._stop_event.is_set():
            try:
                seg = self._segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            yield seg

    # ---------------- protocol-specific hooks (override) ----------------

    def _open_websocket(self) -> None:
        raise NotImplementedError

    def _send_session_config(self, resume: bool = False) -> None:
        raise NotImplementedError

    def _send_audio_chunk(self, audio: np.ndarray) -> None:
        raise NotImplementedError

    def _handle_event(self, event: dict) -> None:
        raise NotImplementedError

    def _clean(self, text: str) -> str:
        """確定/暫定セグメントのテキスト整形フック (既定は素通し)."""
        return text

    # ---------------- audio capture ----------------

    def _open_audio_stream(self) -> None:
        """macOS/Linux: sounddevice の入力ストリームを開く."""
        self._capture_rate = self.SAMPLE_RATE
        self._stream = self._sd.InputStream(
            device=self._device_index,
            samplerate=self.SAMPLE_RATE,
            channels=2,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def _open_loopback_stream_windows(self) -> None:
        """Windows: PyAudioWPatch で WASAPI loopback ストリームを開く.

        sounddevice は loopback 非対応のため PyAudioWPatch を使う。デバイスの
        ネイティブレートでステレオ int16 を取得し、コールバック内で mono 化 +
        SAMPLE_RATE へリサンプルして `_audio_queue` に積む。
        """
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
        target_rate = self.SAMPLE_RATE
        logger.info(
            "WASAPI loopback device: [%s] %s (channels=%d, rate=%d -> %d)",
            device["index"], device["name"], channels,
            self._capture_rate, target_rate,
        )

        def _pa_callback(in_data, frame_count, time_info, status):
            try:
                arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    arr = arr.reshape(-1, channels)
                mono = to_mono(arr)
                mono = resample_linear(mono, self._capture_rate, target_rate)
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

    def _start_threads(self) -> None:
        self._send_thread = threading.Thread(
            target=self._send_loop, name=f"{self.THREAD_PREFIX}-send", daemon=True
        )
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name=f"{self.THREAD_PREFIX}-recv", daemon=True
        )
        self._emit_thread = threading.Thread(
            target=self._emit_loop, name=f"{self.THREAD_PREFIX}-emit", daemon=True
        )
        self._send_thread.start()
        self._recv_thread.start()
        self._emit_thread.start()

    # ---------------- send loop ----------------

    def _send_loop(self) -> None:
        while not self._stop_event.is_set():
            self._maybe_proactive_reconnect()
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

    def _maybe_proactive_reconnect(self) -> None:
        """接続が制限に近づいたら、サーバ切断を待たずに自分から張り替える.

        ws を閉じるだけで実際の再接続は recv_loop が駆動する (再接続経路を一本化)。
        `PROACTIVE_RECONNECT_SECONDS` が None のバックエンドでは何もしない。
        """
        if self.PROACTIVE_RECONNECT_SECONDS is None:
            return
        if self._reconnecting.is_set() or self._connected_at == 0.0:
            return
        if time.monotonic() - self._connected_at < self.PROACTIVE_RECONNECT_SECONDS:
            return
        logger.info("%s proactive reconnect (approaching connection limit)", self.LOG_NAME)
        self._reconnecting.set()
        self._flush_pending_on_disconnect()
        with self._ws_lock:
            try:
                if self._ws is not None:
                    self._ws.close()  # recv_loop が検知して _reconnect() を実行
            except Exception:
                logger.debug("proactive ws close failed", exc_info=True)

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
        sent_seconds = self._level_samples / self.SAMPLE_RATE
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

    # ---------------- receive loop / reconnect ----------------

    def _recv_loop(self) -> None:
        while not self._stop_event.is_set():
            assert self._ws is not None
            try:
                raw = self._ws.recv()
            except Exception:
                # ユーザ起因の終了なら静かに抜ける。
                if self._closing.is_set() or self._stop_event.is_set():
                    return
                # サーバ切断 or プロアクティブ close。進行中ターンを確定してから再接続。
                logger.info("%s connection lost; attempting reconnect...", self.LOG_NAME)
                self._flush_pending_on_disconnect()
                if self._reconnect():
                    continue  # 新しい ws で受信再開
                logger.error("%s reconnection failed; stopping.", self.LOG_NAME)
                self._stop_event.set()
                return
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("non-JSON ws frame: %r", str(raw)[:200])
                continue
            self._handle_event(event)

    def _flush_pending_on_disconnect(self) -> None:
        """切断時に進行中ターンを確定して取りこぼしを防ぐ."""
        with self._pending_lock:
            if self._pending_turn is not None and self._pending_turn.has_content():
                self._emit_pending_locked()

    def _reconnect(self, max_attempts: int = 5) -> bool:
        """新しい WebSocket を張り直して session config を再送する.

        成功で True. closing/stop が立ったら False を返して諦める。
        指数バックオフ (0.5, 1, 2, 4, 8 秒上限) でリトライ。recv_loop からのみ呼ぶ。
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
                        self._connected_at = time.monotonic()
                        self._send_session_config(resume=True)
                    logger.info("%s reconnected (attempt %d)", self.LOG_NAME, attempt)
                    return True
                except Exception:
                    logger.warning(
                        "%s reconnect attempt %d/%d failed; retrying in %.1fs",
                        self.LOG_NAME, attempt, max_attempts, delay,
                    )
                    if self._stop_event.wait(timeout=delay):
                        return False
                    delay = min(delay * 2, 8.0)
            return False
        finally:
            self._reconnecting.clear()

    # ---------------- pending turn management ----------------

    def _ensure_pending_locked(self) -> None:
        if self._pending_turn is None:
            now = time.monotonic()
            offset = now - (self._capture_start_monotonic or now)
            self._pending_turn = _PendingTurn(
                start_offset_seconds=offset,
                started_at=now,
                last_activity_at=now,
            )

    def _append_input(self, delta: str) -> None:
        with self._pending_lock:
            self._ensure_pending_locked()
            assert self._pending_turn is not None
            self._pending_turn.source_parts.append(delta)
            self._pending_turn.last_activity_at = time.monotonic()
            self._emit_partial_locked()

    def _append_output(self, delta: str) -> None:
        with self._pending_lock:
            self._ensure_pending_locked()
            assert self._pending_turn is not None
            self._pending_turn.target_parts.append(delta)
            self._pending_turn.last_activity_at = time.monotonic()
            self._emit_partial_locked()

    # ---------------- emit loop (debounce) ----------------

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

    def _emit_partial_locked(self) -> None:
        """Caller must hold `_pending_lock`. delta 受信ごとに現在状態を partial で push."""
        if self._pending_turn is None:
            return
        turn = self._pending_turn
        src = self._clean(turn.source())
        tgt = self._clean(turn.target())
        if not src and not tgt:
            return
        duration = max(0.0, turn.last_activity_at - turn.started_at)
        self._segment_queue.put(
            TranslatedSegment(
                start_offset_seconds=turn.start_offset_seconds,
                duration_seconds=duration,
                source=src,
                target=tgt,
                is_partial=True,
            )
        )

    def _emit_pending_locked(self) -> None:
        """Caller must hold `_pending_lock`. debounce 経過で final として確定."""
        if self._pending_turn is None:
            return
        turn = self._pending_turn
        self._pending_turn = None
        src = self._clean(turn.source())
        tgt = self._clean(turn.target())
        if not src and not tgt:
            return
        duration = max(0.0, turn.last_activity_at - turn.started_at)
        self._segment_queue.put(
            TranslatedSegment(
                start_offset_seconds=turn.start_offset_seconds,
                duration_seconds=duration,
                source=src,
                target=tgt,
                is_partial=False,
            )
        )
