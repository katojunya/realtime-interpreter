"""OpenAI gpt-realtime-translate バックエンド.

WebSocket 経由で 24kHz PCM16 音声をストリームし、source 言語の転写と target 言語訳の
delta を受け取って累積する。translation session には `*.completed` / `*.done` イベントが
**存在しない** ため、ターン終了は「一定時間 delta が来ないこと」を debounce で
判定する (= 発話の切れ目).

エンドポイント: wss://api.openai.com/v1/realtime/translations?model=...
価格 (2026-05 時点): gpt-realtime-translate $0.034/min + 入力転写 $0.017/min
公式リファレンス:
- https://developers.openai.com/api/docs/guides/realtime-translation
- https://developers.openai.com/api/reference/resources/realtime/translation-server-events

公式に定義されているサーバイベント:
- session.created / session.updated / session.closed
- session.input_transcript.delta   (source 言語; 入力転写を有効化したとき)
- session.output_transcript.delta  (target 言語; 翻訳テキスト)
- session.output_audio.delta       (翻訳音声; 本実装では使用しない)
- error
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
from realtime_interpreter.i18n import DEFAULT_SOURCE, DEFAULT_TARGET

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-realtime-translate"
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime/translations"
OPENAI_SAMPLE_RATE = 24000
INPUT_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"

# delta が来なくなってからこのミリ秒経過 → ターン終了とみなして commit.
# このモデル (gpt-realtime-translate) は EN 1 文を JA 複数文 (例: 挨拶を独立文として
# 切ってから本文を続ける) で訳すことがあるため、文単位での即時 commit を行うと
# EN/JA の対応関係が崩れる. そのため発話の切れ目 (= 翻訳がひと段落して delta が
# 来なくなった瞬間) を唯一の commit トリガとし、対応の整合を優先する。
TURN_DEBOUNCE_MS = 800

# 連続発話 (ポーズなし) で 1 チャンクが肥大化するのを防ぐ強制 commit 上限.
# この秒数に達したら delta が継続中でも強制的に commit する.
# 強制 commit 時は EN/JA が若干ズレる可能性があるが、表示の可読性を優先.
# 0 を指定すると無効 (debounce のみで commit).
OPENAI_MAX_SEGMENT_SECONDS = 8.0

# 文末判定ヘルパは保持 (将来 sentence-level splitting を再有効化する場合の参考実装).
# 現在は EN/JA の切れ目が一致しないため commit には使用しない。
JA_SENTENCE_ENDS = "。？！?!"
EN_SENTENCE_ENDS = ".!?"

# debounce が早く commit したあとに遅れて届いた文末記号が、次セグメントの先頭に
# 流れ込むことがある (例: "[07:43] 。18年..." の先頭の "。"). これを除去するための
# 行頭文字セット. 半角/全角の句読点・記号 + 各種空白。
_LEADING_PUNCT = " 　.,。、!?！？;；:：…・\t\n"


def _clean_leading(text: str) -> str:
    """先頭の句読点・記号・空白を除去する.

    本来は前セグメントの末尾に属するはずだった記号が、ストリームの遅延で
    次セグメントの先頭に来てしまったものを取り除く。中間・末尾の記号は保持。
    """
    return text.lstrip(_LEADING_PUNCT)


def _first_complete_sentence_end_ja(text: str) -> int | None:
    """JA の最初の文末記号位置 (記号の直後の index) を返す. 無ければ None.

    JA は省略記号や略語の曖昧性が無視できるレベルなので単純に走査する。
    """
    for i, ch in enumerate(text):
        if ch in JA_SENTENCE_ENDS:
            return i + 1
    return None


def _first_complete_sentence_end_en(text: str) -> int | None:
    """EN の最初の文末記号位置 (記号の直後の index) を返す. 無ければ None.

    曖昧性を避けるための条件:
    - "..." (省略記号) の `.` ではマッチしない (次の文字も `.` ならスキップ)
    - 略語 (Mr./Dr./e.g. 等) の `.` を避けるため、文末記号の直後に
      空白 or 改行が必要 (= 文末確定)
    - バッファ末尾の `.` は次の delta を待つ (確定不能)
    """
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in EN_SENTENCE_ENDS:
            continue
        # 省略記号 "..." の一部はスキップ (前後どちらかに `.` が隣接)
        if ch == ".":
            prev_is_dot = i > 0 and text[i - 1] == "."
            next_is_dot = i + 1 < n and text[i + 1] == "."
            if prev_is_dot or next_is_dot:
                continue
        # 文末確定には直後に空白が必要 (略語の `Mr.Smith` を回避)
        if i + 1 >= n:
            # バッファ末尾 → 次 delta を待つ
            return None
        if text[i + 1].isspace():
            return i + 1
        # 略語の途中 (例: "Mr.Smith" や "U.S.A") はスキップ
    return None


def _find_input_device(name: str, sd_module: ModuleType) -> int:
    devices = sd_module.query_devices()
    for index, device in enumerate(devices):
        if name in device["name"] and device["max_input_channels"] > 0:
            return index
    available = [d["name"] for d in devices]
    raise RuntimeError(f"Device '{name}' not found. Available: {available}")


def _is_windows() -> bool:
    return sys.platform == "win32"


def _resolve_output_device_for_loopback(
    name: str | None, sd_module: ModuleType
) -> int:
    """WASAPI loopback 用に「出力デバイス」を解決する (Windows).

    - name=None or 既定デバイス名(DEVICE_NAME) のとき: 既定の出力デバイスを採用
      (= ユーザが今聞いているスピーカー/ヘッドホンの音をそのまま loopback 録音)
    - name 指定時: 出力チャンネルを持つデバイスを部分一致で検索
    """
    devices = sd_module.query_devices()
    # 既定値 (BlackHole) のままか未指定なら、システム既定の出力を使う
    if not name or name == DEVICE_NAME:
        default_out = sd_module.default.device[1]
        if default_out is None or default_out < 0:
            raise RuntimeError("No default output device found for WASAPI loopback.")
        return int(default_out)
    for index, device in enumerate(devices):
        if name in device["name"] and device["max_output_channels"] > 0:
            return index
    available = [
        d["name"] for d in devices if d["max_output_channels"] > 0
    ]
    raise RuntimeError(
        f"Output device '{name}' not found for loopback. Available outputs: {available}"
    )


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


class OpenAIRealtimeBackend:
    """OpenAI gpt-realtime-translate を使った WebSocket ベースの翻訳バックエンド.

    スレッド構成:
        - sounddevice コールバック: 音声を `_audio_queue` に入れる
        - 送信スレッド: queue から取り出して PCM16 に変換し WebSocket で送信
        - 受信スレッド: WebSocket からイベントを読み、transcript delta を累積
        - emit スレッド: debounce で「一定時間 delta が来ない」状態を検出して
                       完結したターンを `_segment_queue` に push
        - メインスレッド: stream_segments() で `_segment_queue` を消費

    サーバ VAD によりターン区切りは OpenAI 側で検出されるが、ターン完了通知
    イベントが API に存在しないため、delta の debounce で代用している。
    """

    def __init__(
        self,
        sd_module: ModuleType,
        device_name: str,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        turn_debounce_ms: int = TURN_DEBOUNCE_MS,
        max_segment_seconds: float = OPENAI_MAX_SEGMENT_SECONDS,
        source_lang: str = DEFAULT_SOURCE,
        target_lang: str = DEFAULT_TARGET,
        loopback: bool | None = None,
    ) -> None:
        self._sd = sd_module
        self._device_name = device_name
        self._device_index: int | None = None
        # loopback=None なら Windows のみ自動有効 (案 2a). 明示指定があればそれに従う。
        self._loopback = _is_windows() if loopback is None else loopback
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Set the env var or pass --openai-api-key (not recommended)."
            )
        self._model = model
        # 内部表現は秒. CLI は ms で受け取って秒に変換するためここでも秒に直す。
        self._turn_debounce_seconds = turn_debounce_ms / 1000.0
        self._max_segment_seconds = max_segment_seconds
        # source は Whisper の auto-detect に任せる. target は session config で指定.
        self.source_lang = source_lang
        self.target_lang = target_lang

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._segment_queue: queue.Queue[TranslatedSegment] = queue.Queue()
        self._stop_event = threading.Event()

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

        self._capture_start_monotonic: float | None = None

    # ---------------- context manager ----------------

    def __enter__(self) -> "OpenAIRealtimeBackend":
        self._capture_start_monotonic = time.monotonic()
        # Windows は WASAPI loopback で「出力デバイス」を録音対象にする (案 1a+2a)。
        # それ以外 (macOS/Linux) は従来どおり入力デバイスを開く。
        if self._loopback:
            self._device_index = _resolve_output_device_for_loopback(
                self._device_name, self._sd
            )
        else:
            self._device_index = _find_input_device(self._device_name, self._sd)
        self._open_websocket()
        self._send_session_config()
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
                self._stream.stop()
                self._stream.close()
        except Exception:
            logger.debug("audio stream close failed", exc_info=True)
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

    # ---------------- public iterator ----------------

    def stream_segments(self) -> Iterator[TranslatedSegment]:
        while not self._stop_event.is_set():
            try:
                seg = self._segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            yield seg

    # ---------------- setup ----------------

    def _open_websocket(self) -> None:
        import websocket

        url = f"{OPENAI_REALTIME_URL}?model={self._model}"
        self._ws = websocket.WebSocket()
        self._ws.connect(
            url,
            header=[
                f"Authorization: Bearer {self._api_key}",
            ],
        )
        logger.info("connected to %s", url)

    def _send_session_config(self) -> None:
        # 出力言語は self.target_lang (ISO 639-1) を `audio.output.language` に指定.
        # 入力 (source) の transcript は gpt-realtime-whisper で. source 言語は Whisper
        # の auto-detect 任せ (multilingual で安定). 明示したい場合は input.transcription.language
        # に渡す手もあるが、auto のほうが多くの音声で堅牢。
        msg = {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {
                        "transcription": {"model": INPUT_TRANSCRIPTION_MODEL},
                        "noise_reduction": {"type": "near_field"},
                    },
                    "output": {"language": self.target_lang},
                },
            },
        }
        assert self._ws is not None
        self._ws.send(json.dumps(msg))
        logger.debug("session.update sent: %s", json.dumps(msg))

    def _open_audio_stream(self) -> None:
        extra_settings = None
        if self._loopback:
            # Windows: 出力デバイスを WASAPI loopback で「録音」として開く。
            # これによりスピーカー再生中の音 (システム音声) をそのまま取り込める。
            # 追加ソフト・管理者権限不要。
            try:
                extra_settings = self._sd.WasapiSettings(loopback=True)
            except Exception as e:
                raise RuntimeError(
                    "WASAPI loopback is unavailable on this system "
                    f"({e}). Loopback capture requires Windows + WASAPI host."
                )
        self._stream = self._sd.InputStream(
            device=self._device_index,
            samplerate=OPENAI_SAMPLE_RATE,
            channels=2,
            dtype="float32",
            callback=self._audio_callback,
            extra_settings=extra_settings,
        )
        self._stream.start()

    def _start_threads(self) -> None:
        self._send_thread = threading.Thread(
            target=self._send_loop, name="openai-send", daemon=True
        )
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name="openai-recv", daemon=True
        )
        self._emit_thread = threading.Thread(
            target=self._emit_loop, name="openai-emit", daemon=True
        )
        self._send_thread.start()
        self._recv_thread.start()
        self._emit_thread.start()

    # ---------------- audio capture ----------------

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

    # ---------------- send loop ----------------

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

    def _send_audio_chunk(self, audio: np.ndarray) -> None:
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
        msg = {"type": "session.input_audio_buffer.append", "audio": b64}
        assert self._ws is not None
        self._ws.send(json.dumps(msg))

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
        sent_seconds = self._level_samples / OPENAI_SAMPLE_RATE
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

    # ---------------- receive loop ----------------

    def _recv_loop(self) -> None:
        while not self._stop_event.is_set():
            assert self._ws is not None
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
                logger.warning("non-JSON ws frame: %r", raw[:200])
                continue
            self._handle_event(event)

    def _handle_event(self, event: dict) -> None:
        etype = event.get("type", "")

        if etype == "error" or etype.endswith(".error"):
            logger.error("OpenAI error: %s", event)
            return

        # 翻訳音声は使わない. ノイズ抑制のため明示スキップ.
        if etype.endswith("output_audio.delta") or etype.endswith("output_audio.done"):
            return

        # 英語 (入力) 転写の delta
        if "input_transcript.delta" in etype or "input_audio_transcription.delta" in etype:
            delta = event.get("delta", "")
            if delta:
                self._append_input(delta)
            return

        # 日本語 (出力) 転写の delta
        if (
            "output_transcript.delta" in etype
            or "output_audio_transcript.delta" in etype
            or "output_text.delta" in etype
        ):
            delta = event.get("delta", "")
            if delta:
                self._append_output(delta)
            return

        # サーバ VAD は本 API では明示イベントが無い可能性. 来たらログだけ出す.
        if etype.endswith("speech_started"):
            logger.info("server VAD: speech_started")
            return
        if etype.endswith("speech_stopped"):
            logger.info("server VAD: speech_stopped")
            return

        # session.created / session.updated / session.closed なども含めて
        # 未マッチのイベントは debug に payload と共に残す
        logger.debug("unhandled event type=%s payload=%s", etype, _summarize_event(event))

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
            # 文末での即時 commit は EN/JA のズレを生むので使わない. 進行中表示のみ更新。
            self._emit_partial_locked()

    def _append_output(self, delta: str) -> None:
        with self._pending_lock:
            self._ensure_pending_locked()
            assert self._pending_turn is not None
            self._pending_turn.target_parts.append(delta)
            self._pending_turn.last_activity_at = time.monotonic()
            self._emit_partial_locked()

    def _process_sentence_boundaries_locked(self) -> None:
        """Pending turn から完結した文を抽出して final として emit する.

        - JA と EN の両方に文末記号が見つかったら、それぞれの最初の文末で切って
          1 つの確定セグメント (final) として queue に push.
        - 残り (carryover) は新しい pending turn として継続. carryover に
          まだ文末が含まれていればループで連続コミット.
        - 文末が見つからない場合は partial を emit して in-progress 表示を更新.
        """
        while True:
            if self._pending_turn is None:
                return
            tgt_text = self._pending_turn.target()
            src_text = self._pending_turn.source()

            tgt_idx = _first_complete_sentence_end_ja(tgt_text)
            src_idx = _first_complete_sentence_end_en(src_text)

            if tgt_idx is None or src_idx is None:
                # 両方に文末が無いと commit できない. 現状を partial として表示.
                if self._pending_turn.has_content():
                    self._emit_partial_locked()
                return

            # 両方に文末あり → 切って commit
            self._commit_prefix_as_final_locked(src_idx, tgt_idx)
            # ループ: carryover にさらなる文末が含まれているかチェック

    def _commit_prefix_as_final_locked(self, src_end: int, tgt_end: int) -> None:
        """Pending turn の en[:en_end] / ja[:ja_end] を final で emit, 残りは新 pending."""
        assert self._pending_turn is not None
        turn = self._pending_turn
        src_full = turn.source()
        tgt_full = turn.target()

        src_commit = src_full[:src_end].strip()
        tgt_commit = tgt_full[:tgt_end].strip()

        if src_commit or tgt_commit:
            duration = max(0.0, turn.last_activity_at - turn.started_at)
            seg = TranslatedSegment(
                start_offset_seconds=turn.start_offset_seconds,
                duration_seconds=duration,
                source=src_commit,
                target=tgt_commit,
                is_partial=False,
            )
            self._segment_queue.put(seg)

        src_rem = src_full[src_end:]
        tgt_rem = tgt_full[tgt_end:]

        if not src_rem.strip() and not tgt_rem.strip():
            # carryover なし → pending クリア (in-progress 表示も次の commit() でクリア済み)
            self._pending_turn = None
            return

        # carryover を新しい pending として開始. start_offset は現在時刻に更新。
        now = time.monotonic()
        offset = now - (self._capture_start_monotonic or now)
        self._pending_turn = _PendingTurn(
            start_offset_seconds=offset,
            started_at=now,
            last_activity_at=now,
            source_parts=[src_rem] if src_rem else [],
            target_parts=[tgt_rem] if tgt_rem else [],
        )

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
        assert self._pending_turn is not None
        turn = self._pending_turn
        src = _clean_leading(turn.source())
        tgt = _clean_leading(turn.target())
        # 句読点のみ (clean 後に両方空) のセグメントは表示しない
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
        """Caller must hold `_pending_lock`. debounce 経過で final として確定."""
        assert self._pending_turn is not None
        turn = self._pending_turn
        src = _clean_leading(turn.source())
        tgt = _clean_leading(turn.target())
        self._pending_turn = None
        # 句読点のみ (clean 後に両方空) のセグメントは表示しない
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


def _summarize_event(event: dict, max_field_len: int = 200) -> str:
    """大きなフィールド (base64 audio など) を切り詰めた JSON 文字列を返す."""
    redacted: dict[str, object] = {}
    for k, v in event.items():
        if isinstance(v, str) and len(v) > max_field_len:
            redacted[k] = f"<{len(v)} chars truncated>"
        else:
            redacted[k] = v
    try:
        return json.dumps(redacted, ensure_ascii=False)
    except Exception:
        return repr(redacted)
