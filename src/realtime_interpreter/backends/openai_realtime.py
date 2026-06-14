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

共通のキャプチャ/送受信/emit/再接続機構は `WebSocketStreamingBackend` に集約され、
本モジュールは OpenAI 固有のプロトコル差分のみを実装する。
"""

from __future__ import annotations

import base64
import json
import logging
import os
from types import ModuleType

import numpy as np

from realtime_interpreter.backends._ws_streaming_base import WebSocketStreamingBackend
from realtime_interpreter.i18n import DEFAULT_SOURCE, DEFAULT_TARGET

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-realtime-translate"
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime/translations"
OPENAI_SAMPLE_RATE = 24000
INPUT_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"

# OpenAI Realtime API は 1 接続あたり最大 60 分でサーバ側から切断される。
# その手前 (PROACTIVE) で自分から新しい接続へ張り替えることで、ハード制限到達による
# 切れ目をほぼゼロにする。到達後の切断 (REACTIVE) やネット瞬断も recv_loop が拾って
# 同じ再接続経路で復帰する。OpenAI Realtime にセッション再開ハンドルは無いため、
# 再接続は新規セッションの張り直し (要約履歴はアプリ側保持なので継続)。
OPENAI_PROACTIVE_RECONNECT_SECONDS = 55 * 60

# delta が来なくなってからこのミリ秒経過 → ターン終了とみなして commit.
# このモデル (gpt-realtime-translate) は EN 1 文を JA 複数文 (例: 挨拶を独立文として
# 切ってから本文を続ける) で訳すことがあるため、文単位での即時 commit を行うと
# EN/JA の対応関係が崩れる. そのため発話の切れ目 (= 翻訳がひと段落して delta が
# 来なくなった瞬間) を唯一の commit トリガとし、対応の整合を優先する。
TURN_DEBOUNCE_MS = 800

# 連続発話 (ポーズなし) で 1 チャンクが肥大化するのを防ぐ強制 commit 上限.
# この秒数に達したら delta が継続中でも強制的に commit する.
# 0 を指定すると無効 (debounce のみで commit).
OPENAI_MAX_SEGMENT_SECONDS = 8.0

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


class OpenAIRealtimeBackend(WebSocketStreamingBackend):
    """OpenAI gpt-realtime-translate を使った WebSocket ベースの翻訳バックエンド.

    サーバ VAD によりターン区切りは OpenAI 側で検出されるが、ターン完了通知
    イベントが API に存在しないため、delta の debounce で代用している。
    """

    SAMPLE_RATE = OPENAI_SAMPLE_RATE
    PROACTIVE_RECONNECT_SECONDS = OPENAI_PROACTIVE_RECONNECT_SECONDS
    LOG_NAME = "OpenAI"
    THREAD_PREFIX = "openai"

    def __init__(
        self,
        sd_module: ModuleType | None,
        device_name: str,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        turn_debounce_ms: int = TURN_DEBOUNCE_MS,
        max_segment_seconds: float = OPENAI_MAX_SEGMENT_SECONDS,
        source_lang: str = DEFAULT_SOURCE,
        target_lang: str = DEFAULT_TARGET,
        loopback: bool | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key. Set the OPENAI_API_KEY environment variable, "
                "or pass --openai-realtime-api-key (alias --openai-rt-api-key)."
            )
        super().__init__(
            sd_module=sd_module,
            device_name=device_name,
            api_key=resolved_key,
            model=model,
            turn_debounce_ms=turn_debounce_ms,
            max_segment_seconds=max_segment_seconds,
            source_lang=source_lang,
            target_lang=target_lang,
            loopback=loopback,
        )

    # ---------------- protocol-specific hooks ----------------

    def _open_websocket(self) -> None:
        import websocket

        url = f"{OPENAI_REALTIME_URL}?model={self._model}"
        self._ws = websocket.WebSocket()
        self._ws.connect(url, header=[f"Authorization: Bearer {self._api_key}"])
        logger.info("connected to %s", url)

    def _send_session_config(self, resume: bool = False) -> None:
        # 出力言語は self.target_lang (ISO 639-1) を `audio.output.language` に指定.
        # 入力 (source) の transcript は gpt-realtime-whisper で. source 言語は Whisper
        # の auto-detect 任せ (multilingual で安定)。OpenAI Realtime にセッション再開
        # ハンドルは無いので resume は無視する (再接続=新規セッション)。
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

    def _send_audio_chunk(self, audio: np.ndarray) -> None:
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
        msg = {"type": "session.input_audio_buffer.append", "audio": b64}
        assert self._ws is not None
        self._ws.send(json.dumps(msg))

    def _clean(self, text: str) -> str:
        return _clean_leading(text)

    def _handle_event(self, event: dict) -> None:
        etype = event.get("type", "")

        if etype == "error" or etype.endswith(".error"):
            logger.error("OpenAI error: %s", event)
            return

        # 翻訳音声は使わない. ノイズ抑制のため明示スキップ.
        if etype.endswith("output_audio.delta") or etype.endswith("output_audio.done"):
            return

        # source (入力) 転写の delta
        if "input_transcript.delta" in etype or "input_audio_transcription.delta" in etype:
            delta = event.get("delta", "")
            if delta:
                self._append_input(delta)
            return

        # target (出力) 訳の delta
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

        # 未マッチのイベントは debug に payload と共に残す
        logger.debug("unhandled event type=%s payload=%s", etype, _summarize_event(event))
