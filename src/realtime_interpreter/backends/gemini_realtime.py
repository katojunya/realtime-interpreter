"""Gemini Multimodal Live API バックエンド.

WebSocket 経由で 16kHz PCM16 音声をストリームし、source 言語の転写(inputAudioTranscription)と
target 言語訳(modelTurn)の delta を受け取る。セグメントの確定は emit_loop の debounce
(一定時間 delta が来ない=発話の切れ目)と max_segment(連続発話の強制カット上限)で行う。
サーバの `turnComplete` は短い翻訳単位ごとに頻発し過剰分割を招くため、確定トリガとしては
使わない(連続ターンを束ねる)。

共通のキャプチャ/送受信/emit/再接続機構は `WebSocketStreamingBackend` に集約され、
本モジュールは Gemini 固有のプロトコル差分(session resumption / context window
compression / live-translate setup)のみを実装する。
"""

from __future__ import annotations

import base64
import json
import logging
import os
from types import ModuleType

import numpy as np

from realtime_interpreter.backends._ws_streaming_base import WebSocketStreamingBackend
from realtime_interpreter.i18n import DEFAULT_SOURCE, DEFAULT_TARGET, language_name

logger = logging.getLogger(__name__)

GEMINI_SAMPLE_RATE = 16000
GEMINI_REALTIME_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
DEFAULT_GEMINI_MODEL = "models/gemini-3.5-live-translate-preview"


class GeminiRealtimeBackend(WebSocketStreamingBackend):
    """Gemini Multimodal Live API を使った WebSocket ベースの翻訳バックエンド."""

    SAMPLE_RATE = GEMINI_SAMPLE_RATE
    PROACTIVE_RECONNECT_SECONDS = None  # goAway/切断検知後の reactive 再接続のみ
    LOG_NAME = "Gemini"
    THREAD_PREFIX = "gemini"

    def __init__(
        self,
        sd_module: ModuleType | None,
        device_name: str,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        turn_debounce_ms: int = 800,
        max_segment_seconds: float = 8.0,
        source_lang: str = DEFAULT_SOURCE,
        target_lang: str = DEFAULT_TARGET,
        loopback: bool | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY environment variable."
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
        # Session Resumption: 切断をまたいでセッションを継続するための handle。
        # サーバが sessionResumptionUpdate.newHandle で送ってくる。
        self._resumption_handle: str | None = None

    # ---------------- protocol-specific hooks ----------------

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
                        "echoTargetLanguage": False,
                    },
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
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
                                f"You are a professional simultaneous interpreter from "
                                f"{language_name(self.source_lang)} to "
                                f"{language_name(self.target_lang)}. "
                                f"Translate the incoming audio stream to "
                                f"{language_name(self.target_lang)}. "
                                f"Output only the translation, with no extra text or commentary."
                            )
                        }
                    ]
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
            }

        # Session Resumption を有効化. 初回は空 ({}) でハンドル発行を要求し、
        # 再接続時は保存済みハンドルを渡して同一セッションを継続する。
        if resume and self._resumption_handle:
            setup["sessionResumption"] = {"handle": self._resumption_handle}
        else:
            setup["sessionResumption"] = {}

        # Context Window Compression を有効化 (sliding window, 既定パラメータ).
        # session resumption が「~10分の接続切断」を乗り越えるのに対し、こちらは
        # 「15分のセッション時間 / 128k トークン上限」を回避してセッションを延長する。
        setup["contextWindowCompression"] = {"slidingWindow": {}}

        assert self._ws is not None
        self._ws.send(json.dumps({"setup": setup}))
        logger.debug("setup sent (resume=%s)", resume)

    def _send_audio_chunk(self, audio: np.ndarray) -> None:
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
        # 公式 live-translate ドキュメントの現行形式 `realtimeInput.audio` を使う。
        msg = {
            "realtimeInput": {
                "audio": {
                    "data": b64,
                    "mimeType": f"audio/pcm;rate={GEMINI_SAMPLE_RATE}",
                }
            }
        }
        assert self._ws is not None
        self._ws.send(json.dumps(msg))

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

        # goAway: サーバが接続をまもなく切ることの予告. 記録のみ (切断は recv で検知)。
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

        # 1. source 転写 (inputTranscription / inputAudioTranscription)
        input_transcription = server_content.get("inputTranscription") or server_content.get(
            "inputAudioTranscription"
        )
        if input_transcription:
            text = input_transcription.get("text", "")
            if text:
                self._append_input(text)

        # 2. target 訳 (outputTranscription / outputAudioTranscription)
        output_transcription = server_content.get("outputTranscription") or server_content.get(
            "outputAudioTranscription"
        )
        if output_transcription:
            text = output_transcription.get("text", "")
            if text:
                self._append_output(text)

        model_turn = server_content.get("modelTurn")
        if model_turn:
            for part in model_turn.get("parts", []):
                text_part = part.get("text", "")
                if text_part:
                    self._append_output(text_part)

        # turnComplete は確定トリガに使わない (過剰分割対策; debounce/max_segment に委譲)。
