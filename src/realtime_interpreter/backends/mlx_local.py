"""ローカル MLX バックエンド.

BlackHole 2ch から音声をキャプチャし、Silero VAD で発話セグメントを切り出して
mlx-vlm + Gemma 4 で英語転写と日本語訳を 1 回の推論で生成する。
"""

from __future__ import annotations

import logging
from types import ModuleType, TracebackType
from typing import Iterator, Callable

from rich.text import Text

from realtime_interpreter.audio import SAMPLE_RATE, SpeechSegmentCapture
from realtime_interpreter.backends.base import TranslatedSegment
from realtime_interpreter.translator import GemmaAudioTranslator

logger = logging.getLogger(__name__)


class LocalMLXBackend:
    """ローカル MLX バックエンド. オフライン・無料・高品質."""

    def __init__(
        self,
        sd_module: ModuleType,
        translator: GemmaAudioTranslator,
        device_name: str,
        end_silence_ms: int,
        max_segment_seconds: float,
        diarizer: object | None = None,
    ) -> None:
        self.translator = translator
        self._comm_cb: Callable[[object], None] | None = None
        # 話者ダイアライゼーション (--diarize). None なら無効。
        self._diarizer = diarizer
        self._capture = SpeechSegmentCapture(
            sd_module=sd_module,
            device_name=device_name,
            end_silence_ms=end_silence_ms,
            max_segment_seconds=max_segment_seconds,
        )

    def set_status_callback(
        self,
        audio_cb: Callable[[object], None],
        comm_cb: Callable[[object], None],
    ) -> None:
        # 左スロット(音声メーター)は capture が、右スロット(通信)は本 backend が更新。
        self._comm_cb = comm_cb
        self._capture.status_callback = audio_cb

    def _model_lbl(self) -> str:
        return self.translator.model_id.split("/")[-1]

    def _comm_translating(self) -> str:
        """右スロット用: GPU 推論中."""
        return f"> Translating ({self._model_lbl()} Local GPU Inference)..."

    def _comm_listening(self) -> str:
        """右スロット用: 推論完了後の発話待ち (idle)."""
        return f"> Listening ({self._model_lbl()} Local GPU)..."

    def __enter__(self) -> "LocalMLXBackend":
        if self._comm_cb:
            self._comm_cb(f"... Loading Model ({self._model_lbl()} Local GPU)")
        self.translator.load()
        self._capture.backend_name = "Local MLX"
        self._capture.__enter__()
        # 最初の発話待ちから Listening を表示する。
        if self._comm_cb:
            self._comm_cb(self._comm_listening())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._capture.__exit__(exc_type, exc_val, exc_tb)

    def update_context(self, summary: str) -> None:
        """ローリング要約を翻訳の参照文脈として translator へ供給する (main から呼ばれる)."""
        self.translator.update_context(summary)

    def stream_segments(self) -> Iterator[TranslatedSegment]:
        for segment in self._capture.segments():
            try:
                if self._comm_cb:
                    self._comm_cb(self._comm_translating())
                result = self.translator.translate(segment.audio)
            except Exception:
                logger.exception("translation failed")
                if self._comm_cb:
                    self._comm_cb(self._comm_listening())
                continue
            # 翻訳完了 → 次の発話待ち (Listening) へ戻す。
            if self._comm_cb:
                self._comm_cb(self._comm_listening())
            speaker = (
                self._diarizer.assign(segment.audio, SAMPLE_RATE)
                if self._diarizer is not None
                else None
            )
            yield TranslatedSegment(
                start_offset_seconds=segment.start_offset_seconds,
                duration_seconds=segment.duration_seconds,
                source=result.source,
                target=result.target,
                is_partial=False,
                speaker=speaker,
            )
