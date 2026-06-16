"""ローカル MLX バックエンド.

BlackHole 2ch から音声をキャプチャし、Silero VAD で発話セグメントを切り出して
mlx-vlm + Gemma 4 で英語転写と日本語訳を 1 回の推論で生成する。
"""

from __future__ import annotations

import logging
from types import ModuleType, TracebackType
from typing import Iterator

from realtime_interpreter.audio import SpeechSegmentCapture
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
    ) -> None:
        self.translator = translator
        self._capture = SpeechSegmentCapture(
            sd_module=sd_module,
            device_name=device_name,
            end_silence_ms=end_silence_ms,
            max_segment_seconds=max_segment_seconds,
        )

    def __enter__(self) -> "LocalMLXBackend":
        self.translator.load()
        self._capture.__enter__()
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
                result = self.translator.translate(segment.audio)
            except Exception:
                logger.exception("translation failed")
                continue
            yield TranslatedSegment(
                start_offset_seconds=segment.start_offset_seconds,
                duration_seconds=segment.duration_seconds,
                source=result.source,
                target=result.target,
                is_partial=False,
            )
