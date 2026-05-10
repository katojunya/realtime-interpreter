"""翻訳バックエンドの切り替えポイント.

- LocalMLXBackend: BlackHole キャプチャ + Gemma 4 (mlx-vlm) で完結する完全ローカル実装
- OpenAIRealtimeBackend: BlackHole キャプチャ + OpenAI gpt-realtime-translate (WebSocket 経由)
"""

from realtime_interpreter.backends.base import (
    BackendConfig,
    TranslatedSegment,
    TranslationBackend,
)

__all__ = ["BackendConfig", "TranslatedSegment", "TranslationBackend"]
