"""バックエンド共通の Protocol とデータクラス."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Iterator, Protocol, runtime_checkable


@dataclass
class TranslatedSegment:
    """1 セグメント分の翻訳結果. バックエンド非依存の出力単位.

    is_partial=True: ストリーミング途中の暫定値. 同じ start_offset_seconds で
                     複数回 yield される. 後続の yield で上書きされる前提.
    is_partial=False: ターン確定 (発話終了). この値以降は同じターンで更新されない。
                      ログ・要約バッファ等の永続処理はこちらの値を使う。

    フィールド `source` / `target` は言語非依存. 翻訳元 (例: 英語) が source, 翻訳先
    (例: 日本語) が target. 表示上は source を上, target を下に出す。
    """

    start_offset_seconds: float
    duration_seconds: float
    source: str
    target: str
    is_partial: bool = False


@dataclass
class BackendConfig:
    """複数バックエンドが共通で参照する設定."""

    device_name: str
    end_silence_ms: int
    max_segment_seconds: float


@runtime_checkable
class TranslationBackend(Protocol):
    """音声→翻訳パイプラインの抽象インターフェース.

    実装は context manager として使い、`stream_segments()` で確定したセグメントを
    順次 yield する。具体的なキャプチャ・推論方法 (ローカル vs 外部 API) はバックエンドが隠蔽する。
    """

    def __enter__(self) -> "TranslationBackend": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    def stream_segments(self) -> Iterator[TranslatedSegment]:
        """確定したセグメントを順次 yield する (無限イテレータ)."""
        ...
