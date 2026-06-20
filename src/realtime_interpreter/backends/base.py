"""バックエンド共通の Protocol とデータクラス."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Iterator, Protocol, runtime_checkable, Callable


class BackendState(Enum):
    """バックエンドの動作状態."""

    LOADING_MODEL = "loading_model"  # ローカルモデル読み込み中
    CONNECTING = "connecting"        # サーバー接続中
    LISTENING = "listening"          # 発話待ち (無音)
    SPEAKING = "speaking"            # 発話検知中 (音声バッファ蓄積中 / 音声送信中)
    TRANSLATING = "translating"      # 翻訳・推論実行中
    RECONNECTING = "reconnecting"    # 再接続試行中


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
    # 話者ラベル (S1/S2…). 話者ダイアライゼーション (--diarize) 有効時のみ設定。
    speaker: str | None = None


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

    def set_status_callback(
        self,
        audio_cb: Callable[[object], None],
        comm_cb: Callable[[object], None],
    ) -> None:
        """ステータス表示の2スロットを更新するコールバックを登録する.

        audio_cb: 左スロット = 音声入力レベル (capture スレッドから随時更新)。
        comm_cb:  右スロット = LLM 通信ステータス (接続/受信/翻訳/再接続 等)。
        """
        ...

