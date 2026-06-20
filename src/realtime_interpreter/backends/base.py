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

    # 任意メソッド (Protocol の必須面には含めない):
    #   def set_audio_output_callback(self, cb: Callable[[bytes, int], None]) -> None
    # モデル生成の翻訳音声 (PCM16) を読み上げ再生へ流すコールバックを登録する。
    # ネイティブ音声を持つ realtime 系 (gemini-realtime / openai-realtime) のみ実装し、
    # gemma4 系 (openai-chat / mlx) は実装しない。呼び出し側は hasattr で存在を確認する。
    # 必須面に含めると音声非対応バックエンドが isinstance(..., TranslationBackend) を
    # 満たさなくなるため、あえて任意扱いとする。

