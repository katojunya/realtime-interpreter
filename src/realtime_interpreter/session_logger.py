"""セッションログ管理.

logs/{タイムスタンプ}.log にセグメント単位で英語転写と日本語訳を記録する。

形式:
    [mm:ss] <english>
    [mm:ss] <japanese>
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path


def format_offset(seconds: float) -> str:
    """秒を mm:ss 形式に整形 (60 分以上は時:分:秒)."""
    seconds = max(0.0, seconds)
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    return f"{total // 60:02d}:{total % 60:02d}"


class SessionLogger:
    """1 セッション分のログをファイルに記録する."""

    def __init__(
        self,
        log_dir: Path | str = "logs",
        timestamp: str | None = None,
        settings: list[tuple[str, str]] | None = None,
    ) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self._dir / f"session_{ts}.log"
        self._start = time.monotonic()
        # 起動設定 (backend/モデル/言語/デバイス 等) を (label, value) で受け取り、
        # ヘッダにコメントとして記録する。API キー等の機密は呼び出し側で含めないこと。
        self._settings = settings or []
        self._fh = self.path.open("w", encoding="utf-8", buffering=1)
        self._write_header()

    def _write_header(self) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        self._fh.write("# realtime-interpreter session\n")
        rows = [("started", now), *self._settings]
        for label, value in rows:
            self._fh.write(f"# {(label + ':'):<14}{value}\n")
        self._fh.write("\n")

    def elapsed(self) -> str:
        """セッション開始からの経過時間を mm:ss で返す."""
        return format_offset(time.monotonic() - self._start)

    def log_segment(
        self, ts: str, source: str, target: str, speaker: str | None = None
    ) -> None:
        """1 セグメントの source 転写と target 訳をペアで記録.

        speaker (話者ラベル) があれば各行に `[S1] ` を前置する。
        """
        prefix = f"[{speaker}] " if speaker else ""
        if source.strip():
            self._fh.write(f"[{ts}] {prefix}{source.strip()}\n")
        if target.strip():
            self._fh.write(f"[{ts}] {prefix}{target.strip()}\n")
        if source.strip() or target.strip():
            self._fh.write("\n")

    def log_summary(self, ts: str, text: str) -> None:
        """N 秒ごとの日本語要約を記録."""
        if not text.strip():
            return
        self._fh.write(f"--- 要約 [{ts}] ---\n{text.strip()}\n---\n\n")

    def log_event(self, message: str) -> None:
        """システムイベント (起動/終了/エラー等) を記録."""
        self._fh.write(f"# [{self.elapsed()}] {message}\n")

    def close(self) -> None:
        if not self._fh.closed:
            self.log_event(f"session ended (duration {self.elapsed()})")
            self._fh.close()

    def __enter__(self) -> SessionLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
