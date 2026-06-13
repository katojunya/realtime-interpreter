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

    def __init__(self, log_dir: Path | str = "logs", timestamp: str | None = None) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self._dir / f"session_{ts}.log"
        self._start = time.monotonic()
        self._fh = self.path.open("w", encoding="utf-8", buffering=1)
        self._write_header()

    def _write_header(self) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        self._fh.write(f"# realtime-interpreter session\n# started: {now}\n\n")

    def elapsed(self) -> str:
        """セッション開始からの経過時間を mm:ss で返す."""
        return format_offset(time.monotonic() - self._start)

    def log_segment(self, ts: str, source: str, target: str) -> None:
        """1 セグメントの source 転写と target 訳をペアで記録."""
        if source.strip():
            self._fh.write(f"[{ts}] {source.strip()}\n")
        if target.strip():
            self._fh.write(f"[{ts}] {target.strip()}\n")
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
            self.log_event("session ended")
            self._fh.close()

    def __enter__(self) -> SessionLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
