"""ストリーミング表示用レンダラ.

Rich Live を使い、進行中のセグメント (英語 + 日本語の 2 行) を in-place で更新表示する。
セグメントが確定 (debounce or VAD 終了) したら、Live 領域から「上の永続表示領域」へ
昇格させる (rich の `console.print` が Live の上にラインを差し込む挙動を利用).

確定したセグメントは色付きで append-only に積み上がり、書き換えは発生しない。
進行中セグメントは斜体で in-place 更新される。
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text


class StreamingRenderer:
    """確定済みセグメントは print, 進行中セグメントは Live update.

    使い方:
        with StreamingRenderer() as r:
            r.update_status("● Listening...")
            # delta 受信
            r.update_current("00:05", "We are", "私たち")
            r.update_current("00:05", "We are driven", "私たちは")
            # ターン確定 (debounce)
            r.commit("00:05", "We are driven by the idea.", "私たちは...")
            r.update_status("● Listening...")
    """

    def __init__(self) -> None:
        self._console = Console()
        # 進行中セグメント. None なら表示なし
        self._current_ts: str | None = None
        self._current_en: str = ""
        self._current_ja: str = ""
        self._status: str = ""
        self._live: Live | None = None

    def __enter__(self) -> "StreamingRenderer":
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=15,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    def _render(self) -> Group:
        items: list[Text] = []
        if self._current_ts is not None and (self._current_en or self._current_ja):
            # 進行中セグメント. 英語(グレー斜体) → 日本語(白斜体)
            if self._current_en:
                items.append(
                    Text(f"[{self._current_ts}] {self._current_en}", style="dim italic")
                )
            if self._current_ja:
                items.append(
                    Text(f"[{self._current_ts}] {self._current_ja}", style="italic")
                )
        if self._status:
            items.append(Text(self._status, style="yellow"))
        return Group(*items)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def update_current(self, ts: str, english: str, japanese: str) -> None:
        """進行中セグメントの状態を上書き (delta 受信ごとに呼ぶ)."""
        self._current_ts = ts
        self._current_en = english
        self._current_ja = japanese
        self._refresh()

    def commit(self, ts: str, english: str, japanese: str) -> None:
        """ターン確定: 永続表示エリアに昇格させ、進行中をクリア.

        Live の上に console.print することで、確定した行が Live エリアの上に
        積み上がる (Rich Live の標準挙動). 進行中表示はクリアされ、次のターンを待つ。
        """
        en = english.strip()
        ja = japanese.strip()
        if en:
            # グレー(英語転写) / 通常色(日本語訳) の append-only 出力
            self._console.print(f"[dim][{ts}] {en}[/dim]")
        if ja:
            self._console.print(f"[{ts}] {ja}")
        if en or ja:
            self._console.print("")
        self._current_ts = None
        self._current_en = ""
        self._current_ja = ""
        self._refresh()

    def emit_summary(self, ts: str, text: str) -> None:
        """要約ブロックを永続表示エリアに出す (要約は in-progress 表示しない)."""
        text = text.strip()
        if not text:
            return
        self._console.print(f"[cyan]--- 要約 [{ts}] ---[/cyan]")
        self._console.print(text)
        self._console.print("[cyan]---[/cyan]")
        self._console.print("")
        self._refresh()

    def update_status(self, text: str) -> None:
        self._status = text
        self._refresh()

    def clear_status(self) -> None:
        self._status = ""
        self._refresh()
