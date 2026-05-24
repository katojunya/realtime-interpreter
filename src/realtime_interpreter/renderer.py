"""ストリーミング表示用レンダラ.

Rich Live を使い、進行中のセグメント (英語 + 日本語の 2 行) を in-place で更新表示する。
セグメントが確定 (debounce or VAD 終了) したら、Live 領域から「上の永続表示領域」へ
昇格させる (rich の `console.print` が Live の上にラインを差し込む挙動を利用).

確定したセグメントは色付きで append-only に積み上がり、書き換えは発生しない。
進行中セグメントは斜体で in-place 更新される。
"""

from __future__ import annotations

from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text


def _wrap_to_width(text: str, max_width: int) -> str:
    """テキストを max_width 「セル幅」で折り返した文字列を返す.

    Break opportunities:
    - ASCII スペース ' ': 英語の単語単位での折り返し
    - 全角 (cell_len >= 2) 文字: 日本語等の文字単位での折り返し

    既存の '\\n' は強制改行として保持する.
    """
    if max_width <= 0:
        return text

    rows: list[str] = []
    current: list[str] = []
    cur_width = 0
    last_break_idx = -1  # current の中で「ここで折ってよい」位置 (含む)

    for ch in text:
        if ch == "\n":
            rows.append("".join(current))
            current = []
            cur_width = 0
            last_break_idx = -1
            continue

        ch_width = cell_len(ch)

        if cur_width + ch_width > max_width and current:
            if last_break_idx >= 0:
                head = current[: last_break_idx + 1]
                tail = current[last_break_idx + 1 :]
                rows.append("".join(head).rstrip(" "))
                current = tail
                cur_width = sum(cell_len(c) for c in tail)
            else:
                # break opportunity が無い (例: ASCII の長い単語) → ハード改行
                rows.append("".join(current))
                current = []
                cur_width = 0
            last_break_idx = -1

        current.append(ch)
        cur_width += ch_width
        # スペース or 全角文字を break 候補として記録 (NBSP は対象外)
        if ch == " " or ch_width >= 2:
            last_break_idx = len(current) - 1

    if current:
        rows.append("".join(current))
    return "\n".join(rows)


def _timestamped_line(ts: str, body: str, body_style: str) -> Text:
    """`[ts] body` 形式の単一行 Text. commit() で使用 (terminal が折り返し)。"""
    prefix = f"[{ts}] "
    line = Text(prefix + body, style=body_style)
    line.stylize("green", 0, len(prefix))
    return line


def _timestamped_wrapped_rows(
    ts: str,
    body: str,
    body_style: str,
    max_width: int,
) -> list[Text]:
    """in-progress 表示用. 全文を `max_width` で予め折り返して複数の Text 行にする.

    Rich Live は受け取ったレンダラブルを wrap せず行単位でそのまま描画するため、
    こちらで cell-aware に折り返した結果を渡すことで:
    - `[mm:ss]` 直後で勝手に折られる現象を回避
    - 日本語は文字単位で、英語は単語単位で折り返される (`_wrap_to_width` の実装)
    - Live が「実際の描画行数」を正確に把握できる (no_wrap 無しで安全)
    """
    prefix = f"[{ts}] "
    full = prefix + body
    wrapped = _wrap_to_width(full, max_width) if max_width > 0 else full
    rows: list[Text] = []
    for i, row in enumerate(wrapped.split("\n")):
        text = Text(row, style=body_style)
        if i == 0 and row.startswith(prefix):
            text.stylize("green", 0, len(prefix))
        rows.append(text)
    return rows


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
            # 進行中セグメント. [mm:ss]=緑 / 英語=グレー斜体 / 日本語=白斜体
            # Rich の word-wrap は CJK ランを 1 単語扱いして [mm:ss] 直後で折る挙動が
            # あるため、こちらで cell-aware に折り返して multi-line Text を渡す。
            # Rich はその multi-line をそのまま描画し追加の wrap はしない。
            # → Live の行数追跡も正確で、ターミナル末尾でも凍結しない。
            max_width = self._console.size.width
            if self._current_en:
                items.extend(
                    _timestamped_wrapped_rows(
                        self._current_ts, self._current_en,
                        body_style="dim italic", max_width=max_width,
                    )
                )
            if self._current_ja:
                items.extend(
                    _timestamped_wrapped_rows(
                        self._current_ts, self._current_ja,
                        body_style="italic", max_width=max_width,
                    )
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

        実装メモ:
        - `Text(...)` で渡すことで `[{ts}]` をマークアップとして解釈させない
          (`console.print(f"[{ts}] ...")` だと `[00:59]` を不明なスタイルタグとして
           誤解釈し、表示が崩れる)
        - `soft_wrap=True` で Rich の word-wrap を無効化し、行折り返しは
          ターミナル任せにする (ASCII↔CJK 境界での Rich の早期折りを回避)
        """
        en = english.strip()
        ja = japanese.strip()
        if en:
            # [mm:ss]=緑 / 英語転写=グレー の append-only 出力
            self._console.print(
                _timestamped_line(ts, en, body_style="dim"),
                soft_wrap=True,
            )
        if ja:
            # [mm:ss]=緑 / 日本語訳=通常色
            self._console.print(
                _timestamped_line(ts, ja, body_style=""),
                soft_wrap=True,
            )
        if en or ja:
            self._console.print("")
        self._current_ts = None
        self._current_en = ""
        self._current_ja = ""
        self._refresh()

    def emit_summary(self, ts: str, text: str) -> None:
        """要約ブロックを永続表示エリアに出す (要約は in-progress 表示しない).

        commit() と同じく Text(...) + soft_wrap=True でマークアップ解釈と
        Rich の word-wrap を回避する。
        """
        text = text.strip()
        if not text:
            return
        self._console.print(Text(f"--- 要約 [{ts}] ---", style="cyan"), soft_wrap=True)
        self._console.print(Text(text), soft_wrap=True)
        self._console.print(Text("---", style="cyan"), soft_wrap=True)
        self._console.print("")
        self._refresh()

    def update_status(self, text: str) -> None:
        self._status = text
        self._refresh()

    def clear_status(self) -> None:
        self._status = ""
        self._refresh()
