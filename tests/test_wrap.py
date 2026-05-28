"""_wrap_to_width の cell-aware char-level/word-level 折り返しテスト."""

from __future__ import annotations

from realtime_interpreter.renderer import _wrap_to_width


def test_no_wrap_when_fits() -> None:
    assert _wrap_to_width("hello world", 100) == "hello world"


def test_ascii_word_wrap_at_space() -> None:
    # "the hottest new" (15 cells) でちょうど 15 幅. 次の word は new に続く
    text = "the hottest new programming language"
    wrapped = _wrap_to_width(text, 15)
    rows = wrapped.split("\n")
    # 最後の単語が次行に行くだけで、単語の途中で切れない
    for row in rows:
        assert len(row) <= 15
    # 単語の途中で分割されていないこと
    for row in rows:
        # 各行は空白で始まらない (前の rstrip で消えている)
        assert not row.startswith(" ")


def test_cjk_char_wrap() -> None:
    # 各全角文字は 2 セル. 幅 10 セル = 5 全角文字
    text = "同じ期間で1日のリクエスト"  # 11 全角 + 1 半角 = まあまあ
    wrapped = _wrap_to_width(text, 10)
    rows = wrapped.split("\n")
    # 各行は 10 セル以下 (全角 5 文字ぶん)
    for row in rows:
        # cell width 計算: 全角=2, 半角=1
        from rich.cells import cell_len
        assert cell_len(row) <= 10


def test_cjk_breaks_at_any_char() -> None:
    """CJK ランは文字単位で折れる ([mm:ss] 直後で全部次行に行く現象が起きないこと)."""
    # prefix 風 + CJK 本文.  幅 20 セルだと prefix(8) + 全角 6 文字(12) でちょうど
    text = "[01:04] " + "あいうえおかきくけこさしすせそ"  # 全角 15 文字 = 30 セル
    wrapped = _wrap_to_width(text, 20)
    rows = wrapped.split("\n")
    # 最初の行に prefix + 少なくとも数文字の CJK が含まれること
    assert rows[0].startswith("[01:04] ")
    # 「[01:04] 」だけで終わって CJK が全部次行、にはなっていないこと
    assert len(rows[0]) > len("[01:04] ")


def test_preserves_existing_newlines() -> None:
    text = "line1\nline2"
    assert _wrap_to_width(text, 100) == "line1\nline2"


def test_zero_or_negative_width_is_noop() -> None:
    assert _wrap_to_width("any text", 0) == "any text"
    assert _wrap_to_width("any text", -1) == "any text"


def test_mixed_ascii_cjk() -> None:
    # 英語 + 日本語の混在
    text = "[01:04] Gemini 3.5 Flashの持つ反重力と遺伝的コーディング能力を検索に統合"
    wrapped = _wrap_to_width(text, 30)
    rows = wrapped.split("\n")
    from rich.cells import cell_len
    for row in rows:
        assert cell_len(row) <= 30


def test_long_ascii_word_hard_wrap() -> None:
    """break opportunity がない長い単語は max_width でハード改行される."""
    text = "supercalifragilisticexpialidocious"  # 1 単語. 幅 10 にハード改行
    wrapped = _wrap_to_width(text, 10)
    rows = wrapped.split("\n")
    for row in rows:
        assert len(row) <= 10
