"""_parse_src_tgt: モデル出力パーサのテスト. mlx-vlm の実体には触れない."""

from __future__ import annotations

from realtime_interpreter.translator import _parse_src_tgt


def test_canonical_format() -> None:
    text = "SRC: Hello world.\nTGT: こんにちは世界。"
    src, tgt = _parse_src_tgt(text)
    assert src == "Hello world."
    assert tgt == "こんにちは世界。"


def test_extra_whitespace() -> None:
    text = "  SRC:   Foo bar  \n  TGT: フー バー   \n"
    src, tgt = _parse_src_tgt(text)
    assert src == "Foo bar"
    assert tgt == "フー バー"


def test_lowercase_tag_accepted() -> None:
    text = "src: hi\ntgt: やあ"
    src, tgt = _parse_src_tgt(text)
    assert src == "hi"
    assert tgt == "やあ"


def test_only_tgt_line() -> None:
    text = "TGT: ハロー"
    src, tgt = _parse_src_tgt(text)
    assert src == ""
    assert tgt == "ハロー"


def test_no_tags_falls_back_to_target() -> None:
    """モデルが指示形式を無視した場合は全体を target とみなす."""
    text = "これは翻訳のみです。"
    src, tgt = _parse_src_tgt(text)
    assert src == ""
    assert tgt == "これは翻訳のみです。"


def test_empty_payload() -> None:
    text = "SRC:\nTGT:"
    src, tgt = _parse_src_tgt(text)
    assert src == ""
    assert tgt == ""


def test_extra_chatter_around_tags() -> None:
    """タグ前後に余計な行があっても拾える."""
    text = (
        "Sure, here is the result:\n"
        "SRC: Mr. Quilter is the apostle.\n"
        "TGT: クウィルター氏は使徒です。\n"
    )
    src, tgt = _parse_src_tgt(text)
    assert src == "Mr. Quilter is the apostle."
    assert tgt == "クウィルター氏は使徒です。"
