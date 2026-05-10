"""_parse_en_ja: モデル出力パーサのテスト. mlx-vlm の実体には触れない."""

from __future__ import annotations

from realtime_interpreter.translator import _parse_en_ja


def test_canonical_format() -> None:
    text = "EN: Hello world.\nJA: こんにちは世界。"
    en, ja = _parse_en_ja(text)
    assert en == "Hello world."
    assert ja == "こんにちは世界。"


def test_extra_whitespace() -> None:
    text = "  EN:   Foo bar  \n  JA: フー バー   \n"
    en, ja = _parse_en_ja(text)
    assert en == "Foo bar"
    assert ja == "フー バー"


def test_lowercase_tag_accepted() -> None:
    text = "en: hi\nja: やあ"
    en, ja = _parse_en_ja(text)
    assert en == "hi"
    assert ja == "やあ"


def test_only_ja_line() -> None:
    text = "JA: ハロー"
    en, ja = _parse_en_ja(text)
    assert en == ""
    assert ja == "ハロー"


def test_no_tags_falls_back_to_ja() -> None:
    """モデルが指示形式を無視した場合は全体を JA とみなす."""
    text = "これは日本語訳のみです。"
    en, ja = _parse_en_ja(text)
    assert en == ""
    assert ja == "これは日本語訳のみです。"


def test_empty_payload() -> None:
    text = "EN:\nJA:"
    en, ja = _parse_en_ja(text)
    assert en == ""
    assert ja == ""


def test_extra_chatter_around_tags() -> None:
    """タグ前後に余計な行があっても拾える."""
    text = "Sure, here is the result:\nEN: Mr. Quilter is the apostle.\nJA: クウィルター氏は使徒です。\n"
    en, ja = _parse_en_ja(text)
    assert en == "Mr. Quilter is the apostle."
    assert ja == "クウィルター氏は使徒です。"
