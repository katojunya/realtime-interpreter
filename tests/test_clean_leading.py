"""_clean_leading: 行頭句読点除去のテスト (OpenAI backend の表示アーティファクト対策)."""

from __future__ import annotations

from realtime_interpreter.backends.openai_realtime import _clean_leading


def test_strips_leading_japanese_period() -> None:
    assert _clean_leading("。18年になります") == "18年になります"


def test_strips_leading_ascii_period_and_space() -> None:
    assert _clean_leading(". Yeah. So") == "Yeah. So"


def test_strips_leading_comma() -> None:
    assert _clean_leading("、その後") == "その後"
    assert _clean_leading(", and then") == "and then"


def test_preserves_mid_and_trailing_punctuation() -> None:
    # 中間・末尾の句読点は保持
    assert _clean_leading("Yeah. So we do.") == "Yeah. So we do."
    assert _clean_leading("私たちは、続けます。") == "私たちは、続けます。"


def test_punctuation_only_becomes_empty() -> None:
    assert _clean_leading("。") == ""
    assert _clean_leading(". ") == ""
    assert _clean_leading(" 、。 ") == ""


def test_mixed_leading_punctuation_and_spaces() -> None:
    assert _clean_leading("  . 、 Hello") == "Hello"


def test_no_leading_punctuation_is_unchanged() -> None:
    assert _clean_leading("Hello world") == "Hello world"
    assert _clean_leading("こんにちは") == "こんにちは"


def test_fullwidth_space_stripped() -> None:
    # 全角スペース
    assert _clean_leading("　。　テキスト") == "テキスト"


def test_ellipsis_stripped() -> None:
    assert _clean_leading("…そして") == "そして"
