"""文末検出ヘルパのテスト."""

from __future__ import annotations

from realtime_interpreter.backends.openai_realtime import (
    _first_complete_sentence_end_en,
    _first_complete_sentence_end_ja,
)


# ---- JA ----


def test_ja_finds_japanese_period() -> None:
    text = "こんにちは。元気ですか？"
    idx = _first_complete_sentence_end_ja(text)
    assert idx is not None and text[:idx] == "こんにちは。"


def test_ja_finds_question_mark() -> None:
    text = "本当ですか？"
    idx = _first_complete_sentence_end_ja(text)
    assert idx is not None and text[:idx] == "本当ですか？"


def test_ja_no_boundary_returns_none() -> None:
    assert _first_complete_sentence_end_ja("これは未完の文") is None


def test_ja_handles_halfwidth_punctuation() -> None:
    idx = _first_complete_sentence_end_ja("Hello!世界")
    assert idx is not None and idx == 6  # "Hello!" まで


# ---- EN ----


def test_en_finds_period_followed_by_space() -> None:
    text = "Hello world. Next sentence"
    idx = _first_complete_sentence_end_en(text)
    assert idx is not None and text[:idx] == "Hello world."


def test_en_skips_ellipsis() -> None:
    """'...' で誤って区切らない."""
    text = "Wait... let me think. Done"
    idx = _first_complete_sentence_end_en(text)
    assert idx is not None and text[:idx] == "Wait... let me think."


def test_en_skips_abbreviation_no_space() -> None:
    """'Mr.Smith' の途中ピリオドではマッチしない."""
    text = "Mr.Smith arrived. OK"
    idx = _first_complete_sentence_end_en(text)
    # 最初に space が続く `.` は "arrived." の後
    assert idx is not None and text[:idx] == "Mr.Smith arrived."


def test_en_no_boundary_at_buffer_end() -> None:
    """末尾の `.` は次 delta を待つ (空白続きが確認できないので確定不能)."""
    assert _first_complete_sentence_end_en("Hello world.") is None


def test_en_finds_question_mark_followed_by_space() -> None:
    text = "How are you? Fine"
    idx = _first_complete_sentence_end_en(text)
    assert idx is not None and text[:idx] == "How are you?"


def test_en_no_boundary_returns_none() -> None:
    assert _first_complete_sentence_end_en("incomplete sentence") is None


def test_en_period_followed_by_newline() -> None:
    text = "Done.\nNext"
    idx = _first_complete_sentence_end_en(text)
    assert idx is not None and text[:idx] == "Done."
