"""build_summary_prompt のテスト. mlx-vlm の実体には触れない."""

from __future__ import annotations

from realtime_interpreter.summarizer import build_summary_prompt


def test_basic_prompt_contains_text_and_duration() -> None:
    prompt = build_summary_prompt("Hello world.", 60)
    assert "Hello world." in prompt
    assert "60" in prompt


def test_prompt_strips_text() -> None:
    prompt = build_summary_prompt("  Hello  \n  ", 30)
    assert "Hello" in prompt
    # 余計な改行や空白がプロンプトに大量混入していないこと
    assert "  Hello  " not in prompt


def test_prompt_mentions_japanese_output() -> None:
    prompt = build_summary_prompt("foo", 60)
    # ルール部に日本語出力指示が含まれること
    assert "日本語" in prompt
