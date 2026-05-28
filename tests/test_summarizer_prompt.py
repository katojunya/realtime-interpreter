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


def test_prompt_mentions_target_language() -> None:
    """既定 (en→ja) の場合、Japanese が target として出てくること."""
    prompt = build_summary_prompt("foo", 60)
    assert "Japanese" in prompt
    assert "English" in prompt


def test_prompt_uses_custom_languages() -> None:
    """source_lang/target_lang を渡すと言語名が差し替わること."""
    prompt = build_summary_prompt("hola", 60, source_lang="es", target_lang="en")
    assert "Spanish" in prompt
    assert "English" in prompt
    # 既定で使われていた Japanese / English (source) は消えていること
    assert "Japanese" not in prompt
