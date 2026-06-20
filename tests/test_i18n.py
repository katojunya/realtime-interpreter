"""i18n の言語コード解決テスト."""

from __future__ import annotations

from realtime_interpreter.i18n import (
    DEFAULT_SOURCE,
    DEFAULT_TARGET,
    LANGUAGES,
    language_name,
    normalize_language_code,
)


def test_defaults_are_en_ja() -> None:
    assert DEFAULT_SOURCE == "en"
    assert DEFAULT_TARGET == "ja"


def test_known_languages_resolve() -> None:
    assert language_name("en") == "English"
    assert language_name("ja") == "Japanese"
    assert language_name("es") == "Spanish"
    assert language_name("zh") == "Chinese"


def test_unknown_language_returns_code() -> None:
    # 未知のコードは検証せずそのまま返す (APIエラーに委ねる方針)
    assert language_name("xx") == "xx"
    assert language_name("zz") == "zz"


def test_case_insensitive() -> None:
    assert language_name("EN") == "English"
    assert language_name("JA") == "Japanese"


def test_normalize_language_code() -> None:
    assert normalize_language_code("EN") == "en"
    assert normalize_language_code("  ja  ") == "ja"
    assert normalize_language_code("ES") == "es"


def test_normalize_preserves_script_subtag_casing() -> None:
    # スクリプトサブタグは Title case を保持 (Gemini の zh-Hans/zh-Hant 要件)
    assert normalize_language_code("zh-Hans") == "zh-Hans"
    assert normalize_language_code("zh-hant") == "zh-Hant"
    assert normalize_language_code("ZH-HANT") == "zh-Hant"
    assert normalize_language_code("  Zh-HaNs ") == "zh-Hans"


def test_normalize_uppercases_region_subtag() -> None:
    assert normalize_language_code("zh-cn") == "zh-CN"
    assert normalize_language_code("ZH-TW") == "zh-TW"
    assert normalize_language_code("pt-br") == "pt-BR"


def test_chinese_variant_names_resolve() -> None:
    assert language_name("zh") == "Chinese"
    assert language_name("zh-Hans") == "Chinese (Simplified)"
    assert language_name("zh-hant") == "Chinese (Traditional)"  # 大小無視
    assert language_name("zh-tw") == "Chinese (Traditional, Taiwan)"
    assert language_name("zh-CN") == "Chinese (Simplified, mainland China)"


def test_languages_dict_contains_defaults() -> None:
    assert DEFAULT_SOURCE in LANGUAGES
    assert DEFAULT_TARGET in LANGUAGES


def test_summary_prompt_uses_language_names() -> None:
    """build_summary_prompt が言語名を展開することを確認."""
    from realtime_interpreter.summarizer import build_summary_prompt

    prompt = build_summary_prompt("hello", 60, source_lang="en", target_lang="ja")
    assert "English" in prompt
    assert "Japanese" in prompt

    # 別ペア
    prompt_es = build_summary_prompt("hola", 60, source_lang="es", target_lang="en")
    assert "Spanish" in prompt_es
    assert "English" in prompt_es


def test_translator_prompt_uses_language_names() -> None:
    """GemmaAudioTranslator の起動時 prompt が言語名で format されることを確認."""
    from realtime_interpreter.translator import (
        TRANSCRIBE_AND_TRANSLATE_PROMPT,
        language_name,
    )

    # template に placeholder があること
    assert "{source_language}" in TRANSCRIBE_AND_TRANSLATE_PROMPT
    assert "{target_language}" in TRANSCRIBE_AND_TRANSLATE_PROMPT

    # ja → es に format
    formatted = TRANSCRIBE_AND_TRANSLATE_PROMPT.format(
        source_language=language_name("ja"),
        target_language=language_name("es"),
    )
    assert "Japanese" in formatted
    assert "Spanish" in formatted
    assert "{source_language}" not in formatted
