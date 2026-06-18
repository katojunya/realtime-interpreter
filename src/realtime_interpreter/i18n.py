"""言語コード → 言語名のマッピング.

ISO 639-1 (2 文字コード) を採用. プロンプト内で人間可読な言語名 (例: "Japanese") に
展開するため、また CLI バリデーションの参照用に使う。

未知コードは `language_name()` でコードそのものを返す (起動時エラーにせず、
バックエンド側でモデルに渡して解釈を試みる方針 - 仕様確認: APIエラーに委ねる)。
"""

from __future__ import annotations

DEFAULT_SOURCE = "en"
DEFAULT_TARGET = "ja"


LANGUAGES: dict[str, str] = {
    # 主要対応言語. 必要なら追記してください。
    "en": "English",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "zh-hk": "Traditional Chinese (Hong Kong)",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "nl": "Dutch",
    "sv": "Swedish",
    "pl": "Polish",
    "tr": "Turkish",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
    "hi": "Hindi",
    "uk": "Ukrainian",
    "el": "Greek",
    "fi": "Finnish",
    "da": "Danish",
    "no": "Norwegian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "he": "Hebrew",
}


def language_name(code: str) -> str:
    """ISO 639-1 コードを英語の言語名に変換. 未知ならコードをそのまま返す."""
    return LANGUAGES.get(code.lower(), code)


def normalize_language_code(code: str) -> str:
    """正規化: lowercase + strip. 未知コードでも例外は投げない."""
    return code.strip().lower()
