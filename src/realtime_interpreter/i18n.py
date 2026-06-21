"""言語コード → 言語名のマッピング.

基本は ISO 639-1 (2 文字コード)。ただし中国語の字体区別など、BCP-47 の
スクリプト/地域サブタグ (例: zh-Hans / zh-Hant / pt-BR) も扱える。プロンプト内で
人間可読な言語名 (例: "Japanese") に展開するため、また CLI 参照用に使う。

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
    # 字体 (スクリプトサブタグ). Gemini live-translate はこの形式で簡体字/繁体字を区別する。
    "zh-Hans": "Chinese (Simplified)",
    "zh-Hant": "Chinese (Traditional)",
    # 地域コード. Gemini では非対応だが利便のため別名として残し、Gemini バックエンド側で
    # zh-Hans / zh-Hant へマップする (openai-chat / mlx ではプロンプトの言語名として使う)。
    "zh-CN": "Chinese (Simplified, mainland China)",
    "zh-TW": "Chinese (Traditional, Taiwan)",
    "zh-HK": "Chinese (Traditional, Hong Kong)",
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
    """言語コードを英語の言語名に変換. 未知なら正規化済みコードをそのまま返す."""
    canon = normalize_language_code(code)
    return LANGUAGES.get(canon, canon)


def normalize_language_code(code: str) -> str:
    """BCP-47 の慣用ケーシングへ正規化する (未知コードでも例外は投げない).

    - 言語サブタグ: 小文字 (zh, en, ja)
    - スクリプトサブタグ (4文字): 先頭大文字 (Hans, Hant, Latn)
    - 地域サブタグ (2文字): 大文字 (CN, TW, US)

    これにより Gemini live-translate が要求する `zh-Hans` / `zh-Hant` の字体区別を
    保持する (従来の一律小文字化では `zh-hans` となり targetLanguageCode に効かなかった)。
    """
    parts = code.strip().split("-")
    if not parts[0]:
        return ""
    out = [parts[0].lower()]
    for sub in parts[1:]:
        if len(sub) == 4 and sub.isalpha():
            out.append(sub.capitalize())  # script subtag: Hans / Hant / Latn
        elif len(sub) == 2 and sub.isalpha():
            out.append(sub.upper())  # region subtag: CN / TW / US
        else:
            out.append(sub.lower())
    return "-".join(out)
