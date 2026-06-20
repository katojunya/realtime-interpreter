"""openai-realtime backend の言語コード正規化テスト.

gpt-realtime-translate は中国語を 1 種類しか出力できないため、zh 系のサブタグは
audio.output.language で ISO-639-1 の "zh" に丸める (README の既知の制限)。
"""

from __future__ import annotations

import json

from realtime_interpreter.backends.openai_realtime import (
    OpenAIRealtimeBackend,
    _openai_output_language,
)


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, payload: str) -> None:
        self.sent.append(payload)


def test_openai_output_language_collapses_chinese() -> None:
    assert _openai_output_language("zh") == "zh"
    assert _openai_output_language("zh-Hans") == "zh"
    assert _openai_output_language("zh-Hant") == "zh"
    assert _openai_output_language("zh-TW") == "zh"
    # 非中国語はそのまま
    assert _openai_output_language("ja") == "ja"
    assert _openai_output_language("en") == "en"
    assert _openai_output_language("pt-BR") == "pt-BR"


def test_openai_setup_collapses_chinese_output_language() -> None:
    backend = OpenAIRealtimeBackend(
        sd_module=None,
        device_name="dummy",
        api_key="dummy_key",
        loopback=False,
        target_lang="zh-Hant",
    )
    backend._ws = _FakeWS()
    backend._send_session_config()
    session = json.loads(backend._ws.sent[-1])["session"]
    assert session["audio"]["output"]["language"] == "zh"
