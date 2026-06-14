"""共通 HTTP ヘルパ `post_json` と要約器 `Summarizer` Protocol のテスト.

ネットワークには触れず、urllib.request.urlopen をモックして検証する。
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from realtime_interpreter import _http
from realtime_interpreter._http import HttpError, post_json


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_post_json_sends_payload_and_headers(monkeypatch) -> None:
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["data"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return _FakeResponse('{"ok": true}')

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)

    result = post_json(
        "https://example.test/api",
        {"hello": "world"},
        headers={"Authorization": "Bearer t"},
        timeout=12.0,
    )

    assert result == {"ok": True}
    assert captured["data"] == {"hello": "world"}
    assert captured["timeout"] == 12.0
    # Content-Type は常に付与、追加ヘッダもマージされる (urllib は header 名を capitalize)
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["Authorization"] == "Bearer t"


def test_post_json_raises_httperror_with_body(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error": "rate"}'),
        )

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(HttpError) as ei:
        post_json("https://example.test/api", {"x": 1})

    assert ei.value.code == 429
    assert ei.value.body == '{"error": "rate"}'
    assert "429" in str(ei.value)


def test_summarizer_protocol_conformance() -> None:
    """具象要約器が `Summarizer` Protocol に構造的に適合することを確認."""
    from realtime_interpreter.summarizer import (
        GeminiRESTSummarizer,
        OpenAIChatSummarizer,
        Summarizer,
    )

    gemini = GeminiRESTSummarizer(model="m", api_key="k")
    openai = OpenAIChatSummarizer(api_key="k")
    assert isinstance(gemini, Summarizer)
    assert isinstance(openai, Summarizer)
