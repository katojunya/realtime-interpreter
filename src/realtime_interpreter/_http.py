"""urllib ベースの JSON POST ヘルパ.

要約器 (Gemini REST / OpenAI 互換 Chat) が個別に持っていた
「Request 組み立て → urlopen → decode → HTTPError ハンドリング」の定型処理を
1 箇所に集約する。レスポンスは JSON として parse して返す。

呼び出し側がエラー文言やログ粒度を制御できるよう、HTTP エラーは本文付きの
`HttpError` (RuntimeError サブクラス) として送出する。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class HttpError(RuntimeError):
    """HTTP 非 2xx 応答. code / reason / body を保持する."""

    def __init__(self, code: int, reason: str, body: str) -> None:
        super().__init__(f"HTTP {code} {reason}: {body}")
        self.code = code
        self.reason = reason
        self.body = body


def post_json(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict:
    """`payload` を JSON として POST し、応答 JSON を dict で返す.

    `Content-Type: application/json` は常に付与する。追加ヘッダ
    (例: ``Authorization``) は `headers` で渡す。HTTP エラー時は
    本文を載せた `HttpError` を送出する。
    """
    data = json.dumps(payload).encode("utf-8")
    merged = {"Content-Type": "application/json"}
    if headers:
        merged.update(headers)
    request = urllib.request.Request(url, data=data, headers=merged)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise HttpError(e.code, str(e.reason), err_body) from e
    return json.loads(body)
