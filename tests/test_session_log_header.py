"""セッションログのヘッダ(起動設定の記録)のテスト."""

from __future__ import annotations

import sys

from realtime_interpreter import main as m
from realtime_interpreter.session_logger import SessionLogger


def _args(argv: list[str], monkeypatch) -> object:
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter", *argv])
    args = m._parse_args()
    args.device = m._resolve_device_arg(args.device)
    # デバイス解決(実機依存)を避け、capture ラベルを固定。
    monkeypatch.setattr(m, "_capture_label", lambda _d: "[4] BlackHole 2ch")
    return args


def test_collect_settings_openai_realtime(monkeypatch) -> None:
    args = _args([], monkeypatch)
    pairs = dict(m._collect_settings(args, object()))
    assert pairs["backend"] == "openai-realtime (gpt-realtime-translate)"
    assert pairs["languages"] == "English (en) -> Japanese (ja)"
    assert pairs["capture"] == "[4] BlackHole 2ch"
    assert pairs["segmentation"].startswith("debounce=")
    assert pairs["summary"].startswith("every 60s (")
    assert "(24h)" in pairs["max_session"]
    assert pairs["tls"] == "bundled CAs"
    assert pairs["version"] and pairs["platform"]
    assert "endpoint" not in pairs  # realtime はエンドポイント無し


def test_collect_settings_openai_chat_has_endpoint(monkeypatch) -> None:
    args = _args(["--backend", "openai-chat", "--openai-chat-model", "gemma4:e2b"], monkeypatch)
    pairs = dict(m._collect_settings(args, object()))
    assert pairs["backend"] == "openai-chat (gemma4:e2b)"
    assert pairs["endpoint"] == "http://localhost:11434/v1"
    assert pairs["segmentation"].startswith("end_silence=")


def test_collect_settings_omits_secrets(monkeypatch) -> None:
    # API キーを渡しても設定に漏れないこと。
    args = _args(["--openai-rt-api-key", "sk-SECRET-123"], monkeypatch)
    pairs = m._collect_settings(args, object())
    labels = {k for k, _ in pairs}
    assert not any("key" in k.lower() for k in labels)
    assert all("sk-SECRET-123" not in v for _, v in pairs)


def test_collect_settings_summary_off_and_unlimited(monkeypatch) -> None:
    args = _args(["--summary-interval-seconds", "0", "--max-session-seconds", "0"], monkeypatch)
    pairs = dict(m._collect_settings(args, object()))
    assert pairs["summary"] == "off"
    assert pairs["max_session"] == "unlimited"


def test_session_logger_writes_settings_header(tmp_path) -> None:
    lg = SessionLogger(
        log_dir=tmp_path,
        timestamp="t",
        settings=[("backend", "openai-chat (gemma4:e4b)"), ("tls", "bundled CAs")],
    )
    lg.log_segment("00:05", "Hello", "やあ")
    lg.close()
    text = lg.path.read_text(encoding="utf-8")
    assert "# realtime-interpreter session" in text
    assert "# started:" in text
    assert "# backend:" in text and "openai-chat (gemma4:e4b)" in text
    assert "# tls:" in text
    assert "[00:05] Hello" in text
    assert "session ended (duration" in text


def test_session_logger_no_settings_still_works(tmp_path) -> None:
    lg = SessionLogger(log_dir=tmp_path, timestamp="t2")
    lg.close()
    text = lg.path.read_text(encoding="utf-8")
    assert "# realtime-interpreter session" in text
    assert "# started:" in text
