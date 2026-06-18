"""`--system-certs` フラグと env honor のテスト.

inject_into_ssl() はプロセス全体の ssl を書き換えるためテストでは呼ばない
(フラグ解析と env truthy 判定のみ検証)。
"""

from __future__ import annotations

import sys

from realtime_interpreter import main as m


def test_env_truthy(monkeypatch):
    monkeypatch.delenv("REALTIME_INTERPRETER_SYSTEM_CERTS", raising=False)
    assert m._env_truthy("REALTIME_INTERPRETER_SYSTEM_CERTS") is False
    for v in ("1", "true", "TRUE", "Yes", "on"):
        monkeypatch.setenv("REALTIME_INTERPRETER_SYSTEM_CERTS", v)
        assert m._env_truthy("REALTIME_INTERPRETER_SYSTEM_CERTS") is True
    for v in ("0", "false", "no", ""):
        monkeypatch.setenv("REALTIME_INTERPRETER_SYSTEM_CERTS", v)
        assert m._env_truthy("REALTIME_INTERPRETER_SYSTEM_CERTS") is False


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("REALTIME_INTERPRETER_SYSTEM_CERTS", raising=False)
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter"])
    args = m._parse_args()
    assert args.system_certs is False


def test_flag_explicit_on(monkeypatch):
    monkeypatch.delenv("REALTIME_INTERPRETER_SYSTEM_CERTS", raising=False)
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter", "--system-certs"])
    args = m._parse_args()
    assert args.system_certs is True


def test_flag_default_from_env(monkeypatch):
    monkeypatch.setenv("REALTIME_INTERPRETER_SYSTEM_CERTS", "1")
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter"])
    args = m._parse_args()
    assert args.system_certs is True


def test_truststore_importable():
    # 依存として入っていること (--system-certs の前提)
    import truststore  # noqa: F401
