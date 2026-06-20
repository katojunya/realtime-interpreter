"""プラットフォーム別の対応バックエンド判定のテスト.

mlx は Apple Silicon の macOS 専用。Intel macOS / Windows は 3 種
(openai-realtime / openai-chat / gemini-realtime)、Linux は 2 種。
"""

from __future__ import annotations

from realtime_interpreter import main as m


# ---------------- _resolve_backend_choices (純関数) ----------------


def test_choices_apple_silicon_includes_mlx() -> None:
    choices = m._resolve_backend_choices(mlx_available=True, openai_chat_available=True)
    assert choices == ("openai-realtime", "openai-chat", "gemini-realtime", "mlx")


def test_choices_intel_mac_or_windows_three_backends() -> None:
    # Intel macOS / Windows: mlx 非対応、openai-chat あり。
    choices = m._resolve_backend_choices(mlx_available=False, openai_chat_available=True)
    assert choices == ("openai-realtime", "openai-chat", "gemini-realtime")
    assert "mlx" not in choices


def test_choices_linux_two_backends() -> None:
    choices = m._resolve_backend_choices(mlx_available=False, openai_chat_available=False)
    assert choices == ("openai-realtime", "gemini-realtime")
    assert "mlx" not in choices
    assert "openai-chat" not in choices


# ---------------- _is_apple_silicon ----------------


def test_is_apple_silicon_true_on_arm64_mac(monkeypatch) -> None:
    monkeypatch.setattr(m, "_is_macos", lambda: True)
    monkeypatch.setattr(m.platform, "machine", lambda: "arm64")
    assert m._is_apple_silicon() is True


def test_is_apple_silicon_false_on_intel_mac(monkeypatch) -> None:
    # Intel macОС (または Rosetta 下の x86_64 Python) → mlx 不可。
    monkeypatch.setattr(m, "_is_macos", lambda: True)
    monkeypatch.setattr(m.platform, "machine", lambda: "x86_64")
    assert m._is_apple_silicon() is False


def test_is_apple_silicon_false_on_non_mac(monkeypatch) -> None:
    monkeypatch.setattr(m, "_is_macos", lambda: False)
    monkeypatch.setattr(m.platform, "machine", lambda: "arm64")
    assert m._is_apple_silicon() is False


def test_intel_mac_resolves_to_three_backends(monkeypatch) -> None:
    """Intel macOS をシミュレートし、choices が 3 種 (mlx 無し) になることを確認."""
    monkeypatch.setattr(m, "_is_macos", lambda: True)
    monkeypatch.setattr(m, "_is_windows", lambda: False)
    monkeypatch.setattr(m.platform, "machine", lambda: "x86_64")

    mlx_available = m._is_apple_silicon()
    openai_chat_available = m._is_macos() or m._is_windows()
    choices = m._resolve_backend_choices(mlx_available, openai_chat_available)

    assert mlx_available is False
    assert choices == ("openai-realtime", "openai-chat", "gemini-realtime")
