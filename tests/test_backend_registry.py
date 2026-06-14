"""バックエンドレジストリ (_BACKENDS) のテスト (候補④).

build/describe の実体は外部依存 (websocket / mlx / network) を伴うため呼ばない。
レジストリの整合性 (キー網羅・可用性導出・describe の整形) のみを検証する。
"""

from __future__ import annotations

import argparse

from realtime_interpreter import main as m


def test_registry_has_all_backends() -> None:
    assert set(m._BACKENDS) == {
        "openai-realtime",
        "openai-chat",
        "mlx",
        "gemini-realtime",
    }
    # 各スペックが必須コールバックを備える
    for spec in m._BACKENDS.values():
        assert callable(spec.available)
        assert callable(spec.build)
        assert callable(spec.describe)


def test_registry_order_drives_choices() -> None:
    # --backend の選択肢はレジストリ挿入順に導出される
    names = list(m._BACKENDS)
    assert names == ["openai-realtime", "openai-chat", "mlx", "gemini-realtime"]


def test_always_available_backends() -> None:
    # openai-realtime / gemini-realtime は全プラットフォームで利用可能
    assert m._BACKENDS["openai-realtime"].available() is True
    assert m._BACKENDS["gemini-realtime"].available() is True


def _ns(**kw) -> argparse.Namespace:
    base = dict(source_lang="en", target_lang="ja", summary_interval_seconds=60)
    base.update(kw)
    return argparse.Namespace(**base)


def test_describe_openai_realtime_format() -> None:
    args = _ns(
        backend="openai-realtime",
        openai_model="gpt-realtime-translate",
        openai_debounce_ms=800,
        openai_max_segment_seconds=8.0,
        openai_summary_model="gpt-5-mini",
    )
    out = m._BACKENDS["openai-realtime"].describe(
        args, "English (en) → Japanese (ja)", "every 60s"
    )
    assert out == (
        "Backend: openai-realtime (gpt-realtime-translate) | "
        "Lang: English (en) → Japanese (ja) | "
        "debounce=800ms, max_segment=8.0s | "
        "summary=every 60s (gpt-5-mini)"
    )


def test_describe_max_segment_and_summary_off() -> None:
    args = _ns(
        backend="gemini-realtime",
        summary_interval_seconds=0,
        gemini_model="models/x",
        gemini_debounce_ms=800,
        gemini_max_segment_seconds=0.0,
        gemini_summary_model="g",
    )
    out = m._BACKENDS["gemini-realtime"].describe(
        args, "English (en) → Japanese (ja)", "off"
    )
    assert "max_segment=off" in out
    assert out.endswith("summary=off")
