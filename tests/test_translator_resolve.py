"""resolve_model_id() のテスト. mlx-vlm の実体には触れない."""

from __future__ import annotations

import pytest

from realtime_interpreter.translator import (
    DEFAULT_ALIAS,
    MODEL_PRESETS,
    resolve_model_id,
)


def test_alias_resolves_to_full_id() -> None:
    assert resolve_model_id("e4b") == MODEL_PRESETS["e4b"][0]
    assert resolve_model_id("e2b") == MODEL_PRESETS["e2b"][0]


def test_full_huggingface_id_passes_through() -> None:
    full = "mlx-community/some-future-model-2bit"
    assert resolve_model_id(full) == full


def test_none_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REALTIME_INTERPRETER_MODEL", raising=False)
    assert resolve_model_id(None) == MODEL_PRESETS[DEFAULT_ALIAS][0]


def test_env_var_used_when_arg_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REALTIME_INTERPRETER_MODEL", "e2b")
    assert resolve_model_id(None) == MODEL_PRESETS["e2b"][0]


def test_arg_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REALTIME_INTERPRETER_MODEL", "e2b")
    assert resolve_model_id("e4b") == MODEL_PRESETS["e4b"][0]


def test_env_var_can_be_full_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REALTIME_INTERPRETER_MODEL", "owner/model-id")
    assert resolve_model_id(None) == "owner/model-id"


def test_unknown_alias_raises() -> None:
    with pytest.raises(ValueError, match="unknown model alias"):
        resolve_model_id("not-a-real-alias")
