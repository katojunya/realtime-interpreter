"""Windows の番号のみ `--device`(出力/入力を自動判定)と 2 セクション一覧のテスト.

PyAudioWPatch は Windows 専用依存のため、関数内 import 用にダミーを差し込む。
"""

from __future__ import annotations

import sys
import types

import pytest

from realtime_interpreter import main as m
from realtime_interpreter.audio import (
    DEVICE_NAME,
    _resolve_windows_capture_device,
    _wasapi_input_devices,
)


WASAPI = 99


def _devices() -> list[dict]:
    return [
        {"index": 0, "name": "Microphone (Realtek)", "maxInputChannels": 2, "hostApi": WASAPI, "defaultSampleRate": 48000},
        {"index": 1, "name": "Line In", "maxInputChannels": 2, "hostApi": WASAPI, "defaultSampleRate": 44100},
        {"index": 2, "name": "Speakers [Loopback]", "maxInputChannels": 2, "hostApi": WASAPI, "defaultSampleRate": 48000},  # loopback
        {"index": 3, "name": "USB Mic (MME)", "maxInputChannels": 1, "hostApi": 1, "defaultSampleRate": 44100},  # 非WASAPI
        {"index": 4, "name": "Speakers (output)", "maxInputChannels": 0, "hostApi": WASAPI, "defaultSampleRate": 48000},  # 入力なし
    ]


class _FakePA:
    def __init__(self, devices, default_in=1, default_out=2):
        self._devices = devices
        self._by_index = {d["index"]: d for d in devices}
        self._default_in = default_in
        self._default_out = default_out

    def get_host_api_info_by_type(self, _t):
        return {"index": WASAPI, "defaultInputDevice": self._default_in, "defaultOutputDevice": self._default_out}

    def get_loopback_device_info_generator(self):
        return iter([self._by_index[2]])  # index 2 が loopback

    def get_device_count(self):
        return len(self._devices)

    def get_device_info_by_index(self, idx):
        return self._by_index[idx]

    def terminate(self):
        pass


@pytest.fixture(autouse=True)
def _fake_pyaudiowpatch(monkeypatch):
    fake = types.ModuleType("pyaudiowpatch")
    fake.paWASAPI = 13
    fake.PyAudio = lambda: _FakePA(_devices())
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake)
    return fake


def _pa() -> _FakePA:
    return _FakePA(_devices())


# ---------------- _wasapi_input_devices ----------------


def test_wasapi_input_devices_excludes_loopback_and_non_wasapi() -> None:
    assert [m["index"] for m in _wasapi_input_devices(_pa())] == [0, 1]


# ---------------- _resolve_windows_capture_device (番号で自動判定) ----------------


def test_resolve_default_is_loopback() -> None:
    dev, is_mic = _resolve_windows_capture_device(_pa(), DEVICE_NAME)
    assert is_mic is False and dev["index"] == 2  # 既定出力の loopback


def test_resolve_loopback_index() -> None:
    dev, is_mic = _resolve_windows_capture_device(_pa(), "2")
    assert is_mic is False and dev["index"] == 2


def test_resolve_input_index_is_mic() -> None:
    dev, is_mic = _resolve_windows_capture_device(_pa(), "0")
    assert is_mic is True and dev["index"] == 0


def test_resolve_unknown_index_raises() -> None:
    with pytest.raises(RuntimeError):
        _resolve_windows_capture_device(_pa(), "3")   # MME (非WASAPI)
    with pytest.raises(RuntimeError):
        _resolve_windows_capture_device(_pa(), "99")  # 存在しない


# ---------------- _windows_capture_label (起動表示: 種別 + [index] + 名前) ----------------


def test_windows_capture_label_mic() -> None:
    kind, idx, name = m._windows_capture_label("0")
    assert kind == "microphone" and idx == 0 and "Realtek" in name


def test_windows_capture_label_loopback_and_default() -> None:
    kind, idx, name = m._windows_capture_label("2")
    assert kind == "WASAPI loopback" and idx == 2 and "[Loopback]" not in name
    # 既定 (DEVICE_NAME) → 既定出力の loopback
    kind, idx, _ = m._windows_capture_label(DEVICE_NAME)
    assert kind == "WASAPI loopback" and idx == 2


def test_windows_capture_label_unknown_returns_none() -> None:
    assert m._windows_capture_label("99") is None


# ---------------- _format_windows_device_list ----------------


def _fmt_loopbacks() -> list[dict]:
    return [
        {"index": 10, "name": "リモート オーディオ [Loopback]", "maxInputChannels": 2, "defaultSampleRate": 44100},
        {"index": 11, "name": "EVF3285 [Loopback]", "maxInputChannels": 2, "defaultSampleRate": 44100},
    ]


def _fmt_mics() -> list[dict]:
    return [
        {"index": 1, "name": "マイク (Realtek)", "maxInputChannels": 2, "defaultSampleRate": 48000},
        {"index": 6, "name": "USB マイク", "maxInputChannels": 1, "defaultSampleRate": 44100},
    ]


def test_format_windows_device_list() -> None:
    text = "\n".join(
        m._format_windows_device_list(_fmt_loopbacks(), _fmt_mics(), "リモート オーディオ", 1)
    )
    assert "Output devices —" in text and "Input devices —" in text
    assert "[Loopback]" not in text          # 接尾辞除去
    assert "リモート オーディオ" in text
    assert "<- default (used when --device omitted)" in text
    assert "<- default mic" in text
    assert "in=2" in text and "rate=48000" in text
    assert "Tips:" in text


def test_format_windows_device_list_empty() -> None:
    text = "\n".join(m._format_windows_device_list([], [], "", -1))
    assert "(no WASAPI loopback devices found)" in text
    assert "(no WASAPI microphone input devices found)" in text
    assert m._strip_loopback_suffix("X [Loopback]") == "X"


# ---------------- _resolve_device_arg / --device CLI ----------------


def test_resolve_device_arg_default_is_sentinel() -> None:
    assert m._resolve_device_arg(None) == DEVICE_NAME


def test_resolve_device_arg_accepts_number() -> None:
    assert m._resolve_device_arg("5") == "5"


def test_resolve_device_arg_rejects_name() -> None:
    with pytest.raises(SystemExit):
        m._resolve_device_arg("BlackHole 2ch")


def test_cli_device_default_none_and_no_mic(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter"])
    args = m._parse_args()
    assert args.device is None          # 省略時は None (後で _resolve_device_arg が既定に)
    assert not hasattr(args, "mic")     # --mic は廃止
