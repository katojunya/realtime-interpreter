"""--device の番号(インデックス)指定のテスト.

同名デバイスが複数あるとき名前の部分一致では区別できないため、--list-devices の
番号で一意指定できる。解決ロジックは audio.py に集約し、realtime 系 backend も委譲する。
"""

from __future__ import annotations

import sys
import types

import pytest

from realtime_interpreter import audio
from realtime_interpreter.audio import (
    find_device,
    _find_loopback_device,
    _input_channels,
    _looks_like_index,
)
from realtime_interpreter.backends import gemini_realtime, openai_realtime


# ---------------- _looks_like_index ----------------


def test_looks_like_index() -> None:
    assert _looks_like_index("12") is True
    assert _looks_like_index(" 3 ") is True
    assert _looks_like_index("BlackHole 2ch") is False
    assert _looks_like_index("4K") is False
    assert _looks_like_index("") is False
    assert _looks_like_index(None) is False


# ---------------- find_device (input, sounddevice) ----------------


class _SD:
    def __init__(self, devices: list[dict]) -> None:
        self._devices = devices

    def query_devices(self, index=None):
        return self._devices if index is None else self._devices[index]


def _devs() -> list[dict]:
    # index 1,2 は同名 (EVF3285) で、名前では区別できない。
    return [
        {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "EVF3285", "max_input_channels": 2, "max_output_channels": 2},
        {"name": "EVF3285", "max_input_channels": 2, "max_output_channels": 2},
        {"name": "Speaker only", "max_input_channels": 0, "max_output_channels": 2},
    ]


def test_find_device_by_index_disambiguates_identical_names() -> None:
    sd = _SD(_devs())
    # 名前では先頭 (index 1) しか選べないが、番号なら index 2 を選べる。
    assert find_device("EVF3285", sd) == 1
    assert find_device("2", sd) == 2
    assert find_device("1", sd) == 1


def test_find_device_by_name_still_works() -> None:
    sd = _SD(_devs())
    assert find_device("BlackHole", sd) == 0


def test_find_device_index_out_of_range_raises() -> None:
    sd = _SD(_devs())
    with pytest.raises(RuntimeError):
        find_device("99", sd)


def test_find_device_index_not_input_capable_raises() -> None:
    sd = _SD(_devs())
    # index 3 は出力専用 (max_input_channels=0) なので入力には使えない。
    with pytest.raises(RuntimeError):
        find_device("3", sd)


# ---------------- _input_channels (モノラル機で開けるか) ----------------


def test_input_channels_matches_device() -> None:
    # ステレオ機 (in=2) → 2
    assert _input_channels(_SD(_devs()), 0) == 2
    # モノラル機 (in=1) → 1 (channels=2 で開くと PortAudio が -9998 で失敗するため)
    mono = _SD([{"name": "Mono Mic", "max_input_channels": 1, "max_output_channels": 0}])
    assert _input_channels(mono, 0) == 1
    # 多チャンネル機 → 2 にクランプ
    multi = _SD([{"name": "Aggregate", "max_input_channels": 8, "max_output_channels": 0}])
    assert _input_channels(multi, 0) == 2


# ---------------- _find_loopback_device (Windows WASAPI) ----------------


class _FakePA:
    def __init__(self, loopbacks: list[dict], default_out_index: int) -> None:
        self._loopbacks = loopbacks
        self._default_out_index = default_out_index
        self._by_index = {d["index"]: d for d in loopbacks}

    def get_host_api_info_by_type(self, _t):
        return {"defaultOutputDevice": self._default_out_index}

    def get_loopback_device_info_generator(self):
        return iter(self._loopbacks)

    def get_device_info_by_index(self, idx):
        return self._by_index[idx]


@pytest.fixture
def _fake_pyaudiowpatch(monkeypatch):
    # macOS には pyaudiowpatch が無いため、関数内 import 用にダミーを差し込む。
    fake = types.ModuleType("pyaudiowpatch")
    fake.paWASAPI = 13
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake)
    return fake


def _loopbacks() -> list[dict]:
    # index 11,12 は同名 (EVF3285) の loopback。
    return [
        {"index": 10, "name": "リモート オーディオ [Loopback]"},
        {"index": 11, "name": "EVF3285 [Loopback]"},
        {"index": 12, "name": "EVF3285 [Loopback]"},
    ]


def test_loopback_by_index_disambiguates(_fake_pyaudiowpatch) -> None:
    pa = _FakePA(_loopbacks(), default_out_index=10)
    # 名前では先頭 (index 11) のみ。番号なら 12 を一意指定できる。
    assert _find_loopback_device(pa, "EVF3285")["index"] == 11
    assert _find_loopback_device(pa, "12")["index"] == 12
    assert _find_loopback_device(pa, "11")["index"] == 11


def test_loopback_index_not_found_raises(_fake_pyaudiowpatch) -> None:
    pa = _FakePA(_loopbacks(), default_out_index=10)
    with pytest.raises(RuntimeError):
        _find_loopback_device(pa, "99")


def test_loopback_default_when_omitted(_fake_pyaudiowpatch) -> None:
    pa = _FakePA(_loopbacks(), default_out_index=10)
    # 既定出力 (index 10) の loopback を返す。
    assert _find_loopback_device(pa, None)["index"] == 10


# ---------------- 集約 (backend が audio.py に委譲しているか) ----------------


def test_backends_delegate_to_audio_resolvers() -> None:
    # 重複定義を排し、index 対応が realtime 系にも効くことを担保する。
    assert gemini_realtime._find_input_device is find_device
    assert gemini_realtime._find_loopback_device is _find_loopback_device
    assert openai_realtime._find_input_device is find_device
    assert openai_realtime._find_loopback_device is _find_loopback_device
