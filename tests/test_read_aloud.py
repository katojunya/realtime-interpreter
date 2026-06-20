"""読み上げ (--read-aloud) 機能のテスト: 再生バッファ / 出力デバイス解決 /
ループバック検証 / バックエンドの音声出力コールバック配線."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from realtime_interpreter.audio import find_output_device
from realtime_interpreter.playback import _BaseAudioPlayer, _to_float32_mono
from realtime_interpreter.backends.gemini_realtime import (
    GeminiRealtimeBackend,
    GEMINI_OUTPUT_SAMPLE_RATE,
    _pcm_rate_from_mime,
)
from realtime_interpreter.backends.openai_realtime import (
    OpenAIRealtimeBackend,
    OPENAI_SAMPLE_RATE,
)
from realtime_interpreter import main as m


# ---------------- _to_float32_mono ----------------


def test_to_float32_mono_int16_bytes() -> None:
    pcm = np.array([16384, -16384, 0], dtype="<i2").tobytes()
    out = _to_float32_mono(pcm)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.5, -0.5, 0.0], atol=1e-4)


def test_to_float32_mono_stereo_ndarray() -> None:
    stereo = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = _to_float32_mono(stereo)
    np.testing.assert_allclose(out, [0.3, 0.7], atol=1e-6)


# ---------------- _BaseAudioPlayer buffer ----------------


def test_player_pull_exact_and_underflow() -> None:
    player = _BaseAudioPlayer(device_rate=24000)
    pcm = np.array([16384, -16384], dtype="<i2").tobytes()
    player.enqueue(pcm, src_rate=24000)  # 同レート → リサンプルなし
    first = player._pull(2)
    np.testing.assert_allclose(first, [0.5, -0.5], atol=1e-4)
    # 枯渇後はゼロ埋め (例外を出さない)
    second = player._pull(4)
    np.testing.assert_array_equal(second, np.zeros(4, dtype=np.float32))


def test_player_pull_spans_chunks() -> None:
    player = _BaseAudioPlayer(device_rate=24000)
    player.enqueue(np.array([8192], dtype="<i2").tobytes(), 24000)
    player.enqueue(np.array([-8192], dtype="<i2").tobytes(), 24000)
    out = player._pull(2)
    np.testing.assert_allclose(out, [0.25, -0.25], atol=1e-4)


def test_player_resamples_to_device_rate() -> None:
    player = _BaseAudioPlayer(device_rate=48000)
    pcm = np.array([1000, 2000, 3000, 4000], dtype="<i2").tobytes()
    player.enqueue(pcm, src_rate=24000)  # 24k -> 48k で約 2 倍に伸びる
    out = player._pull(16)  # 余剰はゼロ埋め
    assert np.count_nonzero(out) >= 6  # おおむね 8 サンプル付近


# ---------------- find_output_device ----------------


class _SD:
    def __init__(self, devices: list[dict]) -> None:
        self._devices = devices

    def query_devices(self, index=None):
        if index is None:
            return self._devices
        return self._devices[index]


def test_find_output_device_picks_output_capable() -> None:
    sd = _SD(
        [
            {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 0},
            {"name": "External Headphones", "max_input_channels": 0, "max_output_channels": 2},
        ]
    )
    assert find_output_device("Headphones", sd) == 1


def test_find_output_device_not_found_raises() -> None:
    sd = _SD([{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}])
    with pytest.raises(RuntimeError):
        find_output_device("Headphones", sd)


# ---------------- _names_overlap / _check_playback_device ----------------


def test_names_overlap() -> None:
    assert m._names_overlap("BlackHole 2ch", "blackhole")
    assert m._names_overlap("Headphones", "External Headphones")
    assert not m._names_overlap("Headphones", "Speakers")
    assert not m._names_overlap("", "x")


def test_check_playback_device_requires_name() -> None:
    with pytest.raises(SystemExit):
        m._check_playback_device(None, "BlackHole 2ch")


class _DefaultDev:
    device = [0, 1]


class _MacSD:
    default = _DefaultDev()

    def query_devices(self, index=None):
        return {"name": "MacBook Pro Speakers"}


def test_check_playback_device_rejects_blackhole(monkeypatch) -> None:
    monkeypatch.setattr(m, "_is_windows", lambda: False)
    monkeypatch.setattr(m, "_load_sounddevice", lambda: _MacSD())
    with pytest.raises(SystemExit):
        m._check_playback_device("BlackHole 2ch", "BlackHole 2ch")


def test_check_playback_device_rejects_capture_overlap(monkeypatch) -> None:
    monkeypatch.setattr(m, "_is_windows", lambda: False)
    monkeypatch.setattr(m, "_load_sounddevice", lambda: _MacSD())
    with pytest.raises(SystemExit):
        m._check_playback_device("Aggregate Device", "Aggregate")


def test_check_playback_device_ok_separate_device(monkeypatch) -> None:
    monkeypatch.setattr(m, "_is_windows", lambda: False)
    monkeypatch.setattr(m, "_load_sounddevice", lambda: _MacSD())
    # 別デバイス・既定出力とも重ならない → 例外も警告もなし
    m._check_playback_device("External Headphones", "BlackHole 2ch")


def test_check_playback_device_warns_on_default_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(m, "_is_windows", lambda: False)
    monkeypatch.setattr(m, "_load_sounddevice", lambda: _MacSD())
    m._check_playback_device("MacBook Pro Speakers", "BlackHole 2ch")
    assert "loop back" in capsys.readouterr().err.lower()


# ---------------- mimeType レート解析 ----------------


def test_pcm_rate_from_mime() -> None:
    assert _pcm_rate_from_mime("audio/pcm;rate=24000") == 24000
    assert _pcm_rate_from_mime("audio/pcm") is None
    assert _pcm_rate_from_mime(None) is None


# ---------------- バックエンドの音声出力コールバック ----------------


def test_gemini_emits_output_audio() -> None:
    backend = GeminiRealtimeBackend(sd_module=None, device_name="dummy", api_key="dummy")
    received: list[tuple[bytes, int]] = []
    backend.set_audio_output_callback(lambda pcm, rate: received.append((pcm, rate)))

    raw = np.array([100, -100], dtype="<i2").tobytes()
    event = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"inlineData": {"data": base64.b64encode(raw).decode(), "mimeType": "audio/pcm;rate=24000"}}
                ]
            }
        }
    }
    backend._handle_event(event)
    assert received == [(raw, 24000)]


def test_gemini_output_audio_default_rate_when_no_mime() -> None:
    backend = GeminiRealtimeBackend(sd_module=None, device_name="dummy", api_key="dummy")
    received: list[tuple[bytes, int]] = []
    backend.set_audio_output_callback(lambda pcm, rate: received.append((pcm, rate)))
    raw = b"\x01\x00\x02\x00"
    backend._emit_output_audio({"data": base64.b64encode(raw).decode()})
    assert received == [(raw, GEMINI_OUTPUT_SAMPLE_RATE)]


def test_gemini_no_callback_is_noop() -> None:
    backend = GeminiRealtimeBackend(sd_module=None, device_name="dummy", api_key="dummy")
    # コールバック未登録でも例外を出さない
    backend._emit_output_audio({"data": base64.b64encode(b"\x00\x00").decode()})


def test_openai_realtime_emits_output_audio_when_enabled() -> None:
    backend = OpenAIRealtimeBackend(sd_module=None, device_name="dummy", api_key="dummy")
    received: list[tuple[bytes, int]] = []
    backend.set_audio_output_callback(lambda pcm, rate: received.append((pcm, rate)))
    raw = np.array([7, -7, 0], dtype="<i2").tobytes()
    backend._handle_event(
        {"type": "response.output_audio.delta", "delta": base64.b64encode(raw).decode()}
    )
    assert received == [(raw, OPENAI_SAMPLE_RATE)]


def test_openai_realtime_skips_output_audio_when_disabled() -> None:
    backend = OpenAIRealtimeBackend(sd_module=None, device_name="dummy", api_key="dummy")
    # コールバック未登録 → 何も起きず例外も出ない
    backend._handle_event(
        {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x00\x00").decode()}
    )
