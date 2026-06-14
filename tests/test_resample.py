"""to_mono / resample_linear のテスト (Windows loopback の音声前処理).

音声ハードウェア不要. 純粋な numpy 計算のみ検証する。
"""

from __future__ import annotations

import numpy as np

from realtime_interpreter.audio import resample_linear, to_mono


def test_to_mono_stereo_average() -> None:
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (2,)
    assert np.allclose(mono, [2.0, 3.0])


def test_to_mono_passthrough_1d() -> None:
    mono_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(to_mono(mono_in), mono_in)


def test_resample_same_rate_is_noop() -> None:
    x = np.linspace(-1, 1, 100, dtype=np.float32)
    out = resample_linear(x, 24000, 24000)
    assert np.allclose(out, x)


def test_resample_downsample_length() -> None:
    # 44100 -> 24000: 出力長は round(N * 24000/44100)
    x = np.zeros(44100, dtype=np.float32)
    out = resample_linear(x, 44100, 24000)
    assert out.shape[0] == round(44100 * 24000 / 44100)
    assert out.shape[0] == 24000


def test_resample_empty() -> None:
    out = resample_linear(np.zeros(0, dtype=np.float32), 44100, 24000)
    assert out.shape[0] == 0


def test_resample_preserves_endpoints() -> None:
    # 線形補間は端点を保持する
    x = np.linspace(0.0, 1.0, 441, dtype=np.float32)
    out = resample_linear(x, 44100, 24000)
    assert np.isclose(out[0], 0.0, atol=1e-4)
    assert np.isclose(out[-1], 1.0, atol=1e-4)


def test_resample_dtype_is_float32() -> None:
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    out = resample_linear(x, 48000, 24000)
    assert out.dtype == np.float32
    # 48000->24000 は約半分
    assert out.shape[0] == 500


def test_resample_sine_roughly_preserved() -> None:
    # 1kHz サイン波を 44100→24000 にダウンサンプルしても振幅レンジが保たれる
    t = np.arange(4410) / 44100.0
    sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    out = resample_linear(sine, 44100, 24000)
    assert 0.9 < float(np.max(np.abs(out))) <= 1.0
