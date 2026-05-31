"""Windows WASAPI ループバック可否の診断スクリプト (PyAudioWPatch 使用).

目的:
  本体を PyAudioWPatch 対応に書き換える前に、お使いの環境 (VDI / ベアメタル) で
  「システム音声をループバック録音できるか」を切り分ける。

確認すること:
  1. WASAPI ホスト API が使えるか
  2. 既定の出力デバイス (スピーカー / リモートオーディオ等) の "loopback 版" が存在するか
  3. 全 loopback デバイスの一覧
  4. 既定出力の loopback デバイスで数秒録音し、音声レベル (peak/RMS dBFS) を測定
     → 音が取れていれば本体の loopback 実装が VDI/ベアメタルで動く見込み

使い方 (Windows, この repo のルートで):
  uv run --with pyaudiowpatch python scripts/diagnose_loopback_win.py

  ※ --with で一時的に PyAudioWPatch をインストールして実行する (本体依存は変更しない)。
  ※ 録音テスト中は YouTube やブラウザ等で英語音声を再生しておくこと。

結果の見方:
  - "loopback device found" が出て、peak が -60 dBFS より大きい (例 -30dBFS) なら成功。
  - loopback device が見つからない / peak=-inf のままなら、その環境では
    WASAPI loopback でシステム音声を取得できない (VDI の制約等)。
"""

from __future__ import annotations

import sys
import time

RECORD_SECONDS = 6.0
CHUNK = 1024


def main() -> int:
    if sys.platform != "win32":
        print("This diagnostic is for Windows only.", file=sys.stderr)
        return 2

    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        print(
            "PyAudioWPatch is not installed.\n"
            "Run with uv's --with flag:\n"
            "  uv run --with pyaudiowpatch python scripts/diagnose_loopback_win.py",
            file=sys.stderr,
        )
        return 2

    import numpy as np

    p = pyaudio.PyAudio()
    try:
        # --- 1. WASAPI ホスト API ---
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("ERROR: WASAPI host API not available on this system.")
            return 1
        print(f"WASAPI host API: index={wasapi_info['index']}, "
              f"name={wasapi_info['name']}")
        print()

        # --- 2. 既定の出力デバイス ---
        default_out_index = wasapi_info["defaultOutputDevice"]
        default_speakers = p.get_device_info_by_index(default_out_index)
        print(f"Default output device: [{default_out_index}] {default_speakers['name']}")
        print()

        # --- 3. loopback デバイス一覧 ---
        print("Loopback devices found:")
        loopbacks = list(p.get_loopback_device_info_generator())
        if not loopbacks:
            print("  (none) — this environment exposes NO WASAPI loopback devices.")
            print("  → System-audio loopback capture is NOT possible here.")
            return 1
        for lb in loopbacks:
            print(f"  [{lb['index']}] {lb['name']} "
                  f"(in={lb['maxInputChannels']}, rate={int(lb['defaultSampleRate'])})")
        print()

        # --- 4. 既定出力に対応する loopback を選ぶ ---
        target = None
        for lb in loopbacks:
            if default_speakers["name"] in lb["name"]:
                target = lb
                break
        if target is None:
            target = loopbacks[0]
            print(f"No loopback matched the default output by name; "
                  f"falling back to first loopback: [{target['index']}] {target['name']}")
        else:
            print(f"Matched loopback for default output: "
                  f"[{target['index']}] {target['name']}")
        print()

        # --- 5. 録音テスト ---
        channels = target["maxInputChannels"]
        rate = int(target["defaultSampleRate"])
        print(f"Recording {RECORD_SECONDS:.0f}s from loopback "
              f"(channels={channels}, rate={rate})...")
        print("  >>> PLAY SOME AUDIO NOW (YouTube, browser, etc.) <<<")

        peak = 0.0
        sumsq = 0.0
        nsamp = 0

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            frames_per_buffer=CHUNK,
            input=True,
            input_device_index=target["index"],
        )
        t_end = time.monotonic() + RECORD_SECONDS
        try:
            while time.monotonic() < t_end:
                data = stream.read(CHUNK, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if arr.size:
                    peak = max(peak, float(np.max(np.abs(arr))))
                    sumsq += float(np.sum(arr * arr))
                    nsamp += arr.size
        finally:
            stream.stop_stream()
            stream.close()

        def dbfs(x: float) -> str:
            import math
            return f"{20 * math.log10(x):+.1f} dBFS" if x > 1e-10 else "-inf dBFS (silence)"

        rms = (sumsq / nsamp) ** 0.5 if nsamp else 0.0
        print()
        print(f"  peak = {dbfs(peak)}")
        print(f"  rms  = {dbfs(rms)}")
        print()
        if peak > 1e-4:
            print("RESULT: ✅ Loopback capture WORKS. System audio was recorded.")
            print("        → The OpenAI backend can be made to work here via PyAudioWPatch.")
            return 0
        else:
            print("RESULT: ⚠ Loopback device opened but captured SILENCE.")
            print("        Either no audio was playing, or this endpoint does not")
            print("        actually carry the system mix (common with some VDI setups).")
            print("        Re-run while audio is clearly playing to confirm.")
            return 1
    finally:
        p.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
