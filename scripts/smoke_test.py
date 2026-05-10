"""Static wav/flac ファイルを Gemma 4 で翻訳する単発スモークテスト.

usage:
    uv run python scripts/smoke_test.py path/to/english.wav [model_alias_or_id]

例:
    uv run python scripts/smoke_test.py samples/test.flac          # 既定 (E4B)
    uv run python scripts/smoke_test.py samples/test.flac e2b      # E2B 軽量版
    uv run python scripts/smoke_test.py samples/test.flac mlx-community/...  # 任意ID

確認項目:
    1. mlx-vlm + Gemma 4 が import できる
    2. モデルがロードできる (初回は数GBのDL発生)
    3. 音声ファイルから日本語訳が返る
    4. レイテンシが許容範囲か
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from realtime_interpreter.translator import GemmaAudioTranslator


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: python scripts/smoke_test.py <audio_file> [model_alias_or_id]",
            file=sys.stderr,
        )
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    model_arg = sys.argv[2] if len(sys.argv) >= 3 else None
    if not wav_path.exists():
        print(f"file not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        try:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        except ImportError:
            print(
                f"sample rate is {sr} Hz; install librosa or provide a 16kHz wav",
                file=sys.stderr,
            )
            sys.exit(1)

    duration = len(audio) / 16000
    print(f"loaded: {wav_path.name} ({duration:.1f}s)")

    translator = GemmaAudioTranslator(model=model_arg)
    print(f"loading model: {translator.model_id}")
    t0 = time.perf_counter()
    translator.load()
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    print("translating...")
    result = translator.translate(audio.astype(np.float32))
    print(f"\n--- {result.latency_seconds:.2f}s ---")
    print(f"EN: {result.english}")
    print(f"JA: {result.japanese}")
    print("---")


if __name__ == "__main__":
    main()
