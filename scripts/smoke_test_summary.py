"""Summarizer のスモークテスト.

固定の英文テキストを Gemma 4 (テキスト専用モード) で要約させ、
- mlx-vlm の text-only generate が動作するか
- 日本語要約として妥当か
を確認する。

usage:
    uv run python scripts/smoke_test_summary.py
"""

from __future__ import annotations

from realtime_interpreter.summarizer import Summarizer
from realtime_interpreter.translator import GemmaAudioTranslator


SAMPLE_TEXT = (
    "Welcome to AWS re:Invent 2024. Today I want to talk about Apple's cloud "
    "infrastructure strategy. My team is responsible for building services like "
    "the App Store, Apple Music, Apple TV, and Podcasts. These services run on "
    "a combination of AWS and our own data centers. We believe that running on "
    "AWS gives us the flexibility to scale globally while focusing on what we "
    "do best, which is delighting our customers with great products."
)


def main() -> None:
    print("Loading model...")
    translator = GemmaAudioTranslator()
    translator.load()

    summarizer = Summarizer(translator)

    print(f"Input ({len(SAMPLE_TEXT)} chars):")
    print(SAMPLE_TEXT)
    print()

    print("Summarizing...")
    result = summarizer.summarize(SAMPLE_TEXT, duration_seconds=60)
    print(f"\n--- {result.latency_seconds:.2f}s ---")
    print(result.text)
    print("---")


if __name__ == "__main__":
    main()
