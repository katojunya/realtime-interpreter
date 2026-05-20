"""OpenAIChatSummarizer のスモークテスト.

OpenAI Chat Completions API (gpt-5-mini など) で要約が生成できるか確認する。
OPENAI_API_KEY を環境変数で設定して実行する。

usage:
    OPENAI_API_KEY=sk-... uv run python scripts/smoke_test_openai_summary.py [model]
"""

from __future__ import annotations

import sys

from realtime_interpreter.summarizer import (
    DEFAULT_OPENAI_SUMMARY_MODEL,
    OpenAIChatSummarizer,
)


SAMPLE_TEXT = (
    "Welcome to AWS re:Invent 2024. Today I want to talk about Apple's cloud "
    "infrastructure strategy. My team is responsible for building services like "
    "the App Store, Apple Music, Apple TV, and Podcasts. These services run on "
    "a combination of AWS and our own data centers."
)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_OPENAI_SUMMARY_MODEL
    print(f"model: {model}")
    print(f"input ({len(SAMPLE_TEXT)} chars):")
    print(SAMPLE_TEXT)
    print()

    summarizer = OpenAIChatSummarizer(model=model)
    print("summarizing...")
    result = summarizer.summarize(SAMPLE_TEXT, duration_seconds=60)
    print(f"\n--- {result.latency_seconds:.2f}s ---")
    print(result.text or "(empty — see logs for errors)")
    print("---")


if __name__ == "__main__":
    main()
