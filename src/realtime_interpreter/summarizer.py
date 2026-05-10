"""日本語要約モジュール.

Translator と同じ Gemma 4 モデルをテキスト専用モードで使い、
過去 N 秒の英文書き起こしから日本語要約を生成する。

モデルを共有するため追加の RAM は必要ない。要約推論中は翻訳推論が止まる
(MLX GPU stream の競合) ので、セッションあたり数秒のオーバーヘッドが発生する。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 要約モデルのサンプリング設定
SUMMARY_MAX_TOKENS = 256
SUMMARY_TEMPERATURE = 0.3
SUMMARY_TOP_P = 0.9


SUMMARY_PROMPT_TEMPLATE = (
    "あなたは英語の発話を日本語に要約するプロです。"
    "以下は過去 {duration} 秒間の英語発話の書き起こしです。\n"
    "\n"
    "## 出力ルール\n"
    "- 日本語で 2〜4 文に簡潔にまとめる\n"
    "- 出力は要約本文のみ. 前置き・引用符・「要約:」のようなプレフィックスは禁止\n"
    "- 技術用語 (CPU / AWS / GPU など) は英語のままで良い\n"
    "- 入力が短すぎる、または意味を成さない場合は何も出力しない\n"
    "\n"
    "## 英語書き起こし\n"
    "{text}\n"
)


@dataclass
class SummaryResult:
    """1 回の要約結果."""

    text: str
    latency_seconds: float


def build_summary_prompt(english_text: str, duration_seconds: int) -> str:
    """要約プロンプトを組み立てる."""
    return SUMMARY_PROMPT_TEMPLATE.format(
        text=english_text.strip(), duration=duration_seconds
    )


class Summarizer:
    """共有された Gemma 4 モデルで日本語要約を生成する.

    Translator が既にロードしたモデルインスタンスをそのまま使う。
    """

    def __init__(
        self,
        translator,  # GemmaAudioTranslator (循環 import を避けるため型ヒント省略)
        max_tokens: int = SUMMARY_MAX_TOKENS,
        temperature: float = SUMMARY_TEMPERATURE,
        top_p: float = SUMMARY_TOP_P,
    ) -> None:
        self._translator = translator
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def summarize(self, english_text: str, duration_seconds: int) -> SummaryResult:
        """過去 N 秒の英文を日本語要約に変換."""
        if not english_text.strip():
            return SummaryResult(text="", latency_seconds=0.0)

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt_text = build_summary_prompt(english_text, duration_seconds)
        # num_audios=0 でテキスト専用. audio kwarg を omit して generate を呼ぶ。
        prompt = apply_chat_template(
            self._translator.processor,
            self._translator.model.config,
            prompt_text,
            num_audios=0,
        )

        t0 = time.perf_counter()
        result = generate(
            model=self._translator.model,
            processor=self._translator.processor,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            verbose=False,
        )
        latency = time.perf_counter() - t0

        text = _extract_text(result).strip()
        logger.debug("summary %.2fs / %s", latency, text[:80])
        return SummaryResult(text=text, latency_seconds=latency)


def _extract_text(result: object) -> str:
    if isinstance(result, str):
        return result
    text_attr = getattr(result, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(result)
