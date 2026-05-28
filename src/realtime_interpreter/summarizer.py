"""要約モジュール.

2 つのバックエンドを提供する:

- `Summarizer` (ローカル): 翻訳と同じ Gemma 4 モデルをテキスト専用モードで再利用.
  MLX バックエンド利用時に翻訳モデルを共有する想定. 追加 RAM 不要.
- `OpenAIChatSummarizer` (クラウド): OpenAI Chat Completions API で gpt-5-mini 等を呼ぶ.
  OpenAI バックエンドと同じ API キーで使える. Realtime API とは別経路.

どちらも同じ `summarize(source_text, duration_seconds) -> SummaryResult` インターフェース。
プロンプトは英語で、source/target 言語名を placeholder で埋める方式。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from realtime_interpreter.i18n import DEFAULT_SOURCE, DEFAULT_TARGET, language_name

logger = logging.getLogger(__name__)

# 要約モデルのサンプリング設定 (ローカル Gemma 用)
SUMMARY_MAX_TOKENS = 256
SUMMARY_TEMPERATURE = 0.3
SUMMARY_TOP_P = 0.9

# OpenAI Chat Completions 用の既定モデル.
# - gpt-5-mini: 品質/コスト/レイテンシのバランスが良い. ~$0.02-0.05/h 想定 (60s ごとに要約)
# - gpt-4o-mini: より安価. ~$0.018/h
# - gpt-5: 過剰品質 (要約に reasoning モデル相当はオーバーキル)
DEFAULT_OPENAI_SUMMARY_MODEL = "gpt-5-mini"
# 出力上限. gpt-5 系は reasoning モデルで内部 thinking に大量のトークンを使うため、
# 余裕を持たせて 2048. 通常の要約 (300 トークン程度) + thinking (1000+) を許容する。
# 注: max_completion_tokens は thinking + 出力の合計に対する上限なので、小さすぎると
# thinking で使い切って content が空になる。
OPENAI_SUMMARY_MAX_COMPLETION_TOKENS = 2048


SUMMARY_PROMPT_TEMPLATE = (
    "You are an expert at summarizing {source_language} speech into {target_language}. "
    "Below is the transcript of the past {duration} seconds of {source_language} speech.\n"
    "\n"
    "## Output rules\n"
    "- Write a concise summary in {target_language} (2-4 sentences).\n"
    "- Output ONLY the summary body. No prefix (e.g. 'Summary:'), no quotes, no preamble.\n"
    "- Keep technical terms (CPU, AWS, GPU, etc.) in their original form where natural.\n"
    "- If the input is too short or unintelligible, output nothing.\n"
    "\n"
    "## {source_language} transcript\n"
    "{text}\n"
)


@dataclass
class SummaryResult:
    """1 回の要約結果."""

    text: str
    latency_seconds: float


def build_summary_prompt(
    source_text: str,
    duration_seconds: int,
    source_lang: str = DEFAULT_SOURCE,
    target_lang: str = DEFAULT_TARGET,
) -> str:
    """要約プロンプトを組み立てる. 言語名は ISO コードから英名へ展開."""
    return SUMMARY_PROMPT_TEMPLATE.format(
        text=source_text.strip(),
        duration=duration_seconds,
        source_language=language_name(source_lang),
        target_language=language_name(target_lang),
    )


class Summarizer:
    """共有された Gemma 4 モデルで target 言語の要約を生成する.

    Translator が既にロードしたモデルインスタンスをそのまま使う。
    source_lang / target_lang はプロンプト言語名に展開される。
    """

    def __init__(
        self,
        translator,  # GemmaAudioTranslator (循環 import を避けるため型ヒント省略)
        max_tokens: int = SUMMARY_MAX_TOKENS,
        temperature: float = SUMMARY_TEMPERATURE,
        top_p: float = SUMMARY_TOP_P,
        source_lang: str = DEFAULT_SOURCE,
        target_lang: str = DEFAULT_TARGET,
    ) -> None:
        self._translator = translator
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.source_lang = source_lang
        self.target_lang = target_lang

    def summarize(self, source_text: str, duration_seconds: int) -> SummaryResult:
        """過去 N 秒の source 言語テキストを target 言語要約に変換."""
        if not source_text.strip():
            return SummaryResult(text="", latency_seconds=0.0)

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt_text = build_summary_prompt(
            source_text, duration_seconds,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )
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


class OpenAIChatSummarizer:
    """OpenAI Chat Completions による target 言語要約器.

    Realtime API (gpt-realtime-translate) とは別経路で Chat API を叩く。
    既存の OPENAI_API_KEY をそのまま再利用するため追加認証は不要。

    `Summarizer` (ローカル Gemma) と同じ `.summarize(text, duration) -> SummaryResult`
    インターフェースを提供するので、main.py の要約呼び出しコードは共通化できる。
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_SUMMARY_MODEL,
        api_key: str | None = None,
        max_completion_tokens: int = OPENAI_SUMMARY_MAX_COMPLETION_TOKENS,
        source_lang: str = DEFAULT_SOURCE,
        target_lang: str = DEFAULT_TARGET,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Required for the OpenAI Chat summarizer."
            )
        self._max_completion_tokens = max_completion_tokens
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._client = None  # type: ignore[assignment]

    def _ensure_client(self) -> None:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)

    def summarize(self, source_text: str, duration_seconds: int) -> SummaryResult:
        if not source_text.strip():
            return SummaryResult(text="", latency_seconds=0.0)

        self._ensure_client()
        assert self._client is not None
        prompt_text = build_summary_prompt(
            source_text, duration_seconds,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )

        t0 = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt_text}],
                max_completion_tokens=self._max_completion_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("openai summary failed (model=%s)", self._model)
            return SummaryResult(text="", latency_seconds=time.perf_counter() - t0)
        latency = time.perf_counter() - t0

        # トークン使用量を診断目的でログ. reasoning モデルでは completion_tokens の
        # うち reasoning_tokens が大半を占めるケースがあるため、空応答時の原因切り分けに役立つ。
        usage_info = ""
        usage = getattr(response, "usage", None)
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None) if details else None
            usage_info = (
                f" tokens: prompt={getattr(usage, 'prompt_tokens', '?')}, "
                f"completion={getattr(usage, 'completion_tokens', '?')}"
                + (f", reasoning={reasoning}" if reasoning is not None else "")
            )

        if not text:
            # 空応答は max_completion_tokens 不足 (reasoning 食い潰し) が最有力. WARN で残す。
            finish_reason = getattr(response.choices[0], "finish_reason", "?")
            logger.warning(
                "openai summary returned empty content (model=%s, finish_reason=%s).%s "
                "If finish_reason='length', try a larger --openai-summary-model "
                "budget or switch to gpt-4o-mini.",
                self._model,
                finish_reason,
                usage_info,
            )
        else:
            logger.info(
                "openai summary (%s) %.2fs%s / %s",
                self._model,
                latency,
                usage_info,
                text[:80],
            )
        return SummaryResult(text=text, latency_seconds=latency)
