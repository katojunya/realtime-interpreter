"""mlx-vlm + Gemma 4 による直接音声→(英語転写, 日本語訳) 出力.

VAD で確定した発話セグメントを受け取り、Gemma 4 に「英語の書き起こし」と
「日本語訳」の両方を 1 回の推論で生成させる。
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# キュレートされたモデルプリセット.
# audio 入力をサポートするローカル MLX モデルのみを掲載する。
MODEL_PRESETS: dict[str, tuple[str, str]] = {
    "e4b": (
        "mlx-community/gemma-4-e4b-it-4bit",
        "Gemma 4 E4B (effective 4B params, 4-bit) — 既定. 品質と速度のバランス",
    ),
    "e4b-bf16": (
        "mlx-community/gemma-4-e4b-it-bf16",
        "Gemma 4 E4B (bf16) — 量子化なし, 高品質, 重い",
    ),
    "e2b": (
        "mlx-community/gemma-4-e2b-it-4bit",
        "Gemma 4 E2B (effective 2B params, 4-bit) — 軽量・高速, 品質は E4B より下",
    ),
    "e2b-bf16": (
        "mlx-community/gemma-4-e2b-it-bf16",
        "Gemma 4 E2B (bf16) — 量子化なし",
    ),
}

DEFAULT_ALIAS = "e4b"


def resolve_model_id(name: str | None) -> str:
    """エイリアス・環境変数・フル ID のいずれを与えても HuggingFace ID を返す.

    優先順位:
        1. 引数 `name` が非 None: エイリアスとして解決を試み、無ければそのまま使う
        2. 環境変数 REALTIME_INTERPRETER_MODEL: エイリアス or フル ID として解決
        3. DEFAULT_ALIAS

    "/" を含む文字列はフル ID とみなしエイリアス解決をスキップする。
    """
    candidate = name or os.environ.get("REALTIME_INTERPRETER_MODEL") or DEFAULT_ALIAS
    if "/" in candidate:
        return candidate
    if candidate in MODEL_PRESETS:
        return MODEL_PRESETS[candidate][0]
    available = ", ".join(MODEL_PRESETS.keys())
    raise ValueError(
        f"unknown model alias {candidate!r}. "
        f"known aliases: {available}. "
        f"or pass a full HuggingFace ID containing '/'."
    )


# 英語転写と日本語訳を 1 回の推論で取り出すための構造化プロンプト.
TRANSCRIBE_AND_TRANSLATE_PROMPT = (
    "You are a professional simultaneous interpreter from English to Japanese. "
    "The audio contains English speech.\n"
    "\n"
    "Transcribe the speech in English, then translate it to Japanese.\n"
    "Output EXACTLY in this format, with no extra text or commentary:\n"
    "EN: <verbatim English transcription>\n"
    "JA: <natural Japanese translation>\n"
    "\n"
    "Rules:\n"
    "- Keep technical terms (CPU, AWS, GPU, API, etc.) in English where natural.\n"
    "- Do not invent content. Translate only what is clearly audible.\n"
    "- Do not repeat words or phrases.\n"
    "- If the audio is silent or unintelligible, output exactly:\n"
    "  EN:\n"
    "  JA:\n"
)


@dataclass
class TranslationResult:
    """1 セグメント分の推論結果."""

    english: str
    japanese: str
    latency_seconds: float


_EN_LINE = re.compile(r"^\s*EN\s*:\s*(.*)$", re.IGNORECASE)
_JA_LINE = re.compile(r"^\s*JA\s*:\s*(.*)$", re.IGNORECASE)


def _parse_en_ja(text: str) -> tuple[str, str]:
    """モデル出力から EN: / JA: 行を抽出する.

    モデルが指示形式から外れた場合のフォールバックとして:
    - "EN:" / "JA:" タグが見つからなければ、全体を JA とみなす (英語空)。
    - 同じタグが複数行ある場合は最後を採用 (まれに前置きを生成するため).
    """
    en = ""
    ja = ""
    found_tag = False
    for line in text.splitlines():
        m = _EN_LINE.match(line)
        if m:
            en = m.group(1).strip()
            found_tag = True
            continue
        m = _JA_LINE.match(line)
        if m:
            ja = m.group(1).strip()
            found_tag = True
            continue
    if not found_tag:
        # 想定形式から外れた → 全体を JA として扱う
        ja = text.strip()
    return en, ja


class GemmaAudioTranslator:
    """Gemma 4 (mlx-vlm) を使った音声→(英語転写, 日本語訳) ペア生成器.

    各 translate() 呼び出しは独立で、状態を持たない。
    モデルロードは初回 translate() で遅延実行する。
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        """Args:
            model: モデル指定. エイリアス (e4b/e2b等), 完全な HuggingFace ID, または None.
                   None の場合は環境変数 REALTIME_INTERPRETER_MODEL → DEFAULT_ALIAS の順で解決。
        """
        self.model_id = resolve_model_id(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._processor = None

    def load(self) -> None:
        """モデルとプロセッサをロード (キャッシュ済なら数秒)."""
        if self._model is not None:
            return
        logger.info("Loading model: %s", self.model_id)
        from mlx_vlm import load

        t0 = time.perf_counter()
        self._model, self._processor = load(self.model_id)
        logger.info("Model loaded in %.1fs", time.perf_counter() - t0)

    @property
    def model(self) -> object:
        """ロード済みモデル (Summarizer 等で共有するための公開アクセス)."""
        self.load()
        return self._model

    @property
    def processor(self) -> object:
        """ロード済みプロセッサ (Summarizer 等で共有するための公開アクセス)."""
        self.load()
        return self._processor

    def translate(self, audio: np.ndarray) -> TranslationResult:
        """音声波形 (モノラル float32 @ 16kHz) を英語転写 + 日本語訳に変換."""
        self.load()

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # mlx-vlm の audio 引数は file path / BytesIO を期待するため、
        # 一度 WAV としてメモリ上にエンコードして渡す。
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)

        prompt = apply_chat_template(
            self._processor,
            self._model.config,
            TRANSCRIBE_AND_TRANSLATE_PROMPT,
            num_audios=1,
        )

        t0 = time.perf_counter()
        result = generate(
            model=self._model,
            processor=self._processor,
            prompt=prompt,
            audio=[buf],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            verbose=False,
        )
        latency = time.perf_counter() - t0

        raw = _extract_text(result)
        en, ja = _parse_en_ja(raw)
        logger.debug("raw model output: %r", raw)
        return TranslationResult(english=en, japanese=ja, latency_seconds=latency)


def _extract_text(result: object) -> str:
    """mlx-vlm の generate() 戻り値からテキスト本体を取り出す.

    バージョンによっては str を直接返す場合と、.text 属性を持つオブジェクトを返す場合がある。
    """
    if isinstance(result, str):
        return result
    text_attr = getattr(result, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(result)
