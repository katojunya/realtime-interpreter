"""mlx-vlm + Gemma 4 による直接音声→(source 転写, target 訳) 出力.

VAD で確定した発話セグメントを受け取り、Gemma 4 に「source 言語の書き起こし」と
「target 言語の翻訳」の両方を 1 回の推論で生成させる。
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from dataclasses import dataclass

import numpy as np

from realtime_interpreter.i18n import DEFAULT_SOURCE, DEFAULT_TARGET, language_name

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


# source 転写と target 訳を 1 回の推論で取り出すための構造化プロンプト.
# `{source_language}` / `{target_language}` を .format() で展開して使う。
TRANSCRIBE_AND_TRANSLATE_PROMPT = (
    "You are a professional simultaneous interpreter from {source_language} to {target_language}. "
    "The audio contains {source_language} speech.\n"
    "\n"
    "Transcribe the speech in {source_language}, then translate it to {target_language}.\n"
    "Output EXACTLY in this format, with no extra text or commentary:\n"
    "SRC: <verbatim {source_language} transcription>\n"
    "TGT: <natural {target_language} translation>\n"
    "\n"
    "Rules:\n"
    "- Keep technical terms (CPU, AWS, GPU, API, etc.) in their original form where natural.\n"
    "- Do not invent content. Translate only what is clearly audible.\n"
    "- Do not repeat words or phrases.\n"
    "- If the audio is silent or unintelligible, output exactly:\n"
    "  SRC:\n"
    "  TGT:\n"
)


@dataclass
class TranslationResult:
    """1 セグメント分の推論結果. 言語非依存 (source/target)."""

    source: str
    target: str
    latency_seconds: float


_SRC_LINE = re.compile(r"^\s*SRC\s*:\s*(.*)$", re.IGNORECASE)
_TGT_LINE = re.compile(r"^\s*TGT\s*:\s*(.*)$", re.IGNORECASE)


def parse_src_tgt(text: str) -> tuple[str, str]:
    """モデル出力から SRC: / TGT: 行を抽出する.

    モデルが指示形式から外れた場合のフォールバックとして:
    - "SRC:" / "TGT:" タグが見つからなければ、全体を target とみなす (source 空)。
    - 同じタグが複数行ある場合は最後を採用 (まれに前置きを生成するため).
    """
    src = ""
    tgt = ""
    found_tag = False
    for line in text.splitlines():
        m = _SRC_LINE.match(line)
        if m:
            src = m.group(1).strip()
            found_tag = True
            continue
        m = _TGT_LINE.match(line)
        if m:
            tgt = m.group(1).strip()
            found_tag = True
            continue
    if not found_tag:
        # 想定形式から外れた → 全体を target として扱う
        tgt = text.strip()
    return src, tgt


class GemmaAudioTranslator:
    """Gemma 4 (mlx-vlm) を使った音声→(source 転写, target 訳) ペア生成器.

    各 translate() 呼び出しは独立で、状態を持たない。
    モデルロードは初回 translate() で遅延実行する。
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 0.9,
        source_lang: str = DEFAULT_SOURCE,
        target_lang: str = DEFAULT_TARGET,
    ) -> None:
        """Args:
            model: モデル指定. エイリアス (e4b/e2b等), 完全な HuggingFace ID, または None.
                   None の場合は環境変数 REALTIME_INTERPRETER_MODEL → DEFAULT_ALIAS の順で解決。
            source_lang: 翻訳元言語 ISO 639-1 コード (例: "en").
            target_lang: 翻訳先言語 ISO 639-1 コード (例: "ja").
        """
        self.model_id = resolve_model_id(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.source_lang = source_lang
        self.target_lang = target_lang
        # プロンプトを起動時に 1 回だけ format して以後再利用
        self._prompt = TRANSCRIBE_AND_TRANSLATE_PROMPT.format(
            source_language=language_name(source_lang),
            target_language=language_name(target_lang),
        )
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
        """音声波形 (モノラル float32 @ 16kHz) を source 転写 + target 訳に変換."""
        self.load()

        # mlx 系 / soundfile は macOS 専用依存. 遅延 import (Windows では translator
        # 自体使われないが、モジュール import は通る必要がある)。
        import soundfile as sf
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
            self._prompt,
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
        src, tgt = parse_src_tgt(raw)
        logger.debug("raw model output: %r", raw)
        return TranslationResult(source=src, target=tgt, latency_seconds=latency)


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
