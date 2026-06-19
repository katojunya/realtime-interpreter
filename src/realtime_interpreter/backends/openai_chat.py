"""OpenAI-compatible Chat Completions REST backend.

This backend is intended for local OpenAI-compatible servers such as Ollama.
Unlike the OpenAI Realtime backend, the API is request/response based, so audio
is segmented locally with Silero VAD and each finalized segment is sent as a
small WAV attachment.

Ollama 0.24.0 with Gemma 4 accepts WAV audio through the OpenAI-compatible
`input_audio` message part:

    {
      "type": "input_audio",
      "input_audio": {"data": "<base64 wav>", "format": "wav"}
    }
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Iterator, Callable

from rich.text import Text

import numpy as np

from realtime_interpreter.audio import (
    END_SILENCE_MS,
    MAX_SEGMENT_SECONDS,
    SAMPLE_RATE,
    SpeechSegmentCapture,
    WindowsLoopbackSpeechSegmentCapture,
)
from realtime_interpreter.backends.base import TranslatedSegment
from realtime_interpreter.i18n import DEFAULT_SOURCE, DEFAULT_TARGET, language_name
from realtime_interpreter.summarizer import (
    SummaryResult,
    build_summary_prompt,
)
from realtime_interpreter.translator import _parse_src_tgt

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_CHAT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OPENAI_CHAT_MODEL = "gemma4:e4b"
DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS = 120.0
DEFAULT_OPENAI_CHAT_MAX_TOKENS = 384
DEFAULT_OPENAI_CHAT_SUMMARY_MAX_TOKENS = 256
DEFAULT_OPENAI_CHAT_TEMPERATURE = 0.0


OPENAI_CHAT_AUDIO_PROMPT = (
    "You are a professional simultaneous interpreter from {source_language} to "
    "{target_language}. The attached audio contains {source_language} speech.\n"
    "\n"
    "Transcribe the speech in {source_language}, then translate it to "
    "{target_language}.\n"
    "Output EXACTLY in this format, with no extra text or commentary:\n"
    "SRC: <verbatim {source_language} transcription>\n"
    "TGT: <natural {target_language} translation>\n"
    "\n"
    "Rules:\n"
    "- Transcribe ONLY actual spoken words that you can clearly hear.\n"
    "- Keep technical terms (CPU, AWS, GPU, API, etc.) in their original form "
    "where natural.\n"
    "- Do not invent, summarize, continue, or guess beyond what is audible.\n"
    "- Do not repeat words or phrases.\n"
    "- If the audio is silence, music, applause, noise, or any non-speech "
    "sound, output empty SRC and TGT. Do NOT describe the sound.\n"
    "- NEVER output filler or placeholder sentences such as "
    '"I\'m going to give you an overview...", "Let me show you...", or '
    '"I\'m going to give you a little bit of background...". '
    "If no words are spoken, leave both lines empty.\n"
    "\n"
    "Example when the audio has no speech:\n"
    "SRC:\n"
    "TGT:\n"
)


# 幻覚抑制 (層2a): 直近セグメントと逐語一致する転写を弾く。
# 小型ローカルモデルは音楽/拍手/無音の区間で "I'm going to give you a little
# bit of background..." のような尤もらしい定型文を生成し、それらはセッション内で
# 逐語反復する傾向がある。実発話で同一文が短期間に逐語一致するのは稀なので、
# 直近 N 件と一致したセグメントを幻覚とみなして破棄する。
_DEDUP_WS_RE = re.compile(r"\s+")
_DEDUP_STRIP = " 　.,!?;:。、！？…・「」『』\"'“”’‘()（）"


def _normalize_for_dedup(text: str) -> str:
    """重複判定用の正規化 (小文字化・空白圧縮・前後の記号除去)."""
    norm = _DEDUP_WS_RE.sub(" ", text.strip().lower())
    return norm.strip(_DEDUP_STRIP)


class _RepetitionGuard:
    """過去の転写と逐語一致するセグメントを幻覚として検出する.

    幻覚の定型句はセッション内で数十分離れて再出現するため、既定では
    セッション全体 (history=None) で逐語一致を判定する。history に整数を渡すと
    直近 N 件の窓だけを対象にする (近接重複のみ弾きたい場合)。

    min_chars 未満の短い発話 (相づち・固有名詞など) は、正当な反復を誤って
    弾かないよう判定対象外にする。長い定型句 (= 厄介な幻覚) のみを束ねる。
    """

    def __init__(self, history: int | None = None, min_chars: int = 12) -> None:
        self._min_chars = min_chars
        self._seen: set[str] = set()
        self._window: deque[str] | None = (
            deque(maxlen=history) if history else None
        )

    def is_repeat(self, source: str) -> bool:
        norm = _normalize_for_dedup(source)
        if len(norm) < self._min_chars:
            return False
        if self._window is None:
            if norm in self._seen:
                return True
            self._seen.add(norm)
            return False
        if norm in self._window:
            return True
        self._window.append(norm)
        return False


@dataclass
class OpenAIChatTranslationResult:
    """1 REST call worth of audio translation output."""

    source: str
    target: str
    latency_seconds: float


def _chat_completions_url(base_url: str) -> str:
    """Return the OpenAI-compatible chat completions endpoint URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _api_key(explicit: str | None) -> str:
    """Resolve API key for OpenAI-compatible servers.

    Ollama ignores the bearer token but OpenAI-compatible clients generally
    expect one to be present. Use a harmless default for local Ollama.
    """
    return (
        explicit
        or os.environ.get("OPENAI_CHAT_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "ollama"
    )


def _encode_wav_base64(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Encode mono float32 audio as base64 RIFF/WAV PCM16."""
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim != 1:
        mono = np.mean(mono, axis=1).astype(np.float32)
    pcm16 = np.clip(mono, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_chat_content(payload: dict) -> str:
    """Extract assistant content from a Chat Completions-compatible response."""
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts).strip()
    return ""


def _post_json(
    url: str,
    payload: dict,
    api_key: str,
    timeout_seconds: float,
) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"openai-chat HTTP {e.code}: {body}") from e
    return json.loads(body)


class OpenAIChatAudioTranslator:
    """Send finalized WAV audio chunks to an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_CHAT_MODEL,
        base_url: str = DEFAULT_OPENAI_CHAT_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS,
        max_tokens: int = DEFAULT_OPENAI_CHAT_MAX_TOKENS,
        temperature: float = DEFAULT_OPENAI_CHAT_TEMPERATURE,
        source_lang: str = DEFAULT_SOURCE,
        target_lang: str = DEFAULT_TARGET,
    ) -> None:
        self.model = model
        self.url = _chat_completions_url(base_url)
        self.api_key = _api_key(api_key)
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._prompt = OPENAI_CHAT_AUDIO_PROMPT.format(
            source_language=language_name(source_lang),
            target_language=language_name(target_lang),
        )
        # ローリング要約を「参照文脈」として翻訳プロンプトに前置きする (空なら無効)。
        self.session_context = ""

    def update_context(self, summary: str) -> None:
        """直近の累積要約を翻訳の参照文脈として設定する (main から呼ばれる)."""
        self.session_context = summary or ""

    def _prompt_with_context(self) -> str:
        if self.session_context:
            return (
                f"{self._prompt}\n\n"
                "## Background context (disambiguation only)\n"
                "The text below summarizes earlier speech. Use it ONLY to disambiguate "
                "terms and names you actually hear in the audio. Do NOT transcribe, "
                "translate, repeat, or output this text. If the audio has no clear "
                f"speech, still output empty SRC and TGT.\n{self.session_context}"
            )
        return self._prompt

    def translate(self, audio: np.ndarray) -> OpenAIChatTranslationResult:
        wav_b64 = _encode_wav_base64(audio)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt_with_context()},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": wav_b64,
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # Ollama supports this OpenAI-compatible field for thinking models.
            "reasoning_effort": "none",
        }

        t0 = time.perf_counter()
        response = _post_json(
            self.url,
            payload,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        latency = time.perf_counter() - t0
        raw = _extract_chat_content(response)
        src, tgt = _parse_src_tgt(raw)
        logger.debug("openai-chat raw output: %r", raw)
        return OpenAIChatTranslationResult(
            source=src,
            target=tgt,
            latency_seconds=latency,
        )


class OpenAIChatBackend:
    """Local VAD + OpenAI-compatible Chat Completions audio backend.

    macOS/Linux use a normal sounddevice input device. Windows uses WASAPI
    loopback so the backend can translate system audio without a virtual audio
    driver.
    """

    def __init__(
        self,
        sd_module: ModuleType | None,
        device_name: str,
        translator: OpenAIChatAudioTranslator,
        end_silence_ms: int = END_SILENCE_MS,
        max_segment_seconds: float = MAX_SEGMENT_SECONDS,
    ) -> None:
        self.translator = translator
        self._comm_cb: Callable[[object], None] | None = None
        self._repetition_guard = _RepetitionGuard()
        if sys.platform == "win32":
            self._capture = WindowsLoopbackSpeechSegmentCapture(
                device_name=device_name,
                end_silence_ms=end_silence_ms,
                max_segment_seconds=max_segment_seconds,
            )
        else:
            self._capture = SpeechSegmentCapture(
                sd_module=sd_module,
                device_name=device_name,
                end_silence_ms=end_silence_ms,
                max_segment_seconds=max_segment_seconds,
            )

    def set_status_callback(
        self,
        audio_cb: Callable[[object], None],
        comm_cb: Callable[[object], None],
    ) -> None:
        # 左スロット(音声メーター)は capture が、右スロット(通信)は本 backend が更新。
        self._comm_cb = comm_cb
        self._capture.status_callback = audio_cb

    def __enter__(self) -> "OpenAIChatBackend":
        self._capture.backend_name = "OpenAI Chat"
        self._capture.__enter__()
        # 最初の発話待ちから Listening を表示する。
        if self._comm_cb:
            self._comm_cb(self._comm_listening())
        return self

    def _comm_translating(self) -> Text:
        """右スロット用: API リクエスト中 (推論中)."""
        return Text("> ", style="cyan bold").append(
            "Translating (OpenAI Chat API Request)... [Waiting API]", style="bold"
        )

    def _comm_listening(self) -> Text:
        """右スロット用: 推論完了後の発話待ち (idle)."""
        return Text("> ", style="cyan bold").append(
            "Listening (OpenAI Chat API)...", style="bold"
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._capture.__exit__(exc_type, exc_val, exc_tb)

    def update_context(self, summary: str) -> None:
        """ローリング要約を翻訳の参照文脈として translator へ供給する (main から呼ばれる)."""
        self.translator.update_context(summary)

    def stream_segments(self) -> Iterator[TranslatedSegment]:
        for segment in self._capture.segments():
            try:
                if self._comm_cb:
                    self._comm_cb(self._comm_translating())
                result = self.translator.translate(segment.audio)
            except Exception:
                logger.exception("openai-chat translation failed")
                if self._comm_cb:
                    self._comm_cb(self._comm_listening())
                continue
            # 翻訳完了 → 次の発話待ち (Listening) へ戻す。
            if self._comm_cb:
                self._comm_cb(self._comm_listening())
            # 幻覚抑制 (層2a): 直近と逐語一致する転写は破棄する。
            if self._repetition_guard.is_repeat(result.source):
                logger.debug(
                    "dropped repeated (likely hallucinated) segment: %r",
                    result.source,
                )
                continue
            yield TranslatedSegment(
                start_offset_seconds=segment.start_offset_seconds,
                duration_seconds=segment.duration_seconds,
                source=result.source,
                target=result.target,
                is_partial=False,
            )


class OpenAIChatCompatibleSummarizer:
    """Text summarizer backed by the same OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_CHAT_MODEL,
        base_url: str = DEFAULT_OPENAI_CHAT_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS,
        max_tokens: int = DEFAULT_OPENAI_CHAT_SUMMARY_MAX_TOKENS,
        temperature: float = DEFAULT_OPENAI_CHAT_TEMPERATURE,
        source_lang: str = DEFAULT_SOURCE,
        target_lang: str = DEFAULT_TARGET,
    ) -> None:
        self.model = model
        self.url = _chat_completions_url(base_url)
        self.api_key = _api_key(api_key)
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.source_lang = source_lang
        self.target_lang = target_lang

    def summarize(self, source_text: str, duration_seconds: int, prev_summary: str = "") -> SummaryResult:
        if not source_text.strip():
            return SummaryResult(text="", latency_seconds=0.0)

        prompt = build_summary_prompt(
            source_text,
            duration_seconds,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            prev_summary=prev_summary,
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": "none",
        }

        t0 = time.perf_counter()
        try:
            response = _post_json(
                self.url,
                payload,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception:
            logger.exception("openai-chat summary failed")
            return SummaryResult(text="", latency_seconds=time.perf_counter() - t0)
        latency = time.perf_counter() - t0
        return SummaryResult(
            text=_extract_chat_content(response),
            latency_seconds=latency,
        )
