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
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Iterator

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
    "- Keep technical terms (CPU, AWS, GPU, API, etc.) in their original form "
    "where natural.\n"
    "- Do not invent content. Translate only what is clearly audible.\n"
    "- Do not repeat words or phrases.\n"
    "- If the audio is silent or unintelligible, output exactly:\n"
    "  SRC:\n"
    "  TGT:\n"
)


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

    def translate(self, audio: np.ndarray) -> OpenAIChatTranslationResult:
        wav_b64 = _encode_wav_base64(audio)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
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

    def __enter__(self) -> "OpenAIChatBackend":
        self._capture.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._capture.__exit__(exc_type, exc_val, exc_tb)

    def stream_segments(self) -> Iterator[TranslatedSegment]:
        for segment in self._capture.segments():
            try:
                result = self.translator.translate(segment.audio)
            except Exception:
                logger.exception("openai-chat translation failed")
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

    def summarize(self, source_text: str, duration_seconds: int) -> SummaryResult:
        if not source_text.strip():
            return SummaryResult(text="", latency_seconds=0.0)

        prompt = build_summary_prompt(
            source_text,
            duration_seconds,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
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
