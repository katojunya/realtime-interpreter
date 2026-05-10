"""CLI エントリポイント.

BlackHole 2ch から音声をキャプチャし、選択したバックエンドで英語転写と日本語訳を生成する。
バックエンドは TranslatedSegment をストリームし、`is_partial=True` (進行中) と
`is_partial=False` (確定) を区別して出力する。

- 進行中セグメント (OpenAI バックエンドの delta): Rich Live で in-place 更新
- 確定セグメント: append-only で永続表示 + ログ + 要約バッファ反映

バックエンド:
    mlx     ローカル MLX (Gemma 4 + mlx-vlm). 確定単位でしか出力しない (is_partial=False のみ)
    openai  OpenAI gpt-realtime-translate (WebSocket). delta 単位で in-place 更新

出力形式:
    [mm:ss] <英語転写>
    [mm:ss] <日本語訳>
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import sounddevice as sd

from realtime_interpreter.audio import (
    DEVICE_NAME,
    END_SILENCE_MS,
    MAX_SEGMENT_SECONDS,
)
from realtime_interpreter.backends.base import (
    TranslatedSegment,
    TranslationBackend,
)
from realtime_interpreter.backends.mlx_local import LocalMLXBackend
from realtime_interpreter.backends.openai_realtime import (
    DEFAULT_OPENAI_MODEL,
    OPENAI_MAX_SEGMENT_SECONDS,
    TURN_DEBOUNCE_SECONDS,
    OpenAIRealtimeBackend,
)
from realtime_interpreter.renderer import StreamingRenderer
from realtime_interpreter.session_logger import SessionLogger, format_offset
from realtime_interpreter.summarizer import Summarizer
from realtime_interpreter.translator import (
    DEFAULT_ALIAS,
    MODEL_PRESETS,
    GemmaAudioTranslator,
)

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_INTERVAL_SECONDS = 60
SUPPORTED_BACKENDS = ("mlx", "openai")


def _check_audio_output() -> None:
    """起動時に音声出力先を確認."""
    default_out = sd.default.device[1]
    device_name = sd.query_devices(default_out)["name"]
    print(f"Output device: {device_name}", file=sys.stderr)
    if "複数出力" not in device_name and "multi" not in device_name.lower():
        print(
            "⚠ Please switch macOS output to a Multi-Output Device including BlackHole 2ch.",
            file=sys.stderr,
        )
        try:
            subprocess.run(
                ["open", "x-apple.systempreferences:com.apple.Sound-Settings.extension"],
                check=False,
            )
        except Exception:
            pass
        input("  Press Enter when ready: ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Low-latency EN→JA simultaneous interpreter. "
            "Switchable backend: local MLX (Gemma 4) or OpenAI gpt-realtime-translate."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default="mlx",
        help="Translation backend (default: mlx)",
    )
    parser.add_argument(
        "--device",
        default=DEVICE_NAME,
        help=f"Audio input device name (default: {DEVICE_NAME!r})",
    )

    mlx_group = parser.add_argument_group("mlx backend")
    mlx_group.add_argument(
        "--model",
        default=None,
        help=(
            f"[mlx] Model alias ({'/'.join(MODEL_PRESETS.keys())}) or full HuggingFace ID. "
            f"Default: env REALTIME_INTERPRETER_MODEL or {DEFAULT_ALIAS!r}. "
            "Use --list-models to see all presets."
        ),
    )
    mlx_group.add_argument(
        "--list-models",
        action="store_true",
        help="[mlx] List available model presets and exit",
    )
    mlx_group.add_argument(
        "--end-silence-ms",
        type=int,
        default=END_SILENCE_MS,
        help=(
            "[mlx] Silence (ms) that ends a speech segment. "
            "Lower = smaller chunks, faster output, more risk of mid-sentence cuts. "
            f"(default: {END_SILENCE_MS})"
        ),
    )
    mlx_group.add_argument(
        "--max-segment-seconds",
        type=float,
        default=MAX_SEGMENT_SECONDS,
        help=(
            "[mlx] Hard cap (seconds) for a single segment when there is no silence. "
            f"Lower = smaller chunks for continuous speech. (default: {MAX_SEGMENT_SECONDS})"
        ),
    )

    openai_group = parser.add_argument_group("openai backend")
    openai_group.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"[openai] Realtime model id (default: {DEFAULT_OPENAI_MODEL!r})",
    )
    openai_group.add_argument(
        "--openai-debounce-seconds",
        type=float,
        default=TURN_DEBOUNCE_SECONDS,
        help=(
            "[openai] Commit a chunk after delta has been quiet for this many seconds. "
            "Smaller = more frequent (but higher risk of mid-translation commit causing "
            "EN/JA misalignment). Larger = fewer, longer chunks but properly aligned. "
            f"(default: {TURN_DEBOUNCE_SECONDS})"
        ),
    )
    openai_group.add_argument(
        "--openai-max-segment-seconds",
        type=float,
        default=OPENAI_MAX_SEGMENT_SECONDS,
        help=(
            "[openai] Force-commit a chunk after this many seconds of continuous "
            "accumulation, even when delta is still arriving. Prevents huge single "
            "chunks during long monologues. 0 to disable (debounce only). "
            f"(default: {OPENAI_MAX_SEGMENT_SECONDS})"
        ),
    )

    parser.add_argument(
        "--summary-interval-seconds",
        type=int,
        default=DEFAULT_SUMMARY_INTERVAL_SECONDS,
        help=(
            "Generate a Japanese summary of the last N seconds of English transcription. "
            "0 to disable. Requires the mlx backend to be available "
            "(local Gemma 4 used as summarizer). "
            f"(default: {DEFAULT_SUMMARY_INTERVAL_SECONDS})"
        ),
    )

    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for session logs (default: logs)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging to stderr",
    )
    return parser.parse_args()


def _print_model_presets() -> None:
    print("Available model presets (mlx backend):")
    print()
    for alias, (model_id, desc) in MODEL_PRESETS.items():
        marker = "*" if alias == DEFAULT_ALIAS else " "
        print(f"  {marker} {alias:<10} {model_id}")
        print(f"    {' ':<10} {desc}")
        print()
    print("Use any alias above with --model, or pass a full HuggingFace ID")
    print("(must contain '/'). Set REALTIME_INTERPRETER_MODEL env var to change default.")


def _build_backend(args: argparse.Namespace) -> tuple[TranslationBackend, Summarizer | None]:
    summary_enabled = args.summary_interval_seconds > 0

    if args.backend == "mlx":
        if args.end_silence_ms <= 0:
            raise SystemExit("error: --end-silence-ms must be positive")
        if args.max_segment_seconds <= 0:
            raise SystemExit("error: --max-segment-seconds must be positive")

        translator = GemmaAudioTranslator(model=args.model)
        print(f"Loading {translator.model_id} (this may take a minute)...", file=sys.stderr)
        translator.load()
        print("Model loaded.", file=sys.stderr)

        backend = LocalMLXBackend(
            sd_module=sd,
            translator=translator,
            device_name=args.device,
            end_silence_ms=args.end_silence_ms,
            max_segment_seconds=args.max_segment_seconds,
        )
        summarizer = Summarizer(translator) if summary_enabled else None
        return backend, summarizer

    if args.backend == "openai":
        if args.openai_debounce_seconds <= 0:
            raise SystemExit("error: --openai-debounce-seconds must be positive")
        if args.openai_max_segment_seconds < 0:
            raise SystemExit("error: --openai-max-segment-seconds must be >= 0 (0 disables)")
        backend = OpenAIRealtimeBackend(
            sd_module=sd,
            device_name=args.device,
            model=args.openai_model,
            turn_debounce_seconds=args.openai_debounce_seconds,
            max_segment_seconds=args.openai_max_segment_seconds,
        )
        summarizer = None
        if summary_enabled:
            print(
                "⚠ --summary-interval-seconds is set but ignored on the openai backend "
                "(local Gemma 4 not loaded). Use --backend mlx to enable summarization, "
                "or --summary-interval-seconds 0 to silence this warning.",
                file=sys.stderr,
            )
        return backend, summarizer

    raise SystemExit(f"unknown backend: {args.backend}")


def _emit_settings(args: argparse.Namespace) -> None:
    summary_str = (
        f"every {args.summary_interval_seconds}s"
        if args.summary_interval_seconds > 0
        else "off"
    )
    if args.backend == "mlx":
        print(
            f"Backend: mlx | "
            f"VAD: end_silence={args.end_silence_ms}ms, "
            f"max_segment={args.max_segment_seconds}s | "
            f"summary={summary_str}",
            file=sys.stderr,
        )
    else:
        max_seg_str = (
            f"{args.openai_max_segment_seconds}s"
            if args.openai_max_segment_seconds > 0
            else "off"
        )
        print(
            f"Backend: openai ({args.openai_model}) | "
            f"debounce={args.openai_debounce_seconds}s, "
            f"max_segment={max_seg_str} | "
            f"summary={'off (openai backend)' if summary_str != 'off' else 'off'}",
            file=sys.stderr,
        )


def _maybe_summarize(
    summarizer: Summarizer,
    english_buffer: list[tuple[float, str]],
    since_offset: float,
    until_offset: float,
    duration_seconds: int,
    session_logger: SessionLogger,
    renderer: StreamingRenderer,
) -> None:
    items = [text for ts, text in english_buffer if ts >= since_offset]
    if not items:
        return
    en_concat = " ".join(items)

    renderer.update_status(f"⏳ Summarizing last {duration_seconds}s...")
    try:
        summary = summarizer.summarize(en_concat, duration_seconds)
    except Exception:
        logger.exception("summarization failed")
        renderer.update_status("● Listening...")
        return
    renderer.update_status("● Listening...")

    if summary.text:
        ts = format_offset(until_offset)
        logger.debug(
            "summary %s infer=%.2fs / %s",
            ts,
            summary.latency_seconds,
            summary.text[:80],
        )
        renderer.emit_summary(ts, summary.text)
        session_logger.log_summary(ts, summary.text)


def main() -> None:
    args = _parse_args()
    if args.list_models:
        _print_model_presets()
        return

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.summary_interval_seconds < 0:
        print("error: --summary-interval-seconds must be >= 0", file=sys.stderr)
        sys.exit(2)

    _check_audio_output()

    try:
        backend, summarizer = _build_backend(args)
    except SystemExit:
        raise
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    session_logger = SessionLogger(log_dir=Path(args.log_dir))
    print(f"Log: {session_logger.path}", file=sys.stderr)
    _emit_settings(args)
    print("", file=sys.stderr)

    english_buffer: list[tuple[float, str]] = []
    summary_last_offset = 0.0
    summary_interval = float(args.summary_interval_seconds)

    try:
        with backend, StreamingRenderer() as renderer:
            renderer.update_status("● Listening...")
            for seg in backend.stream_segments():
                ts = format_offset(seg.start_offset_seconds)
                if seg.is_partial:
                    # in-place 更新. ログ・要約バッファには触らない。
                    renderer.update_current(ts, seg.english, seg.japanese)
                    continue

                # 確定セグメント: 永続表示・ログ・要約バッファ反映
                renderer.commit(ts, seg.english, seg.japanese)
                session_logger.log_segment(ts, seg.english, seg.japanese)
                if summarizer is not None and seg.english.strip():
                    english_buffer.append(
                        (seg.start_offset_seconds, seg.english.strip())
                    )

                segment_end = seg.start_offset_seconds + seg.duration_seconds
                if (
                    summarizer is not None
                    and segment_end - summary_last_offset >= summary_interval > 0
                ):
                    _maybe_summarize(
                        summarizer,
                        english_buffer,
                        summary_last_offset,
                        segment_end,
                        int(summary_interval),
                        session_logger,
                        renderer,
                    )
                    summary_last_offset = segment_end
                    cutoff = segment_end - summary_interval * 2
                    english_buffer[:] = [
                        (ts_, t) for ts_, t in english_buffer if ts_ >= cutoff
                    ]

                renderer.update_status("● Listening...")

    except KeyboardInterrupt:
        pass
    finally:
        session_logger.close()
        print(f"\nLog: {session_logger.path}", file=sys.stderr)


if __name__ == "__main__":
    main()
