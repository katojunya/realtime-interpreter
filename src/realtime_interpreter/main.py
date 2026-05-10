"""CLI エントリポイント.

BlackHole 2ch
  → SpeechSegmentCapture (VAD でセグメントを区切る)
  → Gemma 4 が「英語転写 + 日本語訳」を 1 回の推論で生成
  → 確定したセグメントごとに append-only で 2 行出力 (英語 + 日本語)

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
    SpeechSegmentCapture,
)
from realtime_interpreter.session_logger import SessionLogger, format_offset
from realtime_interpreter.summarizer import Summarizer
from realtime_interpreter.translator import (
    DEFAULT_ALIAS,
    MODEL_PRESETS,
    GemmaAudioTranslator,
)

logger = logging.getLogger(__name__)

# --- ANSI エスケープ ---
DIM = "\033[90m"  # 英語転写用 (グレー)
YELLOW = "\033[93m"  # ステータス
CYAN = "\033[96m"  # 要約用
RESET = "\033[0m"
CLEAR_LINE = "\r\033[K"

DEFAULT_SUMMARY_INTERVAL_SECONDS = 60


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
        description="Low-latency EN→JA simultaneous interpreter (Gemma 4 + mlx-vlm)",
    )
    parser.add_argument(
        "--device",
        default=DEVICE_NAME,
        help=f"Audio input device name (default: {DEVICE_NAME!r})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Model alias ({'/'.join(MODEL_PRESETS.keys())}) or full HuggingFace ID. "
            f"Default: env REALTIME_INTERPRETER_MODEL or {DEFAULT_ALIAS!r}. "
            "Use --list-models to see all presets."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model presets and exit",
    )
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=END_SILENCE_MS,
        help=(
            "Silence (ms) that ends a speech segment. "
            "Lower = smaller chunks, faster output, more risk of mid-sentence cuts. "
            f"(default: {END_SILENCE_MS})"
        ),
    )
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=MAX_SEGMENT_SECONDS,
        help=(
            "Hard cap (seconds) for a single segment when there is no silence. "
            f"Lower = smaller chunks for continuous speech. (default: {MAX_SEGMENT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--summary-interval-seconds",
        type=int,
        default=DEFAULT_SUMMARY_INTERVAL_SECONDS,
        help=(
            "Generate a Japanese summary of the last N seconds of English transcription. "
            "0 to disable. "
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
    print("Available model presets:")
    print()
    for alias, (model_id, desc) in MODEL_PRESETS.items():
        marker = "*" if alias == DEFAULT_ALIAS else " "
        print(f"  {marker} {alias:<10} {model_id}")
        print(f"    {' ':<10} {desc}")
        print()
    print("Use any alias above with --model, or pass a full HuggingFace ID")
    print("(must contain '/'). Set REALTIME_INTERPRETER_MODEL env var to change default.")


def _print_status(text: str) -> None:
    """1 行ステータスを上書き表示 (改行しない)."""
    print(f"{CLEAR_LINE}{YELLOW}{text}{RESET}", end="", flush=True)


def _clear_status() -> None:
    print(CLEAR_LINE, end="", flush=True)


def _emit_segment(ts: str, english: str, japanese: str) -> None:
    """確定したセグメントの 2 行を append-only で出力."""
    if english.strip():
        print(f"{DIM}[{ts}] {english.strip()}{RESET}", flush=True)
    if japanese.strip():
        print(f"[{ts}] {japanese.strip()}", flush=True)
    if english.strip() or japanese.strip():
        print("", flush=True)


def _emit_summary(ts: str, text: str) -> None:
    """N 秒ごとの要約をブロックで出力."""
    if not text.strip():
        return
    print(f"{CYAN}--- 要約 [{ts}] ---", flush=True)
    print(text.strip(), flush=True)
    print(f"---{RESET}", flush=True)
    print("", flush=True)


def _maybe_summarize(
    summarizer: Summarizer,
    english_buffer: list[tuple[float, str]],
    since_offset: float,
    until_offset: float,
    duration_seconds: int,
    session_logger: SessionLogger,
) -> None:
    """要約発火: since_offset 以降の英文を集約して 1 つの要約を生成・表示."""
    items = [text for ts, text in english_buffer if ts >= since_offset]
    if not items:
        return
    en_concat = " ".join(items)

    _print_status(f"⏳ Summarizing last {duration_seconds}s...")
    try:
        summary = summarizer.summarize(en_concat, duration_seconds)
    except Exception:
        logger.exception("summarization failed")
        _clear_status()
        return
    _clear_status()

    if summary.text:
        ts = format_offset(until_offset)
        logger.debug(
            "summary %s infer=%.2fs / %s",
            ts,
            summary.latency_seconds,
            summary.text[:80],
        )
        _emit_summary(ts, summary.text)
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

    _check_audio_output()

    try:
        translator = GemmaAudioTranslator(model=args.model)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"Loading {translator.model_id} (this may take a minute)...", file=sys.stderr)
    translator.load()
    print("Model loaded.", file=sys.stderr)

    session_logger = SessionLogger(log_dir=Path(args.log_dir))
    print(f"Log: {session_logger.path}\n", file=sys.stderr)

    if args.end_silence_ms <= 0:
        print("error: --end-silence-ms must be positive", file=sys.stderr)
        sys.exit(2)
    if args.max_segment_seconds <= 0:
        print("error: --max-segment-seconds must be positive", file=sys.stderr)
        sys.exit(2)
    if args.summary_interval_seconds < 0:
        print("error: --summary-interval-seconds must be >= 0", file=sys.stderr)
        sys.exit(2)

    summary_enabled = args.summary_interval_seconds > 0
    summarizer = Summarizer(translator) if summary_enabled else None

    print(
        f"VAD: end_silence={args.end_silence_ms}ms, "
        f"max_segment={args.max_segment_seconds}s, "
        f"summary={'every ' + str(args.summary_interval_seconds) + 's' if summary_enabled else 'off'}",
        file=sys.stderr,
    )

    # 過去の英語転写をオフセット秒付きで保持. 要約発火時に最新ウィンドウぶんを抽出する。
    english_buffer: list[tuple[float, str]] = []
    summary_last_offset = 0.0
    summary_interval = float(args.summary_interval_seconds)

    try:
        with SpeechSegmentCapture(
            sd_module=sd,
            device_name=args.device,
            end_silence_ms=args.end_silence_ms,
            max_segment_seconds=args.max_segment_seconds,
        ) as capture:
            _print_status("● Listening...")
            # MLX の GPU stream はスレッドローカル. translate() はメインスレッドで呼ぶ。
            # 推論中は capture のジェネレータが止まるが、sounddevice コールバックは別スレッドで
            # サンプルを内部キューに溜め続けるため、推論完了後に取りこぼしなく消化される。
            for segment in capture.segments():
                _print_status(f"⏳ Translating {segment.duration_seconds:.1f}s...")
                try:
                    result = translator.translate(segment.audio)
                except Exception:
                    logger.exception("translation failed")
                    _clear_status()
                    _print_status("● Listening...")
                    continue

                _clear_status()
                ts = format_offset(segment.start_offset_seconds)
                logger.debug(
                    "segment %s dur=%.2fs infer=%.2fs",
                    ts,
                    segment.duration_seconds,
                    result.latency_seconds,
                )
                _emit_segment(ts, result.english, result.japanese)
                session_logger.log_segment(ts, result.english, result.japanese)

                # 要約用バッファに英文を追加
                if result.english.strip():
                    english_buffer.append(
                        (segment.start_offset_seconds, result.english.strip())
                    )

                # 要約発火判定: セグメント終了時刻が最後の要約から interval 秒経過していれば
                segment_end_offset = (
                    segment.start_offset_seconds + segment.duration_seconds
                )
                if (
                    summarizer is not None
                    and segment_end_offset - summary_last_offset >= summary_interval
                ):
                    _maybe_summarize(
                        summarizer,
                        english_buffer,
                        summary_last_offset,
                        segment_end_offset,
                        int(summary_interval),
                        session_logger,
                    )
                    summary_last_offset = segment_end_offset
                    # バッファ整理: 直近 2 周期ぶんだけ残す
                    cutoff = segment_end_offset - summary_interval * 2
                    english_buffer[:] = [
                        (ts_, t) for ts_, t in english_buffer if ts_ >= cutoff
                    ]

                _print_status("● Listening...")

    except KeyboardInterrupt:
        pass
    finally:
        _clear_status()
        session_logger.close()
        print(f"\nLog: {session_logger.path}", file=sys.stderr)


if __name__ == "__main__":
    main()
