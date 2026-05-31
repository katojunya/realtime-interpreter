"""CLI エントリポイント.

BlackHole 2ch から音声をキャプチャし、選択したバックエンドで source 言語の転写と
target 言語の訳を生成する。バックエンドは TranslatedSegment をストリームし、
`is_partial=True` (進行中) と `is_partial=False` (確定) を区別して出力する。

- 進行中セグメント (OpenAI バックエンドの delta): Rich Live で in-place 更新
- 確定セグメント: append-only で永続表示 + ログ + 要約バッファ反映

バックエンド:
    mlx     ローカル MLX (Gemma 4 + mlx-vlm). 確定単位でしか出力しない (is_partial=False のみ)
    openai  OpenAI gpt-realtime-translate (WebSocket). delta 単位で in-place 更新

出力形式:
    [mm:ss] <source 転写>
    [mm:ss] <target 訳>
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sounddevice as sd

from realtime_interpreter.audio import (
    DEVICE_NAME,
    END_SILENCE_MS,
    MAX_SEGMENT_SECONDS,
)
from realtime_interpreter.i18n import (
    DEFAULT_SOURCE,
    DEFAULT_TARGET,
    LANGUAGES,
    language_name,
    normalize_language_code,
)
from realtime_interpreter.backends.base import (
    TranslatedSegment,
    TranslationBackend,
)
from realtime_interpreter.backends.openai_realtime import (
    DEFAULT_OPENAI_MODEL,
    OPENAI_MAX_SEGMENT_SECONDS,
    TURN_DEBOUNCE_MS,
    OpenAIRealtimeBackend,
)
from realtime_interpreter.renderer import StreamingRenderer
from realtime_interpreter.session_logger import SessionLogger, format_offset
from realtime_interpreter.summarizer import (
    DEFAULT_OPENAI_SUMMARY_MODEL,
    OpenAIChatSummarizer,
)
# NOTE: mlx バックエンド固有の import (LocalMLXBackend / GemmaAudioTranslator / Summarizer)
# は _build_backend() の mlx 分岐内で遅延 import する。これにより mlx 未インストールの
# Windows/Linux でも --backend openai が動く。
# 定数 (DEVICE_NAME 等, MODEL_PRESETS, DEFAULT_ALIAS) は軽量なので下で参照する。
from realtime_interpreter.translator import DEFAULT_ALIAS, MODEL_PRESETS

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_INTERVAL_SECONDS = 60
SUPPORTED_BACKENDS = ("mlx", "openai")


def _is_windows() -> bool:
    return sys.platform == "win32"


def _list_devices() -> None:
    """`--device` に指定できるデバイスを表示する.

    - Windows: WASAPI loopback で「出力デバイス」を録音対象にするため、出力デバイスを列挙
    - macOS/Linux: 入力デバイス (max_input_channels > 0) を列挙
    """
    devices = sd.query_devices()
    if _is_windows():
        default_out = sd.default.device[1]
        print("Output devices (WASAPI loopback capture targets) for --device:")
        print()
        found = False
        for index, dev in enumerate(devices):
            if dev["max_output_channels"] <= 0:
                continue
            found = True
            marker = "  <- default-out (used when --device omitted)" if index == default_out else ""
            print(f"  {index:>2}: {dev['name']} [out={dev['max_output_channels']}]{marker}")
        if not found:
            print("  (no output-capable devices found)")
        print()
        print("On Windows, system audio is captured via WASAPI loopback on an OUTPUT device.")
        print("Omit --device to loopback the default speaker, or pass an output device name.")
        return

    default_in = sd.default.device[0]
    print("Input devices selectable via --device (index: name [in channels]):")
    print()
    found = False
    for index, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue  # 出力専用デバイスは --device に指定できないので除外
        found = True
        marker = "  <- default-in" if index == default_in else ""
        print(
            f"  {index:>2}: {dev['name']} [in={dev['max_input_channels']}]{marker}"
        )
    if not found:
        print("  (no input-capable devices found)")
    print()
    print("Pass a device name (substring match) to --device to choose the capture input.")
    print("On macOS, capture system audio via BlackHole 2ch (a Multi-Output Device routes")
    print("speaker audio into BlackHole so this program can read it as an input).")


def _check_input_device(device_name: str) -> None:
    """起動時にキャプチャ対象を表示し、必要なら設定を促す (platform 別)."""
    if _is_windows():
        # Windows は WASAPI loopback. 既定スピーカー (or 指定出力) の音を取り込む。
        # 追加設定は不要なので、対象だけ表示して続行。
        if not device_name or device_name == DEVICE_NAME:
            target = "default output device (speaker)"
        else:
            target = repr(device_name)
        print(
            f"Capture (WASAPI loopback): {target}. "
            "Play audio from any app to translate it.",
            file=sys.stderr,
        )
        return

    # macOS/Linux: BlackHole 等の入力デバイスを開く。
    # macOS ではスピーカー出力を Multi-Output Device 経由で BlackHole に流す必要があるため、
    # 現在の出力先が Multi-Output でなければ切替を促す。
    default_out = sd.default.device[1]
    output_name = sd.query_devices(default_out)["name"]
    print(f"Input device: {device_name}", file=sys.stderr)
    if "複数出力" not in output_name and "multi" not in output_name.lower():
        print(
            f"⚠ Current system output is {output_name!r}. "
            "Switch macOS output to a Multi-Output Device including BlackHole 2ch "
            "so audio reaches this program. (Use --no-device-check to skip this prompt.)",
            file=sys.stderr,
        )
        if sys.platform == "darwin":
            try:
                subprocess.run(
                    ["open", "x-apple.systempreferences:com.apple.Sound-Settings.extension"],
                    check=False,
                )
            except Exception:
                pass
        input("  Press Enter when ready: ")


def _parse_args() -> argparse.Namespace:
    # mlx バックエンドは macOS (Apple Silicon) 専用. Windows では mlx 関連オプションを
    # 一切登録しない (= --help に出ない & 渡すと "unrecognized arguments" でエラー)。
    mlx_available = not _is_windows()

    if mlx_available:
        description = (
            "Low-latency simultaneous interpreter. Default: English→Japanese. "
            "Use --source-lang / --target-lang (aliases: --from / --to, -s / -t) "
            "to change languages. Backends: local MLX (Gemma 4) or OpenAI gpt-realtime-translate."
        )
        backend_choices = SUPPORTED_BACKENDS
        backend_default = "mlx"
        backend_help = "Translation backend (default: mlx)"
    else:
        description = (
            "Low-latency simultaneous interpreter (Windows: OpenAI backend only). "
            "Default: English→Japanese. Use --source-lang / --target-lang "
            "(aliases: --from / --to, -s / -t) to change languages."
        )
        # Windows では openai のみ. mlx は選択肢から除外する。
        backend_choices = ("openai",)
        backend_default = "openai"
        backend_help = "Translation backend (Windows: openai only)"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--backend",
        choices=backend_choices,
        default=backend_default,
        help=backend_help,
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--no-device-check",
        action="store_true",
        help=(
            "Skip the startup output-device check / Multi-Output prompt. "
            "Use when you manage audio routing yourself."
        ),
    )
    parser.add_argument(
        "--device",
        default=DEVICE_NAME,
        help=f"Audio input device name (default: {DEVICE_NAME!r})",
    )

    # 共通の言語切替フラグ. ISO 639-1 2文字コード.
    parser.add_argument(
        "--source-lang", "--from", "-s",
        dest="source_lang",
        default=DEFAULT_SOURCE,
        metavar="CODE",
        help=(
            "Source (spoken) language ISO 639-1 code. "
            "Affects prompt for mlx backend; input transcription is auto-detected on openai backend. "
            f"(default: {DEFAULT_SOURCE!r}). Aliases: --from, -s. Use --list-languages."
        ),
    )
    parser.add_argument(
        "--target-lang", "--to", "-t",
        dest="target_lang",
        default=DEFAULT_TARGET,
        metavar="CODE",
        help=(
            "Target (translation) language ISO 639-1 code. "
            "Affects prompt for mlx backend and audio.output.language for openai backend. "
            f"(default: {DEFAULT_TARGET!r}). Aliases: --to, -t. Use --list-languages."
        ),
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List known language codes and exit",
    )

    # mlx 関連オプションは macOS のみ登録. Windows では表示も受付もしない。
    if mlx_available:
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
        "--openai-debounce-ms",
        type=int,
        default=TURN_DEBOUNCE_MS,
        help=(
            "[openai] Commit a chunk after delta has been quiet for this many milliseconds. "
            "Smaller = more frequent (but higher risk of mid-translation commit causing "
            "EN/JA misalignment). Larger = fewer, longer chunks but properly aligned. "
            f"(default: {TURN_DEBOUNCE_MS})"
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
    openai_group.add_argument(
        "--openai-summary-model",
        default=DEFAULT_OPENAI_SUMMARY_MODEL,
        help=(
            "[openai] Chat Completions model used for periodic summaries. "
            "Requires --summary-interval-seconds > 0. Uses the same OPENAI_API_KEY. "
            f"(default: {DEFAULT_OPENAI_SUMMARY_MODEL!r})"
        ),
    )

    parser.add_argument(
        "--summary-interval-seconds",
        type=int,
        default=DEFAULT_SUMMARY_INTERVAL_SECONDS,
        help=(
            "Generate a periodic summary (target language) of the last N seconds. "
            "0 to disable. mlx backend uses local Gemma 4; openai backend uses "
            "Chat Completions (--openai-summary-model). "
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


def _print_languages() -> None:
    print("Known language codes (ISO 639-1):")
    print()
    for code, name in LANGUAGES.items():
        marker = ""
        if code == DEFAULT_SOURCE:
            marker = " (default source)"
        if code == DEFAULT_TARGET:
            marker += " (default target)"
        print(f"  {code:<4} {name}{marker}")
    print()
    print("Pass any code (incl. those not listed) via --source-lang/--from/-s")
    print("or --target-lang/--to/-t. Unknown codes are passed to the backend as-is.")
    print()
    print("Note: gpt-realtime-translate supports ~13 target languages.")
    print("Use the openai backend with an unsupported target → API will error.")


def _build_backend(
    args: argparse.Namespace,
) -> tuple[TranslationBackend, Summarizer | OpenAIChatSummarizer | None]:
    summary_enabled = args.summary_interval_seconds > 0
    src = normalize_language_code(args.source_lang)
    tgt = normalize_language_code(args.target_lang)

    if args.backend == "mlx":
        if args.end_silence_ms <= 0:
            raise SystemExit("error: --end-silence-ms must be positive")
        if args.max_segment_seconds <= 0:
            raise SystemExit("error: --max-segment-seconds must be positive")

        # mlx 系は macOS 専用依存. ここで遅延 import (Windows では到達しない)。
        try:
            from realtime_interpreter.backends.mlx_local import LocalMLXBackend
            from realtime_interpreter.summarizer import Summarizer
            from realtime_interpreter.translator import GemmaAudioTranslator
        except ImportError as e:
            raise SystemExit(
                f"error: mlx backend unavailable ({e}). "
                "The mlx backend requires macOS (Apple Silicon) with mlx-vlm installed. "
                "On Windows/Linux use --backend openai."
            )

        translator = GemmaAudioTranslator(
            model=args.model,
            source_lang=src,
            target_lang=tgt,
        )
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
        summarizer = (
            Summarizer(translator, source_lang=src, target_lang=tgt)
            if summary_enabled
            else None
        )
        return backend, summarizer

    if args.backend == "openai":
        if args.openai_debounce_ms <= 0:
            raise SystemExit("error: --openai-debounce-ms must be positive")
        if args.openai_max_segment_seconds < 0:
            raise SystemExit("error: --openai-max-segment-seconds must be >= 0 (0 disables)")
        backend = OpenAIRealtimeBackend(
            sd_module=sd,
            device_name=args.device,
            model=args.openai_model,
            turn_debounce_ms=args.openai_debounce_ms,
            max_segment_seconds=args.openai_max_segment_seconds,
            source_lang=src,
            target_lang=tgt,
        )
        summarizer = (
            OpenAIChatSummarizer(
                model=args.openai_summary_model,
                source_lang=src,
                target_lang=tgt,
            )
            if summary_enabled
            else None
        )
        return backend, summarizer

    raise SystemExit(f"unknown backend: {args.backend}")


def _emit_settings(args: argparse.Namespace) -> None:
    summary_str = (
        f"every {args.summary_interval_seconds}s"
        if args.summary_interval_seconds > 0
        else "off"
    )
    src = normalize_language_code(args.source_lang)
    tgt = normalize_language_code(args.target_lang)
    lang_label = f"{language_name(src)} ({src}) → {language_name(tgt)} ({tgt})"
    if args.backend == "mlx":
        print(
            f"Backend: mlx | "
            f"Lang: {lang_label} | "
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
        if args.summary_interval_seconds > 0:
            summary_label = (
                f"every {args.summary_interval_seconds}s ({args.openai_summary_model})"
            )
        else:
            summary_label = "off"
        print(
            f"Backend: openai ({args.openai_model}) | "
            f"Lang: {lang_label} | "
            f"debounce={args.openai_debounce_ms}ms, "
            f"max_segment={max_seg_str} | "
            f"summary={summary_label}",
            file=sys.stderr,
        )


def _submit_summary_task(
    executor: ThreadPoolExecutor,
    summarizer: Summarizer | OpenAIChatSummarizer,
    source_buffer: list[tuple[float, str]],
    since_offset: float,
    until_offset: float,
    duration_seconds: int,
    session_logger: SessionLogger,
    renderer: StreamingRenderer,
) -> None:
    """要約タスクをバックグラウンド executor に投げる.

    Submit 後すぐに return するためメインループ (翻訳パイプライン) はブロックしない.
    ワーカー側で完了時に renderer / session_logger に直接書き込む.
    Rich Live は内部ロックで thread-safe なので別スレッドからの emit_summary は問題ない。
    """
    items = [text for ts, text in source_buffer if ts >= since_offset]
    if not items:
        return
    src_concat = " ".join(items)

    def _worker() -> None:
        try:
            summary = summarizer.summarize(src_concat, duration_seconds)
        except Exception:
            logger.exception("summary task failed")
            return
        if not summary.text:
            # summarizer 側で原因 (空応答 / max_completion_tokens 不足 等) を WARN ログ済み
            return
        ts = format_offset(until_offset)
        renderer.emit_summary(ts, summary.text)
        session_logger.log_summary(ts, summary.text)

    executor.submit(_worker)


def main() -> None:
    args = _parse_args()
    # --list-models は mlx 専用フラグ. Windows では未登録なので getattr で安全に参照。
    if getattr(args, "list_models", False):
        _print_model_presets()
        return
    if args.list_languages:
        _print_languages()
        return
    if args.list_devices:
        _list_devices()
        return

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.summary_interval_seconds < 0:
        print("error: --summary-interval-seconds must be >= 0", file=sys.stderr)
        sys.exit(2)

    if args.no_device_check:
        print(f"Input device: {args.device}", file=sys.stderr)
    else:
        _check_input_device(args.device)

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

    # 要約用バッファ: 過去 N 秒の source 言語転写を (offset, text) で保持
    source_buffer: list[tuple[float, str]] = []
    summary_last_offset = 0.0
    summary_interval = float(args.summary_interval_seconds)

    # 要約はメインループから切り離して別スレッドで実行 (1〜3秒 API 待ちで翻訳が止まらないように)。
    # max_workers=1: 同時に複数の要約が走らないようにシリアル化 (連続発火しても順次処理).
    summary_executor: ThreadPoolExecutor | None = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="summary")
        if summarizer is not None
        else None
    )

    try:
        with backend, StreamingRenderer() as renderer:
            renderer.update_status("● Listening...")
            for seg in backend.stream_segments():
                ts = format_offset(seg.start_offset_seconds)
                if seg.is_partial:
                    # in-place 更新. ログ・要約バッファには触らない。
                    renderer.update_current(ts, seg.source, seg.target)
                    continue

                # 確定セグメント: 永続表示・ログ・要約バッファ反映
                renderer.commit(ts, seg.source, seg.target)
                session_logger.log_segment(ts, seg.source, seg.target)
                if summarizer is not None and seg.source.strip():
                    source_buffer.append(
                        (seg.start_offset_seconds, seg.source.strip())
                    )

                segment_end = seg.start_offset_seconds + seg.duration_seconds
                if (
                    summarizer is not None
                    and summary_executor is not None
                    and segment_end - summary_last_offset >= summary_interval > 0
                ):
                    # 非同期投入. メインループは即座に次セグメント処理へ戻る.
                    _submit_summary_task(
                        summary_executor,
                        summarizer,
                        list(source_buffer),  # worker に渡すスナップショット
                        summary_last_offset,
                        segment_end,
                        int(summary_interval),
                        session_logger,
                        renderer,
                    )
                    summary_last_offset = segment_end
                    cutoff = segment_end - summary_interval * 2
                    source_buffer[:] = [
                        (ts_, t) for ts_, t in source_buffer if ts_ >= cutoff
                    ]

                renderer.update_status("● Listening...")

    except KeyboardInterrupt:
        pass
    finally:
        if summary_executor is not None:
            # 進行中の要約タスクは完了まで待つ (まだ表示されてない要約を取りこぼさない).
            summary_executor.shutdown(wait=True)
        session_logger.close()
        print(f"\nLog: {session_logger.path}", file=sys.stderr)


if __name__ == "__main__":
    main()
