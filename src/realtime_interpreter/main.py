"""CLI エントリポイント.

BlackHole 2ch から音声をキャプチャし、選択したバックエンドで source 言語の転写と
target 言語の訳を生成する。バックエンドは TranslatedSegment をストリームし、
`is_partial=True` (進行中) と `is_partial=False` (確定) を区別して出力する。

- 進行中セグメント (OpenAI バックエンドの delta): Rich Live で in-place 更新
- 確定セグメント: append-only で永続表示 + ログ + 要約バッファ反映

バックエンド:
    openai-realtime  OpenAI gpt-realtime-translate (WebSocket). delta 単位で in-place 更新 (既定)
    openai-chat      OpenAI-compatible Chat Completions REST. ローカル VAD で確定単位出力
    mlx              ローカル MLX (Gemma 4 + mlx-vlm, macOS のみ). 確定単位で出力

出力形式:
    [mm:ss] <source 転写>
    [mm:ss] <target 訳>
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 注意: sounddevice はモジュール先頭で import しない。
# sounddevice は PortAudio をロードするが、ARM64 Windows ではその DLL の依存が欠けて
# import 自体が失敗する (error 0x7e)。Windows の音声処理は PyAudioWPatch で完結するため
# sounddevice は不要。mlx 経路と macOS/Linux の openai 入力経路でのみ遅延 import する。


def _load_sounddevice():
    """sounddevice を遅延 import する (macOS/Linux 経路専用).

    Windows では呼ばれない想定。万一呼ばれて失敗した場合は分かりやすいエラーにする。
    """
    try:
        import sounddevice as sd
    except OSError as e:
        raise SystemExit(
            f"error: failed to load sounddevice/PortAudio ({e}). "
            "On Windows use --backend openai-realtime (which uses PyAudioWPatch, not sounddevice)."
        )
    return sd

from realtime_interpreter.audio import (
    DEVICE_NAME,
    END_SILENCE_MS,
    MAX_SEGMENT_SECONDS,
    _looks_like_index,
    _resolve_windows_capture_device,
    _wasapi_input_devices,
    find_device,
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
from realtime_interpreter.backends.openai_chat import (
    DEFAULT_OPENAI_CHAT_BASE_URL,
    DEFAULT_OPENAI_CHAT_MAX_TOKENS,
    DEFAULT_OPENAI_CHAT_MODEL,
    DEFAULT_OPENAI_CHAT_TEMPERATURE,
    DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS,
    OpenAIChatAudioTranslator,
    OpenAIChatBackend,
    OpenAIChatCompatibleSummarizer,
)
from realtime_interpreter.renderer import StreamingRenderer
from realtime_interpreter.session_logger import SessionLogger, format_offset
from realtime_interpreter.summarizer import (
    DEFAULT_OPENAI_SUMMARY_MODEL,
    OpenAIChatSummarizer,
)
# NOTE: mlx バックエンド固有の import (LocalMLXBackend / GemmaAudioTranslator / Summarizer)
# は _build_backend() の mlx 分岐内で遅延 import する。これにより mlx 未インストールの
# Windows/Linux でも mlx 以外のバックエンドが動く。
# 定数 (DEVICE_NAME 等, MODEL_PRESETS, DEFAULT_ALIAS) は軽量なので下で参照する。
from realtime_interpreter.translator import DEFAULT_ALIAS, MODEL_PRESETS

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_INTERVAL_SECONDS = 60
# セッション最大時間 (コスト安全弁). 既定 24 時間. 0 で無制限。
# 長時間バックエンド (gemini/openai realtime) は再接続しながら走り続けるため、
# 無人運用での課金暴走を防ぐ上限として設ける。
DEFAULT_MAX_SESSION_SECONDS = 24 * 60 * 60  # 86400
SUPPORTED_BACKENDS = ("openai-realtime", "openai-chat", "gemini-realtime", "mlx")

DEFAULT_GEMINI_MODEL = "models/gemini-3.5-live-translate-preview"
# 要約は軽いテキストタスク. flash-lite は無料枠 RPD が大きく (実測 500 RPD)、
# 60秒間隔の長時間会議でも枯渇しにくいため既定に採用。
# (gemini-3.5-flash は無料枠 RPD=20 と少なく、20分程度で 429 に達する)
DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_DEBOUNCE_MS = 800
DEFAULT_GEMINI_MAX_SEGMENT_SECONDS = 8.0


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_apple_silicon() -> bool:
    """mlx が動作可能な環境 (Apple Silicon の macOS) か判定する.

    mlx / mlx-vlm は Apple Silicon (Metal + ユニファイドメモリ) 専用で、Intel Mac では
    動作しない。Rosetta 下の x86_64 Python も machine()=="x86_64" となり mlx の wheel が
    無いため、ここで False になるのは正しい (mlx を提供しない)。
    """
    return _is_macos() and platform.machine() == "arm64"


def _resolve_backend_choices(
    mlx_available: bool, openai_chat_available: bool
) -> tuple[str, ...]:
    """対応バックエンドの choices をプラットフォーム可用性から決定する.

    - Apple Silicon macOS: 4種 (mlx 含む)
    - Intel macOS / Windows: 3種 (openai-realtime, openai-chat, gemini-realtime)
    - Linux など: 2種 (openai-realtime, gemini-realtime)
    """
    if mlx_available:
        return ("openai-realtime", "openai-chat", "gemini-realtime", "mlx")
    if openai_chat_available:
        return ("openai-realtime", "openai-chat", "gemini-realtime")
    return ("openai-realtime", "gemini-realtime")


def _env_truthy(name: str) -> bool:
    """環境変数が truthy (1/true/yes/on) かを判定する."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _enable_system_certs() -> None:
    """ランタイムの TLS 検証を OS の証明書ストアに切り替える (--system-certs).

    truststore を ssl モジュールへ inject することで、stdlib ssl を使う urllib /
    websocket-client と、httpx を使う openai SDK のすべてが OS ストア (Windows 証明書
    ストア / macOS キーチェーン / Linux システムCA) でルート CA を検証するようになる。
    他の TLS 利用より前 (起動直後) に呼ぶこと。truststore 未導入時は警告のみで継続する。
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        logger.info("system certificate store enabled (truststore.inject_into_ssl)")
    except Exception as e:  # noqa: BLE001
        print(
            f"warning: --system-certs requested but could not enable OS trust store "
            f"({e}). Falling back to bundled CAs. Install with `uv sync`.",
            file=sys.stderr,
        )


def _resolve_device_arg(device: str | None) -> str:
    """`--device` (番号のみ) を検証し、省略時はプラットフォーム既定 (DEVICE_NAME) を返す.

    番号文字列はそのまま返す。None は DEVICE_NAME(センチネル: Win=既定出力 loopback /
    macOS=BlackHole)。数字でない値が来たら SystemExit (名前指定は廃止)。
    """
    if device is None:
        return DEVICE_NAME
    if not _looks_like_index(device):
        raise SystemExit(
            "error: --device must be an index number from --list-devices "
            "(device names are no longer accepted)."
        )
    return device


def _strip_loopback_suffix(name: str) -> str:
    """表示用に PyAudioWPatch の '[Loopback]' 接尾辞を除く (出力名を素で見せる)."""
    return name.replace("[Loopback]", "").rstrip()


def _format_windows_device_list(
    loopbacks: list[dict],
    mics: list[dict],
    default_out_name: str,
    default_in_index: int,
) -> list[str]:
    """Windows の --list-devices 表示行を組み立てる (出力 + 入力の2セクションを常に表示).

    どちらも `--device <番号>` で選ぶ。番号は出力/入力で一意なので自動判定される。
    """
    lines: list[str] = ["Audio devices (Windows, WASAPI)", ""]

    lines.append("Output devices — system audio via loopback (default). Select with --device <index>:")
    lines.append("")
    if not loopbacks:
        lines.append("  (no WASAPI loopback devices found)")
    for lb in loopbacks:
        name = _strip_loopback_suffix(lb["name"])
        is_default = bool(default_out_name) and default_out_name in lb["name"]
        marker = "   <- default (used when --device omitted)" if is_default else ""
        lines.append(
            f"  [{lb['index']:>2}] {name}  rate={int(lb['defaultSampleRate'])}{marker}"
        )
    lines.append("")

    lines.append("Input devices — microphones. Select with --device <index>:")
    lines.append("")
    if not mics:
        lines.append("  (no WASAPI microphone input devices found)")
    for m in mics:
        marker = "   <- default mic" if m["index"] == default_in_index else ""
        lines.append(
            f"  [{m['index']:>2}] {m['name']}  in={m['maxInputChannels']}  "
            f"rate={int(m['defaultSampleRate'])}{marker}"
        )
    lines.append("")

    lines.append("Tips:")
    lines.append("  - Pass a device index to --device. A number auto-selects output (loopback)")
    lines.append("    or microphone; numbers are unique across both lists.")
    lines.append("  - Omit --device to capture the default output (speaker).")
    return lines


def _windows_capture_label(device_name: str) -> tuple[str, int, str] | None:
    """Windows のキャプチャ対象を解決し (種別, index, 名前) を返す (起動表示用).

    種別は "microphone" / "WASAPI loopback"。解決に失敗したら None
    (呼び出し側でフォールバック表示する)。
    """
    try:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            device, is_mic = _resolve_windows_capture_device(pa, device_name)
            kind = "microphone" if is_mic else "WASAPI loopback"
            return kind, int(device["index"]), _strip_loopback_suffix(device["name"])
        finally:
            pa.terminate()
    except Exception:
        return None


def _macos_input_label(device_name: str) -> str | None:
    """macOS/Linux のキャプチャ対象を解決し "[index] name" を返す (起動表示用).

    解決に失敗したら None (呼び出し側でフォールバック表示する)。
    """
    try:
        sd = _load_sounddevice()
        idx = find_device(device_name, sd)
        name = sd.query_devices(idx)["name"]
        return f"[{idx}] {name}"
    except Exception:
        return None


def _list_devices() -> None:
    """`--device <番号>` に指定できるデバイスを表示する.

    - Windows: WASAPI の出力(loopback 取り込み対象)と入力(マイク)を両方列挙
      (sounddevice は使わない — ARM64 で DLL がロードできないため)
    - macOS/Linux: sounddevice で入力デバイス (max_input_channels > 0) を列挙
    """
    if _is_windows():
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out_index = wasapi_info["defaultOutputDevice"]
            default_out_name = pa.get_device_info_by_index(default_out_index)["name"]
            loopbacks = list(pa.get_loopback_device_info_generator())
            mics = _wasapi_input_devices(pa)
            default_in_index = wasapi_info.get("defaultInputDevice", -1)
            for line in _format_windows_device_list(
                loopbacks, mics, default_out_name, default_in_index
            ):
                print(line)
        finally:
            pa.terminate()
        return

    sd = _load_sounddevice()
    devices = sd.query_devices()
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
    print("Pass a device index number above to --device to choose the capture input.")
    print("Omit --device to use the default (BlackHole 2ch).")
    print("On macOS, capture system audio via BlackHole 2ch (a Multi-Output Device routes")
    print("speaker audio into BlackHole so this program can read it as an input).")


def _print_capture_target(device_name: str) -> None:
    """起動時にキャプチャ対象 (番号 + 名前) を platform に応じて表示する.

    番号で指定した場合も、解決したデバイスの「[index] 名前」を併記する。解決に
    失敗した場合は簡易表示にフォールバックする (起動表示で落とさない)。
    """
    if _is_windows():
        resolved = _windows_capture_label(device_name)
        if resolved is not None:
            kind, idx, name = resolved
            hint = (
                "Speak into the mic to translate."
                if kind == "microphone"
                else "Play audio from any app to translate it."
            )
            print(f"Capture ({kind}): [{idx}] {name}. {hint}", file=sys.stderr)
        else:
            target = (
                "default output device (speaker)"
                if (not device_name or device_name == DEVICE_NAME)
                else f"device #{device_name}"
            )
            print(f"Capture (WASAPI loopback): {target}.", file=sys.stderr)
    else:
        print(f"Input device: {_macos_input_label(device_name) or device_name}", file=sys.stderr)


def _check_input_device(device_name: str) -> None:
    """起動時にキャプチャ対象を表示し、必要なら設定を促す (platform 別)."""
    if _is_windows():
        # Windows は WASAPI loopback. 追加設定は不要なので、対象だけ表示して続行。
        _print_capture_target(device_name)
        return

    # macOS/Linux: BlackHole 等の入力デバイスを開く。
    # macOS ではスピーカー出力を Multi-Output Device 経由で BlackHole に流す必要があるため、
    # 現在の出力先が Multi-Output でなければ切替を促す。
    sd = _load_sounddevice()
    default_out = sd.default.device[1]
    output_name = sd.query_devices(default_out)["name"]
    print(f"Input device: {_macos_input_label(device_name) or device_name}", file=sys.stderr)
    if "複数出力" not in output_name and "multi" not in output_name.lower():
        print(
            f"⚠ Current system output is {output_name!r}. "
            "Switch macOS output to a Multi-Output Device including BlackHole 2ch "
            "so audio reaches this program. (This prompt appears only with --device-check.)",
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
    # mlx バックエンドは macOS (Apple Silicon) 専用. mlx が使えない環境では
    # mlx モデル関連オプションを登録しない (= --help に出ない & 渡すと
    # unrecognized arguments).
    # mlx は Apple Silicon の macOS 専用 (Intel Mac では mlx/mlx-vlm が動作しない)。
    mlx_available = _is_apple_silicon()
    openai_chat_available = _is_macos() or _is_windows()
    backend_choices = _resolve_backend_choices(mlx_available, openai_chat_available)
    backend_default = "openai-realtime"
    backend_help = "Translation backend (default: openai-realtime)"

    if mlx_available:
        description = (
            "Low-latency simultaneous interpreter. Default: English→Japanese. "
            "Use --source-lang / --target-lang (aliases: --from / --to, -s / -t) "
            "to change languages. Backends: OpenAI gpt-realtime-translate, "
            "OpenAI-compatible Chat Completions, local MLX (Gemma 4), "
            "or Gemini Multimodal Live API."
        )
    elif openai_chat_available:
        # Intel macOS / Windows: mlx 非対応。残り 3 種。
        description = (
            "Low-latency simultaneous interpreter (OpenAI realtime, "
            "OpenAI-compatible Chat Completions, or Gemini Live API). "
            "On macOS the local MLX backend requires Apple Silicon. "
            "Default: English→Japanese. Use --source-lang / --target-lang "
            "(aliases: --from / --to, -s / -t) to change languages."
        )
    else:
        description = (
            "Low-latency simultaneous interpreter. "
            "Default: English→Japanese. Use --source-lang / --target-lang "
            "(aliases: --from / --to, -s / -t) to change languages."
        )

    # usage/help の折り返し幅を 80 に固定 (ターミナル幅に依存させない)。
    # 長いオプションでも 1 行が 80 文字を超えないよう、メタ変数名も短くしている。
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, width=80),
    )
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
        "--device-check",
        action="store_true",
        help=(
            "Check the output-device routing at startup and prompt to switch to a "
            "Multi-Output Device if needed (macOS). Skipped by default."
        ),
    )
    # --device は番号(--list-devices のインデックス)のみ。省略時はプラットフォーム既定
    # (Windows=既定スピーカーの loopback / macOS=BlackHole)。名前指定は廃止。
    if _is_windows():
        device_help = (
            "Capture device index number from --list-devices. A number auto-selects "
            "output (loopback) or microphone. Omit to capture the default output (speaker)."
        )
    else:
        device_help = (
            "Capture device index number from --list-devices. "
            "Omit to use the default (BlackHole 2ch)."
        )
    parser.add_argument(
        "--device",
        default=None,
        metavar="INDEX",
        help=device_help,
    )

    # 共通の言語切替フラグ. ISO 639-1 2文字コード.
    parser.add_argument(
        "--source-lang", "--from", "-s",
        dest="source_lang",
        default=DEFAULT_SOURCE,
        metavar="CODE",
        help=(
            "Source (spoken) language ISO 639-1 code. "
            "Affects prompt for mlx/openai-chat backends when available; input "
            "transcription is auto-detected on openai realtime backend. "
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
            "Affects prompt for mlx/openai-chat backends when available and "
            "audio.output.language for openai realtime backend. "
            f"(default: {DEFAULT_TARGET!r}). Aliases: --to, -t. Use --list-languages."
        ),
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List known language codes and exit",
    )

    # --help でのバックエンド別グループ表示順は openai-realtime → openai-chat → mlx。
    # オプションは --openai-realtime-* (正式) と --openai-rt-* (短縮) の 2 表記を受け付ける。
    # dest は従来名 (openai_model 等) を維持し、下流コードの参照を変えない。
    openai_group = parser.add_argument_group("openai-realtime backend")
    openai_group.add_argument(
        "--openai-realtime-model",
        "--openai-rt-model",
        dest="openai_model",
        metavar="MODEL",
        default=DEFAULT_OPENAI_MODEL,
        help=f"[openai-realtime] Realtime model id (default: {DEFAULT_OPENAI_MODEL!r})",
    )
    openai_group.add_argument(
        "--openai-realtime-debounce-ms",
        "--openai-rt-debounce-ms",
        dest="openai_debounce_ms",
        metavar="MS",
        type=int,
        default=TURN_DEBOUNCE_MS,
        help=(
            "[openai-realtime] Commit a chunk after delta has been quiet for this many "
            "milliseconds. Smaller = more frequent (but higher risk of mid-translation "
            "commit causing EN/JA misalignment). Larger = fewer, longer chunks but "
            f"properly aligned. (default: {TURN_DEBOUNCE_MS})"
        ),
    )
    openai_group.add_argument(
        "--openai-realtime-max-segment-seconds",
        "--openai-rt-max-segment-seconds",
        dest="openai_max_segment_seconds",
        metavar="SECS",
        type=float,
        default=OPENAI_MAX_SEGMENT_SECONDS,
        help=(
            "[openai-realtime] Force-commit a chunk after this many seconds of continuous "
            "accumulation, even when delta is still arriving. Prevents huge single "
            "chunks during long monologues. 0 to disable (debounce only). "
            f"(default: {OPENAI_MAX_SEGMENT_SECONDS})"
        ),
    )
    openai_group.add_argument(
        "--openai-realtime-summary-model",
        "--openai-rt-summary-model",
        dest="openai_summary_model",
        metavar="MODEL",
        default=DEFAULT_OPENAI_SUMMARY_MODEL,
        help=(
            "[openai-realtime] Chat Completions model used for periodic summaries. "
            "Requires --summary-interval-seconds > 0. Uses the same OPENAI_API_KEY. "
            f"(default: {DEFAULT_OPENAI_SUMMARY_MODEL!r})"
        ),
    )
    openai_group.add_argument(
        "--openai-realtime-api-key",
        "--openai-rt-api-key",
        dest="openai_api_key",
        metavar="KEY",
        default=None,
        help=(
            "[openai-realtime] OpenAI API key. Defaults to the OPENAI_API_KEY "
            "environment variable (recommended; avoids leaking the key into "
            "shell history)."
        ),
    )

    if openai_chat_available:
        openai_chat_group = parser.add_argument_group("openai-chat backend")
        openai_chat_group.add_argument(
            "--openai-chat-base-url",
            metavar="URL",
            default=DEFAULT_OPENAI_CHAT_BASE_URL,
            help=(
                "[openai-chat] OpenAI-compatible base URL. "
                f"(default: {DEFAULT_OPENAI_CHAT_BASE_URL!r})"
            ),
        )
        openai_chat_group.add_argument(
            "--openai-chat-model",
            metavar="MODEL",
            default=DEFAULT_OPENAI_CHAT_MODEL,
            help=(
                "[openai-chat] Chat Completions model id. "
                f"(default: {DEFAULT_OPENAI_CHAT_MODEL!r})"
            ),
        )
        openai_chat_group.add_argument(
            "--openai-chat-api-key",
            metavar="TOKEN",
            default=None,
            help=(
                "[openai-chat] Bearer token. Defaults to OPENAI_CHAT_API_KEY, "
                "then OPENAI_API_KEY, then 'ollama'."
            ),
        )
        openai_chat_group.add_argument(
            "--openai-chat-timeout-seconds",
            metavar="SECS",
            type=float,
            default=DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS,
            help=(
                "[openai-chat] Per-request timeout in seconds. "
                f"(default: {DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS:g})"
            ),
        )
        openai_chat_group.add_argument(
            "--openai-chat-max-tokens",
            metavar="N",
            type=int,
            default=DEFAULT_OPENAI_CHAT_MAX_TOKENS,
            help=(
                "[openai-chat] Max output tokens per audio segment. "
                f"(default: {DEFAULT_OPENAI_CHAT_MAX_TOKENS})"
            ),
        )
        openai_chat_group.add_argument(
            "--openai-chat-temperature",
            metavar="TEMP",
            type=float,
            default=DEFAULT_OPENAI_CHAT_TEMPERATURE,
            help=(
                "[openai-chat] Sampling temperature. "
                f"(default: {DEFAULT_OPENAI_CHAT_TEMPERATURE:g})"
            ),
        )

    gemini_group = parser.add_argument_group("gemini-realtime backend")
    gemini_group.add_argument(
        "--gemini-realtime-model",
        "--gemini-rt-model",
        dest="gemini_model",
        metavar="MODEL",
        default=DEFAULT_GEMINI_MODEL,
        help=f"[gemini-realtime] Gemini Live model id (default: {DEFAULT_GEMINI_MODEL!r})",
    )
    gemini_group.add_argument(
        "--gemini-realtime-api-key",
        "--gemini-rt-api-key",
        dest="gemini_api_key",
        metavar="KEY",
        default=None,
        help=(
            "[gemini-realtime] Gemini API key. Defaults to the GEMINI_API_KEY "
            "environment variable."
        ),
    )
    gemini_group.add_argument(
        "--gemini-realtime-summary-model",
        "--gemini-rt-summary-model",
        dest="gemini_summary_model",
        metavar="MODEL",
        default=DEFAULT_GEMINI_SUMMARY_MODEL,
        help=(
            "[gemini-realtime] Model used for periodic summaries. "
            "Requires --summary-interval-seconds > 0. Uses the same GEMINI_API_KEY. "
            f"(default: {DEFAULT_GEMINI_SUMMARY_MODEL!r})"
        ),
    )
    gemini_group.add_argument(
        "--gemini-realtime-debounce-ms",
        "--gemini-rt-debounce-ms",
        dest="gemini_debounce_ms",
        type=int,
        metavar="MS",
        default=DEFAULT_GEMINI_DEBOUNCE_MS,
        help=(
            f"[gemini-realtime] Silence duration in milliseconds to commit "
            f"current segment. (default: {DEFAULT_GEMINI_DEBOUNCE_MS}ms)"
        ),
    )
    gemini_group.add_argument(
        "--gemini-realtime-max-segment-seconds",
        "--gemini-rt-max-segment-seconds",
        dest="gemini_max_segment_seconds",
        type=float,
        metavar="SECONDS",
        default=DEFAULT_GEMINI_MAX_SEGMENT_SECONDS,
        help=(
            "[gemini-realtime] Force-commit segment duration to split long monologue "
            f"chunks. 0 to disable. (default: {DEFAULT_GEMINI_MAX_SEGMENT_SECONDS}s)"
        ),
    )

    # mlx モデル関連オプションは macOS のみ登録. Windows では表示も受付もしない。
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

    # mlx / openai-chat が共有するローカル VAD 区切り設定 (両者ともローカルで発話区切り)。
    if mlx_available or openai_chat_available:
        local_vad_group = parser.add_argument_group(
            "local VAD segmentation (mlx / openai-chat)"
        )
        local_vad_group.add_argument(
            "--end-silence-ms",
            metavar="MS",
            type=int,
            default=END_SILENCE_MS,
            help=(
                "[mlx/openai-chat] Silence (ms) that ends a speech segment. "
                "Lower = smaller chunks, faster output, more risk of mid-sentence cuts. "
                f"(default: {END_SILENCE_MS})"
            ),
        )
        local_vad_group.add_argument(
            "--max-segment-seconds",
            metavar="SECS",
            type=float,
            default=MAX_SEGMENT_SECONDS,
            help=(
                "[mlx/openai-chat] Hard cap (seconds) for a single segment when "
                "there is no silence. "
                f"Lower = smaller chunks for continuous speech. (default: {MAX_SEGMENT_SECONDS})"
            ),
        )

    parser.add_argument(
        "--summary-interval-seconds",
        metavar="SECS",
        type=int,
        default=DEFAULT_SUMMARY_INTERVAL_SECONDS,
        help=(
            "Generate a periodic summary (target language) of the last N seconds. "
            "0 to disable. mlx backend uses local Gemma 4; openai realtime uses "
            "Chat Completions (--openai-summary-model); openai-chat, when "
            "available, uses the same OpenAI-compatible endpoint. "
            f"(default: {DEFAULT_SUMMARY_INTERVAL_SECONDS})"
        ),
    )

    parser.add_argument(
        "--max-session-seconds",
        metavar="MAX",
        type=int,
        default=DEFAULT_MAX_SESSION_SECONDS,
        help=(
            "Stop automatically after this many seconds (cost safety valve for "
            "unattended long sessions). 0 = unlimited. "
            f"(default: {DEFAULT_MAX_SESSION_SECONDS} = 24h)"
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
    # OS の証明書ストアで TLS を検証する (企業 TLS インターセプト対応)。uv の --system-certs
    # と同形式の opt-in。env REALTIME_INTERPRETER_SYSTEM_CERTS が truthy なら既定 on。
    parser.add_argument(
        "--system-certs",
        action="store_true",
        default=_env_truthy("REALTIME_INTERPRETER_SYSTEM_CERTS"),
        help=(
            "Verify TLS using the OS certificate store (Windows store / macOS Keychain "
            "/ Linux system CAs) instead of the bundled CAs. Needed behind corporate TLS "
            "interception. Env: REALTIME_INTERPRETER_SYSTEM_CERTS=1."
        ),
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
) -> tuple[
    TranslationBackend,
    Summarizer | OpenAIChatSummarizer | OpenAIChatCompatibleSummarizer | None,
]:
    summary_enabled = args.summary_interval_seconds > 0
    src = normalize_language_code(args.source_lang)
    tgt = normalize_language_code(args.target_lang)

    # sounddevice は Windows では import しない (ARM64 で PortAudio DLL がロードできず、
    # かつ Windows の各バックエンドは PyAudioWPatch ベースのキャプチャを使うため不要)。
    # Windows では sd=None を渡す。各バックエンドの Windows 経路は sd を参照しない。
    sd = None if _is_windows() else _load_sounddevice()

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
                "The mlx backend requires macOS on Apple Silicon (M-series) with mlx-vlm "
                "installed; it does not run on Intel Macs. "
                "On Intel macOS or Windows use --backend openai-realtime or --backend "
                "openai-chat; on Linux use --backend openai-realtime."
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

    if args.backend == "openai-realtime":
        if args.openai_debounce_ms <= 0:
            raise SystemExit("error: --openai-debounce-ms must be positive")
        if args.openai_max_segment_seconds < 0:
            raise SystemExit("error: --openai-max-segment-seconds must be >= 0 (0 disables)")
        backend = OpenAIRealtimeBackend(
            sd_module=sd,
            device_name=args.device,
            api_key=args.openai_api_key,
            model=args.openai_model,
            turn_debounce_ms=args.openai_debounce_ms,
            max_segment_seconds=args.openai_max_segment_seconds,
            source_lang=src,
            target_lang=tgt,
        )
        summarizer = (
            OpenAIChatSummarizer(
                model=args.openai_summary_model,
                api_key=args.openai_api_key,
                source_lang=src,
                target_lang=tgt,
            )
            if summary_enabled
            else None
        )
        return backend, summarizer

    if args.backend == "gemini-realtime":
        if args.gemini_debounce_ms <= 0:
            raise SystemExit("error: --gemini-debounce-ms must be positive")
        if args.gemini_max_segment_seconds < 0:
            raise SystemExit("error: --gemini-max-segment-seconds must be >= 0 (0 disables)")

        from realtime_interpreter.backends.gemini_realtime import GeminiRealtimeBackend
        from realtime_interpreter.summarizer import GeminiRESTSummarizer

        backend = GeminiRealtimeBackend(
            sd_module=sd,
            device_name=args.device,
            api_key=args.gemini_api_key,
            model=args.gemini_model,
            turn_debounce_ms=args.gemini_debounce_ms,
            max_segment_seconds=args.gemini_max_segment_seconds,
            source_lang=src,
            target_lang=tgt,
        )
        summarizer = (
            GeminiRESTSummarizer(
                model=args.gemini_summary_model,
                api_key=args.gemini_api_key,
                source_lang=src,
                target_lang=tgt,
            )
            if summary_enabled
            else None
        )
        return backend, summarizer

    if args.backend == "openai-chat":
        if args.end_silence_ms <= 0:
            raise SystemExit("error: --end-silence-ms must be positive")
        if args.max_segment_seconds <= 0:
            raise SystemExit("error: --max-segment-seconds must be positive")
        if args.openai_chat_timeout_seconds <= 0:
            raise SystemExit("error: --openai-chat-timeout-seconds must be positive")
        if args.openai_chat_max_tokens <= 0:
            raise SystemExit("error: --openai-chat-max-tokens must be positive")
        if args.openai_chat_temperature < 0:
            raise SystemExit("error: --openai-chat-temperature must be >= 0")

        translator = OpenAIChatAudioTranslator(
            model=args.openai_chat_model,
            base_url=args.openai_chat_base_url,
            api_key=args.openai_chat_api_key,
            timeout_seconds=args.openai_chat_timeout_seconds,
            max_tokens=args.openai_chat_max_tokens,
            temperature=args.openai_chat_temperature,
            source_lang=src,
            target_lang=tgt,
        )
        try:
            backend = OpenAIChatBackend(
                sd_module=sd,
                device_name=args.device,
                translator=translator,
                end_silence_ms=args.end_silence_ms,
                max_segment_seconds=args.max_segment_seconds,
            )
        except ImportError as e:
            raise SystemExit(
                f"error: openai-chat backend unavailable ({e}). "
                "It requires local VAD dependencies for audio segmentation."
            )
        summarizer = (
            OpenAIChatCompatibleSummarizer(
                model=args.openai_chat_model,
                base_url=args.openai_chat_base_url,
                api_key=args.openai_chat_api_key,
                timeout_seconds=args.openai_chat_timeout_seconds,
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
    elif args.backend == "openai-realtime":
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
            f"Backend: openai-realtime ({args.openai_model}) | "
            f"Lang: {lang_label} | "
            f"debounce={args.openai_debounce_ms}ms, "
            f"max_segment={max_seg_str} | "
            f"summary={summary_label}",
            file=sys.stderr,
        )
    elif args.backend == "gemini-realtime":
        if args.summary_interval_seconds > 0:
            summary_label = (
                f"every {args.summary_interval_seconds}s ({args.gemini_summary_model})"
            )
        else:
            summary_label = "off"
        max_seg_str = (
            f"{args.gemini_max_segment_seconds}s"
            if args.gemini_max_segment_seconds > 0
            else "off"
        )
        print(
            f"Backend: gemini-realtime ({args.gemini_model}) | "
            f"Lang: {lang_label} | "
            f"debounce={args.gemini_debounce_ms}ms, "
            f"max_segment={max_seg_str} | "
            f"summary={summary_label}",
            file=sys.stderr,
        )
    elif args.backend == "openai-chat":
        print(
            f"Backend: openai-chat ({args.openai_chat_model}) | "
            f"Base URL: {args.openai_chat_base_url} | "
            f"Lang: {lang_label} | "
            f"VAD: end_silence={args.end_silence_ms}ms, "
            f"max_segment={args.max_segment_seconds}s | "
            f"summary={summary_str}",
            file=sys.stderr,
        )


class _SummaryState:
    """直近の累積要約をスレッドセーフに保持するホルダー.

    要約ワーカー (max_workers=1 でシリアル) が完了時に書き込み、次回ワーカーが
    開始時に読み出してローリング要約 (prev_summary) として引き継ぐ。
    """

    def __init__(self) -> None:
        self._text = ""
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            return self._text

    def set(self, text: str) -> None:
        with self._lock:
            self._text = text


def _apply_translation_context(backend: TranslationBackend, summary: str) -> None:
    """要約をリアルタイム翻訳の文脈として backend へ渡す (対応 backend のみ).

    openai-chat / mlx は `update_context` を実装しており、次以降の翻訳プロンプトに
    「参照文脈」として要約を前置きする。openai-realtime / gemini-realtime は
    実装しない (要約のローリングのみ対象)。
    """
    update = getattr(backend, "update_context", None)
    if callable(update):
        try:
            update(summary)
        except Exception:
            logger.exception("update_context failed")


def _submit_summary_task(
    executor: ThreadPoolExecutor,
    summarizer: Summarizer | OpenAIChatSummarizer | OpenAIChatCompatibleSummarizer,
    source_buffer: list[tuple[float, str]],
    since_offset: float,
    until_offset: float,
    duration_seconds: int,
    session_logger: SessionLogger,
    renderer: StreamingRenderer,
    summary_state: _SummaryState,
    backend: TranslationBackend,
) -> None:
    """要約タスクをバックグラウンド executor に投げる.

    Submit 後すぐに return するためメインループ (翻訳パイプライン) はブロックしない.
    ワーカー側で完了時に renderer / session_logger に直接書き込む.
    Rich Live は内部ロックで thread-safe なので別スレッドからの emit_summary は問題ない。

    ローリング要約: ワーカー開始時に `summary_state` から前回の累積要約を読み、
    `prev_summary` として渡す。生成後は state を更新し、対応 backend には翻訳文脈として供給する。
    """
    items = [text for ts, text in source_buffer if ts >= since_offset]
    if not items:
        return
    src_concat = " ".join(items)

    def _worker() -> None:
        prev_summary = summary_state.get()
        try:
            summary = summarizer.summarize(
                src_concat, duration_seconds, prev_summary=prev_summary
            )
        except Exception:
            logger.exception("summary task failed")
            return
        if not summary.text:
            # summarizer 側で原因 (空応答 / max_completion_tokens 不足 等) を WARN ログ済み
            return
        summary_state.set(summary.text)
        _apply_translation_context(backend, summary.text)
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

    # --device は番号 (--list-devices のインデックス) のみ。非数字はエラー。省略時は
    # プラットフォーム既定 (DEVICE_NAME センチネル: Win=既定出力 loopback / mac=BlackHole)。
    args.device = _resolve_device_arg(args.device)

    import datetime as dt
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    debug_log_path = log_dir / f"session_{timestamp}.debug.log"

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            filename=str(debug_log_path),
            filemode="w",
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        print(f"Debug log: {debug_log_path}", file=sys.stderr)
    else:
        logging.basicConfig(
            level=logging.WARNING,
            stream=sys.stderr,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    # OS 証明書ストア検証は、いずれの TLS 利用 (バックエンド構築・ネットワーク) よりも前に有効化する。
    if args.system_certs:
        _enable_system_certs()

    if args.summary_interval_seconds < 0:
        print("error: --summary-interval-seconds must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.max_session_seconds < 0:
        print("error: --max-session-seconds must be >= 0 (0 = unlimited)", file=sys.stderr)
        sys.exit(2)

    # 既定はチェックなし (platform に応じたキャプチャ対象の表示のみ)。
    # --device-check 指定時のみ出力ルーティングの確認プロンプトを出す。
    if args.device_check:
        _check_input_device(args.device)
    else:
        _print_capture_target(args.device)

    try:
        backend, summarizer = _build_backend(args)
    except SystemExit:
        raise
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    session_logger = SessionLogger(log_dir=log_dir, timestamp=timestamp)
    print(f"Log: {session_logger.path}", file=sys.stderr)
    _emit_settings(args)
    if args.system_certs:
        print("TLS: OS certificate store (truststore)", file=sys.stderr)
    print("", file=sys.stderr)

    # 要約用バッファ: 過去 N 秒の source 言語転写を (offset, text) で保持
    source_buffer: list[tuple[float, str]] = []
    summary_last_offset = 0.0
    summary_interval = float(args.summary_interval_seconds)
    # ローリング要約 + 翻訳文脈注入のための共有状態 (直近の累積要約)
    summary_state = _SummaryState()

    # 要約はメインループから切り離して別スレッドで実行 (1〜3秒 API 待ちで翻訳が止まらないように)。
    # max_workers=1: 同時に複数の要約が走らないようにシリアル化 (連続発火しても順次処理).
    summary_executor: ThreadPoolExecutor | None = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="summary")
        if summarizer is not None
        else None
    )

    # コスト安全弁: 既定 24h で自動停止. 0 = 無制限。
    # セグメント受信ごとに経過時間を確認し、上限到達でループを抜ける
    # (無音中は次セグメント到着時に判定 = 多少のズレは許容範囲)。
    session_deadline = (
        None
        if args.max_session_seconds == 0
        else time.monotonic() + args.max_session_seconds
    )

    try:
        with StreamingRenderer() as renderer:
            if hasattr(backend, "set_status_callback"):
                # 左=音声入力レベル / 右=LLM通信ステータス の 2 スロットへ配線
                backend.set_status_callback(
                    renderer.update_audio_status, renderer.update_comm_status
                )
            with backend:
                for seg in backend.stream_segments():
                    if session_deadline is not None and time.monotonic() >= session_deadline:
                        logger.info(
                            "max-session-seconds reached (%ds); stopping.",
                            args.max_session_seconds,
                        )
                        print(
                            f"\n--max-session-seconds ({args.max_session_seconds}s) reached. "
                            "Stopping. (pass 0 for unlimited)",
                            file=sys.stderr,
                        )
                        break
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
                            summary_state,
                            backend,
                        )
                        summary_last_offset = segment_end
                        cutoff = segment_end - summary_interval * 2
                        source_buffer[:] = [
                            (ts_, t) for ts_, t in source_buffer if ts_ >= cutoff
                        ]

    except KeyboardInterrupt:
        pass
    finally:
        # Commit any remaining final segments left in the queue (e.g. emitted during shutdown)
        if hasattr(backend, "_segment_queue"):
            try:
                time.sleep(0.1)
                while not backend._segment_queue.empty():
                    seg = backend._segment_queue.get_nowait()
                    if not seg.is_partial:
                        ts = format_offset(seg.start_offset_seconds)
                        session_logger.log_segment(ts, seg.source, seg.target)
                        print(f"[{ts}] {seg.source.strip()}")
                        print(f"[{ts}] {seg.target.strip()}\n")
            except Exception:
                pass

        if summary_executor is not None:
            # 進行中の要約タスクは完了まで待つ (まだ表示されてない要約を取りこぼさない).
            summary_executor.shutdown(wait=True)
        session_logger.close()
        print(f"\nLog: {session_logger.path}", file=sys.stderr)


if __name__ == "__main__":
    main()
