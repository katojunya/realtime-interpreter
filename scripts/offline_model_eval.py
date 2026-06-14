"""オフライン・複数モデル比較ハーネス.

benchmark の .webm(や任意の音声/動画ファイル)を入力に、ライブと同一の VAD で
**1 度だけ**発話セグメントに分割し、その同一チャンク列を **複数モデル**へ順に与えて
翻訳(SRC 転写 + TGT 訳)させ、各モデルの出力ログと **遅延統計** を書き出す。

ライブ録音不要・完全オフライン。Ollama でも、OpenAI 互換サーバ(vLLM /
mlx-omni-server 等)でも、`input_audio` を受ける chat completions エンドポイントなら
そのまま比較できる(本体 openai-chat バックエンドと同一の送信フォーマット)。

使い方:
  # 既定: benchmark/ の .webm を自動検出、Ollama(localhost:11434)で 2 モデル比較
  uv run python scripts/offline_model_eval.py --models gemma4:e2b gemma4:e4b

  # サーバをモデルごとに変える (name@baseurl 構文)。例: Gemma=Ollama, Qwen=MLXサーバ
  uv run python scripts/offline_model_eval.py \
      --models gemma4:e4b@http://localhost:11434/v1 \
               qwen2.5-omni:7b@http://localhost:10240/v1

  # 先頭 5 分だけで素早く試す + セグメントをキャッシュ
  uv run python scripts/offline_model_eval.py --models gemma4:e4b \
      --start 0 --duration 300 --cache .eval_cache.npz

  # mlx-vlm を in-process 実行(Ollama 非経由)。Qwen3-Omni 等の omni 音声モデルを
  # HuggingFace ID で評価。macOS/Apple Silicon 専用。
  uv run python scripts/offline_model_eval.py --backend mlxvlm \
      --models mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit \
      --start 0 --duration 300 --cache .eval_cache.npz

出力 (--out-dir, 既定 eval_out/):
  <model>.log    … [mm:ss] SRC / [mm:ss] TGT 形式 + 遅延サマリ
  summary.md     … 全モデルの遅延・出力統計の比較表
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

# リポジトリ src を import path に追加 (uv run であれば不要だが直接実行にも耐える)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from realtime_interpreter.audio import (  # noqa: E402
    END_SILENCE_MS,
    MAX_SEGMENT_SECONDS,
    SAMPLE_RATE,
    SpeechSegment,
)
from realtime_interpreter.backends.openai_chat import (  # noqa: E402
    DEFAULT_OPENAI_CHAT_MAX_TOKENS,
    DEFAULT_OPENAI_CHAT_TEMPERATURE,
    DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS,
    OpenAIChatAudioTranslator,
    _RepetitionGuard,
)
from realtime_interpreter.i18n import DEFAULT_SOURCE, DEFAULT_TARGET, normalize_language_code  # noqa: E402
from realtime_interpreter.offline_capture import OfflineSegmentCapture  # noqa: E402

DEFAULT_BASE_URL = "http://localhost:11434/v1"


@dataclass
class ModelSpec:
    name: str
    base_url: str


def _fmt_offset(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _safe_name(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in model)


def decode_audio(path: str, sample_rate: int, start: float | None, duration: float | None) -> np.ndarray:
    """ffmpeg で任意の音声/動画をモノラル float32 @ sample_rate にデコードする."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]  # -i の前に置いて高速シーク
    cmd += ["-i", path]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[:500]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    end_silence_ms: int,
    max_segment_seconds: float,
) -> list[SpeechSegment]:
    cap = OfflineSegmentCapture(
        sample_rate=sample_rate,
        end_silence_ms=end_silence_ms,
        max_segment_seconds=max_segment_seconds,
    )
    return list(cap.segments_from(audio))


def save_cache(path: str, segs: list[SpeechSegment]) -> None:
    if not segs:
        return
    concat = np.concatenate([s.audio for s in segs]).astype(np.float32)
    lengths = np.array([s.audio.size for s in segs], dtype=np.int64)
    offsets = np.array([s.start_offset_seconds for s in segs], dtype=np.float64)
    np.savez(path, audio=concat, lengths=lengths, offsets=offsets)


def load_cache(path: str, sample_rate: int) -> list[SpeechSegment]:
    z = np.load(path)
    concat, lengths, offsets = z["audio"], z["lengths"], z["offsets"]
    segs: list[SpeechSegment] = []
    pos = 0
    for length, off in zip(lengths, offsets):
        chunk = concat[pos : pos + length]
        pos += length
        segs.append(
            SpeechSegment(
                audio=chunk,
                start_offset_seconds=float(off),
                duration_seconds=length / sample_rate,
            )
        )
    return segs


def parse_models(entries: list[str], default_base_url: str) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    for e in entries:
        if "@" in e:
            name, url = e.split("@", 1)
        else:
            name, url = e, default_base_url
        out.append(ModelSpec(name=name, base_url=url))
    return out


@dataclass
class ModelResult:
    name: str
    latencies: list[float]
    empty_tgt: int
    errors: int
    dropped_repeats: int
    total_chars: int


def run_model(
    name: str,
    info: str,
    translator,
    segments: list[SpeechSegment],
    out_dir: str,
    use_repetition_guard: bool,
    audio_seconds: float,
) -> ModelResult:
    """事前構築済み translator (.translate(audio)->.source/.target) で全セグメントを処理.

    遅延は backend 横断で公平になるよう **harness 側で translate() 全体を計測**する
    (音声エンコード+前処理+推論/往復を含む)。
    """
    guard = _RepetitionGuard() if use_repetition_guard else None

    log_path = os.path.join(out_dir, f"{_safe_name(name)}.log")
    latencies: list[float] = []
    empty_tgt = errors = dropped = chars = 0
    total = len(segments)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# offline model eval\n")
        f.write(f"# model: {name}\n")
        f.write(f"# {info}\n")
        f.write(f"# segments: {total}  guard: {use_repetition_guard}\n\n")
        f.flush()

        for idx, seg in enumerate(segments, 1):
            ts = _fmt_offset(seg.start_offset_seconds)
            t0 = time.perf_counter()
            try:
                res = translator.translate(seg.audio)
            except Exception as e:  # noqa: BLE001
                errors += 1
                f.write(f"[{ts}] <<ERROR: {type(e).__name__}: {str(e)[:160]}>>\n\n")
                f.flush()
                # 初回から全滅(サーバ未起動/モデル未取得)なら早期に打ち切る
                if errors >= 3 and idx == errors:
                    f.write("# aborted: first 3 segments all errored "
                            "(server down or model not loadable?)\n")
                    print(f"  ! {name}: 連続エラーで打ち切り", file=sys.stderr)
                    break
                continue
            latencies.append(time.perf_counter() - t0)

            src_text = getattr(res, "source", "") or ""
            tgt_text = getattr(res, "target", "") or ""
            if guard and src_text and guard.is_repeat(src_text):
                dropped += 1
                f.write(f"[{ts}] <<dropped repeat: {src_text[:60]!r}>>\n\n")
                f.flush()
                continue

            chars += len(src_text) + len(tgt_text)
            if not tgt_text.strip():
                empty_tgt += 1
            f.write(f"[{ts}] {src_text}\n")
            f.write(f"[{ts}] {tgt_text}\n\n")
            f.flush()

            if idx % 25 == 0 or idx == total:
                avg = statistics.mean(latencies) if latencies else 0.0
                print(f"  {name}: {idx}/{total}  avg latency {avg:.2f}s",
                      file=sys.stderr)

        # 遅延サマリを末尾に追記
        f.write(_latency_block(latencies, audio_seconds))

    return ModelResult(
        name=name,
        latencies=latencies,
        empty_tgt=empty_tgt,
        errors=errors,
        dropped_repeats=dropped,
        total_chars=chars,
    )


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def _latency_block(latencies: list[float], audio_seconds: float) -> str:
    if not latencies:
        return "\n# latency: (no successful segments)\n"
    total_infer = sum(latencies)
    return (
        "\n# latency (seconds per segment): "
        f"mean={statistics.mean(latencies):.2f} "
        f"median={statistics.median(latencies):.2f} "
        f"p90={_pct(latencies, 90):.2f} "
        f"max={max(latencies):.2f}\n"
        f"# total inference: {total_infer:.1f}s for {audio_seconds:.1f}s of audio "
        f"(RTF={total_infer / audio_seconds:.2f}, lower=faster)\n"
    )


def write_summary(
    out_dir: str,
    results: list[ModelResult],
    n_segments: int,
    audio_seconds: float,
    input_path: str,
) -> str:
    lines = [
        "# Offline model comparison",
        "",
        f"- input: `{os.path.basename(input_path)}`",
        f"- segments: {n_segments}  (audio {audio_seconds:.0f}s)",
        "",
        "| model | ok | err | empty TGT | dropped | "
        "lat mean | median | p90 | max | RTF | chars |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        lat = r.latencies
        ok = len(lat)
        mean = statistics.mean(lat) if lat else 0.0
        med = statistics.median(lat) if lat else 0.0
        p90 = _pct(lat, 90)
        mx = max(lat) if lat else 0.0
        rtf = (sum(lat) / audio_seconds) if (lat and audio_seconds) else 0.0
        lines.append(
            f"| {r.name} | {ok} | {r.errors} | {r.empty_tgt} | "
            f"{r.dropped_repeats} | {mean:.2f} | {med:.2f} | {p90:.2f} | "
            f"{mx:.2f} | {rtf:.2f} | {r.total_chars} |"
        )
    lines += [
        "",
        "- **lat** = 1 セグメントあたりの翻訳遅延(秒)。低遅延ほど良い。",
        "- **RTF** = 総推論時間 / 音声長。<1 で全体として実時間より速い。",
        "- **empty TGT** = 訳が空のセグメント数(無音/非音声/失敗の指標)。",
        "",
    ]
    text = "\n".join(lines)
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None,
                    help="音声/動画ファイル。省略時は benchmark/*.webm を自動検出")
    ap.add_argument("--models", nargs="+", required=True,
                    help="モデル名のリスト。openai-chat では name または name@baseurl、"
                         "mlxvlm では mlx-vlm が読める HuggingFace ID(例 "
                         "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit)")
    ap.add_argument("--backend", choices=["openai-chat", "mlxvlm"], default="openai-chat",
                    help="openai-chat: OpenAI互換HTTP(Ollama等)。"
                         "mlxvlm: mlx-vlm を in-process で実行(Qwen3-Omni 等、macOS専用)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help=f"既定の OpenAI 互換エンドポイント (default: {DEFAULT_BASE_URL})")
    ap.add_argument("--api-key", default=None, help="Bearer トークン (Ollama は不要)")
    ap.add_argument("--source-lang", default=DEFAULT_SOURCE)
    ap.add_argument("--target-lang", default=DEFAULT_TARGET)
    ap.add_argument("--start", type=float, default=None, help="開始秒 (部分評価)")
    ap.add_argument("--duration", type=float, default=None, help="長さ秒 (部分評価)")
    ap.add_argument("--end-silence-ms", type=int, default=END_SILENCE_MS)
    ap.add_argument("--max-segment-seconds", type=float, default=MAX_SEGMENT_SECONDS)
    ap.add_argument("--timeout-seconds", type=float, default=DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_OPENAI_CHAT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_OPENAI_CHAT_TEMPERATURE)
    ap.add_argument("--repetition-guard", action="store_true",
                    help="本番同様に逐語重複を破棄する (既定 off = 素のモデル出力)")
    ap.add_argument("--limit-segments", type=int, default=None,
                    help="先頭 N セグメントだけ処理 (スモークテスト用)")
    ap.add_argument("--cache", default=None,
                    help="セグメントの .npz キャッシュパス (再実行で再利用)")
    ap.add_argument("--out-dir", default="eval_out")
    args = ap.parse_args()

    # 入力決定
    input_path = args.input
    if not input_path:
        cands = sorted(glob.glob("benchmark/*.webm"))
        if not cands:
            raise SystemExit("error: --input 未指定かつ benchmark/*.webm が見つかりません")
        input_path = cands[0]
    if not os.path.exists(input_path):
        raise SystemExit(f"error: 入力が存在しません: {input_path}")

    src = normalize_language_code(args.source_lang)
    tgt = normalize_language_code(args.target_lang)
    os.makedirs(args.out_dir, exist_ok=True)

    # セグメント取得 (キャッシュ優先)
    if args.cache and os.path.exists(args.cache):
        print(f"segments: キャッシュから読み込み {args.cache}", file=sys.stderr)
        segments = load_cache(args.cache, SAMPLE_RATE)
    else:
        print(f"decode: {input_path} "
              f"(start={args.start}, duration={args.duration}) ...", file=sys.stderr)
        t0 = time.perf_counter()
        audio = decode_audio(input_path, SAMPLE_RATE, args.start, args.duration)
        print(f"  {audio.size / SAMPLE_RATE:.1f}s decoded in {time.perf_counter()-t0:.1f}s",
              file=sys.stderr)
        print("segment: Silero VAD ...", file=sys.stderr)
        t0 = time.perf_counter()
        segments = segment_audio(audio, SAMPLE_RATE, args.end_silence_ms, args.max_segment_seconds)
        print(f"  {len(segments)} segments in {time.perf_counter()-t0:.1f}s", file=sys.stderr)
        if args.cache:
            save_cache(args.cache, segments)
            print(f"  cached -> {args.cache}", file=sys.stderr)

    if args.limit_segments:
        segments = segments[: args.limit_segments]
    if not segments:
        raise SystemExit("error: セグメントが 0 件です")

    audio_seconds = sum(s.duration_seconds for s in segments)
    specs = parse_models(args.models, args.base_url)

    results: list[ModelResult] = []
    for spec in specs:
        if args.backend == "mlxvlm":
            # in-process mlx-vlm (macOS/Apple Silicon)。GemmaAudioTranslator は
            # 実体が汎用 mlx-vlm 音声翻訳器で、フル HF ID をそのまま受ける。
            from realtime_interpreter.translator import GemmaAudioTranslator

            print(f"\n=== {spec.name}  (mlx-vlm in-process, loading on first segment...) ===",
                  file=sys.stderr)
            translator = GemmaAudioTranslator(
                model=spec.name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                source_lang=src,
                target_lang=tgt,
            )
            info = "in-process mlx-vlm"
        else:
            print(f"\n=== {spec.name}  ({spec.base_url}) ===", file=sys.stderr)
            translator = OpenAIChatAudioTranslator(
                model=spec.name,
                base_url=spec.base_url,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                source_lang=src,
                target_lang=tgt,
            )
            info = f"base_url: {spec.base_url}"

        r = run_model(
            spec.name, info, translator, segments, args.out_dir,
            use_repetition_guard=args.repetition_guard,
            audio_seconds=audio_seconds,
        )
        results.append(r)

    text = write_summary(args.out_dir, results, len(segments), audio_seconds, input_path)
    print("\n" + text)
    print(f"\nログ: {args.out_dir}/*.log  サマリ: {args.out_dir}/summary.md", file=sys.stderr)


if __name__ == "__main__":
    main()
