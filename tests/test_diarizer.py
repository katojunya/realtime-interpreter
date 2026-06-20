"""話者ダイアライゼーション (DIALIZATION-1, 実験的) のテスト.

実モデル (resemblyzer) は使わず、ダミー埋め込みを注入して決定論的に検証する。
ダミー埋め込みは「音声先頭 3 サンプル」をそのまま埋め込みベクトルとして返すため、
セグメント先頭の値で話者の声色を擬似的に表現できる。
"""

from __future__ import annotations

import io
import sys

import numpy as np

from realtime_interpreter import main as m
from realtime_interpreter.backends.base import TranslatedSegment
from realtime_interpreter.diarizer import (
    DEFAULT_SIMILARITY_THRESHOLD,
    UNKNOWN_LABEL,
    Diarizer,
)
from realtime_interpreter.renderer import StreamingRenderer
from realtime_interpreter.session_logger import SessionLogger

SR = 16000


def _emb_first3(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """音声先頭 3 サンプルを埋め込みとして返すダミー (Diarizer 側で L2 正規化される)."""
    return np.asarray(audio[:3], dtype=np.float32)


def _seg(vec: list[float], n: int = SR) -> np.ndarray:
    """先頭に vec を埋めた長さ n の mono float32 セグメントを作る (min_seconds を満たす)."""
    a = np.zeros(n, dtype=np.float32)
    a[: len(vec)] = vec
    return a


# --- Diarizer コア: クラスタリング ---


def test_assigns_same_label_to_similar_voice() -> None:
    d = Diarizer(_emb_first3)  # default threshold (0.80)
    assert d.assign(_seg([1, 0, 0]), SR) == "S1"
    assert d.num_speakers == 1
    # 近い声色 → 同一話者
    assert d.assign(_seg([0.9, 0.1, 0]), SR) == "S1"
    assert d.num_speakers == 1


def test_assigns_new_label_to_different_voice() -> None:
    d = Diarizer(_emb_first3)
    assert d.assign(_seg([1, 0, 0]), SR) == "S1"
    assert d.assign(_seg([0, 1, 0]), SR) == "S2"  # 直交 → 別話者
    assert d.num_speakers == 2
    # それぞれの話者に正しく戻る
    assert d.assign(_seg([0.95, 0.05, 0]), SR) == "S1"
    assert d.assign(_seg([0.05, 0.95, 0]), SR) == "S2"
    assert d.num_speakers == 2


def test_threshold_controls_new_speaker_creation() -> None:
    # 同じ声色ペア (cos≈0.71) でも、しきい値次第で同一話者にも別話者にもなる。
    a = _seg([1, 0, 0])
    b = _seg([0.7, 0.7, 0])  # 正規化後 [0.707, 0.707, 0] → cos(a,b)≈0.707
    # 緩いしきい値: 同一話者にまとめる
    loose = Diarizer(_emb_first3, threshold=0.70)
    assert loose.assign(a, SR) == "S1"
    assert loose.assign(b, SR) == "S1"
    assert loose.num_speakers == 1
    # 厳しいしきい値: 別話者として分ける
    strict = Diarizer(_emb_first3, threshold=0.99)
    assert strict.assign(a, SR) == "S1"
    assert strict.assign(b, SR) == "S2"
    assert strict.num_speakers == 2


# --- Diarizer コア: フォールバック ---


def test_short_segment_follows_last_speaker() -> None:
    d = Diarizer(_emb_first3)  # min_seconds=0.6 → 9600 サンプル未満は判定しない
    short = _seg([1, 0, 0], n=100)
    # 直前話者が無ければ不確定
    assert d.assign(short, SR) == UNKNOWN_LABEL
    # 通常セグメントで話者確定後は、短いセグメントは直前話者を踏襲
    assert d.assign(_seg([1, 0, 0]), SR) == "S1"
    assert d.assign(short, SR) == "S1"
    assert d.num_speakers == 1  # 短いセグメントは新話者を作らない


def test_embedding_failure_falls_back_gracefully() -> None:
    def _boom(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        raise RuntimeError("embedding model exploded")

    d = Diarizer(_boom)
    assert d.assign(_seg([1, 0, 0]), SR) == UNKNOWN_LABEL  # 例外 → 不確定
    assert d.num_speakers == 0  # 失敗時は話者を作らない


# --- パイプライン統合: speaker フィールド伝播 ---


def test_translated_segment_speaker_defaults_none() -> None:
    seg = TranslatedSegment(0.0, 1.0, "Hello", "やあ")
    assert seg.speaker is None


def test_translated_segment_speaker_can_be_set() -> None:
    seg = TranslatedSegment(0.0, 1.0, "Hello", "やあ", speaker="S2")
    assert seg.speaker == "S2"


# --- 表示: renderer の話者プレフィクス ---


def test_renderer_commit_prefixes_speaker() -> None:
    r = StreamingRenderer()
    buf = io.StringIO()
    from rich.console import Console

    r._console = Console(file=buf, width=200)  # Live は開始しないので _refresh は no-op
    r.commit("00:02", "Hello there", "やあ", speaker="S3")
    out = buf.getvalue()
    assert "[S3] Hello there" in out
    assert "[S3] やあ" in out


def test_renderer_commit_without_speaker_has_no_prefix() -> None:
    r = StreamingRenderer()
    buf = io.StringIO()
    from rich.console import Console

    r._console = Console(file=buf, width=200)
    r.commit("00:03", "Plain text", "プレーン")
    out = buf.getvalue()
    assert "Plain text" in out
    assert "[S" not in out  # 話者ラベルは付かない ([00:03] は "[0" なので誤検出しない)


# --- ログ: session_logger の話者プレフィクス ---


def test_session_logger_prefixes_speaker(tmp_path) -> None:
    lg = SessionLogger(log_dir=tmp_path, timestamp="diar")
    lg.log_segment("00:01", "Hello", "こんにちは", speaker="S2")
    lg.close()
    text = lg.path.read_text(encoding="utf-8")
    assert "[00:01] [S2] Hello" in text
    assert "[00:01] [S2] こんにちは" in text


def test_session_logger_no_speaker_no_prefix(tmp_path) -> None:
    lg = SessionLogger(log_dir=tmp_path, timestamp="diar2")
    lg.log_segment("00:01", "Hello", "こんにちは")
    lg.close()
    text = lg.path.read_text(encoding="utf-8")
    assert "[00:01] Hello" in text
    assert "[S" not in text


# --- CLI: --diarize フラグと設定行 ---


def _args(argv: list[str], monkeypatch) -> object:
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter", *argv])
    args = m._parse_args()
    args.device = m._resolve_device_arg(args.device)
    monkeypatch.setattr(m, "_capture_label", lambda _d: "[4] BlackHole 2ch")
    return args


def test_diarize_flag_defaults_false(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter"])
    args = m._parse_args()
    assert args.diarize is False


def test_diarize_flag_parses_true(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter", "--diarize"])
    args = m._parse_args()
    assert args.diarize is True


def test_build_diarizer_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter"])
    args = m._parse_args()
    assert m._build_diarizer(args) is None


def test_collect_settings_adds_diarization_when_enabled(monkeypatch) -> None:
    args = _args(["--backend", "openai-chat", "--diarize"], monkeypatch)
    pairs = dict(m._collect_settings(args, object()))
    assert pairs.get("diarization", "").startswith("on")
    assert "experimental" in pairs["diarization"]
    assert "threshold=0.8" in pairs["diarization"]  # 既定しきい値が記録される


def test_diarize_threshold_defaults_to_constant(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter"])
    args = m._parse_args()
    assert args.diarize_threshold == DEFAULT_SIMILARITY_THRESHOLD == 0.80


def test_diarize_threshold_parses_custom(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["realtime-interpreter", "--diarize-threshold", "0.85"]
    )
    args = m._parse_args()
    assert args.diarize_threshold == 0.85


def test_build_diarizer_forwards_threshold(monkeypatch) -> None:
    # resemblyzer の実ロードを避け、threshold が make_default_diarizer に渡るか検証。
    monkeypatch.setattr(
        sys,
        "argv",
        ["realtime-interpreter", "--diarize", "--diarize-threshold", "0.83"],
    )
    args = m._parse_args()
    captured: dict[str, float] = {}

    def fake_make(threshold: float = DEFAULT_SIMILARITY_THRESHOLD, min_seconds: float = 0.6):
        captured["threshold"] = threshold
        return "DIARIZER_SENTINEL"

    monkeypatch.setattr(
        "realtime_interpreter.diarizer.make_default_diarizer", fake_make
    )
    result = m._build_diarizer(args)
    assert result == "DIARIZER_SENTINEL"
    assert captured["threshold"] == 0.83


def test_collect_settings_omits_diarization_when_disabled(monkeypatch) -> None:
    args = _args(["--backend", "openai-chat"], monkeypatch)
    pairs = dict(m._collect_settings(args, object()))
    assert "diarization" not in pairs


def test_collect_settings_omits_diarization_for_realtime(monkeypatch) -> None:
    # realtime 系は未対応 (警告して無効化) なので、設定行にも出さない。
    args = _args(["--backend", "openai-realtime", "--diarize"], monkeypatch)
    pairs = dict(m._collect_settings(args, object()))
    assert "diarization" not in pairs
