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

    def fake_make(
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_seconds: float = 0.6,
        split: bool = False,
    ):
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


# --- 変化点検出 (--diarize-split) ---


class _SplitDummyEmbedder:
    """分割テスト用の注入埋め込み.

    - embed_partials(): コンストラクタで与えた固定の部分埋め込み列/スパンを返す
      (= 境界検出を決定論的に制御)。
    - __call__(): 音声の平均符号で話者を擬似判定 (+ → 話者A / − → 話者B)。
      → 分割後の各ピースをラベル付けする assign() を決定論化する。
    """

    def __init__(self, partials: list[list[float]], spans: list[tuple[int, int]]):
        self._partials = np.asarray(partials, dtype=np.float32)
        self._spans = spans

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        mean = float(np.mean(audio)) if np.size(audio) else 0.0
        return (
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if mean >= 0
            else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )

    def embed_partials(self, audio, sample_rate, rate=2.5):
        return self._partials, self._spans


def _windows(n: int, win: int = SR) -> list[tuple[int, int]]:
    return [(i * win, (i + 1) * win) for i in range(n)]


def test_split_detects_speaker_change() -> None:
    # 6窓 [A,A,A,B,B,B] → 窓2-3 間で類似度が落ちる → 1境界 (中央 48000) で2分割。
    A, B = [1, 0, 0], [0, 1, 0]
    emb = _SplitDummyEmbedder([A, A, A, B, B, B], _windows(6))
    d = Diarizer(
        emb, split_enabled=True, energy_snap=False,
        split_min_seconds=1.0, boundary_threshold=0.75,
    )
    audio = np.empty(6 * SR, dtype=np.float32)
    audio[: 3 * SR] = 1.0   # 前半 = 話者A
    audio[3 * SR :] = -1.0  # 後半 = 話者B
    pieces = d.split_and_label(audio, SR)
    assert [len(a) for a, _ in pieces] == [3 * SR, 3 * SR]
    assert [label for _, label in pieces] == ["S1", "S2"]
    assert d.num_speakers == 2


def test_split_disabled_returns_single_piece() -> None:
    emb = _SplitDummyEmbedder([[1, 0, 0]] * 4, _windows(4))
    d = Diarizer(emb, split_enabled=False)
    audio = np.ones(4 * SR, dtype=np.float32)
    pieces = d.split_and_label(audio, SR)
    assert len(pieces) == 1
    assert pieces[0][1] == "S1"


def test_split_no_boundary_single_speaker() -> None:
    # 全窓が同一話者 → 類似度の谷なし → 分割しない。
    emb = _SplitDummyEmbedder([[1, 0, 0]] * 4, _windows(4))
    d = Diarizer(emb, split_enabled=True, energy_snap=False)
    pieces = d.split_and_label(np.ones(4 * SR, dtype=np.float32), SR)
    assert len(pieces) == 1


def test_split_fallback_without_partial_embedder() -> None:
    # plain callable は embed_partials を持たない → split 有効でも分割せず1ピース。
    def emb(a: np.ndarray, sr: int) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    d = Diarizer(emb, split_enabled=True)
    pieces = d.split_and_label(np.ones(6 * SR, dtype=np.float32), SR)
    assert len(pieces) == 1


def test_split_too_short_returns_single_piece() -> None:
    emb = _SplitDummyEmbedder([[1, 0, 0], [0, 1, 0]], _windows(2, SR // 2))
    d = Diarizer(emb, split_enabled=True, split_min_seconds=1.0)
    # 1.0s < 2*split_min_seconds(2.0s) → 短すぎて分割対象外。
    pieces = d.split_and_label(np.ones(SR, dtype=np.float32), SR)
    assert len(pieces) == 1


def test_split_respects_max_splits() -> None:
    # 5境界が立つ窓列でも max_splits=1 なら2ピースまで。
    A, B = [1, 0, 0], [0, 1, 0]
    emb = _SplitDummyEmbedder([A, B, A, B, A, B], _windows(6))
    d = Diarizer(
        emb, split_enabled=True, energy_snap=False,
        split_min_seconds=1.0, boundary_threshold=0.75, max_splits=1,
    )
    audio = np.ones(6 * SR, dtype=np.float32)
    pieces = d.split_and_label(audio, SR)
    assert len(pieces) == 2  # max_splits=1 → 最大2ピース


def test_snap_to_energy_min_lands_in_silence() -> None:
    d = Diarizer(lambda a, sr: np.array([1.0, 0.0, 0.0], np.float32), split_enabled=True)
    audio = np.ones(2 * SR, dtype=np.float32)
    audio[15000:17000] = 0.0  # ~1.0s 付近に短い無音
    snapped = d._snap_to_energy_min(audio, SR, SR)  # cut≈1.0s 付近
    assert 15000 <= snapped < 17000  # 無音区間へスナップ


# --- CLI: --diarize-split ---


def test_diarize_split_flag_defaults_false(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter"])
    assert m._parse_args().diarize_split is False


def test_diarize_split_flag_parses_true(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["realtime-interpreter", "--diarize-split"])
    assert m._parse_args().diarize_split is True


def test_build_diarizer_forwards_split(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["realtime-interpreter", "--diarize", "--diarize-split"]
    )
    args = m._parse_args()
    captured: dict[str, bool] = {}

    def fake_make(
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_seconds: float = 0.6,
        split: bool = False,
    ):
        captured["split"] = split
        return "DIARIZER_SENTINEL"

    monkeypatch.setattr(
        "realtime_interpreter.diarizer.make_default_diarizer", fake_make
    )
    assert m._build_diarizer(args) == "DIARIZER_SENTINEL"
    assert captured["split"] is True


def test_collect_settings_shows_split_when_enabled(monkeypatch) -> None:
    args = _args(["--backend", "openai-chat", "--diarize", "--diarize-split"], monkeypatch)
    pairs = dict(m._collect_settings(args, object()))
    assert "change-point split" in pairs["diarization"]
