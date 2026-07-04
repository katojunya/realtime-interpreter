"""話者ダイアライゼーション (DIALIZATION-1, 実験的・逐次発話向け PoC).

確定した発話セグメントの音声から話者埋め込み (speaker embedding) を計算し、
セッション内でオンライン・クラスタリングして話者ラベル (S1 / S2 / …) を割り当てる。

スコープ: 交互に話す「逐次発話」を対象とする。1 セグメント = 1 話者として扱い、
重なり発話 (カクテルパーティ) の分離は行わない (調査の結論として現状非現実的)。
realtime 系バックエンドはローカルの区切り音声を持たないため対象外で、本機能は
per-segment 音声を持つ mlx / openai-chat バックエンドでのみ使う。

埋め込みモデルは `Embedder` (callable) として差し替え可能にし、テストでは
ダミー埋め込みを注入して決定論的に検証できる。既定の実モデルは resemblyzer を
遅延ロードする (オプション依存。未導入なら親切なエラー)。
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# (audio: mono float32, sample_rate) -> 1次元 embedding ベクトル
Embedder = Callable[[np.ndarray, int], np.ndarray]

# コサイン類似度がこの値以上なら同一話者に併合、未満なら新話者。低いほど併合されやすい。
# resemblyzer は話者間マージンが狭く (同一話者≈0.86–0.93 / 異話者≈0.65–0.78)、実素材の
# 検証では 0.70 だと別話者まで併合された。0.80 で交互発話 (男女) を正しく分離できたため既定とする。
# 素材により最適値は変わるので CLI (--diarize-threshold) で調整可能。
DEFAULT_SIMILARITY_THRESHOLD = 0.80
# これより短いセグメントは埋め込みが不安定なため話者判定しない (直前話者を踏襲)。
DEFAULT_MIN_SECONDS = 0.6
UNKNOWN_LABEL = "S?"

# --- 変化点検出 (実験的 --diarize-split): 無音切れ目ゼロで話者交代した max_segment
# チャンクを、窓ごとの埋め込み変化からセグメント内分割する。---
# 隣接窓コサインがこの値未満の「谷」を話者境界候補とする (併合しきい値とは別物)。
# 実音声キャリブレーション (2話者連結 8s vs 単一話者 8s, rate=1.0+音量正規化):
# 異話者境界の谷≈0.62 / 同一話者内の最小≈0.76 → 0.70 が両者を分離する。
DEFAULT_BOUNDARY_THRESHOLD = 0.70
# 分割後ピースの最小長 & セグメント端/他境界からの最小距離 (秒)。
DEFAULT_SPLIT_MIN_SECONDS = 1.0
# 1 セグメントあたり最大分割数 (= 最大 max_splits+1 ピース)。翻訳回数の上限にもなる。
DEFAULT_MAX_SPLITS = 2
# resemblyzer 部分埋め込みの窓レート (窓/秒)。窓長は固定 1.6s なので、低いほど重なりが
# 減って話者交代の谷が深くなる (高いと隣接窓が大半を共有して谷が浅まり検出漏れ)。
# 下限 0.625。実測では 1.0 が「谷の深さ」と「境界の時間分解能」のバランスが良い。
DEFAULT_PARTIAL_RATE = 1.0
# 検出境界をこの半径 (秒) 内の局所エネルギー最小点へスナップし、語の途中切りを緩和する。
_SNAP_RADIUS_SECONDS = 0.35


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class Diarizer:
    """セグメント音声 → 話者ラベルのオンライン割当て.

    コサイン類似度がしきい値以上で最も近い既存話者に割当て、無ければ新話者を採番する。
    割当て後はその話者セントロイドを移動平均で更新する (話者の声の変動に追従)。
    """

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_seconds: float = DEFAULT_MIN_SECONDS,
        update_weight: float = 0.2,
        split_enabled: bool = False,
        boundary_threshold: float = DEFAULT_BOUNDARY_THRESHOLD,
        split_min_seconds: float = DEFAULT_SPLIT_MIN_SECONDS,
        max_splits: int = DEFAULT_MAX_SPLITS,
        partial_rate: float = DEFAULT_PARTIAL_RATE,
        energy_snap: bool = True,
    ) -> None:
        self._embed = embedder
        self._threshold = threshold
        self._min_seconds = min_seconds
        self._update_weight = update_weight
        # 変化点検出 (--diarize-split) 用パラメータ
        self._split_enabled = split_enabled
        self._boundary_threshold = boundary_threshold
        self._split_min_seconds = split_min_seconds
        self._max_splits = max_splits
        self._partial_rate = partial_rate
        self._energy_snap = energy_snap
        self._centroids: list[np.ndarray] = []  # L2 正規化済み
        self._labels: list[str] = []            # _centroids と並行
        self._last_label: str | None = None

    @property
    def num_speakers(self) -> int:
        return len(self._centroids)

    def assign(self, audio: np.ndarray, sample_rate: int) -> str:
        """セグメント音声に話者ラベルを割り当てて返す."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < int(self._min_seconds * sample_rate):
            # 短すぎる: 埋め込みが不安定。直前話者を踏襲 (無ければ不確定)。
            return self._last_label or UNKNOWN_LABEL
        try:
            emb = _l2_normalize(self._embed(audio, sample_rate))
        except Exception:
            logger.exception("speaker embedding failed; skipping diarization for segment")
            return self._last_label or UNKNOWN_LABEL
        if emb.size == 0:
            return self._last_label or UNKNOWN_LABEL

        if self._centroids:
            sims = [float(np.dot(emb, c)) for c in self._centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self._threshold:
                self._update_centroid(best, emb)
                self._last_label = self._labels[best]
                return self._last_label
        # 新話者
        label = f"S{len(self._centroids) + 1}"
        self._centroids.append(emb)
        self._labels.append(label)
        self._last_label = label
        return label

    def _update_centroid(self, idx: int, emb: np.ndarray) -> None:
        w = self._update_weight
        updated = (1.0 - w) * self._centroids[idx] + w * emb
        self._centroids[idx] = _l2_normalize(updated)

    # ------------------------------------------------------------------
    # 変化点検出 (実験的): セグメント内の話者交代で分割する
    # ------------------------------------------------------------------
    def split_and_label(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[tuple[np.ndarray, str]]:
        """セグメント音声を話者の変わり目で分割し、各ピースに (音声, ラベル) を付ける.

        分割無効・分割不能・話者交代なし のときは 1 ピース [(audio, label)] を返す
        (従来の assign 1 回と同じ。後方互換)。各ピースのラベルは assign() で採番するため
        セッション全体の話者ID (S1/S2…) と整合する。
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if (
            not self._split_enabled
            or audio.size < int(2 * self._split_min_seconds * sample_rate)
            or not hasattr(self._embed, "embed_partials")
        ):
            return [(audio, self.assign(audio, sample_rate))]
        try:
            partials, spans = self._embed.embed_partials(
                audio, sample_rate, rate=self._partial_rate
            )
        except Exception:
            logger.exception("partial embedding failed; not splitting segment")
            return [(audio, self.assign(audio, sample_rate))]

        cuts = self._find_boundaries(partials, spans, audio.size, sample_rate)
        if self._energy_snap and cuts:
            cuts = sorted(
                {self._snap_to_energy_min(audio, c, sample_rate) for c in cuts}
            )
        if not cuts:
            return [(audio, self.assign(audio, sample_rate))]

        pieces: list[tuple[np.ndarray, str]] = []
        bounds = [0, *cuts, audio.size]
        for s, e in zip(bounds[:-1], bounds[1:]):
            sub = audio[s:e]
            if sub.size == 0:
                continue
            pieces.append((sub, self.assign(sub, sample_rate)))
        return pieces or [(audio, self.assign(audio, sample_rate))]

    def _find_boundaries(
        self,
        partials: np.ndarray,
        spans: list[tuple[int, int]],
        n: int,
        sample_rate: int,
    ) -> list[int]:
        """部分埋め込み列から話者境界 (サンプル位置) を検出して返す.

        隣接窓のコサイン類似度が boundary_threshold 未満かつ局所最小の点を境界候補とし、
        類似度が低い (= 変化が強い) 順に、端と既採用境界から split_min_seconds 以上離れた
        ものを max_splits 個まで採用する。
        """
        partials = np.asarray(partials, dtype=np.float32)
        if partials.ndim != 2 or len(partials) < 2 or len(spans) != len(partials):
            return []
        norm = np.vstack([_l2_normalize(p) for p in partials])
        adj = np.sum(norm[:-1] * norm[1:], axis=1)  # 隣接窓のコサイン (N-1,)
        centers = [(s + e) / 2.0 for s, e in spans]
        min_gap = int(self._split_min_seconds * sample_rate)

        candidates: list[tuple[float, int]] = []
        for i in range(len(adj)):
            if adj[i] >= self._boundary_threshold:
                continue
            left_ok = i == 0 or adj[i] <= adj[i - 1]
            right_ok = i == len(adj) - 1 or adj[i] <= adj[i + 1]
            if left_ok and right_ok:
                boundary = int((centers[i] + centers[i + 1]) / 2.0)
                candidates.append((float(adj[i]), boundary))

        candidates.sort(key=lambda c: c[0])  # 変化が強い (低類似) 順
        accepted: list[int] = []
        for _sim, b in candidates:
            if b < min_gap or b > n - min_gap:
                continue
            if any(abs(b - a) < min_gap for a in accepted):
                continue
            accepted.append(b)
            if len(accepted) >= self._max_splits:
                break
        return sorted(accepted)

    def _snap_to_energy_min(
        self, audio: np.ndarray, cut: int, sample_rate: int
    ) -> int:
        """境界 cut を ±_SNAP_RADIUS_SECONDS 内の局所エネルギー最小点へスナップする."""
        radius = int(_SNAP_RADIUS_SECONDS * sample_rate)
        lo = max(0, cut - radius)
        hi = min(len(audio), cut + radius)
        if hi - lo < 2:
            return cut
        seg = audio[lo:hi]
        win = max(1, int(0.02 * sample_rate))  # 20ms 移動平均パワー
        power = seg * seg
        kernel = np.ones(win, dtype=np.float32) / win
        smooth = np.convolve(power, kernel, mode="same")
        return lo + int(np.argmin(smooth))


class ResemblyzerEmbedder:
    """resemblyzer (VoiceEncoder) による話者埋め込み (既定の実モデル, 遅延ロード).

    resemblyzer は 16kHz mono float32 を前提とするため、SpeechSegment.audio をそのまま
    渡せる。オプション依存 (`uv sync --extra diarize`)。未導入なら親切なエラーにする。
    """

    def __init__(self) -> None:
        try:
            from resemblyzer import VoiceEncoder
        except ImportError as e:  # noqa: BLE001
            raise SystemExit(
                "error: --diarize requires the optional 'diarize' dependencies. "
                "Install with: uv sync --extra diarize"
            ) from e
        self._encoder = VoiceEncoder(verbose=False)

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(np.asarray(audio, dtype=np.float32), source_sr=sample_rate)
        return self._encoder.embed_utterance(wav)

    def embed_partials(
        self, audio: np.ndarray, sample_rate: int, rate: float = DEFAULT_PARTIAL_RATE
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """変化点検出用: 窓ごとの部分埋め込みと各窓の (開始, 終了) サンプルを返す.

        無音トリム無しの生音声に embed_utterance(return_partials=True) を適用することで、
        窓スライスを元音声のサンプル位置へ 1:1 で対応づける (preprocess_wav はトリムで
        位置がずれるため使わない)。resemblyzer は 16kHz 前提 = 本パイプラインの SAMPLE_RATE。
        """
        from resemblyzer import normalize_volume

        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        # 音量正規化のみ適用 (VAD トリム無し): 長さを保つのでスライス位置が元音声に対応し、
        # かつ埋め込みのS/N が上がって話者交代の谷が深くなる (キャリブで確認)。
        a = normalize_volume(a, target_dBFS=-30, increase_only=False).astype(np.float32)
        _, partials, wav_slices = self._encoder.embed_utterance(
            a, return_partials=True, rate=rate
        )
        spans = [(int(s.start), int(min(s.stop, a.size))) for s in wav_slices]
        return np.asarray(partials, dtype=np.float32), spans


def make_default_diarizer(
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    split: bool = False,
) -> Diarizer:
    """既定の実モデル (resemblyzer) を用いた Diarizer を生成する.

    split=True で変化点検出 (--diarize-split) を有効化する。
    """
    return Diarizer(
        ResemblyzerEmbedder(),
        threshold=threshold,
        min_seconds=min_seconds,
        split_enabled=split,
    )
