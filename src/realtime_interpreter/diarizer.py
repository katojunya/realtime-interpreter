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
    ) -> None:
        self._embed = embedder
        self._threshold = threshold
        self._min_seconds = min_seconds
        self._update_weight = update_weight
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


def make_default_diarizer(
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_seconds: float = DEFAULT_MIN_SECONDS,
) -> Diarizer:
    """既定の実モデル (resemblyzer) を用いた Diarizer を生成する."""
    return Diarizer(ResemblyzerEmbedder(), threshold=threshold, min_seconds=min_seconds)
