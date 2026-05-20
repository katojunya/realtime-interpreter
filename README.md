# realtime-interpreter

英語音声をマルチモーダル LLM に直接入力して、英語転写 + 日本語訳をリアルタイムにストリーム表示する CLI。

**バックエンド切替対応**:

| バックエンド | 実装 | 特徴 |
|---|---|---|
| `mlx` (既定) | mlx-vlm + Gemma 4 (ローカル) | オフライン・無料・Apple Silicon ネイティブ |
| `openai` | OpenAI gpt-realtime-translate (WebSocket) | クラウド・$0.034/分・低レイテンシ・サーバ VAD |

姉妹プロジェクト [`realtime-transcriber`](../realtime-transcriber/) が「Whisper 文字起こし → Ollama 翻訳」の 2 段直列で 7〜15 秒チャンクを処理するのに対し、本ツールは **Whisper を経由せず** Gemma 4 (mlx-vlm) に音声を直接入力し、1 回の推論で **英語転写と日本語訳を同時生成** します。OpenAI バックエンドではサーバ側で同等の処理が走ります。

## 出力フォーマット

VAD で検出された発話セグメント単位で、確定した結果を 2 行で append-only に出力します。一度出力した行は書き換えません。

```
[00:21] We're driven by the idea that the products we create should help unleash creativity.
[00:21] 私たちは、私たちが作る製品が創造性を解き放つ手助けをするべきだという考えに突き動かされています。

[00:27] We build some of the largest internet services on the planet and many of them run on AWS.
[00:27] 私たちは世界最大級のインターネットサービスを構築しており、その多くは AWS 上で稼働しています。
```

英語転写はグレー、日本語訳は通常色で表示されます。同じ `[mm:ss]` がペアで付くのは、両者が同じセグメントを指すためです (時刻はセグメント開始時)。

## アーキテクチャ

### MLX バックエンド (既定, ローカル)

```
BlackHole 2ch
  → SpeechSegmentCapture (Silero VAD で発話を区切る. 無音 300ms or 最大 8s で finalize)
  → GemmaAudioTranslator (mlx-vlm + google/gemma-4-e4b-it 4bit)
       1 回の推論で "EN: ... / JA: ..." を生成
  → 確定したセグメント単位で append-only 出力
  → SessionLogger に同形式でログ記録

  [60秒ごと]
  → Summarizer (Gemma 4 をテキスト専用で再利用)
       過去 60 秒の英語転写から日本語要約を生成
  → 要約ブロックを表示 + ログ記録
```

### OpenAI バックエンド (クラウド)

```
BlackHole 2ch
  → 24kHz PCM16 で WebSocket ストリーム送信
  → OpenAI gpt-realtime-translate
       サーバ側で VAD + 転写 + 翻訳 + (TTS) を一括実行
  → input_transcript.delta (英語) / output_transcript.delta (日本語) を delta 単位で受信
  → 受信ごとに Rich Live が in-place で「進行中」行を更新表示
  → 1.5 秒 delta が来なくなったら debounce でターン確定
  → 確定行を append-only に永続表示 + ログ
```

**ストリーミング表示 + 発話単位の commit (alignment-first)**: delta が届くたびに英語 + 日本語の 2 行が in-place で伸びていきます (Rich Live)。**delta が一定時間 (既定 300ms) 来なくなった時点でその発話を確定** として上に積み上げ、次の発話を新しい in-progress 行として表示します。

文単位 (JA `。` / EN `.`) でリアルタイム commit する設計も試したが、本モデルは **EN 1 文を JA 複数文 (例: 挨拶を独立文として切る) で訳す** ことがあり、文単位 commit を行うと EN1 ↔ JA1 だけがペアになって後続がズレるため、現状は debounce ベースに統一している。

`--openai-debounce-ms` で発話の切れ目検出感度を調整できます (既定 300):

| 値 | トレードオフ |
|---|---|
| 200〜300 (既定 300) | レスポンス重視. ただし翻訳途中で commit して EN/JA がズレる可能性あり |
| 500〜1000 | バランス. 通常の発話間ポーズで commit |
| 1500〜2500 | より長い文・段落単位で commit. 確実に対応が取れるが表示は遅延する |

`--openai-max-segment-seconds` で **連続発話時の強制 commit 上限** を設定できます (既定 8.0). ポーズなしの長台詞でも N 秒で 1 チャンクに切り分けます:

| 値 | 効果 |
|---|---|
| `8` (既定) | 連続発話でも約 8 秒で強制カット. 読みやすさ重視 |
| `15` | 中庸 |
| `30` | 段落単位の長文を許容 |
| `0` | 強制カット無効. debounce のみで判定 (連続発話で巨大化する可能性) |

強制 commit 時は EN/JA に若干ズレが生じる可能性がありますが、文単位の即時 commit と比べると軽症です。

**要約機能** は OpenAI Chat Completions API (既定 `gpt-5-mini`) を使って実装されています。Realtime API とは別経路ですが、**同じ `OPENAI_API_KEY` で動作** するため追加設定不要。コストは要約 1 回あたり概ね $0.0003〜$0.0005、60 秒に 1 回の頻度なら **$0.02〜$0.05/時** ほどで、翻訳コスト ($3/h) に対して誤差レベル。`--openai-summary-model` で gpt-4o-mini など他モデルへ切り替え可能 (`gpt-4o-mini` ならさらに半額)。

## 必要なディスク容量

- Gemma 4 E4B 4-bit MLX: 約 5 GB (初回起動時に `~/.cache/huggingface/hub/` へ自動ダウンロード)
- Python 依存関係: 数百 MB

## 前提条件

- macOS (Apple Silicon)
- Python 3.11〜3.13
- [uv](https://docs.astral.sh/uv/)
- [BlackHole 2ch](https://existential.audio/blackhole/) と Multi-Output Device 設定 (姉妹プロジェクト README 参照)

## セットアップ

```bash
uv sync
```

初回の `uv run realtime-interpreter` 実行時に Gemma 4 のモデルファイル (約 5GB) が HuggingFace から自動ダウンロードされます。

## 使い方

### MLX バックエンド (既定)

```bash
uv run realtime-interpreter
# = uv run realtime-interpreter --backend mlx
```

### OpenAI バックエンド

```bash
export OPENAI_API_KEY=sk-...
uv run realtime-interpreter --backend openai
```

OpenAI バックエンドは **WebSocket ベースのストリーミング** で、サーバ側 VAD によって自動的に発話の切れ目が検出されます。要約も同じ API キーで OpenAI Chat Completions 経由 (既定 `gpt-5-mini`) で生成されます。

出力デバイスが Multi-Output Device になっていない場合、起動時に切替を促します。
終了は `Ctrl+C`。

### オプション

| フラグ | 説明 | 既定 | 対象 |
|---|---|---|---|
| `--backend` | `mlx` or `openai` | `mlx` | 共通 |
| `--device` | 入力デバイス名 | `BlackHole 2ch` | 共通 |
| `--summary-interval-seconds` | N 秒ごとに過去 N 秒分の英文を日本語要約. `0` で無効. MLX バックエンドは Gemma 4 を再利用、OpenAI バックエンドは Chat Completions (`--openai-summary-model`) を使用 | `60` | mlx のみ |
| `--log-dir` | セッションログの出力先 | `logs/` | 共通 |
| `--debug` | 詳細ログを stderr に出す | (off) | 共通 |
| `--model` | モデルエイリアス or 完全な HuggingFace ID | `e4b` | mlx |
| `--list-models` | プリセット一覧を表示して終了 | — | mlx |
| `--end-silence-ms` | この長さの無音で発話セグメントを区切る | `300` | mlx |
| `--max-segment-seconds` | 連続発話時のセグメント最大長 (秒) | `8.0` | mlx |
| `--openai-model` | Realtime モデル ID | `gpt-realtime-translate` | openai |

環境変数:
- `REALTIME_INTERPRETER_MODEL`: MLX デフォルトモデルの上書き (エイリアスでも完全 ID でも可)
- `OPENAI_API_KEY`: OpenAI バックエンド使用時に必須

### モデルの切り替え

```bash
# 既定 (E4B 4bit, 品質と速度のバランス)
uv run realtime-interpreter

# 軽量・高速 (E2B 4bit)
uv run realtime-interpreter --model e2b

# 高品質 (E4B bf16, 量子化なし)
uv run realtime-interpreter --model e4b-bf16

# プリセット一覧
uv run realtime-interpreter --list-models

# 任意の HuggingFace ID も渡せる ("/" を含む文字列)
uv run realtime-interpreter --model mlx-community/some-other-model

# 環境変数で永続的に上書き
export REALTIME_INTERPRETER_MODEL=e2b
uv run realtime-interpreter
```

| エイリアス | モデル ID | 用途 |
|---|---|---|
| `e4b` (既定) | `mlx-community/gemma-4-e4b-it-4bit` | 通常運用 |
| `e2b` | `mlx-community/gemma-4-e2b-it-4bit` | レイテンシ優先 |
| `e4b-bf16` | `mlx-community/gemma-4-e4b-it-bf16` | 量子化なしの最高品質 |
| `e2b-bf16` | `mlx-community/gemma-4-e2b-it-bf16` | 量子化なし軽量 |

新しい音声入力対応 MLX モデルを使う場合は `src/realtime_interpreter/translator.py` の `MODEL_PRESETS` に追記してください。

## 定期要約 (1 分ごと)

既定で 60 秒ごとに、過去 60 秒間の **英語転写** を日本語で要約してターミナルに表示します。要約モデルは翻訳と同じ Gemma 4 をテキスト専用モードで再利用するため、追加 RAM は不要です。

```
[01:23] We're driven by the idea that the products we create should help unleash creativity.
[01:23] 私たちは、私たちが作る製品が創造性を解き放つ手助けをするべきだという考えに突き動かされています。

--- 要約 [01:24] ---
過去1分間は、Apple のクラウドインフラストラクチャ戦略についての導入であった。
App Store や Apple Music などのサービスが AWS と自社データセンターを組み合わせて
運用されていることが説明された。
---
```

要約推論には 1〜3 秒かかる間、翻訳パイプラインが一時停止します (MLX GPU stream を共有するため)。1 分に 1 回なので体感はほぼ気にならないはず。

```bash
# 要約を無効化
uv run realtime-interpreter --summary-interval-seconds 0

# 30 秒ごとに要約 (短い会話の検証用)
uv run realtime-interpreter --summary-interval-seconds 30
```

要約モデルのプロンプトは [`src/realtime_interpreter/summarizer.py`](src/realtime_interpreter/summarizer.py) の `SUMMARY_PROMPT_TEMPLATE` で調整できます。

## チャンクサイズの調整

「文の刻みをもう少し細かくしたい / 長台詞を待たずに早く出したい」場合は、CLI フラグで簡単に切り替えられます。

```bash
# 短い間 (200ms) で区切る → 反応が早いが文が分断されやすい
uv run realtime-interpreter --end-silence-ms 200

# 連続発話でも 5 秒で強制カット
uv run realtime-interpreter --max-segment-seconds 5

# 両方併用してかなり細切れに
uv run realtime-interpreter --end-silence-ms 250 --max-segment-seconds 6
```

| 設定例 | 体感 |
|---|---|
| `--end-silence-ms 500 --max-segment-seconds 15` | 文単位でしっかり区切られる. 翻訳品質優先 |
| `--end-silence-ms 300 --max-segment-seconds 8` (既定) | レスポンス重視のバランス型 |
| `--end-silence-ms 300 --max-segment-seconds 8` | やや早めに区切る. バランス型 |
| `--end-silence-ms 200 --max-segment-seconds 5` | 細切れ. 反応速度優先, 文が途中で切れることがある |

### その他のチューニング (ソース直書き)

`src/realtime_interpreter/audio.py` の冒頭で調整:

| 定数 | 既定 | 意味 |
|---|---|---|
| `VAD_THRESHOLD` | 0.3 | 発話判定の確率しきい値 (低くすると小さい音も発話扱い) |
| `MIN_SEGMENT_SECONDS` | 0.5 | これより短いセグメントはノイズとして捨てる |

## スモークテスト (静的 wav/flac)

リアルタイムキャプチャを始める前に、音声ファイルで翻訳が機能するか確認できます:

```bash
# 既定モデル (e4b) で実行
uv run python scripts/smoke_test.py samples/test_en_librispeech.flac

# 別モデルで試す
uv run python scripts/smoke_test.py samples/test_en_librispeech.flac e2b
```

`samples/test_en_librispeech.flac` は LibriSpeech の標準テストサンプル (5.86 秒)。

## ユニットテスト

```bash
uv run pytest
```

含まれるテスト:
- `tests/test_translator_resolve.py`: モデルエイリアス解決ロジック
- `tests/test_translator_parse.py`: モデル出力 (`EN: / JA:`) のパーサ

## レイテンシ (M3 Max 実測ベース)

| 段階 | 実測 |
|---|---|
| モデルロード (キャッシュ済) | 約 2.6 秒 |
| 5.86 秒音声 → EN+JA 出力 | 約 1.0 秒 |

実機での体感レイテンシは「発話終了 (無音 300ms 検知) + 推論 (発話長に応じて 0.5〜2 秒)」で、目安として **発話終了から 1〜3 秒で画面に表示** されます。

## 料金の目安

### MLX バックエンド (既定)

すべてローカル実行のため **無料**. 電力消費のみ。

### OpenAI バックエンド

主要コストは Realtime API の翻訳分。要約は誤差レベル。

| 項目 | 単価 | 1 時間あたり |
|---|---|---|
| `gpt-realtime-translate` (翻訳) | $0.034/分 | 約 $2.04 |
| `gpt-realtime-whisper` (英語転写) | $0.017/分 | 約 $1.02 |
| `gpt-5-mini` (要約, 60s ごと) | 約 $0.0003/回 | 約 **$0.02** |
| **合計** | | **約 $3.08 / 時** |

```
1 セッション (1時間連続)  ≈ $3.08
1 営業日 (8時間)          ≈ $25
1 ヶ月 (営業日 20 日)     ≈ $500
```

詳細は [OpenAI Realtime 料金ページ](https://developers.openai.com/api/docs/models/gpt-realtime-translate) を参照。

## 既知の制限事項

- 英語→日本語のみ
- macOS (Apple Silicon) 専用 (mlx-vlm に依存. OpenAI バックエンドは macOS 以外でも動くが BlackHole 設定の手順が異なる)
- 入力デバイスは BlackHole 2ch を想定
- Gemma 4 の音声入力は E2B / E4B 系のみ対応 (26B MoE / 31B Dense は不可)
- セグメント完結まで何も表示されない設計 (低レイテンシ最優先ではなく、確定済み出力のストリームを目的とする)
- 要約機能はバックエンドごとに別経路 — MLX: ローカル Gemma 4 (text-only) を再利用 / OpenAI: Chat Completions API (`gpt-5-mini` 等)

## 設計方針

過去の試行で **暫定→確定 (LocalAgreement-2)** UX も実装したが、独立窓ベースだと幻覚が多くなるため廃止した。現行は **「VAD で発話単位を区切り、完結した発話を 1 回だけ翻訳して表示」** という単純な append-only 設計を採用している。

## ライセンス

MIT
