# realtime-interpreter (Windows, macOS 対応 コンソール翻訳アプリ)

音声をマルチモーダル LLM に直接入力して、入力言語の文字起こしと翻訳テキストを生成し、リアルタイムにストリーム表示するコンソールアプリケーション。既定は **英語 → 日本語**。音声からテキストへ文字起こしを行う **Whisper を経由せず**、音声をマルチモーダル LLM に直接渡し、**文字起こしと訳文を同時に生成**します。一定間隔(デフォルトでは1分)ごとに翻訳先の言語で要約を表示します。

画面の様子は OpenAI が GPT-Realtime-Translate のモデルを発表したときの [YouTube 動画](https://www.youtube.com/watch?v=JOu8v6CBjkE)の音声をリアルタイムに翻訳している様子です。

![realtime-interpreter の実行例 (英語音声 → 日本語の文字起こし・訳・要約)](example.png)

## サポートしているバックエンド

| バックエンド | 実装 | 特徴 |
|---|---|---|
| `openai-realtime` (既定) | OpenAI gpt-realtime-translate (WebSocket) | クラウド・低レイテンシ・サーバ VAD・約 $3/時。|
| `gemini-realtime` | Gemini Multimodal Live API (WebSocket) | クラウド・低レイテンシ・サーバ VAD・約 $2.25/時。 |
| `openai-chat` | OpenAI 互換 Chat Completions REST | ローカル VAD で区切った WAV を直接入力。標準は**ローカル Ollama**。エンドポイント変更で OpenAI Webサービスの利用も可 |
| `mlx` (macOS のみ) | mlx-vlm + Gemma 4 (ローカル) | オフライン・Apple Silicon 専用 macOS ネイティブ |

既定バックエンドは**両OSで `openai-realtime`**。利用可能なバックエンドは OS により異なります:

- **macOS**: `openai-realtime` / `gemini-realtime` / `openai-chat` / `mlx`
- **Windows**: `openai-realtime` / `gemini-realtime` / `openai-chat`

## クイックスタート

動かすための最短手順です。詳細 (バックエンド切替・言語切替・モデル選択・チューニング等) は後続の各節を参照してください。

### 1. uv をインストール

Python 本体は `uv` が自動で用意するため、別途インストールは不要です。管理者権限も不要です。
`uv` のインストール後は**新しいシェルを開いて** PATH を反映させてください (`uv --version` で確認)。

#### macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# brew を使っているなら
brew install uv
```

#### Windows

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


### 2. 依存モジュールをインストール

リポジトリのルートで:

```bash
uv sync
```

プラットフォームに応じて必要な依存モジュールがインストールされます。

### 3. 起動する

#### 起動方法

お使いの環境・用途に応じて、各バックエンドでの動かし方を示します。音声入力デバイスの準備は OS ごとに異なります (下記参照)。

Windows:

ブラウザや Zoom 等で音声を再生するとデフォルトの音声出力先のデバイスがタップされて、その音が入力として取り込まれ翻訳を開始します。音声デバイスの設定は不要です。

macOS:

BlackHole 2ch などを用いて音声出力を、入力デバイスへルーティングして下さい。ユーザによる設定が必要です。brew を使っているなら、下記のコマンドでインストール可能です。

```bash
brew install --cask blackhole-2ch
```

#### 起動方法 [**OpenAI Realtime Translate (デフォルト)**]

```bash
# APIキーの設定
## macOS
export OPENAI_API_KEY="sk-..."

## Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."


uv run realtime-interpreter           # --backend openai-realtime の指定と同じ
```

#### 起動方法 [**Gemini 3.5 Live Translate**]

```bash
# APIキーの設定
## macOS
export GEMINI_API_KEY="..."

## Windows PowerShell
$env:GEMINI_API_KEY = "..."


uv run realtime-interpreter --backend gemini-realtime # バックエンドを明示的に指定すること
```

#### 起動方法 [**OpenAI 互換 Chat Completions API**] (ローカル LLM / Ollama)

ローカル LLM で動かす場合は、Ollama / LM Studio などの OpenAI 互換サーバを起動し、`gemma4:e4b` をロードしておきます。デフォルトのベース URI は `http://localhost:11434/v1`、モデル名は `gemma4:e4b` です。

```bash
ollama serve            # 別ターミナルで起動しておく
ollama pull gemma4:e4b
uv run realtime-interpreter --backend openai-chat   # バックエンドは明示的に指定すること
```

## アーキテクチャ

### openai-realtime バックエンド (既定, クラウド)

```
BlackHole 2ch (macOS) / WASAPI loopback (Windows)
  → 24kHz PCM16 で WebSocket ストリーム送信
  → OpenAI gpt-realtime-translate
       サーバ側で VAD + 文字起こし + 翻訳 を一括実行
  → input_transcript.delta (source) / output_transcript.delta (target) を delta 単位で受信
  → 受信ごとに Rich Live が in-place で「進行中」行を更新表示
  → delta が一定時間 (既定 800ms) 来なくなったら debounce で発話確定 (or 最大 8s で強制カット)
  → 確定行を append-only に永続表示 + ログ
  → [60秒ごと] OpenAI Chat Completions (既定 gpt-5-mini) で過去 60 秒を要約
```

**確定ロジック (alignment-first)**: この API には `*.completed` イベントが無いため、**delta が一定時間 (既定 800ms) 来なくなった時点で発話を確定**します。連続発話では `--openai-rt-max-segment-seconds` (既定 8s) で強制カットします。文単位 (JA `。` / EN `.`) の即時 commit も試したが、本モデルは EN 1 文を JA 複数文に訳すことがあり対応がズレるため、debounce ベースに統一しています。

**自動再接続 (60分制限対策)**: OpenAI Realtime API は 1 接続あたり最大 60 分でサーバから切断されます。本実装は **接続から 55 分でプロアクティブに新接続へ張り替え** (切れ目をほぼゼロに)、加えて切断・瞬断を検知すると**指数バックオフで再接続**します。再接続は新規セッションですが (Realtime に再開ハンドルは無い)、要約履歴はアプリ側で保持するため継続します。再接続の瞬間 (1〜2 秒) に 1 フレーズ程度欠落することがあります。

### gemini-realtime バックエンド (クラウド)

```
BlackHole 2ch (macOS) / WASAPI loopback (Windows)
  → 16kHz PCM16 で WebSocket ストリーム送信
  → Gemini Multimodal Live API (models/gemini-3.5-live-translate-preview)
       inputAudioTranscription (source) / 翻訳 (target) を delta 単位で受信
  → debounce (既定 800ms) / 最大 8s で発話確定して append-only 出力
  → [60秒ごと] gemini-3.1-flash-lite で要約
```

**長時間運用**: Gemini Live は接続 (~10分) / セッション (~15分) に上限がありますが、本実装は **session resumption** (切断をまたいで再接続) と **context window compression** (sliding window でトークン上限を回避) の両方を有効化しており、**数時間の連続セッション**が可能です (実機で 1 時間超の連続翻訳を確認済み)。

**過剰分割対策**: live-translate は短い翻訳単位ごとに `turnComplete` を頻発させます。これを確定トリガにすると 1〜数語の細切れが多発するため、本実装では `turnComplete` を確定に使わず debounce / max_segment のみで束ねています。それでも細切れが気になる場合は `--gemini-rt-debounce-ms` を上げてください (例 1500)。

### openai-chat バックエンド (OpenAI 互換 Chat Completions REST)

```
BlackHole 2ch (macOS) / WASAPI loopback (Windows)
  → SpeechSegmentCapture (Silero VAD で発話を区切る. 無音 800ms or 最大 8s で finalize)
  → WAV PCM16 にエンコード
  → OpenAI 互換 /v1/chat/completions
       input_audio として WAV を渡し、1 回の推論で "SRC: ... / TGT: ..." を生成
  → 確定したセグメント単位で append-only 出力
```

既定では Ollama を想定し、`--openai-chat-base-url http://localhost:11434/v1` と `--openai-chat-model gemma4:e4b` を使います。REST 方式のため delta 単位の in-place 表示はなく、発話セグメント確定後に表示されます。

### mlx バックエンド (macOS のみ, ローカル)

```
BlackHole 2ch
  → SpeechSegmentCapture (Silero VAD で発話を区切る. 無音 800ms or 最大 8s で finalize)
  → GemmaAudioTranslator (mlx-vlm + google/gemma-4-e4b-it 4bit)
       1 回の推論で "SRC: ... / TGT: ..." を生成
  → 確定したセグメント単位で append-only 出力
  → [60秒ごと] 同じ Gemma 4 をテキスト専用で再利用して要約
```

## 長時間連続動作に対するセーフティ機構

全バックエンド共通で `--max-session-seconds` (既定 86400 = 24 時間) を超えると自動停止します。従量課金の暴走を防ぐセーフティです。
ユーザーが明示的に制限を解除するなら `--max-session-seconds 0` 指定して下さい。

## 必要なディスク容量

- Gemma 4 E4B 4-bit MLX: 約 5 GB (mlx 初回起動時に `~/.cache/huggingface/hub/` へ自動ダウンロード)
- Python 依存関係: 数百 MB

## 前提条件

### macOS

- macOS (Apple Silicon)
- Python 3.11〜3.13 / [uv](https://docs.astral.sh/uv/)
- [BlackHole 2ch](https://existential.audio/blackhole/) と Multi-Output Device 設定
- バックエンド別:
  - `openai-realtime`: `OPENAI_API_KEY`
  - `gemini-realtime`: `GEMINI_API_KEY`
  - `openai-chat`: [Ollama](https://ollama.com/) 等の OpenAI 互換サーバ
  - `mlx`: 追加不要 (初回に Gemma 4 を自動 DL)

### Windows

- Windows 10/11 / Python 3.11〜3.13 / [uv](https://docs.astral.sh/uv/)
- バックエンド別: `openai-realtime` は `OPENAI_API_KEY`、`gemini-realtime` は `GEMINI_API_KEY`、`openai-chat` は Ollama 等 (API キー不要)
- **追加ソフト不要** — システム音声は WASAPI ループバックで取り込みます (管理者権限・仮想オーディオドライバ不要)。`uv sync` で [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) が自動インストールされます
- VDI (リモートデスクトップの「リモート オーディオ」) でも動作確認済み
- mlx バックエンドは Apple Silicon 専用のため Windows では使えません

## セットアップ

```bash
uv sync
```

`uv sync` はプラットフォームを自動判定します:
- **macOS**: mlx-vlm / silero-vad-lite 等を含む全依存をインストール
- **Windows**: mlx 系はスキップ、PyAudioWPatch と silero-vad-lite をインストール

## 使い方

### openai-realtime (既定)

```bash
export OPENAI_API_KEY=sk-...
uv run realtime-interpreter
# = uv run realtime-interpreter --backend openai-realtime
```

Windows (PowerShell / cmd):

```powershell
$env:OPENAI_API_KEY = "sk-..."
uv run realtime-interpreter
```

```cmd
set OPENAI_API_KEY=sk-...
uv run realtime-interpreter
```

### gemini-realtime

```bash
export GEMINI_API_KEY=...
uv run realtime-interpreter --backend gemini-realtime
```

### openai-chat (Ollama 等)

```bash
ollama serve
ollama pull gemma4:e4b          # Windows は gemma4:e2b
uv run realtime-interpreter --backend openai-chat
```

別の OpenAI 互換サーバやモデルを使う場合:

```bash
uv run realtime-interpreter \
  --backend openai-chat \
  --openai-chat-base-url http://localhost:11434/v1 \
  --openai-chat-model gemma4:e2b
```

### mlx (macOS のみ)

```bash
uv run realtime-interpreter --backend mlx
```

終了は `Ctrl+C`。Windows では既定スピーカーの音を WASAPI ループバックで取り込みます。別の出力デバイスを取り込むには `--device "<出力デバイス名>"`、一覧は `--list-devices`。

### オプション

| フラグ | 説明 | 既定 | 対象 |
|---|---|---|---|
| `--backend` | `openai-realtime` / `gemini-realtime` / `openai-chat` / `mlx` | `openai-realtime` | 共通 |
| `--device` | 入力/取り込みデバイス名 (部分一致) | `BlackHole 2ch` (macOS) | 共通 |
| `--list-devices` | オーディオデバイス一覧を表示して終了 | — | 共通 |
| `--device-check` | 起動時に出力デバイスのルーティングを確認 (macOS)。既定はスキップ | (off) | 共通 |
| `--source-lang` / `--from` / `-s` | 音声の言語 (ISO 639-1) | `en` | 共通 |
| `--target-lang` / `--to` / `-t` | 翻訳先の言語 (ISO 639-1) | `ja` | 共通 |
| `--list-languages` | 既知の言語コードを表示して終了 | — | 共通 |
| `--summary-interval-seconds` | N 秒ごとに過去 N 秒を要約. `0` で無効 | `60` | 共通 |
| `--max-session-seconds` | この秒数で自動停止 (コスト安全弁). `0` で無制限 | `86400` (24h) | 共通 |
| `--log-dir` | セッションログの出力先 | `logs/` | 共通 |
| `--debug` | 詳細ログを stderr に出す | (off) | 共通 |
| `--openai-rt-model` | Realtime モデル ID | `gpt-realtime-translate` | openai-realtime |
| `--openai-rt-debounce-ms` | 発話の切れ目検出 (debounce) | `800` | openai-realtime |
| `--openai-rt-max-segment-seconds` | 連続発話の強制カット上限. `0` で無効 | `8.0` | openai-realtime |
| `--openai-rt-summary-model` | 要約モデル (Chat Completions) | `gpt-5-mini` | openai-realtime |
| `--openai-rt-api-key` | API キー (既定 `OPENAI_API_KEY`) | — | openai-realtime |
| `--gemini-rt-model` | Gemini Live モデル ID | `models/gemini-3.5-live-translate-preview` | gemini-realtime |
| `--gemini-rt-debounce-ms` | 発話の切れ目検出 (debounce) | `800` | gemini-realtime |
| `--gemini-rt-max-segment-seconds` | 連続発話の強制カット上限. `0` で無効 | `8.0` | gemini-realtime |
| `--gemini-rt-summary-model` | 要約モデル | `gemini-3.1-flash-lite` | gemini-realtime |
| `--gemini-rt-api-key` | API キー (既定 `GEMINI_API_KEY`) | — | gemini-realtime |
| `--openai-chat-base-url` | OpenAI 互換 API の base URL | `http://localhost:11434/v1` | openai-chat |
| `--openai-chat-model` | Chat Completions モデル ID | `gemma4:e4b` | openai-chat |
| `--openai-chat-api-key` | Bearer token (既定 `OPENAI_CHAT_API_KEY`→`OPENAI_API_KEY`→`ollama`) | — | openai-chat |
| `--openai-chat-timeout-seconds` | 1 セグメントの REST タイムアウト | `120` | openai-chat |
| `--openai-chat-max-tokens` | 1 セグメントの出力上限 | `384` | openai-chat |
| `--openai-chat-temperature` | サンプリング温度 | `0` | openai-chat |
| `--end-silence-ms` | 無音で発話を区切る長さ | `800` | mlx / openai-chat |
| `--max-segment-seconds` | 連続発話のセグメント最大長 (秒) | `8.0` | mlx / openai-chat |
| `--model` | モデルエイリアス or HuggingFace ID | `e4b` | mlx |
| `--list-models` | プリセット一覧を表示して終了 | — | mlx |

長いフラグには短縮形があります (`--openai-realtime-model` = `--openai-rt-model`、`--gemini-realtime-model` = `--gemini-rt-model` など)。

環境変数:
- `OPENAI_API_KEY`: `openai-realtime` 必須。`openai-chat` では互換サーバが認証を要求する場合のみ
- `GEMINI_API_KEY`: `gemini-realtime` 必須
- `OPENAI_CHAT_API_KEY`: `openai-chat` の Bearer token (Ollama では不要)
- `REALTIME_INTERPRETER_MODEL`: mlx デフォルトモデルの上書き

### 言語の切り替え

ISO 639-1 (2 文字コード) で source / target を指定します。各バックエンド共通。

```bash
uv run realtime-interpreter -s en -t es           # 英語 → スペイン語
uv run realtime-interpreter --from zh --to en     # 中国語 → 英語
uv run realtime-interpreter --list-languages      # 既知の言語コード一覧
```

#### バックエンド別の挙動

| バックエンド | source | target |
|---|---|---|
| **openai-realtime** | `gpt-realtime-whisper` が**自動検出** (source 指定はメタ情報) | `audio.output.language` に target コードを設定 |
| **gemini-realtime** | live-translate が処理 | `translationConfig.targetLanguageCode` に設定 |
| **openai-chat** | プロンプトに言語名を埋め込み (多言語対応はモデル依存) | 同上 |
| **mlx** | プロンプトに言語名を埋め込み (品質は言語ペア依存) | 同上 |

#### OpenAI / Gemini の制約

`gpt-realtime-translate` は **target が ~13 言語に限定** (公式仕様)。範囲外を指定すると API がエラーを返します。本ツールは事前 validation せず API エラーに委ねます。詳細は [OpenAI Realtime Translation ドキュメント](https://developers.openai.com/api/docs/guides/realtime-translation) を参照。

### モデルの切り替え (mlx)

```bash
uv run realtime-interpreter --backend mlx --model e2b   # 軽量・高速
uv run realtime-interpreter --backend mlx --list-models # プリセット一覧
```

| エイリアス | モデル ID | 用途 |
|---|---|---|
| `e4b` (既定) | `mlx-community/gemma-4-e4b-it-4bit` | 通常運用 |
| `e2b` | `mlx-community/gemma-4-e2b-it-4bit` | レイテンシ優先 |
| `e4b-bf16` | `mlx-community/gemma-4-e4b-it-bf16` | 量子化なしの最高品質 |
| `e2b-bf16` | `mlx-community/gemma-4-e2b-it-bf16` | 量子化なし軽量 |

新しい音声入力対応 MLX モデルを使う場合は `src/realtime_interpreter/translator.py` の `MODEL_PRESETS` に追記してください。

## 定期要約 (1 分ごと)

既定で 60 秒ごとに、過去 60 秒間の **source 言語の文字起こし** を target 言語で要約して表示します。要約経路はバックエンドごとに異なります:

- **openai-realtime**: OpenAI Chat Completions (`gpt-5-mini`)。ライブ翻訳と同じ `OPENAI_API_KEY` で動作
- **gemini-realtime**: `gemini-3.1-flash-lite`, APIキーもライブ翻訳と同じ
- **openai-chat**: 同じ OpenAI 互換 REST API
- **mlx**: ライブ翻訳と同じ Gemma 4 をテキスト専用モードで再利用

```bash
uv run realtime-interpreter --summary-interval-seconds 0    # 要約を無効化
uv run realtime-interpreter --summary-interval-seconds 30   # 30 秒ごと
```

要約プロンプトは [`src/realtime_interpreter/summarizer.py`](src/realtime_interpreter/summarizer.py) で調整できます。

## チャンクサイズ / 確定タイミングの調整

「文の刻みを細かくしたい / 長台詞を待たずに早く出したい / 細切れを減らしたい」場合に調整します。フラグはバックエンドごとに独立です。

```bash
# openai-realtime: debounce と強制カット上限
uv run realtime-interpreter --openai-rt-debounce-ms 500 --openai-rt-max-segment-seconds 8

# gemini-realtime: 細切れが気になるなら debounce を上げて束ねる
uv run realtime-interpreter --backend gemini-realtime --gemini-rt-debounce-ms 1500

# mlx / openai-chat: VAD の無音長と最大セグメント長
uv run realtime-interpreter --backend mlx --end-silence-ms 500 --max-segment-seconds 6
```

| debounce / end-silence | トレードオフ |
|---|---|
| 短い (200〜500ms) | レスポンス重視. ただし途中で commit して対応がズレる可能性 |
| 800ms (既定) | バランス. 通常の発話間ポーズで commit |
| 長い (1500〜2500ms) | 段落単位で commit. 対応は確実だが表示は遅延 |

### その他のチューニング (ソース直書き)

`src/realtime_interpreter/audio.py` の冒頭 (mlx / openai-chat の VAD):

| 定数 | 既定 | 意味 |
|---|---|---|
| `VAD_THRESHOLD` | 0.3 | 発話判定の確率しきい値 |
| `MIN_SEGMENT_SECONDS` | 0.5 | これより短いセグメントはノイズとして捨てる |

## スモークテスト / ユニットテスト

```bash
# 音声ファイルで mlx の翻訳を確認 (macOS)
uv run python scripts/smoke_test.py samples/test_en_librispeech.flac

# ユニットテスト一式
uv run pytest
```

## レイテンシ

実機での体感は「発話終了 (無音検知) + 推論」で、目安として**発話終了から 1.5〜3 秒で画面に表示**されます。mlx (M3 Max) は 5.86 秒音声 → EN+JA 出力が約 1.0 秒、モデルロード (キャッシュ済) 約 2.6 秒。クラウドバックエンドはネットワーク往復に依存します。

## 料金の目安

### openai-realtime

| 項目 | 単価 | 1 時間あたり |
|---|---|---|
| `gpt-realtime-translate` (翻訳) | $0.034/分 | 約 $2.04 |
| `gpt-realtime-whisper` (文字起こし) | $0.017/分 | 約 $1.02 |
| `gpt-5-mini` (要約, 60s ごと) | 約 $0.0003/回 | 約 $0.02 |
| **合計** | | **約 $3.08 / 時** (1 営業日 8h ≈ $25) |

詳細は [OpenAI Realtime 料金ページ](https://developers.openai.com/api/docs/models/gpt-realtime-translate)。

### gemini-realtime

**無料枠**: live-translate プレビューは無料枠で利用でき、要約の `gemini-3.1-flash-lite` も無料枠の RPD が大きい (実測 ~500 RPD)。無料枠の制限内なら実質無料で運用できます。

**無料枠を使わない場合 (有料ティア)**:

| 項目 | 単価 | 1 時間あたり |
|---|---|---|
| `gemini-3.5-live-translate-preview` 入力音声 | $3.50/1M tok (≈ $0.0053/分) | 約 $0.32 |
| `gemini-3.5-live-translate-preview` 出力音声 | $21.00/1M tok (≈ $0.0315/分) | 約 $1.89 |
| `gemini-3.1-flash-lite` (要約, 60s ごと) | $0.25/1M (入力) + $1.50/1M (出力) | 約 $0.04 |
| **合計** | | **約 $2.25 / 時** |

出力音声 (TTS) は本ツールでは使わず翻訳テキストのみ利用しますが、`responseModalities=["AUDIO"]` のため課金対象になり、コストの約 85% を占めます。最新の価格・無料枠は [Gemini API 料金ページ](https://ai.google.dev/gemini-api/docs/pricing) を参照 (preview モデルのため変動の可能性あり)。

### openai-chat / mlx

ローカル実行のため**無料** (電力消費のみ)。

## 既知の制限事項

- mlx バックエンドは macOS (Apple Silicon) 専用。`openai-realtime` / `gemini-realtime` / `openai-chat` は macOS / Windows で動作
- macOS の入力は BlackHole 2ch を想定。Windows は既定スピーカー等を WASAPI loopback で取り込む
- Gemma 4 の音声入力は E2B / E4B 系のみ対応 (26B MoE / 31B Dense は不可)
- リアルタイム両バックエンドは入力文字起こしと訳が別ストリームのため、行頭の対応が 1 セグメント程度ずれることがある
- セグメント完結まで確定行は表示されない設計 (確定済み出力のストリームを目的とする)
- 再接続 (openai-realtime) の瞬間に 1 フレーズ程度欠落することがある

## ライセンス

MIT

本プロジェクトは [SatoshiMoriyama/live-translate (realtime-transcriber)](https://github.com/SatoshiMoriyama/live-translate/tree/main/packages/realtime-transcriber) (MIT License) を基に開始しました。
