# realtime-interpreter

音声をマルチモーダル LLM に直接入力して、**source 言語の文字起こし + target 言語の訳** をリアルタイムにストリーム表示する CLI。既定は **英語 → 日本語** で、`--source-lang` / `--target-lang` (短縮形 `--from` / `--to`, `-s` / `-t`) で任意の言語ペアに切り替え可能。

**バックエンド切替対応**:

| バックエンド | 実装 | 特徴 |
|---|---|---|
| `openai-realtime` (既定) | OpenAI gpt-realtime-translate (WebSocket) | クラウド・$0.034/分・低レイテンシ・サーバ VAD |
| `openai-chat` | OpenAI 互換 Chat Completions REST | ローカル VAD で区切った WAV を直接マルチモーダル LLM に入力。ローカル動作のOllamaの利用が標準設定 |
| `mlx` (macOSのみ) | mlx-vlm + Gemma 4 (ローカル) | オフライン・無料・Apple Silicon ネイティブ |

本ツールは **Whisper を経由せず** Gemma 4 (mlx-vlm) に音声を直接入力し、1 回の推論で **英文の文字起こしと日本語訳を同時生成** します。OpenAI Realtime バックエンドではサーバ側で同等の処理が走ります。`openai-chat` バックエンドのデフォルトは、ローカルの Ollama 等の OpenAI 互換の REST API に WAV 音声を直接渡す設定です。エンドポイント変更することでOpenAIのサービスを使うことも可能です。

## クイックスタート

まず 1 回動かすための最短手順です。詳細 (言語切替・モデル選択・チューニング等) は後続の各節を参照してください。

### 1. uv をインストール

Python 本体は uv が自動で用意するため、別途インストールは不要です。管理者権限も不要です。

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後は**新しいシェルを開いて** PATH を反映させてください (`uv --version` で確認)。

### 2. 依存をインストール

リポジトリのルートで:

```bash
uv sync
```

プラットフォームに応じて必要な依存だけが入ります (macOS は mlx 系も込み、Windows/Linux はスキップ)。

### 3. まず 1 回動かす

お使いの環境に応じて、いちばん手軽な方法でどうぞ。

**macOS — まず音声ファイルで翻訳を確認** (オーディオ設定不要・無料・ローカル):

```bash
uv run python scripts/smoke_test.py samples/test_en_librispeech.flac
```

英文と日本語訳が表示されれば成功です。初回は Gemma 4 モデル (約 5GB) のダウンロードが走ります。ライブ翻訳 (`uv run realtime-interpreter`) には BlackHole の設定が必要です → [前提条件](#前提条件)。

**Windows — クラウドで即実行** (追加のオーディオ設定不要):

```powershell
$env:OPENAI_API_KEY = "sk-..."
uv run realtime-interpreter --backend openai-realtime
```

ブラウザや Zoom 等で英語音声を再生すると、その音が WASAPI ループバックで取り込まれ翻訳が流れます。スピーカーからは普段どおり音が聞こえたままです。

**完全オフライン / ローカル LLM** (Ollama 等の OpenAI 互換サーバ・APIキー不要):

```bash
ollama serve            # 別ターミナルで起動しておく
ollama pull gemma4:e4b
uv run realtime-interpreter --backend openai-chat
```

> うまく動いたら、以降の節で **言語の切り替え**・**バックエンドの選択**・**チャンクサイズの調整**などを確認してください。

## 出力フォーマット

VAD で検出された発話セグメント単位で、確定した結果を 2 行で append-only に出力します。一度出力した行は書き換えません。

```
[00:21] We're driven by the idea that the products we create should help unleash creativity.
[00:21] 私たちは、私たちが作る製品が創造性を解き放つ手助けをするべきだという考えに突き動かされています。

[00:27] We build some of the largest internet services on the planet and many of them run on AWS.
[00:27] 私たちは世界最大級のインターネットサービスを構築しており、その多くは AWS 上で稼働しています。
```

英文の文字起こしはグレー、日本語訳は通常色で表示されます。同じ `[mm:ss]` がペアで付くのは、両者が同じセグメントを指すためです (時刻はセグメント開始時)。

## アーキテクチャ

### MLX バックエンド (既定, ローカル)

```
BlackHole 2ch
  → SpeechSegmentCapture (Silero VAD で発話を区切る. 無音 800ms or 最大 8s で finalize)
  → GemmaAudioTranslator (mlx-vlm + google/gemma-4-e4b-it 4bit)
       1 回の推論で "EN: ... / JA: ..." を生成
  → 確定したセグメント単位で append-only 出力
  → SessionLogger に同形式でログ記録

  [60秒ごと]
  → Summarizer (Gemma 4 をテキスト専用で再利用)
       過去 60 秒の英文の文字起こしから日本語要約を生成
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

**ストリーミング表示 + 発話単位の commit (alignment-first)**: delta が届くたびに英語 + 日本語の 2 行が in-place で伸びていきます (Rich Live)。**delta が一定時間 (既定 800ms) 来なくなった時点でその発話を確定** として上に積み上げ、次の発話を新しい in-progress 行として表示します。

文単位 (JA `。` / EN `.`) でリアルタイム commit する設計も試したが、本モデルは **EN 1 文を JA 複数文 (例: 挨拶を独立文として切る) で訳す** ことがあり、文単位 commit を行うと EN1 ↔ JA1 だけがペアになって後続がズレるため、現状は debounce ベースに統一している。

`--openai-debounce-ms` で発話の切れ目検出感度を調整できます (既定 800):

| 値 | トレードオフ |
|---|---|
| 200〜500 | レスポンス重視. ただし翻訳途中で commit して EN/JA がズレる可能性あり |
| 800 (既定) | バランス. 通常の発話間ポーズで commit |
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

### OpenAI Chat バックエンド (OpenAI 互換 REST)

```
macOS: BlackHole 2ch / Windows: WASAPI loopback
  → SpeechSegmentCapture (Silero VAD で発話を区切る. 無音 800ms or 最大 8s で finalize)
  → WAV PCM16 にエンコード
  → OpenAI-compatible /v1/chat/completions
       input_audio として WAV を渡し、1 回の推論で "SRC: ... / TGT: ..." を生成
  → 確定したセグメント単位で append-only 出力
  → SessionLogger に同形式でログ記録
```

既定では Ollama を想定し、`--openai-chat-base-url http://localhost:11434/v1` と `--openai-chat-model gemma4:e4b` を使います。Ollama 0.24.0 + `gemma4:e4b` / `gemma4:e2b` では WAV 音声の直接入力を確認済みです。REST 方式のため OpenAI Realtime のような delta 単位の in-place 表示はなく、MLX と同じく発話セグメント確定後に表示されます。

## 必要なディスク容量

- Gemma 4 E4B 4-bit MLX: 約 5 GB (初回起動時に `~/.cache/huggingface/hub/` へ自動ダウンロード)
- Python 依存関係: 数百 MB

## 前提条件

### macOS (mlx / openai / openai-chat)

- macOS (Apple Silicon)
- Python 3.11〜3.13
- [uv](https://docs.astral.sh/uv/)
- [BlackHole 2ch](https://existential.audio/blackhole/) と Multi-Output Device 設定 (姉妹プロジェクト README 参照)
- `openai-chat` でローカル推論する場合は [Ollama](https://ollama.com/) 等の OpenAI 互換サーバ

### Windows (openai / openai-chat)

- Windows 10/11
- Python 3.11〜3.13
- [uv](https://docs.astral.sh/uv/)
- `openai` 使用時: OpenAI API キー
- `openai-chat` 使用時: Ollama 等の OpenAI 互換サーバ (Ollama ならAPIキー不要)
- **追加ソフト不要** — システム音声は WASAPI ループバックで取り込みます (管理者権限・仮想オーディオドライバ不要). `uv sync` で [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) (loopback 対応の PyAudio フォーク) が自動インストールされます
- `openai-chat` は `uv sync` でインストールされる `silero-vad-lite` を使ってローカルで発話区切りを検出します
- VDI (リモートデスクトップ等の「リモート オーディオ」) でも動作確認済み

> mlx バックエンド (ローカル Gemma 4 via mlx-vlm) は Apple Silicon 専用のため Windows では使えません。Windows では `--backend openai` または `--backend openai-chat` を使ってください。

## セットアップ

```bash
uv sync
```

`uv sync` はプラットフォームを自動判定します:
- **macOS**: mlx-vlm / silero-vad-lite 等を含む全依存をインストール (mlx / openai / openai-chat 利用可)
- **Windows**: mlx 系はスキップされ、openai / openai-chat に必要な PyAudioWPatch と silero-vad-lite をインストール
- **Linux**: mlx 系はスキップされ、openai-realtime バックエンドに必要な依存のみインストール

(mlx バックエンド初回実行時に Gemma 4 のモデルファイル (約 5GB) が HuggingFace から自動ダウンロードされます。)

### Windows での実行

OpenAI Realtime を使う場合:

```powershell
# PowerShell
$env:OPENAI_API_KEY = "sk-..."
uv run realtime-interpreter --backend openai-realtime
```

```cmd
:: コマンドプロンプト
set OPENAI_API_KEY=sk-...
uv run realtime-interpreter --backend openai-realtime
```

Ollama 等の OpenAI 互換 REST API を使う場合:

```powershell
ollama serve
ollama pull gemma4:e4b
uv run realtime-interpreter --backend openai-chat
```

どちらのバックエンドも、Windowsでは **既定のスピーカー (出力デバイス) の音を WASAPI ループバックで取り込み**、翻訳します。何かアプリ (YouTube, Zoom, Teams 等) で英語音声を再生すると、その音が翻訳されます。スピーカーからは普段どおり音が聞こえたままです。

別の出力デバイスを取り込みたい場合は `--device "<出力デバイス名>"` で指定します。`--list-devices` で取り込み可能な出力デバイス一覧を確認できます。

## 使い方

### MLX バックエンド (既定)

```bash
uv run realtime-interpreter
# = uv run realtime-interpreter --backend mlx
```

### OpenAI バックエンド

```bash
export OPENAI_API_KEY=sk-...
uv run realtime-interpreter --backend openai-realtime
```

OpenAI バックエンドは **WebSocket ベースのストリーミング** で、サーバ側 VAD によって自動的に発話の切れ目が検出されます。要約も同じ API キーで OpenAI Chat Completions 経由 (既定 `gpt-5-mini`) で生成されます。

出力デバイスが Multi-Output Device になっていない場合、起動時に切替を促します。
終了は `Ctrl+C`。

### OpenAI Chat バックエンド (Ollama 等)

```bash
ollama serve
ollama pull gemma4:e4b
uv run realtime-interpreter --backend openai-chat
```

別の OpenAI 互換サーバやモデルを使う場合:

```bash
uv run realtime-interpreter \
  --backend openai-chat \
  --openai-chat-base-url http://localhost:11434/v1 \
  --openai-chat-model gemma4:e2b
```

### オプション

| フラグ | 説明 | 既定 | 対象 |
|---|---|---|---|
| `--backend` | `mlx` / `openai` / `openai-chat` | macOS: `mlx`, Windows: `openai` | 共通 |
| `--device` | 入力デバイス名 (部分一致) | `BlackHole 2ch` | 共通 |
| `--list-devices` | オーディオデバイス一覧を表示して終了 | — | 共通 |
| `--device-check` | 起動時に出力デバイスのルーティングを確認し、必要なら Multi-Output Device への切替を促す (macOS)。既定ではスキップ | (off) | 共通 |
| `--source-lang` / `--from` / `-s` | 音声の言語 (ISO 639-1) | `en` | 共通 |
| `--target-lang` / `--to` / `-t` | 翻訳先の言語 (ISO 639-1) | `ja` | 共通 |
| `--list-languages` | 既知の言語コードを表示して終了 | — | 共通 |
| `--summary-interval-seconds` | N 秒ごとに過去 N 秒分の source 言語テキストを target 言語で要約. `0` で無効. MLX は Gemma 4 を再利用、OpenAI は Chat Completions (`--openai-summary-model`)、openai-chat は同じ OpenAI 互換 API を使用 | `60` | 共通 |
| `--log-dir` | セッションログの出力先 | `logs/` | 共通 |
| `--debug` | 詳細ログを stderr に出す | (off) | 共通 |
| `--model` | モデルエイリアス or 完全な HuggingFace ID | `e4b` | mlx |
| `--list-models` | プリセット一覧を表示して終了 | — | mlx |
| `--end-silence-ms` | この長さの無音で発話セグメントを区切る | `800` | mlx / openai-chat |
| `--max-segment-seconds` | 連続発話時のセグメント最大長 (秒) | `8.0` | mlx / openai-chat |
| `--openai-model` | Realtime モデル ID | `gpt-realtime-translate` | openai-realtime |
| `--openai-chat-base-url` | OpenAI 互換 API の base URL | `http://localhost:11434/v1` | openai-chat |
| `--openai-chat-model` | Chat Completions モデル ID | `gemma4:e4b` | openai-chat |
| `--openai-chat-api-key` | Bearer token. 未指定時は `OPENAI_CHAT_API_KEY` → `OPENAI_API_KEY` → `ollama` | — | openai-chat |
| `--openai-chat-timeout-seconds` | 1 セグメントあたりの REST API タイムアウト | `120` | openai-chat |
| `--openai-chat-max-tokens` | 1 セグメントあたりの出力上限 | `384` | openai-chat |
| `--openai-chat-temperature` | サンプリング温度 | `0` | openai-chat |

環境変数:
- `REALTIME_INTERPRETER_MODEL`: MLX デフォルトモデルの上書き (エイリアスでも完全 ID でも可)
- `OPENAI_API_KEY`: `openai` バックエンド使用時に必須. `openai-chat` では互換サーバが認証を要求する場合のみ使用
- `OPENAI_CHAT_API_KEY`: `openai-chat` バックエンドの Bearer token. Ollama では未指定で可

### 言語の切り替え

ISO 639-1 (2 文字コード) で source / target を指定します. 各バックエンドで共通。

```bash
# 既定: 英語 → 日本語
uv run realtime-interpreter

# 短縮形 (mlx / openai / openai-chat いずれでも)
uv run realtime-interpreter -s en -t es     # 英語 → スペイン語
uv run realtime-interpreter --from zh --to en  # 中国語 → 英語
uv run realtime-interpreter --source-lang ja --target-lang en  # 日本語 → 英語

# 既知の言語コード一覧
uv run realtime-interpreter --list-languages
```

主要な言語コード:

| コード | 言語 | コード | 言語 | コード | 言語 |
|---|---|---|---|---|---|
| `en` | English | `ja` | Japanese | `es` | Spanish |
| `fr` | French | `de` | German | `it` | Italian |
| `pt` | Portuguese | `zh` | Chinese | `ko` | Korean |
| `ru` | Russian | `ar` | Arabic | `nl` | Dutch |
| ... | (他は `--list-languages` で確認) | | | | |

#### バックエンド別の挙動

| バックエンド | source | target |
|---|---|---|
| **mlx** | プロンプトに言語名を埋め込んで Gemma 4 に渡す. 多言語対応だが品質は言語ペアに依存 | 同上 |
| **openai** | `gpt-realtime-whisper` が **自動検出** (source 指定はメタ情報として保持するのみ) | `audio.output.language` に target コードを設定 |
| **openai-chat** | プロンプトに言語名を埋め込んで OpenAI 互換 Chat Completions に渡す. 多言語対応はモデル依存 | 同上 |

#### OpenAI の制約

`gpt-realtime-translate` は **target が ~13 言語に限定** (公式仕様). 範囲外を指定すると API がエラーを返します. 本ツールは事前 validation は行わず API エラーに委ねる方針です. 不明な場合は OpenAI の [Realtime Translation ドキュメント](https://developers.openai.com/api/docs/guides/realtime-translation) を参照してください。

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

既定で 60 秒ごとに、過去 60 秒間の **英文の文字起こし** を日本語で要約してターミナルに表示します。要約モデルは翻訳と同じ Gemma 4 をテキスト専用モードで再利用するため、追加 RAM は不要です。

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
| `--end-silence-ms 800 --max-segment-seconds 8` (既定) | 文単位でしっかり区切られる. 翻訳品質優先 |
| `--end-silence-ms 500 --max-segment-seconds 8` | やや早めに区切る. バランス型 |
| `--end-silence-ms 300 --max-segment-seconds 6` | レスポンス重視. 文が途中で切れやすくなる |
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

実機での体感レイテンシは「発話終了 (無音 800ms 検知) + 推論 (発話長に応じて 0.5〜2 秒)」で、目安として **発話終了から 1.5〜3 秒で画面に表示** されます。

## 料金の目安

### MLX バックエンド (既定)

すべてローカル実行のため **無料**. 電力消費のみ。

### OpenAI バックエンド

主要コストは Realtime API の翻訳分。要約は誤差レベル。

| 項目 | 単価 | 1 時間あたり |
|---|---|---|
| `gpt-realtime-translate` (翻訳) | $0.034/分 | 約 $2.04 |
| `gpt-realtime-whisper` (英文の文字起こし) | $0.017/分 | 約 $1.02 |
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
- mlx バックエンドは macOS (Apple Silicon) 専用 (mlx-vlm に依存). `openai` / `openai-chat` バックエンドは macOS / Windows で動作 (Windows は WASAPI ループバックで音声取り込み)
- macOS の入力デバイスは BlackHole 2ch を想定. Windows は既定スピーカー等の出力デバイスを WASAPI loopback で取り込む
- Gemma 4 の音声入力は E2B / E4B 系のみ対応 (26B MoE / 31B Dense は不可)
- セグメント完結まで何も表示されない設計 (低レイテンシ最優先ではなく、確定済み出力のストリームを目的とする)
- 要約機能はバックエンドごとに別経路 — MLX: ローカル Gemma 4 (text-only) を再利用 / OpenAI: Chat Completions API (`gpt-5-mini` 等) / openai-chat: 同じ OpenAI 互換 REST API

## 設計方針

過去の試行で **暫定→確定 (LocalAgreement-2)** UX も実装したが、独立窓ベースだと幻覚が多くなるため廃止した。現行は **「VAD で発話単位を区切り、完結した発話を 1 回だけ翻訳して表示」** という単純な append-only 設計を採用している。

## ライセンス

MIT
