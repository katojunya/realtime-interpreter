# realtime-interpreter (Windows, macOS 対応 コンソール翻訳アプリ)

## クイックスタート

動かすための最短手順です。詳細 (バックエンド切替・言語切替・モデル選択・チューニング等) は README.md を参照してください。以降の節はバックエンドごとに動かし方を記載しています。まずは、どれで動かすかを決めて下さい。

  1. OpenAI GPT-Realtime-Translate (APIキーが必要)
  2. Google Gemini 3.5 Live Translate (APIキーが必要)
  3. OpenAI Chat Completions 互換 API (ローカルで動くOllamaなどを推論エンジンとして使う)

### 1. uv をインストール

Python 本体は `uv` が自動で用意するため、別途インストールは不要です。管理者権限も不要です。`uv` のインストール後は `PATH` を反映させてください。

#### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
#### macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

macOS で `brew` を使っているなら
```bash
brew install uv
```

### 2. 依存モジュールをインストール

リポジトリのルートで:

```bash
uv sync
```

プラットフォームに応じて必要な依存モジュールがインストールされます。

### 3. 起動する

お使いの環境・用途に応じた各バックエンドごとの動かし方を示します。音声入力デバイスの準備は OS ごとに異なります (下記参照)。

Windows:

ブラウザや Teams, Zoom 等で音声を再生するとデフォルトの音声出力デバイスをタップして、そこに流れる音声を入力として取り込みます。音声入力デバイスを選択する操作は不要です。

macOS:

[BlackHole 2ch](https://existential.audio/blackhole/) などを用いて、音声出力を入力デバイスへルーティングして下さい。BlackHole 2ch使うならユーザによるインストールと設定が必要です。brew ユーザは下記のコマンドでインストール可能です。

```bash
brew install --cask blackhole-2ch
```

realtime-interpreter の macOS 版ではデフォルトの音声入力デバイスは `BlackHole 2ch` となっています。

#### [**OpenAI Realtime Translate (デフォルトのバックエンド)**] の起動方法

OpenAI の APIキー が必要です。環境変数に設定して下さい。

Windows PowerShell での設定
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

macOS での設定
```bash
export OPENAI_API_KEY="sk-..."
```

起動方法は下記の通り。引数なしだと openai-realtime (GPT-Realtime-Translate) が使われます。


```bash
uv run realtime-interpreter           # --backend openai-realtime の指定と同じ
```


#### [**Gemini 3.5 Live Translate**] の起動方法 

Gemini の APIキー が必要です。環境変数に設定して下さい。

Windows PowerShell での設定

```powershell
$env:GEMINI_API_KEY = "..."
```

macOS での設定
```bash
export GEMINI_API_KEY="..."
```

起動方法は下記の通り。オプション `--backend gemini-realtime` を与えることで Gemini 3.5 Live Translate を使います。
```bash
uv run realtime-interpreter --backend gemini-realtime # バックエンドを明示的に指定すること
```

#### [**OpenAI 互換 Chat Completions API**] (ローカル LLM / Ollama) の起動方法

OpenAI Chat APIでも動作しますがローカル LLM 利用を想定しており、localhost で Ollama / LM Studio などの OpenAI 互換サーバを起動しておくことが前提です。デフォルトのベース URI は `http://localhost:11434/v1`、モデル名は `gemma4:e4b` となっています。

```bash
ollama serve            # 別ターミナルで起動しておく
ollama pull gemma4:e4b  # モデルを事前にダウンロードしておく
uv run realtime-interpreter --backend openai-chat  # バックエンドは明示的に指定すること
```
