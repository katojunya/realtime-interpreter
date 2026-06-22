# realtime-interpreter (Windows, macOS 対応 コンソール翻訳アプリ)

## クイックスタート

動かすための最短手順です。詳細 (バックエンド切替・言語切替・モデル選択・チューニング等) は README.md を参照してください。

手順は **「1. OS で準備 → 2. 取得とインストール → 3. バックエンドを選んで起動」** の順です。OS で異なるのは手順 1 だけで、手順 2・3 は共通です。

まず、どのバックエンドで動かすか決めてください:

1. **OpenAI GPT-Realtime-Translate**(APIキーが必要・デフォルト)
2. **Google Gemini 3.5 Live Translate**(APIキーが必要)
3. **OpenAI Chat Completions 互換 API**(ローカルで動く Ollama などを推論エンジンとして使う)

---

## 1. OS で準備

お使いの OS の節だけ読めば OK です。Python 本体は `uv` が自動で用意するため、別途インストールは不要です(管理者権限も不要)。`uv` のインストール後は `PATH` を反映させてください。

### Windows

**uv をインストール**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**音声入力** — 操作は不要です。ブラウザや Teams / Zoom などで音声を再生すると、既定の音声出力デバイスを WASAPI ループバックで取り込みます(仮想オーディオドライバ不要)。

**APIキーの設定書式**(クラウドのバックエンドで使用。どのキーを設定するかは手順 3 参照):

```powershell
$env:OPENAI_API_KEY = "sk-..."   # 例。GEMINI_API_KEY も同じ書式
```

### macOS

**uv をインストール**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`brew` を使っているなら:

```bash
brew install uv
```

**音声入力** — [BlackHole 2ch](https://existential.audio/blackhole/) などで音声出力を入力デバイスへルーティングします。BlackHole 2ch を使うにはインストールと設定が必要です(`brew` ユーザは下記):

```bash
brew install --cask blackhole-2ch
```

macOS 版の既定の音声入力デバイスは `BlackHole 2ch` です。

**APIキーの設定書式**(クラウドのバックエンドで使用。どのキーを設定するかは手順 3 参照):

```bash
export OPENAI_API_KEY="sk-..."   # 例。GEMINI_API_KEY も同じ書式
```

---

## 2. 取得とインストール(共通)

リポジトリを clone し、ルートフォルダで依存をインストールします:

```bash
git clone https://github.com/katojunya/realtime-interpreter.git
cd realtime-interpreter
uv sync
```

プラットフォームに応じて必要な依存モジュールがインストールされます。

---

## 3. バックエンドを選んで起動(共通)

手順 1 の書式で APIキーを設定してから、選んだバックエンドを起動します。終了は `Ctrl+C`。

### OpenAI Realtime Translate(デフォルト)

OpenAI の APIキー `OPENAI_API_KEY` が必要です(設定書式は手順 1)。引数なしだと openai-realtime が使われます。

```bash
uv run realtime-interpreter            # --backend openai-realtime と同じ
```

### Gemini 3.5 Live Translate

Gemini の APIキー `GEMINI_API_KEY` が必要です(設定書式は手順 1)。

```bash
uv run realtime-interpreter --backend gemini-realtime
```

### OpenAI 互換 Chat Completions API(ローカル LLM / Ollama)

ローカル LLM 利用を想定し、localhost で Ollama / LM Studio などの OpenAI 互換サーバを起動しておく前提です(APIキー不要)。既定のベース URL は `http://localhost:11434/v1`、モデル名は `gemma4:e4b`。

```bash
ollama serve            # 別ターミナルで起動しておく
ollama pull gemma4:e4b  # モデルを事前にダウンロードしておく
uv run realtime-interpreter --backend openai-chat
```
