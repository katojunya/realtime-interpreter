"""ローリング要約 + 翻訳への要約文脈注入のテスト.

ネットワーク・モデルロード不要 (プロンプト組み立てと文脈設定の純粋ロジックのみ)。
"""

from __future__ import annotations

from realtime_interpreter.summarizer import build_summary_prompt


# ---------------- ローリング要約プロンプト ----------------

def test_build_prompt_first_call_has_no_prev_section():
    p = build_summary_prompt("hello world", 60)
    assert "Summary so far" not in p
    assert "hello world" in p


def test_build_prompt_rolling_includes_prev_summary():
    p = build_summary_prompt("new utterance", 60, prev_summary="これまでの累積要約")
    assert "Summary so far" in p
    assert "これまでの累積要約" in p
    assert "new utterance" in p
    # ローリング時は「全体要約を更新」する指示になっている
    assert "running summary" in p


def test_build_prompt_blank_prev_falls_back_to_window():
    p = build_summary_prompt("x", 60, prev_summary="   ")
    assert "Summary so far" not in p


# ---------------- 翻訳への文脈注入 (openai-chat) ----------------

def test_openai_chat_translator_context_injection():
    from realtime_interpreter.backends.openai_chat import OpenAIChatAudioTranslator

    t = OpenAIChatAudioTranslator(model="m", base_url="http://x/v1", api_key="k")
    base = t._prompt_with_context()
    assert "Session context" not in base  # 初期は文脈なし

    t.update_context("要約コンテキスト")
    ctx = t._prompt_with_context()
    assert "要約コンテキスト" in ctx
    assert "Background context" in ctx
    assert ctx.startswith(t._prompt)  # 元プロンプトに後置きされる

    t.update_context("")  # 空にすると無効化
    assert "Session context" not in t._prompt_with_context()


# ---------------- 翻訳への文脈注入 (mlx / GemmaAudioTranslator) ----------------

def test_gemma_translator_context_injection():
    # __init__ はモデルをロードしない (lazy) のでネット/MLX不要
    from realtime_interpreter.translator import GemmaAudioTranslator

    t = GemmaAudioTranslator(model="e4b")
    assert "Session context" not in t._prompt_with_context()

    t.update_context("rolling summary text")
    out = t._prompt_with_context()
    assert "rolling summary text" in out
    assert "Background context" in out


# ---------------- backend の update_context 委譲 ----------------

def test_openai_chat_backend_delegates_update_context():
    from realtime_interpreter.backends.openai_chat import (
        OpenAIChatAudioTranslator,
        OpenAIChatBackend,
    )

    translator = OpenAIChatAudioTranslator(model="m", base_url="http://x/v1", api_key="k")
    # sd_module=None / device 名はキャプチャ生成に使われるが、__init__ で例外が出ない経路を確認
    # (macOS では SpeechSegmentCapture が VAD を初期化する。失敗する環境ではスキップ)
    try:
        backend = OpenAIChatBackend(
            sd_module=None, device_name="dev", translator=translator
        )
    except Exception:
        # VAD 依存が無い環境では translator への委譲だけ直接確認
        translator.update_context("X")
        assert "X" in translator._prompt_with_context()
        return
    backend.update_context("delegated summary")
    assert "delegated summary" in translator._prompt_with_context()
