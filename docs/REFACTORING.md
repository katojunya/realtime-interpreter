# リファクタリング設計書

main ブランチのコード整理。重複の単一情報源化と死にコード除去を目的とする。
実施は安全な順(③→②→⑥→①→⑤→④)。各候補ごとにテストを通してからコミットする。

## 候補①: 2つの WebSocket バックエンドの基底クラス化

`openai_realtime.py` と `gemini_realtime.py` は `_PendingTurn`・audio capture・
emit/debounce ループ・level meter・send/recv/reconnect ループがほぼ逐語同一。
共通機構を `backends/_ws_streaming_base.py` の `WebSocketStreamingBackend` に集約し、
各バックエンドはプロトコル固有のフックのみ実装する:

- `_open_websocket()` / `_send_session_config()` / `_send_audio_chunk(audio)` /
  `_handle_event(event)` / `SAMPLE_RATE` / 再接続時の resume 可否 /
  プロアクティブ再接続(OpenAI のみ)。

`audio.py` の `_BaseSpeechSegmentCapture` と同じ手法。

## 候補②: 共有オーディオヘルパの一元化

`_to_mono` / `_resample_linear` / `_find_loopback_device` / `_find_input_device` /
`_is_windows` が audio.py + 2 backend に重複。audio.py に公開関数として集約し、
backend は import する。

## 候補③: openai_realtime の死にコード除去

`_process_sentence_boundaries_locked`(呼び出し0)とその連鎖
(`_commit_prefix_as_final_locked`・`_first_complete_sentence_end_ja/en`・
専用定数 `JA/EN_SENTENCE_ENDS`)を削除。`_clean_leading` は生きた emit 経路で
使用中のため残す。

## 候補④: main.py のバックエンド「3重スイッチ」解消

引数グループ・`_build_backend`・`_emit_settings` の3箇所に散るバックエンド分岐を
レジストリ/スペック化する。

## 候補⑤: 要約レイヤの統合

要約クラスが4つ・2ファイルに分散。`Summarizer` Protocol を定義し、urllib POST を
共通ヘルパに集約する。

## 候補⑥: 私的シンボルのモジュール跨ぎ import

`openai_chat.py` が `translator.py` の private `_parse_src_tgt` を import。
公開 API 化(`parse_src_tgt`)するか共有パースモジュールへ移動する。
