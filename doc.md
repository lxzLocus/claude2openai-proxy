> **📌 このドキュメントについて**  
> このドキュメントは [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) のフォークリポジトリに関連する技術情報です。

# Claude Code + Local LLM 利用時の現状と問題点

## 概要

Claude Code（以下 cc）を Local LLM（Ollama / vLLM / llama.cpp 等）経由で利用した際、  
**本来表示されるはずの Tool 承認 UI（yes / no / edit など）が表示されず、  
ツール実行フローが正常に機能しない問題**が発生している。

本ドキュメントは、現在起きている事象・原因・制約を整理したものである。

---

## 本来の Claude Code の動作（Anthropic Claude 利用時）

1. LLM が Plan Mode で作業計画を出力する
2. LLM が `ExitPlanMode` を呼び出す
3. cc が `allowedPrompts` を解釈する
4. ユーザーに **yes / no / edit** などの選択肢を UI として表示する
5. ユーザーが yes を選択すると、対応する Tool（例: Bash）が実行される

この一連の流れは、Anthropic Claude SDK が返す  
**構造化された tool_call / stop_reason イベント**を前提としている。

---

## 現在起きている事象（Local LLM 利用時）

- LLM の出力ログ上では以下が確認できる
  - `[Tool Use: Write]`
  - `[Tool Use: ExitPlanMode]`
- しかし、以下が発生しない
  - Tool 承認 UI（yes / no / edit）の表示
  - ユーザー選択による Tool 実行

結果として、

- Tool Use が「ログとして表示されるだけ」
- cc 側の状態遷移（Plan → 承認 → 実行）が行われない
- ユーザーは次に何を選択すべきか分からない状態になる

---

## 原因

### 1. Claude Code は「テキスト」ではなく「SDKイベント」で Tool を判定している

Claude Code は以下を前提に実装されている。

- Anthropic SDK が返す
  - `tool_call` オブジェクト
  - `stop_reason`（例: `tool_use`, `end_plan`）
  - tool 名・引数の構造化データ
- これらを **内部イベントとして受信**し、UI や実行フローを制御する

Local LLM はこれらの SDK 内部イベントを再現できず、  
**Tool Use を JSON 風テキストとして出力することしかできない**。

---

### 2. Local LLM の Tool 出力は cc から見ると「ただの文字列」

Local LLM が出力する以下のような内容は：

- `[Tool Use: ExitPlanMode]`
- `allowedPrompts` を含む JSON 風構造

であっても、

- cc から見ると「意味を持たない文字列」
- 承認 UI を起動するトリガにならない

そのため、

- Tool Use ログは表示される
- しかし Tool 承認フローは開始されない

という中途半端な状態になる。

---

## system instruction では解決できない理由

- system instruction は **LLM のテキスト出力の形**しか制御できない
- Claude Code が必要とする以下の要素は制御不能
  - `stop_reason`
  - `tool_call` イベント
  - `ExitPlanMode` による内部状態遷移
  - Tool 承認 UI の起動

つまり、

> system instruction でどれだけ Claude 互換の出力をさせても、  
> cc の Tool 承認 UI を復活させることはできない。

---

## 現在の制約まとめ

- Claude Code + Local LLM では以下が利用不可
  - Plan Mode 承認フロー
  - yes / no / edit の UI 表示
  - Tool 実行の自動トリガ

- これは設定ミスではなく、**設計上の非互換**である。

---

## 現実的な運用方針（暫定）

- Claude Code の Tool 承認機構には依存しない
- 以下を前提とした運用に切り替える
  - Tool Use / ExitPlanMode を出力させない
  - yes / no は通常テキストで確認する
  - 実行コマンドは明示的なテキストブロックとして出力する
  - 実行はユーザーまたは外部ラッパーが行う

---

## 結論

Claude Code は Anthropic Claude 専用に設計されており、  
Local LLM で完全互換動作をさせることはできない。

現在発生している問題は不具合ではなく、  
**SDK レイヤの非互換による仕様上の制約**である。
