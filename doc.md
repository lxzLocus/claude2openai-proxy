# Anthropic互換プロキシ（LiteLLM + LM Studio）  
## 500エラー / max_tokens 問題 切り分け・確認手順書  
（Copilot 作業指示用ドキュメント）

---

## 目的

- `/v1/messages`（Anthropic互換）で `max_tokens` を指定すると 500 が出る問題の原因特定
- **LM Studio / LiteLLM / プロキシ変換層** のどこが壊れているかを切り分ける
- Copilot に「何を確認・修正させるか」を明確化する

---

## 全体構成（想定アーキテクチャ）

```
Client
  → Anthropic互換 API (/v1/messages)
    → FastAPI Proxy
      → LiteLLM
        → LM Studio (OpenAI互換)
```

---

## 結論サマリ（重要）

- `max_tokens` は **正しい入力**
- 500 の直接原因は **プロキシ側の例外処理バグ**
- `json.dumps()` に **litellm.Response / starlette.Response** を渡して落ちている
- LiteLLM → LM Studio 呼び出し自体は **成功している可能性が高い**

---

## 1. LM Studio 単体の健全性確認（最優先）

### 確認内容
- LM Studio が OpenAI互換 API として正常動作しているか

### 実行コマンド
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [
      { "role": "user", "content": "こんにちは" }
    ],
    "max_tokens": 128
  }'
```

### OK条件
- HTTP 200
- `choices[0].message.content` が存在

### NGなら
- LM Studio 側のモデル / API 設定を修正
- この時点でプロキシは無関係

---

## 2. LiteLLM 単体の健全性確認（変換なし）

### 確認内容
- LiteLLM → LM Studio が問題なく動作するか

### 実行コマンド
```bash
curl http://<proxy-host>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [
      { "role": "user", "content": "こんにちは" }
    ],
    "max_tokens": 128
  }'
```

### OK条件
- HTTP 200
- OpenAI互換レスポンスが返る

### NGなら
- LiteLLM provider 設定（base_url / api_key / model名）を修正

---

## 3. Anthropic互換 `/v1/messages` の入力仕様確認

### 推奨リクエスト（安全側）

```json
{
  "model": "kimi-k2.5",
  "max_tokens": 128,
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "こんにちは。自己紹介してください。" }
      ]
    }
  ]
}
```

### 確認ポイント
- `max_tokens` が **int**
- `messages[].content` が
  - string か
  - `{ type: "text", text: "..." }` 配列
- `system` は messages に混ぜていない

---

## 4. create_message() 内の例外処理確認（最重要）

### 問題箇所（典型）

```python
logger.error(
  json.dumps(error_details, indent=2, ensure_ascii=False)
)
```

### NG理由
- `error_details` に以下が含まれる可能性：
  - `litellm.Response`
  - `starlette.responses.Response`
- → JSONシリアライズ不可 → **例外処理中に例外**

---

## 5. 応急修正（必須）

### 修正指示（Copilot向け）

#### 最低限の修正
```python
json.dumps(error_details, ensure_ascii=False, default=str)
```

#### または安全変換
```python
safe_error_details = {
    k: str(v) for k, v in error_details.items()
}
logger.error(json.dumps(safe_error_details, ensure_ascii=False))
```

---

## 6. LiteLLM Response の扱い確認

### 確認事項
- LiteLLM の戻り値の型をログ出力

```python
logger.debug(type(response))
logger.debug(str(response))
```

### 対応方針
- `Response` オブジェクトは
  - `dict(response)`
  - `response.model_dump()`
  - `str(response)`
  のいずれかに変換してから扱う

---

## 7. Anthropic → OpenAI 変換ロジックの確認点

### よく壊れるポイント
- `max_tokens` + prompt_tokens > context_length
- `choices[0].message.content` が null
- streaming=false なのに stream 処理に入る
- tool / function 呼び出し未対応

### 確認指示
- except に入る条件をすべて列挙
- 例外を **raise せず握りつぶしていないか** 確認

---

## 8. デバッグ用フラグ

### LiteLLM
```python
litellm._turn_on_debug()
```

### FastAPI
- request body をそのまま dump（Responseは除外）

---

## 9. 最終判定フローチャート

```
LM Studio OK?
  ├─ NO → LM Studio 修正
  └─ YES
      ↓
LiteLLM /v1/chat/completions OK?
  ├─ NO → LiteLLM 設定修正
  └─ YES
      ↓
/v1/messages + max_tokens で例外？
  ├─ YES → プロキシ変換 / 例外処理バグ
  └─ NO → 解決
```

---

## 10. Copilot への最終指示（そのまま貼れる）

- `json.dumps()` に渡しているオブジェクトを全洗い出し
- `Response` 型を **絶対に dumps しない**
- LiteLLM response を dict / str に変換
- `/v1/chat/completions` と `/v1/messages` の差分をコメント化

---

## 補足

この問題は **max_tokens が原因ではない**。  
「正しい入力で壊れるコード」を直すのが目的。

---