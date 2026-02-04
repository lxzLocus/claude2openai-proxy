# claude-to-openai proxy

> **📌 このリポジトリについて**  
> このリポジトリは [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) からのフォークです。
> 
> **追加された機能：**
> - 🔧 汎用利用のための修正・拡張
> - 📊 トークンカウント機能の追加
> - 🎯 モデルマッピングの強化（環境変数とプレフィックスベース）
> - 📝 詳細なOpenAPI/Swagger UIドキュメント
> - 🔇 ログノイズの削減と整形されたリクエストログ

## Overview
FastAPI proxy translating Anthropic-style /v1/messages to OpenAI via LiteLLM. Handles model mapping, content block conversion, streaming (SSE), tools, and token counting with reduced logging noise.

- Original code came from: https://github.com/1rgs/claude-code-proxy
- fixed for generic usages

## Features
- Model mapping via env and prefixes (anthropic/* passthrough, openai/* normalized)
- Content-block to OpenAI messages conversion (text/image/tool_use/tool_result)
- SSE streaming compatible with Anthropic clients
- Tool calls passthrough; converts to Anthropic tool_use blocks on response
- Token counting endpoint using LiteLLM token_counter
- Minimal logging with filtered noise and pretty request logs
- OpenAPI schema and Swagger UI with Anthropic-compatible models and endpoint docs

## Run
```
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

## API Docs (OpenAPI / Swagger)
- OpenAPI JSON: GET `/_openapi.json` or `/openapi.json` (FastAPI default)
- Swagger UI: GET `/docs`
- ReDoc: GET `/redoc`

App metadata (title/description/version) and schema descriptions are embedded so the docs reflect the Anthropic request/response shapes and streaming behavior.

## Anthropic Compatibility Notes
- Endpoints accept Anthropic Messages API-shaped payloads.
- Streaming when `stream=true` returns Anthropic-style SSE events in this order:
  - `message_start` → `content_block_start` (text) → `content_block_delta`(n) → `content_block_stop`
  - optional tool call blocks: `content_block_start` (tool_use) → `input_json_delta`(n) → `content_block_stop`
  - `message_delta` (includes `stop_reason`, `usage`) → `message_stop` → final `data: [DONE]`
- Tooling:
  - Request tools follow Anthropic `tools[]` with `name`, `description`, `input_schema` (JSON Schema).
  - Responses may include `tool_use` blocks for Claude-family models; for non-Claude models, tool info is appended as text.

## Container

### Build
```
docker build -t claude-proxy .
```

### Run
```
docker run -d \
  --name claude-proxy \
  -p 8082:8082 \
  -e OPENAI_API_BASE="http://10.2.2.10:4000/v1" \
  -e BIG_MODEL=gpt-4.1 \
  -e SMALL_MODEL=gpt-4.1-mini \
  -e MAX_TOKENS=65535 \
  -e BIG_PREFIXES="opus,sonnet" \
  -e SMALL_PREFIXES="haiku" \
  claude-proxy
```

### Docker Compose
docker-compose.yml:
```
version: "3.8"
services:
  claude-proxy:
    image: claude-proxy
    container_name: claude-proxy
    ports:
      - "8082:8082"
    environment:
      OPENAI_API_BASE: "http://10.2.2.10:4000/v1"
      BIG_MODEL: "gpt-4.1"
      SMALL_MODEL: "gpt-4.1-mini"
      MAX_TOKENS: "65535"
      BIG_PREFIXES: "opus,sonnet"
      SMALL_PREFIXES: "haiku"
    restart: unless-stopped
```
Run with:
```
docker compose up -d
```

## Usages

- Claude Code
```
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY=sk-keyhere ANTHROPIC_MODEL="big-size-model" ANTHROPIC_SMALL_FAST_MODEL="small-size-model" claude
```

## Environment
- OPENAI_API_KEY: OpenAI key (or send via X-API-Key header)
- OPENAI_API_BASE: upstream OpenAI-compatible base URL (e.g., https://api.openai.com/v1)
- BASE_URL: external public prefix for this proxy behind LB/reverse-proxy (e.g., https://foo.bar/claude-proxy/v1). Only affects returned links/OpenAPI servers; not used for upstream calls
- BIG_MODEL: default big model (default: gpt-4.1)
- SMALL_MODEL: default small model (default: gpt-4.1-mini)
- BIG_PREFIXES: map prefixes to BIG_MODEL (default: opus,sonnet)
- SMALL_PREFIXES: map prefixes to SMALL_MODEL (default: haiku)
- MAX_TOKENS: cap for OpenAI models (default: 65535)

## Model Resolution
- If body.model exists: used as-is; provider prefix added if missing
- Else: when body.size=="small" use openai/SMALL_MODEL, otherwise openai/BIG_MODEL
- If model is an OpenAI known id without prefix, server prefixes as openai/<model>

## Endpoints
- POST /v1/messages
  - Request body: Anthropic-like { model, max_tokens, messages, system, tools, tool_choice, temperature, stream }
  - Behavior: converts to OpenAI format for LiteLLM, caps max_tokens for OpenAI models to MAX_TOKENS
  - Streaming: when stream=true, emits Anthropic SSE events (message_start, content_block_start/delta/stop, message_delta, message_stop, final [DONE])
  - Auth: Header `X-API-Key: <OpenAI key>` (OpenAPI 문서에 노출됨)
- POST /v1/messages/count_tokens
  - Returns { input_tokens }
  - Auth: Header `X-API-Key: <OpenAI key>`
- GET /
  - Returns { message, docs, redoc, openapi }

## Content Conversion Notes
- User/assistant messages may be string or content blocks; lists are flattened to text for OpenAI
- tool_result blocks in user messages are converted to plain text segments
- image blocks emit a placeholder text when targeting OpenAI text endpoints
- Unsupported fields in messages are removed to satisfy OpenAI API

## Headers/Auth
- Send `X-API-Key: <OpenAI key>`
- If missing, request is rejected with 401

## Examples
Simple completion:
```
curl -sS -X POST http://localhost:8082/v1/messages \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Say hi"}
    ]
  }'
```

Streaming response:
```
curl -N -sS -X POST http://localhost:8082/v1/messages \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "stream": true,
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "Stream a short sentence"}
    ]
  }'
```

Token count:
```
curl -sS -X POST http://localhost:8082/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "How many tokens?"}
    ]
  }'
```

## Logging
- Global WARN level, uvicorn access/error suppressed
- Filters noisy LiteLLM/internal messages
- Pretty per-request log showing Anthropic→OpenAI mapping and counts

## Security Notes
- Do not log secrets; API key is read from header and assigned to process env for the call, then restored
- Validate inputs where possible; unsupported fields are stripped before upstream calls