# Claude Gateway

Anthropic-to-vLLM protocol translation gateway. Enables Claude Code (and any Anthropic-compatible client) to work with a local vLLM inference engine.

## Architecture

```
Claude Code / Anthropic Client
        |
        |  POST /v1/messages (Anthropic format)
        v
   +---------+
   | Gateway |  FastAPI on port 9000
   +---------+
        |
        |  POST /v1/completions (OpenAI format)
        v
   vLLM Server (port 8000)
```

## Features

- **Protocol Translation**: Converts Anthropic `/v1/messages` API to OpenAI `/v1/completions` API
- **SSE Streaming**: Full Server-Sent Events support with real-time event conversion
- **Usage Tracking**: Forces `stream_options: { include_usage: true }` on all upstream requests
- **Structured Logging**: JSON-lines format, one line per request + one line per response
- **Request Tracing**: UUID-based request ID correlation across request/response pairs
- **Token Cache Visibility**: Logs `prompt_tokens`, `completion_tokens`, `cached_tokens`, and `total_tokens`
- **Error Handling**: Graceful 502/504 responses with Anthropic-compatible error format

## Quick Start

### 1. Prerequisites

- Python 3.12+
- vLLM running locally (default: `http://127.0.0.1:8000`)

### 2. Install Dependencies

```bash
cd gateway
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the Gateway

```bash
source venv/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 9000
```

### 4. Configure Claude Code

Create `.claude/settings.local.json` in your project directory:

```json
{
    "env": {
        "ANTHROPIC_AUTH_TOKEN": "fake-token",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9000"
    },
    "model": "vllm-local"
}
```

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `VLLM_UPSTREAM_URL` | `http://127.0.0.1:8000` | vLLM base URL |
| `VLLM_UPSTREAM_MODEL` | *(see config.py)* | vLLM model path |
| `GATEWAY_MODEL` | `vllm-local` | Model name exposed to clients |
| `GATEWAY_PORT` | `9000` | Gateway listen port |
| `GATEWAY_LOG_FILE` | `logs/requests.log` | Log file path |
| `GATEWAY_TIMEOUT` | `300` | Upstream request timeout (seconds) |

Example:

```bash
VLLM_UPSTREAM_MODEL=/path/to/your/model \
GATEWAY_PORT=8080 \
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080
```

## Log Format

Logs are JSON-lines, one entry per line in `logs/requests.log`:

### Request Line

```json
{"event":"request","timestamp":"2026-04-15T10:00:00+00:00","request_id":"abc-123","method":"POST","path":"/v1/messages","model":"vllm-local","max_tokens":32000,"system":"...","messages":[{"role":"user","content":"hello"}]}
```

### Response Line

```json
{"event":"response","timestamp":"2026-04-15T10:00:01+00:00","request_id":"abc-123","status_code":200,"text":"full completion text...","usage":{"prompt_tokens":6125,"completion_tokens":60,"total_tokens":6185,"prompt_tokens_details":{"cached_tokens":48}},"finish_reason":"end_turn","duration_ms":4625.7}
```

### Analyzing Logs

```bash
# Show all request/response pairs with token usage
cat logs/requests.log | python3 -c "
import sys, json
pairs = {}
for line in sys.stdin:
    obj = json.loads(line.strip())
    pairs.setdefault(obj['request_id'], {})[obj['event']] = obj
for rid, data in pairs.items():
    req = data.get('request', {})
    resp = data.get('response', {})
    usage = resp.get('usage', {})
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) if usage.get('prompt_tokens_details') else 0
    print(f'{rid[:8]} | prompt={usage.get(\"prompt_tokens\",0)} cached={cached} completion={usage.get(\"completion_tokens\",0)} total={usage.get(\"total_tokens\",0)} | dur={resp.get(\"duration_ms\",0)}ms')
"
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/v1/messages` | Main endpoint (streaming + non-streaming) |
| `GET` | `/v1/messages/count` | Mock token counting (returns zeros) |

## Project Structure

```
gateway/
├── config.py          # Configuration with env var overrides
├── converter.py       # Anthropic <-> OpenAI protocol conversion
├── sse_handler.py     # SSE streaming consumption and event conversion
├── logger.py          # JSON-lines structured logging
├── main.py            # FastAPI application and route handlers
├── models.py          # Pydantic request/response schemas
├── requirements.txt   # Python dependencies
└── logs/
    └── requests.log   # Runtime request/response trace logs
```

## How It Works

1. **Request**: Claude Code sends `POST /v1/messages` with Anthropic format (messages array, system prompt as top-level field)
2. **Conversion**: Gateway extracts and concatenates system + messages into a single prompt string
3. **Upstream**: Sends `POST /v1/completions` to vLLM with `stream_options: { include_usage: true }`
4. **Streaming**: For streaming responses, converts each OpenAI completion chunk to Anthropic SSE events:
   - `content_block_start` (first text)
   - `content_block_delta` (each text increment)
   - `message_delta` (stop_reason + usage)
   - `[DONE]`
5. **Logging**: Records both request and response as single JSON lines with full usage information
