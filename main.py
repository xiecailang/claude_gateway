from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from config import GatewayConfig
from converter import (
    build_upstream_request,
    build_non_streaming_response,
    convert_messages,
)
from logger import (
    setup_logger,
    setup_debug_logger,
    setup_org_logger,
    log_request,
    log_response,
    log_error,
    log_debug_request,
    log_debug_response,
    log_debug_streaming_response,
    log_org_request,
    log_org_response,
    log_org_stream_chunk,
)
from sse_handler import SSEAccumulator, handle_streaming


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=GatewayConfig.timeout)
    app.state.logger = setup_logger()
    app.state.debug_logger = setup_debug_logger()
    app.state.org_logger = setup_org_logger()
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="vLLM Anthropic Gateway", lifespan=lifespan)


@app.get("/")
async def health():
    return {"status": "ok", "upstream": GatewayConfig.upstream_base_url}


@app.get("/v1/messages/count")
async def count_tokens():
    """Mock token counting endpoint. vLLM does not support this natively."""
    return JSONResponse(content={"input_tokens": 0, "output_tokens": 0})


@app.post("/v1/messages")
async def handle_messages(request: Request):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    logger = request.app.state.logger
    debug_logger = request.app.state.debug_logger
    http_client = request.app.state.http_client
    start_time = time.monotonic()

    body = await request.json()
    log_request(logger, request_id, body)
    org_logger = request.app.state.org_logger
    log_org_request(org_logger, request_id, body)

    # Build prompt preview for debug logging
    system_raw = body.get("system", "")
    messages = body.get("messages", [])
    openai_messages = convert_messages(system_raw, messages)
    built_prompt = "\n".join(
        str(m.get("content", ""))[:200] for m in openai_messages if isinstance(m.get("content"), str)
    )

    upstream_url = f"{GatewayConfig.upstream_base_url}/v1/chat/completions"
    upstream_body = build_upstream_request(body)

    # Debug log the full request
    log_debug_request(
        debug_logger, request_id, body,
        body.get("model"), system_raw, messages, built_prompt,
        upstream_url, upstream_body,
    )

    try:
        if upstream_body.get("stream"):
            return await _handle_streaming(
                logger, debug_logger, org_logger, http_client, request_id,
                upstream_url, upstream_body, start_time,
            )
        else:
            return await _handle_non_streaming(
                logger, debug_logger, org_logger, http_client, request_id,
                upstream_url, upstream_body, start_time,
            )
    except httpx.ConnectError as e:
        duration = (time.monotonic() - start_time) * 1000
        log_error(logger, request_id, str(e), 502, duration)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Upstream vLLM is not reachable at {upstream_url}: {e}",
                },
            },
        )
    except httpx.TimeoutException:
        duration = (time.monotonic() - start_time) * 1000
        log_error(logger, request_id, "Upstream timeout", 504, duration)
        return JSONResponse(
            status_code=504,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Upstream vLLM timed out after {GatewayConfig.timeout}s",
                },
            },
        )
    except httpx.HTTPStatusError as e:
        duration = (time.monotonic() - start_time) * 1000
        error_body = e.response.text
        log_error(logger, request_id, f"HTTP {e.response.status_code}: {error_body}", e.response.status_code, duration)
        return JSONResponse(
            status_code=e.response.status_code,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Upstream error: {error_body}",
                },
            },
        )


async def _handle_streaming(logger, debug_logger, org_logger, http_client, request_id, url, body, start_time):
    response = await http_client.post(url, json=body)

    accumulator = SSEAccumulator()

    async def event_generator():
        try:
            async for event in handle_streaming(response, accumulator, org_logger, request_id):
                yield event
        except Exception:
            pass
        finally:
            duration = (time.monotonic() - start_time) * 1000
            log_response(
                logger,
                request_id,
                200,
                accumulator.full_text,
                accumulator.usage,
                accumulator.finish_reason,
                duration,
                first_token_ts=accumulator.first_token_ts,
                last_token_ts=accumulator.last_token_ts,
            )
            log_debug_streaming_response(
                debug_logger, request_id,
                accumulator.full_text, accumulator.usage,
                accumulator.finish_reason, duration,
                first_token_ts=accumulator.first_token_ts,
                last_token_ts=accumulator.last_token_ts,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-Id": request_id,
        },
    )


async def _handle_non_streaming(logger, debug_logger, org_logger, http_client, request_id, url, body, start_time):
    response = await http_client.post(url, json=body)
    duration = (time.monotonic() - start_time) * 1000
    upstream_body = response.json()

    log_org_response(org_logger, request_id, upstream_body)

    anthropic_response = build_non_streaming_response(
        request_id, upstream_body, finish_reason="end_turn"
    )

    # Extract text from choices for logging
    choices = upstream_body.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    text = message.get("content", "") or ""
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        text += f"\n[tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}]"

    log_response(
        logger,
        request_id,
        200,
        text,
        upstream_body.get("usage"),
        anthropic_response.get("stop_reason"),
        duration,
    )

    log_debug_response(
        debug_logger, request_id, text,
        upstream_body.get("usage"), anthropic_response, duration,
    )

    return JSONResponse(
        status_code=200,
        content=anthropic_response,
        headers={"X-Request-Id": request_id},
    )
