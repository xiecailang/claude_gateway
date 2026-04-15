from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from config import GatewayConfig
from converter import build_upstream_request, build_non_streaming_response
from logger import setup_logger, log_request, log_response, log_error
from sse_handler import SSEAccumulator, handle_streaming


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=GatewayConfig.timeout)
    app.state.logger = setup_logger()
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="vLLM Anthropic Gateway", lifespan=lifespan)


@app.get("/")
async def health():
    return {"status": "ok", "upstream": GatewayConfig.upstream_base_url}


@app.get("/v1/messages/count")
async def count_tokens():
    """Mock token counting endpoint. vLLM completions does not support this."""
    return JSONResponse(content={"input_tokens": 0, "output_tokens": 0})


@app.post("/v1/messages")
async def handle_messages(request: Request):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    logger = request.app.state.logger
    http_client = request.app.state.http_client
    start_time = time.monotonic()

    body = await request.json()
    log_request(logger, request_id, body)

    upstream_url = f"{GatewayConfig.upstream_base_url}/v1/completions"
    upstream_body = build_upstream_request(body)

    try:
        if upstream_body.get("stream"):
            return await _handle_streaming(
                logger, http_client, request_id, upstream_url, upstream_body, start_time
            )
        else:
            return await _handle_non_streaming(
                logger, http_client, request_id, upstream_url, upstream_body, start_time
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


async def _handle_streaming(logger, http_client, request_id, url, body, start_time):
    response = await http_client.post(url, json=body)

    accumulator = SSEAccumulator()

    async def event_generator():
        try:
            async for event in handle_streaming(response, accumulator):
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


async def _handle_non_streaming(logger, http_client, request_id, url, body, start_time):
    response = await http_client.post(url, json=body)
    duration = (time.monotonic() - start_time) * 1000
    upstream_body = response.json()

    anthropic_response = build_non_streaming_response(
        request_id, upstream_body, finish_reason="end_turn"
    )

    choices = upstream_body.get("choices", [])
    text = choices[0].get("text", "") if choices else ""

    log_response(
        logger,
        request_id,
        200,
        text,
        upstream_body.get("usage"),
        "end_turn",
        duration,
    )

    return JSONResponse(
        status_code=200,
        content=anthropic_response,
        headers={"X-Request-Id": request_id},
    )
