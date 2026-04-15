from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from config import GatewayConfig


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("gateway")
    logger.setLevel(logging.INFO)

    log_file = GatewayConfig.log_file
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def setup_debug_logger() -> logging.Logger:
    """Setup gateway.log for full request/response debugging."""
    logger = logging.getLogger("gateway_debug")
    logger.setLevel(logging.DEBUG)

    log_file = GatewayConfig.log_file
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    debug_log_file = os.path.join(log_dir, "gateway.log")
    handler = logging.FileHandler(debug_log_file, mode="a")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_debug_request(
    logger: logging.Logger,
    request_id: str,
    raw_body: dict,
    model: str,
    system_raw: str | list,
    messages: list,
    built_prompt: str,
    upstream_url: str,
    upstream_body: dict,
) -> None:
    """Log full raw request details to gateway.log."""
    sys_len = 0
    sys_type = type(system_raw).__name__
    sys_blocks = 0
    if isinstance(system_raw, str):
        sys_len = len(system_raw)
    elif isinstance(system_raw, list):
        sys_blocks = len(system_raw)
        sys_len = sum(len(b.get("text", "")) for b in system_raw)

    msg_total_len = 0
    msg_details = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            clen = len(content)
        else:
            clen = sum(len(c.get("text", "")) for c in content if isinstance(c, dict))
        msg_total_len += clen
        msg_details.append({"role": m.get("role"), "content_len": clen})

    prompt_preview = built_prompt[:500] + ("..." if len(built_prompt) > 500 else "")

    obj = {
        "event": "DEBUG_REQUEST",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "raw_body_all_keys": list(raw_body.keys()),
        "model": model,
        "system": {
            "type": sys_type,
            "blocks": sys_blocks,
            "total_chars": sys_len,
            "raw": system_raw,
        },
        "messages": {
            "count": len(messages),
            "total_chars": msg_total_len,
            "details": msg_details,
            "raw": messages,
        },
        "total_input_chars": sys_len + msg_total_len,
        "built_prompt_chars": len(built_prompt),
        "built_prompt_preview": prompt_preview,
        "upstream_url": upstream_url,
        "upstream_body_keys": list(upstream_body.keys()),
        "upstream_body_prompt_len": len(upstream_body.get("prompt", "")),
    }

    # Include tools and tool_choice if present in original request
    if "tools" in raw_body:
        obj["tools"] = raw_body["tools"]
    if "tool_choice" in raw_body:
        obj["tool_choice"] = raw_body["tool_choice"]
    if "max_tokens" in raw_body:
        obj["max_tokens"] = raw_body["max_tokens"]

    logger.debug(json.dumps(obj, ensure_ascii=False))


def log_debug_response(
    logger: logging.Logger,
    request_id: str,
    upstream_response_text: str,
    upstream_usage: dict | None,
    anthropic_response: dict,
    duration_ms: float,
) -> None:
    """Log full response details to gateway.log."""
    obj = {
        "event": "DEBUG_RESPONSE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "upstream_response_text": upstream_response_text[:5000],
        "upstream_text_chars": len(upstream_response_text),
        "upstream_usage": upstream_usage,
        "anthropic_response": anthropic_response,
        "duration_ms": round(duration_ms, 1),
    }
    logger.debug(json.dumps(obj, ensure_ascii=False))


def log_debug_streaming_response(
    logger: logging.Logger,
    request_id: str,
    accumulated_text: str,
    usage: dict | None,
    finish_reason: str | None,
    duration_ms: float,
) -> None:
    """Log streaming response details to gateway.log."""
    obj = {
        "event": "DEBUG_STREAMING_RESPONSE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "accumulated_text": accumulated_text[:5000],
        "accumulated_text_chars": len(accumulated_text),
        "usage": usage,
        "finish_reason": finish_reason,
        "duration_ms": round(duration_ms, 1),
    }
    logger.debug(json.dumps(obj, ensure_ascii=False))


def log_request(logger: logging.Logger, request_id: str, request: dict) -> None:
    obj = {
        "event": "request",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "method": "POST",
        "path": "/v1/messages",
        "model": request.get("model"),
        "max_tokens": request.get("max_tokens"),
        "system": request.get("system", ""),
        "messages": request.get("messages"),
    }
    logger.info(json.dumps(obj, ensure_ascii=False))


def log_response(
    logger: logging.Logger,
    request_id: str,
    status_code: int,
    text: str,
    usage: dict | None,
    finish_reason: str | None,
    duration_ms: float,
    max_length: int = GatewayConfig.max_log_text_length,
) -> None:
    if len(text) > max_length:
        text = text[:max_length] + "...(truncated)"

    obj = {
        "event": "response",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "status_code": status_code,
        "text": text,
        "usage": usage,
        "finish_reason": finish_reason,
        "duration_ms": round(duration_ms, 1),
    }
    logger.info(json.dumps(obj, ensure_ascii=False))


def log_error(
    logger: logging.Logger,
    request_id: str,
    error: str,
    status_code: int,
    duration_ms: float,
) -> None:
    obj = {
        "event": "error",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "status_code": status_code,
        "error": error,
        "duration_ms": round(duration_ms, 1),
    }
    logger.info(json.dumps(obj, ensure_ascii=False))
