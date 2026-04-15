from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from config import GatewayConfig


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("gateway")
    logger.setLevel(logging.INFO)

    log_dir = os.path.dirname(GatewayConfig.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handler = logging.FileHandler(GatewayConfig.log_file)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


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
