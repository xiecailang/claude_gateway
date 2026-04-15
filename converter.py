from __future__ import annotations

import uuid
from config import GatewayConfig


def extract_text_content(content: str | list) -> str:
    """Extract text from content which may be a string or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def build_prompt(system: str, messages: list[dict]) -> str:
    """Convert Anthropic messages into a single prompt string for /v1/completions."""
    parts = []
    if system:
        parts.append(system)
    for msg in messages:
        role = msg.get("role", "user")
        text = extract_text_content(msg.get("content", ""))
        parts.append(f"[{role}]: {text}")
    return "\n\n".join(parts)


def build_upstream_request(
    anthropic_body: dict,
) -> dict:
    """Build the JSON body for the upstream /v1/completions request."""
    # system can be a string or list of content blocks (Claude Code sends list)
    system_raw = anthropic_body.get("system", "")
    if isinstance(system_raw, list):
        system_raw = extract_text_content(system_raw)

    prompt = build_prompt(
        system=system_raw,
        messages=anthropic_body.get("messages", []),
    )
    is_stream = anthropic_body.get("stream", False)
    body = {
        "model": GatewayConfig.upstream_model,
        "prompt": prompt,
        "max_tokens": anthropic_body.get("max_tokens", 1024),
        "stream": is_stream,
        "temperature": anthropic_body.get("temperature", 0.7),
        "top_p": anthropic_body.get("top_p", 1.0),
    }
    # vLLM rejects stream_options when stream=False
    if is_stream:
        body["stream_options"] = {"include_usage": True}
    return body


def map_finish_reason(openai_reason: str | None) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
        "tool_calls": "tool_use",
    }
    return mapping.get(openai_reason or "", "end_turn")


def build_non_streaming_response(
    request_id: str,
    upstream_body: dict,
    finish_reason: str | None = None,
) -> dict:
    """Convert a non-streaming /v1/completions response to Anthropic /v1/messages format."""
    choices = upstream_body.get("choices", [])
    text = ""
    if choices:
        text = choices[0].get("text", "")

    usage = upstream_body.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    reason = finish_reason or map_finish_reason(
        choices[0].get("finish_reason") if choices else None
    )

    return {
        "id": f"msg_{request_id.replace('-', '')[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": GatewayConfig.gateway_model,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
        "stop_reason": reason,
        "stop_sequence": None,
    }
