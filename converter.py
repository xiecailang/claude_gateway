from __future__ import annotations

import json

from config import GatewayConfig


def convert_messages(
    system: str | list,
    messages: list[dict],
) -> list[dict]:
    """Convert Anthropic messages to OpenAI chat messages.

    Anthropic uses a separate `system` field; OpenAI embeds it as a
    `role: "system"` message at the beginning of the conversation.
    Anthropic content can be a string or a list of content blocks.
    """
    # System message
    openai_messages: list[dict] = []
    system_text = _extract_text(system) if system else ""
    if system_text:
        openai_messages.append({"role": "system", "content": system_text})

    # Convert each message
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Anthropic assistant messages can contain tool_use blocks which
        # we pass through as-is (they map to OpenAI assistant tool_calls).
        # For user messages with tool_result blocks, convert to OpenAI format.
        if role == "user" and isinstance(content, list):
            # Check if this is a tool_result message
            tool_results = []
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_result":
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
                    elif block.get("type") == "text":
                        text_parts.append(block["text"])
                    else:
                        text_parts.append(str(block))
                else:
                    text_parts.append(str(block))

            if tool_results:
                # OpenAI tool result message
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_results[0]["tool_use_id"],
                    "content": tool_results[0]["content"],
                })
            if text_parts:
                openai_messages.append({
                    "role": "user",
                    "content": "\n".join(text_parts),
                })
        elif role == "assistant" and isinstance(content, list):
            # Assistant message with potential tool_use blocks
            text_parts = []
            tool_uses = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_uses.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })

            assistant_msg: dict = {"role": "assistant"}
            if text_parts:
                assistant_msg["content"] = "\n".join(text_parts)
            if tool_uses:
                assistant_msg["tool_calls"] = tool_uses
                # If there are tool calls, content should be None per OpenAI spec
                assistant_msg.setdefault("content", None)
            openai_messages.append(assistant_msg)
        else:
            # Simple string content
            openai_messages.append({
                "role": role,
                "content": _extract_text(content),
            })

    return openai_messages


def convert_tools(tools: list[dict] | None) -> tuple[list[dict] | None, str | None]:
    """Convert Anthropic tools to OpenAI tools format.

    Returns (openai_tools, tool_choice) tuple.
    """
    if not tools:
        return None, None

    openai_tools = []
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        input_schema = tool.get("input_schema", {})

        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": input_schema,
            },
        })

    return openai_tools, "auto"


def build_upstream_request(anthropic_body: dict) -> dict:
    """Build the JSON body for the upstream /v1/chat/completions request."""
    system_raw = anthropic_body.get("system", "")
    messages = anthropic_body.get("messages", [])
    is_stream = anthropic_body.get("stream", False)

    openai_messages = convert_messages(system_raw, messages)
    openai_tools, tool_choice = convert_tools(anthropic_body.get("tools"))

    body = {
        "model": GatewayConfig.upstream_model,
        "messages": openai_messages,
        "max_tokens": anthropic_body.get("max_tokens", 1024),
        "stream": is_stream,
        "temperature": anthropic_body.get("temperature", 0.7),
        "top_p": anthropic_body.get("top_p", 1.0),
    }

    if openai_tools:
        body["tools"] = openai_tools
        body["tool_choice"] = tool_choice

    # vLLM rejects stream_options when stream=False
    if is_stream:
        body["stream_options"] = {"include_usage": True}

    return body


def _extract_text(content: str | list) -> str:
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
    """Convert a non-streaming /v1/chat/completions response to Anthropic format."""
    choices = upstream_body.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    usage = upstream_body.get("usage", {})

    # Build content blocks from the response
    content = []

    # Text content - check both content and reasoning fields
    text = message.get("content", "") or ""
    reasoning = message.get("reasoning", "") or ""
    if reasoning:
        text = reasoning + text  # reasoning comes first
    if text:
        content.append({"type": "text", "text": text})

    # Tool calls
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        try:
            arguments = tc["function"]["arguments"]
            if isinstance(arguments, str):
                import json
                arguments = json.loads(arguments)
        except Exception:
            arguments = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": tc["function"]["name"],
            "input": arguments,
        })

    if not content:
        content = [{"type": "text", "text": ""}]

    reason = finish_reason or map_finish_reason(
        choices[0].get("finish_reason") if choices else None
    )

    return {
        "id": f"msg_{request_id.replace('-', '')[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": GatewayConfig.gateway_model,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "stop_reason": reason,
        "stop_sequence": None,
    }
