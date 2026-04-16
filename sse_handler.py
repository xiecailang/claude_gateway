from __future__ import annotations

import json
import httpx
from converter import map_finish_reason


def _sse_event(data: dict) -> str:
    """Format an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def content_block_start_event() -> str:
    return _sse_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )


def content_block_delta_event(text: str) -> str:
    return _sse_event(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def message_delta_event(
    stop_reason: str,
    stop_sequence: str | None = None,
    output_tokens: int | None = None,
) -> str:
    data = {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
        },
        "usage": {"output_tokens": output_tokens} if output_tokens is not None else {},
    }
    return _sse_event(data)


class SSEAccumulator:
    """Holds accumulated text, usage, and finish_reason for logging."""
    full_text: str = ""
    usage: dict | None = None
    finish_reason: str | None = None


async def handle_streaming(
    response: httpx.Response,
    accumulator: SSEAccumulator,
) -> httpx.Response | None:
    """
    Consume upstream chat/completions SSE stream and yield Anthropic-formatted SSE events.

    Parses chat.completion.chunk format (delta.content, delta.reasoning,
    delta.tool_calls) and converts to Anthropic SSE events.
    Updates `accumulator` with full_text, usage, and finish_reason.
    """
    sent_content_block_start = False
    tool_block_starts: set[int] = set()
    buffer = ""

    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()

            # Skip empty lines and SSE comments
            if not line or line.startswith(":"):
                continue

            # Parse data: prefix
            if not line.startswith("data: "):
                continue

            data_str = line[6:]  # strip "data: "

            # Handle [DONE]
            if data_str.strip() == "[DONE]":
                yield "data: [DONE]\n\n"
                return

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Check for usage-only chunk (empty choices with usage)
            choices = data.get("choices") or []
            if not choices and data.get("usage"):
                u = data["usage"]
                output_tokens = u.get("completion_tokens", u.get("total_tokens", 0))
                accumulator.usage = u
                yield message_delta_event(
                    stop_reason=accumulator.finish_reason or "end_turn",
                    output_tokens=output_tokens,
                )
                continue

            # Regular completion chunk
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {})

                # Handle text content in delta
                text = delta.get("content", "") or ""

                # Also handle reasoning (some models put reasoning in delta.reasoning)
                reasoning = delta.get("reasoning", "") or ""
                if reasoning:
                    # Prefix reasoning with a marker so client can distinguish
                    text = reasoning + text

                # Check for finish_reason on this chunk
                fr = choice.get("finish_reason")
                if fr:
                    accumulator.finish_reason = map_finish_reason(fr)

                # Emit events for text
                if text:
                    if not sent_content_block_start:
                        yield content_block_start_event()
                        sent_content_block_start = True

                    yield content_block_delta_event(text)
                    accumulator.full_text += text

                # Handle tool_calls in delta
                tool_calls = delta.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        index = tc.get("index", 0)
                        function_info = tc.get("function", {})
                        name = function_info.get("name", "")
                        arguments_str = function_info.get("arguments", "")

                        # Emit content_block_start only once per tool index
                        if index not in tool_block_starts:
                            tool_block_starts.add(index)
                            tool_block = {
                                "type": "content_block_start",
                                "index": index + 1,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc.get("id", ""),
                                    "name": name,
                                    "input": {},
                                },
                            }
                            yield _sse_event(tool_block)

                        # Always emit delta for partial JSON
                        if arguments_str:
                            delta_block = {
                                "type": "content_block_delta",
                                "index": index + 1,
                                "delta": {"type": "input_json_delta", "partial_json": arguments_str},
                            }
                            yield _sse_event(delta_block)

    # Stream ended without [DONE], flush
    yield "\n\n"
