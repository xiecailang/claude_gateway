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
    Consume upstream SSE stream and yield Anthropic-formatted SSE events.

    Updates `accumulator` with full_text, usage, and finish_reason.
    Yields SSE-formatted strings suitable for StreamingResponse.
    """
    sent_content_block_start = False
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

            # Check for usage-only chunk (empty choices)
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
                text = choice.get("text", "")

                # First text: emit content_block_start
                if not sent_content_block_start:
                    yield content_block_start_event()
                    sent_content_block_start = True

                # Emit delta text
                if text:
                    yield content_block_delta_event(text)
                    accumulator.full_text += text

                # Check for finish_reason
                fr = choice.get("finish_reason")
                if fr:
                    accumulator.finish_reason = map_finish_reason(fr)

    # Stream ended without [DONE], flush
    yield "\n\n"
