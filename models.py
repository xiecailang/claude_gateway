from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[TextContent]


class AnthropicMessagesRequest(BaseModel):
    model: str
    max_tokens: int
    system: str = ""
    stream: bool = False
    messages: list[AnthropicMessage]


class AnthropicNonStreamingResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[dict]
    model: str
    usage: dict
    stop_reason: str
    stop_sequence: Optional[str] = None
