from typing import Any, Dict, List, Literal, Optional

import msgspec


class ChatCompletionRequest(
    msgspec.Struct,
    kw_only=True,
    forbid_unknown_fields=False,
):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    run_config: Dict[str, Any] = msgspec.field(default_factory=dict)
    stream_options: Optional[Dict[str, Any]] = None


class ChatCompletionMessage(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=False,
):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatCompletionChoice(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=False,
):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=False,
):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, Any]] = None


class ChatCompletionStreamDelta(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=False,
):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatCompletionStreamChoice(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=False,
):
    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: Optional[str] = None


class ChatCompletionStreamChunk(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=False,
):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[Dict[str, Any]] = None


class ErrorDetails(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=False,
):
    message: str
    type: str
    code: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None


class ErrorResponse(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=False,
):
    error: ErrorDetails
