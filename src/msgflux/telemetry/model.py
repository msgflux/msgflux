"""OpenTelemetry GenAI spans for the shared model HTTP transport."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from contextlib import aclosing, closing
from functools import wraps
from typing import Any

from msgtrace.sdk.tracer import tracer_manager
from opentelemetry.trace import SpanKind, Status, StatusCode

from msgflux.envs import envs
from msgflux.models.usage import default_usage_codec


def _operation(endpoint: str) -> str:
    path = endpoint.split("?", 1)[0].rstrip("/")
    if path.endswith(("/chat/completions", "/responses", "/api/chat", "/v1/messages")):
        return "chat"
    if path.endswith("/embeddings"):
        return "embeddings"
    if path.endswith("/completions"):
        return "text_completion"
    if path.endswith("/rerank"):
        return "rerank"
    return "generate_content"


def _value(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            text
            for part in value
            if isinstance(text := _value(part, "text") or _value(part, "refusal"), str)
        ]
        return "".join(parts) if parts else None
    return None


def _stop_reason(payload: Any) -> str | None:
    status = _value(payload, "status")
    if status == "completed":
        return "stop"
    if status == "incomplete":
        reason = _value(_value(payload, "incomplete_details"), "reason")
        return "length" if reason == "max_output_tokens" else reason or "incomplete"
    if status in {"failed", "cancelled"}:
        return "error"
    return None


class _OutputCollector:
    """Collect text and stop reasons across one response or SSE stream."""

    def __init__(self) -> None:
        self.text: dict[int, str] = {}
        self.tool_calls: dict[int, dict[int, dict[str, str]]] = {}
        self.roles: dict[int, str] = {}
        self.reasons: dict[int, str] = {}

    def record(self, payload: Any) -> None:
        event_type = _value(payload, "type")
        if event_type == "response.output_text.delta":
            self._append(_value(payload, "output_index") or 0, _value(payload, "delta"))
        elif event_type == "response.output_text.done":
            self._replace(_value(payload, "output_index") or 0, _value(payload, "text"))

        if event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            payload = _value(payload, "response") or payload

        choices = _value(payload, "choices")
        if isinstance(choices, list):
            self._choices(choices)

        native_message = _value(payload, "message")
        if native_message is not None and not isinstance(choices, list):
            self._native_message(payload, native_message)

        if _value(payload, "type") == "message" and isinstance(
            _value(payload, "content"), list
        ):
            self._anthropic_message(payload)

        output = _value(payload, "output")
        if isinstance(output, list):
            self._response_output(output)

        reason = _stop_reason(payload)
        if reason:
            self.reasons[0] = reason

    def _native_message(self, payload: Any, message: Any) -> None:
        self._append(0, _text(_value(message, "content")))
        self.roles[0] = _value(message, "role") or "assistant"
        for position, tool in enumerate(_value(message, "tool_calls") or []):
            self._tool_call(0, position, tool, full=True)
        reason = _value(payload, "done_reason")
        if isinstance(reason, str):
            self.reasons[0] = reason

    def _anthropic_message(self, payload: Any) -> None:
        self.roles[0] = _value(payload, "role") or "assistant"
        for position, block in enumerate(_value(payload, "content")):
            block_type = _value(block, "type")
            if block_type == "text":
                self._append(0, _value(block, "text"))
            elif block_type == "tool_use":
                self._tool_call(
                    0,
                    position,
                    {
                        "id": _value(block, "id"),
                        "name": _value(block, "name"),
                        "arguments": _value(block, "input"),
                    },
                    full=True,
                )
        reason = _value(payload, "stop_reason")
        if isinstance(reason, str):
            self.reasons[0] = reason

    def _response_output(self, output: list[Any]) -> None:
        self.text.clear()
        self.tool_calls.clear()
        for position, item in enumerate(output):
            if _value(item, "type") == "message":
                self._append(0, _text(_value(item, "content")))
                self.roles[0] = _value(item, "role") or "assistant"
            elif _value(item, "type") == "function_call":
                self._tool_call(0, position, item, full=True)

    def _choices(self, choices: list[Any]) -> None:
        for position, choice in enumerate(choices):
            index = _value(choice, "index")
            index = index if isinstance(index, int) else position
            message = _value(choice, "message")
            delta = _value(choice, "delta")
            if message is not None:
                self._replace(
                    index,
                    _text(_value(message, "content")) or _value(message, "refusal"),
                )
                self.roles[index] = _value(message, "role") or "assistant"
                for tool_index, tool in enumerate(_value(message, "tool_calls") or []):
                    self._tool_call(index, tool_index, tool, full=True)
            elif delta is not None:
                self._append(index, _text(_value(delta, "content")))
                for tool_index, tool in enumerate(_value(delta, "tool_calls") or []):
                    call_index = _value(tool, "index")
                    self._tool_call(
                        index,
                        call_index if isinstance(call_index, int) else tool_index,
                        tool,
                        full=False,
                    )
            else:
                self._replace(index, _value(choice, "text"))
            reason = _value(choice, "finish_reason")
            if isinstance(reason, str):
                self.reasons[index] = reason

    def _tool_call(self, choice: int, position: int, tool: Any, *, full: bool) -> None:
        call = self.tool_calls.setdefault(choice, {}).setdefault(position, {})
        identifier = _value(tool, "id") or _value(tool, "call_id")
        if isinstance(identifier, str):
            call["id"] = identifier
        function = _value(tool, "function") or tool
        for key in ("name", "arguments"):
            value = _value(function, key)
            if isinstance(value, Mapping):
                call[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, str):
                call[key] = value if full else call.get(key, "") + value

    def _append(self, index: int, value: Any) -> None:
        if isinstance(value, str) and value:
            self.text[index] = self.text.get(index, "") + value

    def _replace(self, index: int, value: Any) -> None:
        if isinstance(value, str) and value:
            self.text[index] = value

    def finish(self, span: Any) -> None:
        if not span.is_recording():
            return
        if self.reasons:
            span.set_attribute(
                "gen_ai.response.finish_reasons",
                [self.reasons[index] for index in sorted(self.reasons)],
            )
        if envs.telemetry_capture_model_output and (self.text or self.tool_calls):
            messages = self._messages()
            if not messages:
                return
            span.set_attribute(
                "gen_ai.output.messages", json.dumps(messages, ensure_ascii=False)
            )

    def _messages(self) -> list[dict[str, Any]]:
        messages = []
        for index in sorted(self.text.keys() | self.tool_calls.keys()):
            parts = []
            if index in self.text:
                parts.append({"type": "text", "content": self.text[index]})
            for position in sorted(self.tool_calls.get(index, {})):
                tool = self.tool_calls[index][position]
                name = tool.get("name")
                if not name:
                    continue
                part: dict[str, Any] = {"type": "tool_call", "name": name}
                if identifier := tool.get("id"):
                    part["id"] = identifier
                if arguments := tool.get("arguments"):
                    try:
                        part["arguments"] = json.loads(arguments)
                    except ValueError:
                        pass
                parts.append(part)
            if parts:
                messages.append(
                    {"role": self.roles.get(index, "assistant"), "parts": parts}
                )
        return messages


def _request_attributes(body: Any, provider: str | None) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for field in (
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    ):
        value = _value(body, field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attributes[f"gen_ai.request.{field}"] = value
    if "gen_ai.request.max_tokens" not in attributes:
        for field in ("max_output_tokens", "max_completion_tokens"):
            value = _value(body, field)
            if isinstance(value, int) and not isinstance(value, bool):
                attributes["gen_ai.request.max_tokens"] = value
                break
    stop_sequences = _value(body, "stop_sequences") or _value(body, "stop")
    if isinstance(stop_sequences, list) and all(
        isinstance(sequence, str) for sequence in stop_sequences
    ):
        attributes["gen_ai.request.stop_sequences"] = stop_sequences
    if isinstance(_value(body, "stream"), bool):
        attributes["gen_ai.request.stream"] = _value(body, "stream")
    _reasoning_attributes(attributes, body, provider)
    return attributes


def _reasoning_attributes(
    attributes: dict[str, Any], body: Any, provider: str | None
) -> None:
    reasoning_level = _value(body, "reasoning_effort")
    if not isinstance(reasoning_level, str):
        reasoning_level = _value(_value(body, "reasoning"), "effort")
    if not isinstance(reasoning_level, str):
        reasoning_level = _value(_value(body, "output_config"), "effort")
    if isinstance(reasoning_level, str) and reasoning_level:
        attributes["gen_ai.request.reasoning.level"] = reasoning_level
    if provider == "anthropic":
        thinking = _value(body, "thinking")
        budget = _value(thinking, "budget_tokens")
        if isinstance(budget, int) and not isinstance(budget, bool):
            attributes["anthropic.request.thinking.budget_tokens"] = budget
        if _value(thinking, "type") == "disabled":
            attributes["gen_ai.request.reasoning.level"] = "none"
    if provider == "ollama":
        think = _value(body, "think")
        if isinstance(think, str) and think:
            attributes["gen_ai.request.reasoning.level"] = think
        elif isinstance(think, bool):
            attributes["ollama.request.think"] = think


def _span_context(owner: Any, endpoint: str, kwargs: dict[str, Any]):
    body = kwargs.get("json") or kwargs.get("data")
    operation = _operation(endpoint)
    model = _value(body, "model") or getattr(owner, "model_id", None)
    provider = getattr(owner, "provider", None)
    attributes: dict[str, Any] = {"gen_ai.operation.name": operation}
    if provider:
        attributes["gen_ai.provider.name"] = str(provider)
    if model:
        attributes["gen_ai.request.model"] = str(model)
    attributes.update(_request_attributes(body, provider))
    return tracer_manager.tracer.start_as_current_span(
        f"{operation} {model}" if model else operation,
        kind=SpanKind.CLIENT,
        attributes=attributes,
    )


def _record_payload(span: Any, payload: Any) -> None:
    if not span.is_recording():
        return
    event_type = _value(payload, "type")
    if event_type in {"response.completed", "response.incomplete", "response.failed"}:
        payload = _value(payload, "response") or payload
    response_id = _value(payload, "id")
    response_model = _value(payload, "model")
    if response_id:
        span.set_attribute("gen_ai.response.id", str(response_id))
    if response_model:
        span.set_attribute("gen_ai.response.model", str(response_model))
    status = _value(payload, "status")
    if isinstance(status, str):
        span.set_attribute("gen_ai.response.status", status)
        if status == "failed":
            span.set_status(Status(StatusCode.ERROR, "model response failed"))
    usage = _value(payload, "usage")
    if _value(payload, "type") == "message" and isinstance(usage, Mapping):
        usage = dict(usage)
        usage["input_tokens"] = sum(
            value
            for field in (
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
            if isinstance(value := usage.get(field), int)
            and not isinstance(value, bool)
        )
    if usage is None:
        native_usage = {
            field: value
            for field in ("prompt_eval_count", "eval_count")
            if isinstance(value := _value(payload, field), int)
            and not isinstance(value, bool)
        }
        usage = native_usage or None
    _record_usage(span, usage)


def _record_usage(span: Any, usage: Any) -> None:
    if usage is not None:
        normalized = default_usage_codec.normalize(usage)
        if normalized is not None:
            raw = normalized.raw
            if any(key in raw for key in default_usage_codec.input_token_fields):
                span.set_attribute("gen_ai.usage.input_tokens", normalized.input_tokens)
            if any(key in raw for key in default_usage_codec.output_token_fields):
                span.set_attribute(
                    "gen_ai.usage.output_tokens", normalized.output_tokens
                )
            details = normalized.input_tokens_details
            if details.cached_tokens is not None:
                span.set_attribute(
                    "gen_ai.usage.cache_read.input_tokens", details.cached_tokens
                )
            if details.cache_write_tokens is not None:
                span.set_attribute(
                    "gen_ai.usage.cache_creation.input_tokens",
                    details.cache_write_tokens,
                )
            output_details = normalized.output_tokens_details
            if output_details.reasoning_tokens is not None:
                span.set_attribute(
                    "gen_ai.usage.reasoning.output_tokens",
                    output_details.reasoning_tokens,
                )


def _record_http_response(span: Any, response: Any) -> None:
    if not span.is_recording():
        return
    span.set_attribute("http.response.status_code", response.status_code)
    if "json" in response.headers.get("content-type", ""):
        try:
            payload = response.json()
        except (ValueError, TypeError):
            pass
        else:
            _record_payload(span, payload)
            output = _OutputCollector()
            output.record(payload)
            output.finish(span)


def trace_model_request(func):
    """Trace one logical request, including retries and the stream lifetime."""
    if inspect.isasyncgenfunction(func):

        @wraps(func)
        async def async_stream(*args, **kwargs):
            with _span_context(args[1], args[2], kwargs) as span:
                output = _OutputCollector()
                try:
                    async with aclosing(func(*args, **kwargs)) as source:
                        async for item in source:
                            _record_payload(span, item)
                            output.record(item)
                            yield item
                finally:
                    output.finish(span)

        return async_stream

    if inspect.isgeneratorfunction(func):

        @wraps(func)
        def stream(*args, **kwargs):
            with _span_context(args[1], args[2], kwargs) as span:
                output = _OutputCollector()
                try:
                    with closing(func(*args, **kwargs)) as source:
                        for item in source:
                            _record_payload(span, item)
                            output.record(item)
                            yield item
                finally:
                    output.finish(span)

        return stream

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_request(*args, **kwargs):
            with _span_context(args[1], args[2], kwargs) as span:
                response = await func(*args, **kwargs)
                _record_http_response(span, response)
                return response

        return async_request

    @wraps(func)
    def request(*args, **kwargs):
        with _span_context(args[1], args[2], kwargs) as span:
            response = func(*args, **kwargs)
            _record_http_response(span, response)
            return response

    return request


def trace_model_call(owner, func):
    """Trace a legacy model call after its retry wrapper has been applied."""
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_call(**kwargs):
            with _span_context(
                owner, owner.endpoint, {"json": {"model": owner.model_id, **kwargs}}
            ) as span:
                response = await func(**kwargs)
                _record_payload(span, response)
                output = _OutputCollector()
                output.record(response)
                output.finish(span)
                return response

        return async_call

    @wraps(func)
    def call(**kwargs):
        with _span_context(
            owner, owner.endpoint, {"json": {"model": owner.model_id, **kwargs}}
        ) as span:
            response = func(**kwargs)
            _record_payload(span, response)
            output = _OutputCollector()
            output.record(response)
            output.finish(span)
            return response

    return call
