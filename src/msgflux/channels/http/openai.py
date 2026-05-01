import asyncio
import time
import uuid
from collections.abc import Mapping as ABCMapping
from typing import Any, AsyncIterator, Dict, Mapping, Optional

import msgspec

from msgflux.channels.exceptions import ChannelError, ForbiddenError, UnauthorizedError
from msgflux.channels.http.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
    ChatCompletionStreamDelta,
    ErrorDetails,
    ErrorResponse,
)
from msgflux.channels.registry import (
    AgentRun,
    ChannelContext,
    ChannelRegistry,
    call_processor,
)
from msgflux.models.response import ModelStreamResponse
from msgflux.utils.msgspec import msgspec_dumps

_REQUEST_DECODER = msgspec.json.Decoder(ChatCompletionRequest)
_ENCODER = msgspec.json.Encoder()


def _as_list(value: Any) -> list[Any]:
    return list(value)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value or {})


def _identity(value: Any) -> Any:
    return value


def decode_chat_completion_request(body: bytes) -> ChatCompletionRequest:
    return _REQUEST_DECODER.decode(body)


def encode_json(obj: Any) -> bytes:
    return _ENCODER.encode(obj)


def encode_error(
    message: str,
    *,
    code: str,
    error_type: str = "invalid_request",
) -> bytes:
    return encode_json(
        ErrorResponse(
            error=ErrorDetails(
                message=message,
                type=error_type,
                code=code,
            )
        )
    )


async def create_chat_completion(
    registry: ChannelRegistry,
    request: ChatCompletionRequest,
    *,
    http_request: Any = None,
) -> ChatCompletionResponse:
    request_id = _make_completion_id()
    context = ChannelContext(
        channel="http",
        agent_name=request.model,
        request_id=request_id,
        request=request,
    )
    if http_request is not None:
        context.state["http_request"] = http_request

    run = None
    try:
        await registry.run_hooks("request_start", request, context)
        await authenticate_request(registry, request, context, http_request)
        agent = registry.get_agent(request.model)
        run = await prepare_agent_run(registry, request, context)
        run.stream = False

        output = await agent.acall(
            messages=run.messages,
            vars=run.vars,
            model_preference=run.model_preference,
            tool_filter=run.tool_filter,
            stream=False,
            **run.kwargs,
        )
        output = await apply_post_processors(
            registry,
            request.model,
            output,
            context,
            run,
        )
        content, reasoning_content = _extract_message_content(output)

        response = ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        content=content,
                        reasoning_content=reasoning_content,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        await registry.run_hooks("request_end", request, context, run, response, None)
        return response
    except asyncio.CancelledError as e:
        await registry.run_hooks("request_end", request, context, run, None, e)
        raise
    except Exception as e:
        await registry.run_hooks("request_end", request, context, run, None, e)
        raise


async def create_chat_completion_stream(
    registry: ChannelRegistry,
    request: ChatCompletionRequest,
    *,
    http_request: Any = None,
) -> AsyncIterator[bytes]:
    request_id = _make_completion_id()
    created = int(time.time())
    context = ChannelContext(
        channel="http",
        agent_name=request.model,
        request_id=request_id,
        request=request,
    )
    if http_request is not None:
        context.state["http_request"] = http_request

    run = None
    try:
        await registry.run_hooks("request_start", request, context)
        await authenticate_request(registry, request, context, http_request)
        agent = registry.get_agent(request.model)
        run = await prepare_agent_run(registry, request, context)
        run.stream = True

        output = await agent.acall(
            messages=run.messages,
            vars=run.vars,
            model_preference=run.model_preference,
            tool_filter=run.tool_filter,
            stream=True,
            **run.kwargs,
        )
        output = await apply_post_processors(
            registry,
            request.model,
            output,
            context,
            run,
        )

        yield await _emit_sse_chunk(
            registry,
            _stream_chunk(
                request_id,
                created,
                request.model,
                delta=ChatCompletionStreamDelta(role="assistant"),
            ),
            context,
            run,
        )

        if isinstance(output, ModelStreamResponse):
            async for chunk in _stream_response_chunks(
                output,
                request_id=request_id,
                created=created,
                model=request.model,
            ):
                yield await _emit_sse_chunk(registry, chunk, context, run)
        else:
            content, reasoning_content = _extract_message_content(output)
            if reasoning_content:
                yield await _emit_sse_chunk(
                    registry,
                    _stream_chunk(
                        request_id,
                        created,
                        request.model,
                        delta=ChatCompletionStreamDelta(
                            reasoning_content=reasoning_content,
                        ),
                    ),
                    context,
                    run,
                )
            if content:
                yield await _emit_sse_chunk(
                    registry,
                    _stream_chunk(
                        request_id,
                        created,
                        request.model,
                        delta=ChatCompletionStreamDelta(content=content),
                    ),
                    context,
                    run,
                )

        yield await _emit_sse_chunk(
            registry,
            _stream_chunk(
                request_id,
                created,
                request.model,
                delta=ChatCompletionStreamDelta(),
                finish_reason=_finish_reason(output),
            ),
            context,
            run,
        )
        yield b"data: [DONE]\n\n"
        await registry.run_hooks("request_end", request, context, run, output, None)
    except asyncio.CancelledError as e:
        await registry.run_hooks("request_end", request, context, run, None, e)
        raise
    except Exception as e:
        await registry.run_hooks("request_end", request, context, run, None, e)
        raise


async def authenticate_request(
    registry: ChannelRegistry,
    request: ChatCompletionRequest,
    context: ChannelContext,
    http_request: Any = None,
) -> None:
    principal = None
    auth_handler = registry.auth_handler()
    if auth_handler is not None:
        principal = await call_processor(auth_handler, http_request, request, context)
        if principal is False:
            raise UnauthorizedError("Unauthorized")

    context.state["principal"] = principal
    context.state["auth"] = principal
    for authorizer in registry.authorizers(request.model):
        result = await call_processor(authorizer, request, context, principal)
        if result is False:
            raise ForbiddenError("Forbidden")
        if isinstance(result, ABCMapping):
            context.state.update(result)


async def _emit_sse_chunk(
    registry: ChannelRegistry,
    chunk: ChatCompletionStreamChunk,
    context: ChannelContext,
    run: AgentRun,
) -> bytes:
    await registry.run_hooks("stream_chunk", chunk, context, run)
    return _sse_chunk(chunk)


async def prepare_agent_run(
    registry: ChannelRegistry,
    request: ChatCompletionRequest,
    context: ChannelContext,
) -> AgentRun:
    run_config = _merged_run_config(request)
    run = AgentRun(
        messages=list(request.messages),
        vars=dict(run_config.get("vars") or {}),
        stream=request.stream,
        model_preference=run_config.get("model_preference"),
        tool_filter=run_config.get("tool_filter"),
    )

    for processor in registry.pre_processors(request.model):
        update = await call_processor(processor, request, context, run)
        run = _apply_run_update(run, update)

    return run


async def apply_post_processors(
    registry: ChannelRegistry,
    agent_name: str,
    output: Any,
    context: ChannelContext,
    run: AgentRun,
) -> Any:
    for processor in registry.post_processors(agent_name):
        update = await call_processor(processor, output, context, run)
        if update is not None:
            output = update
    return output


def _apply_run_update(run: AgentRun, update: Any) -> AgentRun:
    if update is None:
        return run
    if isinstance(update, AgentRun):
        return update
    if not isinstance(update, ABCMapping):
        raise ChannelError(
            "Pre processors must return None, AgentRun, or a mapping of run fields"
        )

    field_updates = {
        "messages": _as_list,
        "vars": _as_dict,
        "stream": _identity,
        "model_preference": _identity,
        "tool_filter": _identity,
        "kwargs": _as_dict,
    }
    target_fields = {"vars": "vars"}

    for field_name, transform in field_updates.items():
        if field_name in update:
            setattr(
                run,
                target_fields.get(field_name, field_name),
                transform(update[field_name]),
            )
    return run


def _merged_run_config(request: ChatCompletionRequest) -> Dict[str, Any]:
    return dict(request.run_config)


async def _stream_response_chunks(
    response: ModelStreamResponse,
    *,
    request_id: str,
    created: int,
    model: str,
) -> AsyncIterator[ChatCompletionStreamChunk]:
    generators = {
        "content": response.consume(),
        "reasoning": response.consume_reasoning(),
    }
    tasks = {
        asyncio.create_task(_read_next(generator)): (name, generator)
        for name, generator in generators.items()
    }

    while tasks:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            name, generator = tasks.pop(task)
            chunk, finished = task.result()
            if finished:
                continue

            if name == "reasoning":
                delta = ChatCompletionStreamDelta(reasoning_content=str(chunk))
            else:
                delta = ChatCompletionStreamDelta(content=_stringify(chunk))

            yield _stream_chunk(request_id, created, model, delta=delta)
            tasks[asyncio.create_task(_read_next(generator))] = (
                name,
                generator,
            )


async def _read_next(generator: AsyncIterator[Any]) -> tuple[Any, bool]:
    try:
        return await generator.__anext__(), False
    except StopAsyncIteration:
        return None, True


def _stream_chunk(
    request_id: str,
    created: int,
    model: str,
    *,
    delta: ChatCompletionStreamDelta,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> ChatCompletionStreamChunk:
    return ChatCompletionStreamChunk(
        id=request_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _sse_chunk(chunk: ChatCompletionStreamChunk) -> bytes:
    return b"data: " + encode_json(chunk) + b"\n\n"


def _extract_message_content(output: Any) -> tuple[str, Optional[str]]:
    reasoning_content = None
    if isinstance(output, ABCMapping):
        reasoning_content = output.get("reasoning") or output.get("reasoning_content")
        if "answer" in output:
            output = output["answer"]
        elif "response" in output:
            output = output["response"]

    return (
        _stringify(output),
        _stringify(reasoning_content) if reasoning_content else None,
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return msgspec_dumps(value)


def _finish_reason(output: Any) -> str:
    metadata = getattr(output, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get("finish_reason") or metadata.get("stop_reason") or "stop"
    return "stop"


def _make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"
