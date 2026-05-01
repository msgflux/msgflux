import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any

import msgspec

from msgflux.channels.exceptions import (
    ChannelError,
    PayloadTooLargeError,
    RequestTimeoutError,
)
from msgflux.channels.http.msgspec import make_msgspec_classes
from msgflux.channels.http.openai import (
    create_chat_completion,
    create_chat_completion_stream,
    decode_chat_completion_request,
    encode_error,
    encode_json,
)
from msgflux.channels.registry import ChannelRegistry, call_processor


def create_app(registry: ChannelRegistry, **fastapi_kwargs: Any):
    try:
        fastapi_cls = import_module("fastapi").FastAPI
        request_cls = import_module("fastapi").Request
        responses = import_module("fastapi.responses")
        response_cls = responses.Response
        streaming_response_cls = responses.StreamingResponse
    except ImportError as e:
        raise ImportError(
            "The msgflux server requires FastAPI. Install it with "
            "`pip install msgflux[server]`."
        ) from e

    settings = registry.settings()
    if not settings.enable_docs:
        fastapi_kwargs.setdefault("docs_url", None)
        fastapi_kwargs.setdefault("redoc_url", None)
        fastapi_kwargs.setdefault("openapi_url", None)
    if registry.has_lifespan_hooks() or fastapi_kwargs.get("lifespan") is not None:
        fastapi_kwargs["lifespan"] = _build_lifespan(
            registry,
            fastapi_kwargs.get("lifespan"),
        )

    msgspec_json_response, _, msgspec_route = make_msgspec_classes()
    fastapi_kwargs.setdefault("default_response_class", msgspec_json_response)
    app = fastapi_cls(**fastapi_kwargs)
    app.router.route_class = msgspec_route
    _configure_cors(app, settings)
    _configure_otel(app, settings)

    _register_routes(
        app,
        registry,
        request_cls,
        response_cls,
        streaming_response_cls,
        msgspec_json_response,
        settings,
    )
    return app


def _register_routes(
    app: Any,
    registry: ChannelRegistry,
    request_cls: Any,
    response_cls: Any,
    streaming_response_cls: Any,
    msgspec_json_response: Any,
    settings: Any,
) -> None:
    @app.get("/")
    async def home():
        return {
            "status": "ok",
            "agents": "/agents",
            "health": "/health",
            "chat_completions": "/v1/chat/completions",
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/agents")
    async def agents():
        return {"agents": sorted(registry.agents())}

    @app.post("/v1/chat/completions", response_class=msgspec_json_response)
    async def chat_completions(http_request: request_cls):
        return await _handle_chat_completions(
            http_request,
            registry,
            response_cls,
            streaming_response_cls,
            settings,
        )


async def _handle_chat_completions(
    http_request: Any,
    registry: ChannelRegistry,
    response_cls: Any,
    streaming_response_cls: Any,
    settings: Any,
):
    try:
        body = await _read_body(http_request, settings.max_request_bytes)
        request = decode_chat_completion_request(body)
    except msgspec.ValidationError as e:
        return response_cls(
            content=encode_error(str(e), code="invalid_request"),
            status_code=400,
            media_type="application/json",
        )
    except Exception as e:
        handled = await _exception_response(registry, e, response_cls)
        if handled is not None:
            return handled
        raise

    try:
        if request.stream:
            chunks = create_chat_completion_stream(
                registry,
                request,
                http_request=http_request,
            )
            first_chunk, chunks = await _first_stream_chunk(
                chunks,
                timeout_s=settings.request_timeout_s,
            )
            chunks = _with_stream_timeout(
                chunks,
                timeout_s=settings.request_timeout_s,
            )
            return streaming_response_cls(
                _prepend_stream_chunk(first_chunk, chunks),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await _with_timeout(
            create_chat_completion(registry, request, http_request=http_request),
            timeout_s=settings.request_timeout_s,
        )
        return response_cls(
            content=encode_json(response),
            media_type="application/json",
        )
    except Exception as e:
        handled = await _exception_response(registry, e, response_cls)
        if handled is not None:
            return handled
        raise


def _build_lifespan(registry: ChannelRegistry, user_lifespan: Any):
    @asynccontextmanager
    async def lifespan(app: Any):
        await registry.run_startup_hooks(app)
        try:
            if user_lifespan is None:
                yield
            else:
                async with user_lifespan(app):
                    yield
        finally:
            await registry.run_shutdown_hooks(app)

    return lifespan


def _configure_cors(app: Any, settings: Any) -> None:
    if not settings.cors:
        return
    cors_middleware = import_module("fastapi.middleware.cors").CORSMiddleware
    app.add_middleware(
        cors_middleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=list(settings.cors_allowed_methods),
        allow_headers=list(settings.cors_allowed_headers),
    )


def _configure_otel(app: Any, settings: Any) -> None:
    if not settings.enable_otel:
        return
    try:
        instrumentor_cls = import_module(
            "opentelemetry.instrumentation.fastapi"
        ).FastAPIInstrumentor
    except ImportError as e:
        raise ImportError(
            "FastAPI OpenTelemetry instrumentation requires "
            "`opentelemetry-instrumentation-fastapi`."
        ) from e

    instrumentor_cls.instrument_app(app, **settings.otel_kwargs)


async def _read_body(http_request: Any, max_request_bytes: int | None) -> bytes:
    if max_request_bytes is not None:
        content_length = http_request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                length = None
        else:
            length = None
        if length is not None and length > max_request_bytes:
            raise PayloadTooLargeError(
                f"Request body exceeds {max_request_bytes} bytes"
            )

    body = await http_request.body()
    if max_request_bytes is not None and len(body) > max_request_bytes:
        raise PayloadTooLargeError(f"Request body exceeds {max_request_bytes} bytes")
    return body


async def _with_timeout(awaitable: Any, *, timeout_s: float | None) -> Any:
    if timeout_s is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise RequestTimeoutError(f"Request exceeded {timeout_s} seconds") from e


async def _first_stream_chunk(
    chunks: AsyncIterator[bytes],
    *,
    timeout_s: float | None,
) -> tuple[bytes | None, AsyncIterator[bytes]]:
    iterator = chunks.__aiter__()
    try:
        if timeout_s is None:
            return await iterator.__anext__(), iterator
        return await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s), iterator
    except StopAsyncIteration:
        return None, iterator
    except asyncio.TimeoutError as e:
        raise RequestTimeoutError(f"Request exceeded {timeout_s} seconds") from e


async def _prepend_stream_chunk(
    first_chunk: bytes | None,
    chunks: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    if first_chunk is not None:
        yield first_chunk
    async for chunk in chunks:
        yield chunk


async def _with_stream_timeout(
    chunks: AsyncIterator[bytes],
    *,
    timeout_s: float | None,
) -> AsyncIterator[bytes]:
    if timeout_s is None:
        async for chunk in chunks:
            yield chunk
        return

    iterator = chunks.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            timeout = RequestTimeoutError(f"Request exceeded {timeout_s} seconds")
            yield _sse_error(timeout)
            return
        except ChannelError as e:
            yield _sse_error(e)
            return
        yield chunk


def _sse_error(error: ChannelError) -> bytes:
    return b"data: " + encode_error(error.message, code=error.code) + b"\n\n"


async def _exception_response(
    registry: ChannelRegistry,
    exc: BaseException,
    response_cls: Any,
):
    for _, handler in reversed(registry.error_handlers(exc)):
        mapped = await call_processor(handler, exc)
        response = _mapped_error_response(mapped, response_cls)
        if response is not None:
            return response

    if isinstance(exc, ChannelError):
        return _channel_error_response(exc, response_cls)
    return None


def _mapped_error_response(mapped: Any, response_cls: Any):
    if mapped is None:
        return None
    if isinstance(mapped, ChannelError):
        return _channel_error_response(mapped, response_cls)
    if hasattr(mapped, "status_code") and hasattr(mapped, "body"):
        return mapped

    status_code = 500
    payload = mapped
    if isinstance(mapped, tuple) and len(mapped) == 2:
        payload, status_code = mapped

    if isinstance(payload, Mapping):
        status_code = int(payload.get("status_code", status_code))
        if "body" in payload:
            payload = payload["body"]
        elif "message" in payload:
            return response_cls(
                content=encode_error(
                    str(payload["message"]),
                    code=str(payload.get("code") or "server_error"),
                    error_type=str(payload.get("type") or "server_error"),
                ),
                status_code=status_code,
                media_type="application/json",
            )
        else:
            payload = {
                key: value for key, value in payload.items() if key != "status_code"
            }

    return response_cls(
        content=encode_json(payload),
        status_code=int(status_code),
        media_type="application/json",
    )


def _channel_error_response(error: ChannelError, response_cls: Any):
    return response_cls(
        content=encode_error(error.message, code=error.code),
        status_code=error.status_code,
        media_type="application/json",
    )
