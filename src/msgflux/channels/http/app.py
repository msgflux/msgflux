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
    resolve_request_metadata,
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
    fastapi_kwargs.setdefault("title", settings.title)
    fastapi_kwargs.setdefault("description", settings.description)
    if not settings.enable_docs:
        fastapi_kwargs.setdefault("docs_url", None)
        fastapi_kwargs.setdefault("redoc_url", None)
        fastapi_kwargs.setdefault("openapi_url", None)
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
    _validate_routes(registry, settings)

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
        payload = {
            "status": "ok",
            "title": settings.title,
            "subtitle": settings.subtitle,
            "agents": "/agents",
            "health": "/health",
            "ready": "/ready",
        }
        if not settings.disable_chat_completions:
            payload["chat_completions"] = "/v1/chat/completions"
        social_routes = {
            channel: f"/social/{channel}/webhook"
            for channel in registry.social_boundary().adapters()
        }
        if social_routes:
            payload["social"] = social_routes
        return payload

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return response_cls(status_code=204)

    @app.get("/ready")
    async def ready():
        readiness = registry.readiness()
        return response_cls(
            content=encode_json(
                {
                    "status": readiness.status,
                    "ready": readiness.ready,
                    "error": readiness.error,
                }
            ),
            status_code=200 if readiness.ready else 503,
            media_type="application/json",
        )

    @app.get("/agents")
    async def agents():
        return {
            "agents": [
                _agent_metadata_payload(metadata)
                for metadata in sorted(
                    registry.agents_metadata().values(),
                    key=lambda item: item.name,
                )
            ]
        }

    _register_chat_completion_route(
        app,
        registry,
        request_cls,
        response_cls,
        streaming_response_cls,
        msgspec_json_response,
        settings,
    )

    for social_channel in registry.social_boundary().adapters():

        @app.post(f"/social/{social_channel}/webhook")
        async def social_webhook(
            http_request: request_cls,
            channel: str = social_channel,
        ):
            return await _handle_social_webhook(
                channel,
                http_request,
                registry,
                response_cls,
                settings,
            )


async def _handle_chat_completions(
    http_request: Any,
    registry: ChannelRegistry,
    response_cls: Any,
    streaming_response_cls: Any,
    settings: Any,
):
    request_metadata = resolve_request_metadata(http_request)
    try:
        body = await _read_body(http_request, settings.max_request_bytes)
        request = decode_chat_completion_request(body)
    except msgspec.ValidationError as e:
        return response_cls(
            content=encode_error(
                str(e),
                code="invalid_request",
                request_id=request_metadata["request_id"],
                correlation_id=request_metadata["correlation_id"],
            ),
            status_code=400,
            media_type="application/json",
            headers=_response_headers(request_metadata),
        )
    except Exception as e:
        handled = await _exception_response(
            registry,
            e,
            response_cls,
            request_metadata,
        )
        if handled is not None:
            return handled
        raise

    try:
        if request.stream:
            chunks = create_chat_completion_stream(
                registry,
                request,
                http_request=http_request,
                request_metadata=request_metadata,
            )
            first_chunk, chunks = await _first_stream_chunk(
                chunks,
                timeout_s=settings.request_timeout_s,
            )
            chunks = _with_stream_timeout(
                chunks,
                timeout_s=settings.request_timeout_s,
                request_metadata=request_metadata,
            )
            return streaming_response_cls(
                _prepend_stream_chunk(first_chunk, chunks),
                media_type="text/event-stream",
                headers={
                    **_response_headers(request_metadata),
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await _with_timeout(
            create_chat_completion(
                registry,
                request,
                http_request=http_request,
                request_metadata=request_metadata,
            ),
            timeout_s=settings.request_timeout_s,
        )
        return response_cls(
            content=encode_json(response),
            media_type="application/json",
            headers=_response_headers(request_metadata),
        )
    except Exception as e:
        handled = await _exception_response(
            registry,
            e,
            response_cls,
            request_metadata,
        )
        if handled is not None:
            return handled
        raise


def _register_chat_completion_route(
    app: Any,
    registry: ChannelRegistry,
    request_cls: Any,
    response_cls: Any,
    streaming_response_cls: Any,
    msgspec_json_response: Any,
    settings: Any,
) -> None:
    if settings.disable_chat_completions:
        return

    @app.post("/v1/chat/completions", response_class=msgspec_json_response)
    async def chat_completions(http_request: request_cls):
        return await _handle_chat_completions(
            http_request,
            registry,
            response_cls,
            streaming_response_cls,
            settings,
        )


async def _handle_social_webhook(
    channel: str,
    http_request: Any,
    registry: ChannelRegistry,
    response_cls: Any,
    settings: Any,
):
    request_metadata = resolve_request_metadata(http_request)
    try:
        body = await _read_body(http_request, settings.max_request_bytes)
        event_count = await registry.social_boundary().handle_webhook(
            channel,
            body,
            http_request,
        )
        return response_cls(
            content=encode_json({"status": "accepted", "events": event_count}),
            media_type="application/json",
            headers=_response_headers(request_metadata),
        )
    except Exception as e:
        handled = await _exception_response(
            registry,
            e,
            response_cls,
            request_metadata,
        )
        if handled is not None:
            return handled
        raise


def _build_lifespan(registry: ChannelRegistry, user_lifespan: Any):
    @asynccontextmanager
    async def lifespan(app: Any):
        registry.mark_starting()
        social_boundary = registry.social_boundary()
        try:
            await registry.run_startup_hooks(app)
            await social_boundary.start()
            if user_lifespan is None:
                registry.mark_ready()
                yield
            else:
                async with user_lifespan(app):
                    registry.mark_ready()
                    yield
        except Exception as e:
            status = "error" if registry.readiness().ready else "startup_failed"
            registry.mark_not_ready(status, e)
            raise
        finally:
            if registry.readiness().ready:
                registry.mark_not_ready("stopping")
            try:
                await social_boundary.stop()
                await registry.run_shutdown_hooks(app)
            finally:
                if registry.readiness().status != "startup_failed":
                    registry.mark_not_ready("stopped")

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


def _validate_routes(registry: ChannelRegistry, settings: Any) -> None:
    if settings.social_debounce_s is not None and settings.social_debounce_s < 0:
        raise ValueError("social_debounce_s must be >= 0")
    chat_disabled_without_social = (
        settings.disable_chat_completions
        and not registry.social_boundary().has_adapters()
    )
    if chat_disabled_without_social:
        raise ChannelError(
            "disable_chat_completions=True requires at least one social adapter"
        )


def _agent_metadata_payload(metadata: Any) -> dict[str, Any]:
    return {
        "name": metadata.name,
        "description": metadata.description,
        "tags": list(metadata.tags),
        "capabilities": dict(metadata.capabilities),
    }


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
    request_metadata: Mapping[str, Any],
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
            yield _sse_error(timeout, request_metadata)
            return
        except ChannelError as e:
            yield _sse_error(e, request_metadata)
            return
        yield chunk


def _sse_error(error: ChannelError, request_metadata: Mapping[str, Any]) -> bytes:
    return (
        b"data: "
        + encode_error(
            error.message,
            code=error.code,
            error_type=error.error_type,
            request_id=request_metadata["request_id"],
            correlation_id=request_metadata["correlation_id"],
        )
        + b"\n\n"
    )


async def _exception_response(
    registry: ChannelRegistry,
    exc: BaseException,
    response_cls: Any,
    request_metadata: Mapping[str, Any],
):
    for _, handler in reversed(registry.error_handlers(exc)):
        mapped = await call_processor(handler, exc)
        response = _mapped_error_response(mapped, response_cls, request_metadata)
        if response is not None:
            return response

    if isinstance(exc, ChannelError):
        return _channel_error_response(exc, response_cls, request_metadata)
    return None


def _mapped_error_response(
    mapped: Any,
    response_cls: Any,
    request_metadata: Mapping[str, Any],
):
    if mapped is None:
        return None
    if isinstance(mapped, ChannelError):
        return _channel_error_response(mapped, response_cls, request_metadata)
    if hasattr(mapped, "status_code") and hasattr(mapped, "body"):
        return mapped

    status_code = 500
    headers = {}
    payload = mapped
    if isinstance(mapped, tuple) and len(mapped) == 2:
        payload, status_code = mapped
    elif isinstance(mapped, tuple) and len(mapped) == 3:
        payload, status_code, headers = mapped

    if isinstance(payload, Mapping):
        status_code = int(payload.get("status_code", status_code))
        headers = dict(payload.get("headers") or headers)
        if "body" in payload:
            payload = payload["body"]
        elif "message" in payload:
            return response_cls(
                content=encode_error(
                    str(payload["message"]),
                    code=str(payload.get("code") or "server_error"),
                    error_type=str(payload.get("type") or "server_error"),
                    request_id=request_metadata["request_id"],
                    correlation_id=request_metadata["correlation_id"],
                ),
                status_code=status_code,
                media_type="application/json",
                headers=_response_headers(request_metadata, headers),
            )
        else:
            payload = {
                key: value
                for key, value in payload.items()
                if key not in {"headers", "status_code"}
            }

    return response_cls(
        content=encode_json(payload),
        status_code=int(status_code),
        media_type="application/json",
        headers=_response_headers(request_metadata, headers),
    )


def _channel_error_response(
    error: ChannelError,
    response_cls: Any,
    request_metadata: Mapping[str, Any],
):
    return response_cls(
        content=encode_error(
            error.message,
            code=error.code,
            error_type=error.error_type,
            request_id=request_metadata["request_id"],
            correlation_id=request_metadata["correlation_id"],
        ),
        status_code=error.status_code,
        media_type="application/json",
        headers=_response_headers(request_metadata, error.headers),
    )


def _response_headers(
    request_metadata: Mapping[str, Any],
    extra_headers: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    headers = {
        "X-Request-ID": str(request_metadata["request_id"]),
        "X-Correlation-ID": str(request_metadata["correlation_id"]),
    }
    if request_metadata.get("traceparent"):
        headers["traceparent"] = str(request_metadata["traceparent"])
    headers.update(
        {str(key): str(value) for key, value in (extra_headers or {}).items()}
    )
    return headers
