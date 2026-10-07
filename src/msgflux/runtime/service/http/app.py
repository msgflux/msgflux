"""Optional authenticated Litestar adapter for :class:`AgentService`."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from litestar import Litestar, Request, Response, get, post
from litestar.background_tasks import BackgroundTask
from litestar.exceptions import HTTPException, ValidationException
from litestar.middleware import DefineMiddleware
from litestar.params import FromPath
from litestar.response.sse import ServerSentEvent, ServerSentEventMessage

from msgflux.exceptions import EventBufferOverflowError
from msgflux.logger import logger
from msgflux.runtime.event_buffer import validate_event_buffer_limit
from msgflux.runtime.service.api import AgentService
from msgflux.runtime.service.http.records import (
    AgentsResponse,
    ErrorResponse,
    HealthRecord,
    InterruptResponse,
    OpenThreadRequest,
    PromptRequest,
    ResumeRequest,
    RunsResponse,
    SteerRequest,
    ThreadsResponse,
)
from msgflux.runtime.service.http.serialization import (
    encode_json,
    event_record,
    snapshot_record,
)
from msgflux.runtime.service.records import (
    ServiceBusyError,
    ServiceConflictError,
    ServiceRecoveryRequiredError,
)

PathValue = FromPath[str]


def _response(value: object, status_code: int = 200) -> Response:
    return Response(
        content=encode_json(value),
        status_code=status_code,
        media_type="application/json",
    )


def _error(code: str, message: str, status_code: int) -> Response:
    return _response(ErrorResponse(code=code, message=message), status_code)


class _BearerAuthMiddleware:
    """Authenticate before Litestar parses request bodies or calls handlers."""

    def __init__(self, app, *, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        presented = headers.get(b"authorization", b"")
        if not hmac.compare_digest(presented, self.expected):
            body = encode_json(ErrorResponse("unauthorized", "Bearer token required"))
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def create_service_app(  # noqa: C901
    service: AgentService,
    *,
    token: str,
    close_service: bool = False,
    event_buffer_limit: int | None = 1024,
    instance_id: str | None = None,
) -> Litestar:
    """Create an authenticated HTTP/SSE app around an existing service.

    The service is borrowed by default. Set ``close_service`` only when this
    application owns its lifecycle.
    """
    validate_event_buffer_limit(event_buffer_limit)
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")

    if instance_id is None:
        instance_id = uuid4().hex
    elif not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("instance_id must be a non-empty string")

    @asynccontextmanager
    async def lifespan(_: Litestar) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if close_service:
                await service.aclose()

    @get("/v1/health")
    async def health() -> Response:
        # Check service availability without invoking a configured Agent factory.
        service.threads()
        return _response(HealthRecord(instance_id))

    @get("/v1/agents")
    async def list_agents() -> Response:
        return _response(AgentsResponse(agents=service.agents()))

    @get("/v1/threads")
    async def list_threads() -> Response:
        return _response(ThreadsResponse(threads=service.threads()))

    @post("/v1/threads")
    async def open_thread(data: OpenThreadRequest) -> Response:
        thread = await service.open_thread(
            data.agent_id, thread_id=data.thread_id, cwd=data.cwd
        )
        return _response(thread, 201)

    @get("/v1/threads/{thread_id:str}/snapshot")
    async def get_snapshot(thread_id: PathValue) -> Response:
        return _response(snapshot_record(await service.snapshot(thread_id)))

    @get("/v1/threads/{thread_id:str}/runs")
    async def list_runs(thread_id: PathValue) -> Response:
        return _response(RunsResponse(runs=await service.runs(thread_id)))

    @post("/v1/threads/{thread_id:str}/prompt")
    async def prompt(thread_id: PathValue, data: PromptRequest) -> Response:
        receipt = await service.prompt(
            thread_id, data.prompt, request_id=data.request_id
        )
        return _response(receipt, 202)

    @get("/v1/threads/{thread_id:str}/requests/{request_id:str}")
    async def get_receipt(thread_id: PathValue, request_id: PathValue) -> Response:
        return _response(service.receipt(thread_id, request_id))

    @post("/v1/threads/{thread_id:str}/runs/{run_id:str}/interrupt")
    async def interrupt(thread_id: PathValue, run_id: PathValue) -> Response:
        return _response(
            InterruptResponse(interrupted=await service.interrupt(thread_id, run_id))
        )

    @post("/v1/threads/{thread_id:str}/runs/{run_id:str}/steer")
    async def steer(
        thread_id: PathValue, run_id: PathValue, data: SteerRequest
    ) -> Response:
        notification = await service.steer(thread_id, run_id, data.content)
        return _response(notification)

    @post("/v1/threads/{thread_id:str}/runs/{run_id:str}/resume")
    async def resume(
        thread_id: PathValue, run_id: PathValue, data: ResumeRequest
    ) -> Response:
        # HTTP callers cannot assert worker quiescence. This only admits the
        # service's safe default recovery path.
        del data
        return _response(
            await service.resume_checkpoint(thread_id, run_id, worker_stopped=False),
            202,
        )

    @get("/v1/threads/{thread_id:str}/watch")
    async def watch(request: Request, thread_id: PathValue) -> Response:
        if request.headers.get("last-event-id") is not None:
            return _error(
                "invalid_request",
                "Last-Event-ID replay is unsupported; reconnect for a fresh snapshot",
                422,
            )

        try:
            # Resolve the thread and trusted session before sending SSE headers.
            # The watcher still captures the snapshot atomically with attach.
            await service.session(thread_id)
        except Exception as exc:
            return _exception_response(exc)

        async def records() -> AsyncIterator[ServerSentEventMessage]:
            async with service.watch(
                thread_id, event_buffer_limit=event_buffer_limit
            ) as watcher:
                # AgentService.watch attaches the observer and captures its
                # initial snapshot atomically, avoiding a gap before event 1.
                yield ServerSentEventMessage(
                    event="snapshot",
                    data=encode_json(snapshot_record(watcher.snapshot)).decode(),
                )
                try:
                    async for event in watcher:
                        yield ServerSentEventMessage(
                            event="event",
                            data=encode_json(event_record(event)).decode(),
                        )
                except EventBufferOverflowError:
                    yield ServerSentEventMessage(
                        event="error",
                        data=encode_json(
                            ErrorResponse(
                                "observer_overflow",
                                "Observation buffer overflowed; "
                                "reconnect for a fresh snapshot",
                            )
                        ).decode(),
                    )

        # ASGIStreamingResponse does not explicitly close its async iterator if
        # a send is cancelled after yielding a chunk. Close it after the
        # response to ensure the observer detaches on a disconnect.
        stream = records()
        return ServerSentEvent(stream, background=BackgroundTask(stream.aclose))

    return Litestar(
        route_handlers=[
            health,
            list_agents,
            list_threads,
            open_thread,
            get_snapshot,
            list_runs,
            prompt,
            get_receipt,
            interrupt,
            steer,
            resume,
            watch,
        ],
        middleware=[DefineMiddleware(_BearerAuthMiddleware, token=token)],
        lifespan=[lifespan],
        exception_handlers={
            ServiceBusyError: _domain_error,
            ServiceConflictError: _domain_error,
            ServiceRecoveryRequiredError: _domain_error,
            KeyError: _domain_error,
            ValueError: _domain_error,
            ValidationException: _validation_error,
            HTTPException: _http_error,
            Exception: _unexpected_error,
        },
    )


def _domain_error(request: Request, exc: Exception) -> Response:
    del request
    return _exception_response(exc)


def _exception_response(exc: Exception) -> Response:
    if isinstance(exc, ServiceBusyError):
        return _error("service_busy", str(exc), 409)
    if isinstance(exc, ServiceConflictError):
        return _error("service_conflict", str(exc), 409)
    if isinstance(exc, ServiceRecoveryRequiredError):
        return _error("recovery_required", str(exc), 409)
    if isinstance(exc, KeyError):
        return _error("not_found", "The requested resource was not found", 404)
    if isinstance(exc, ValueError):
        return _error("invalid_request", str(exc), 422)
    return _error("internal_error", "The service could not complete the request", 500)


def _validation_error(request: Request, exc: ValidationException) -> Response:
    del request, exc
    return _error("invalid_request", "The request body is invalid", 422)


def _http_error(request: Request, exc: HTTPException) -> Response:
    del request
    if exc.status_code == 404:
        return _error("not_found", "The requested resource was not found", 404)
    if exc.status_code in {400, 422}:
        return _error("invalid_request", "The request is invalid", 422)
    if 400 <= exc.status_code < 500:
        return _error("http_error", "The HTTP request is unsupported", exc.status_code)
    return _error("internal_error", "The service could not complete the request", 500)


def _unexpected_error(request: Request, exc: Exception) -> Response:
    del request
    logger.error("AgentService HTTP request failed", exc_info=exc)
    return _error("internal_error", "The service could not complete the request", 500)
