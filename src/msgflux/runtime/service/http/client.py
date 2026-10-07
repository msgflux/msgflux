"""Native asynchronous HTTP/SSE client for a remote :class:`AgentService`."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx2
import msgspec

from msgflux.runtime.approvals import ApprovalConflictError
from msgflux.runtime.service import (
    AdmissionReceipt,
    ApprovalReview,
    RunSummary,
    ServiceBusyError,
    ServiceConflictError,
    ServiceRecoveryRequiredError,
    ServiceThread,
)
from msgflux.runtime.service.http.records import (
    AgentsResponse,
    ApprovalDecisionRequest,
    ApprovalReviewsResponse,
    ErrorResponse,
    EventRecord,
    HealthRecord,
    InterruptResponse,
    OpenThreadRequest,
    PromptRequest,
    ResumeRequest,
    RunsResponse,
    SnapshotRecord,
    SteerRequest,
    ThreadsResponse,
)


class AgentServiceHTTPError(RuntimeError):
    """An HTTP or wire-protocol failure from a remote AgentService."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AgentServiceProtocolError(AgentServiceHTTPError):
    """The server returned a malformed or incompatible response."""


class RemoteThreadWatcher:
    """One open observation stream. Closing it never interrupts the run."""

    def __init__(self, response, snapshot: SnapshotRecord, events) -> None:
        self.snapshot = snapshot
        self._response = response
        self._events = self._iter_events(events)
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> EventRecord:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await anext(self._events)
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def _iter_events(self, events):
        async for name, data in events:
            if name == "error":
                try:
                    error = msgspec.json.decode(data, type=ErrorResponse)
                except msgspec.DecodeError as exc:
                    raise AgentServiceProtocolError("Invalid SSE error record") from exc
                raise AgentServiceHTTPError(error.message)
            if name != "event":
                continue
            try:
                yield msgspec.json.decode(data, type=EventRecord)
            except (msgspec.DecodeError, TypeError, ValueError) as exc:
                raise AgentServiceProtocolError(
                    "Invalid event record in SSE stream"
                ) from exc

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._response.aclose()


class AgentServiceClient:
    """Async client for the versioned AgentService HTTP API.

    A client created here owns its HTTP connection pool. An injected client is
    borrowed and remains open after :meth:`aclose`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        client: httpx2.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not isinstance(token, str) or not token:
            raise ValueError("token must be a non-empty string")
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._authorization = f"Bearer {token}"
        self._client = client or httpx2.AsyncClient(
            timeout=timeout,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> HealthRecord:
        """Return the authenticated runtime instance and protocol identity."""
        return await self._json("GET", "/v1/health", HealthRecord)

    async def agents(self) -> tuple[str, ...]:
        result = await self._json("GET", "/v1/agents", AgentsResponse)
        return result.agents

    async def threads(self) -> tuple[ServiceThread, ...]:
        result = await self._json("GET", "/v1/threads", ThreadsResponse)
        return result.threads

    async def open_thread(
        self,
        agent_id: str,
        *,
        thread_id: str | None = None,
        cwd: str | Path | None = None,
    ) -> ServiceThread:
        request = OpenThreadRequest(
            agent_id=agent_id,
            thread_id=thread_id,
            cwd=str(cwd) if isinstance(cwd, Path) else cwd,
        )
        return await self._json("POST", "/v1/threads", ServiceThread, request)

    async def snapshot(self, thread_id: str) -> SnapshotRecord:
        return await self._json(
            "GET", self._thread_path(thread_id) + "/snapshot", SnapshotRecord
        )

    async def runs(self, thread_id: str) -> tuple[RunSummary, ...]:
        """Return typed saved-run metadata without exposing checkpoint state."""
        result = await self._json(
            "GET", self._thread_path(thread_id) + "/runs", RunsResponse
        )
        return result.runs

    async def prompt(
        self, thread_id: str, prompt: str, *, request_id: str
    ) -> AdmissionReceipt:
        request = PromptRequest(prompt=prompt, request_id=request_id)
        return await self._json(
            "POST", self._thread_path(thread_id) + "/prompt", AdmissionReceipt, request
        )

    async def receipt(self, thread_id: str, request_id: str) -> AdmissionReceipt:
        path = self._thread_path(thread_id) + "/requests/" + _segment(request_id)
        return await self._json("GET", path, AdmissionReceipt)

    async def interrupt(self, thread_id: str, run_id: str) -> bool:
        path = self._thread_path(thread_id) + "/runs/" + _segment(run_id) + "/interrupt"
        result = await self._json("POST", path, InterruptResponse)
        return result.interrupted

    async def steer(self, thread_id: str, run_id: str, content: str) -> dict[str, Any]:
        path = self._thread_path(thread_id) + "/runs/" + _segment(run_id) + "/steer"
        response = await self._request(
            "POST", path, json=msgspec.to_builtins(SteerRequest(content))
        )
        try:
            return msgspec.json.decode(response.content, type=dict[str, Any])
        except (msgspec.DecodeError, TypeError, ValueError) as exc:
            raise AgentServiceProtocolError("Invalid steer response") from exc

    async def resume_checkpoint(self, thread_id: str, run_id: str) -> AdmissionReceipt:
        path = self._thread_path(thread_id) + "/runs/" + _segment(run_id) + "/resume"
        return await self._json("POST", path, AdmissionReceipt, ResumeRequest())

    async def approval_reviews(
        self, thread_id: str, run_id: str
    ) -> tuple[ApprovalReview, ...]:
        """List reviewable approval requests for one run."""
        path = self._thread_path(thread_id) + "/runs/" + _segment(run_id) + "/approvals"
        result = await self._json("GET", path, ApprovalReviewsResponse)
        return result.approvals

    async def decide_approval(
        self,
        thread_id: str,
        run_id: str,
        request_id: str,
        *,
        approved: bool,
        expected_revision: int,
    ) -> ApprovalReview:
        """Record a reviewer decision without resuming the paused run."""
        path = (
            self._thread_path(thread_id)
            + "/runs/"
            + _segment(run_id)
            + "/approvals/"
            + _segment(request_id)
            + "/decision"
        )
        request = ApprovalDecisionRequest(approved, expected_revision)
        return await self._json("POST", path, ApprovalReview, request)

    @asynccontextmanager
    async def watch(self, thread_id: str):
        """Connect and yield a watcher after receiving its atomic first snapshot.

        Reconnection is caller controlled; each new connection starts with a
        fresh snapshot and does not replay missed deltas.
        """
        path = self._thread_path(thread_id) + "/watch"
        request = self._client.build_request(
            "GET",
            self.base_url + path,
            headers={
                "Accept": "text/event-stream",
                "Authorization": self._authorization,
            },
            timeout=httpx2.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
        )
        response = await self._client.send(request, stream=True)
        try:
            await self._raise_for_response(response)
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise AgentServiceProtocolError("Watch response is not an SSE stream")
            events = _sse_events(response.aiter_lines())
            try:
                name, data = await anext(events)
            except StopAsyncIteration as exc:
                raise AgentServiceProtocolError(
                    "Watch stream ended before its snapshot"
                ) from exc
            if name != "snapshot":
                raise AgentServiceProtocolError(
                    "Watch stream did not begin with a snapshot"
                )
            try:
                snapshot = msgspec.json.decode(data, type=SnapshotRecord)
            except (msgspec.DecodeError, TypeError, ValueError) as exc:
                raise AgentServiceProtocolError(
                    "Invalid initial snapshot in SSE stream"
                ) from exc
            watcher = RemoteThreadWatcher(response, snapshot, events)
            try:
                yield watcher
            finally:
                await watcher.aclose()
        except BaseException:
            await response.aclose()
            raise

    async def _json(self, method: str, path: str, result_type, body=None):
        payload = msgspec.to_builtins(body) if body is not None else None
        response = await self._request(method, path, json=payload)
        try:
            return msgspec.json.decode(response.content, type=result_type)
        except (msgspec.DecodeError, TypeError, ValueError) as exc:
            raise AgentServiceProtocolError(
                "Invalid JSON response from AgentService"
            ) from exc

    async def _request(self, method: str, path: str, **kwargs):
        kwargs.setdefault("headers", {})["Authorization"] = self._authorization
        response = await self._client.request(method, self.base_url + path, **kwargs)
        await self._raise_for_response(response)
        return response

    async def _raise_for_response(self, response: httpx2.Response) -> None:
        if response.is_success:
            return
        await response.aread()
        message, code = _error_payload(response)
        if code == "service_busy":
            raise ServiceBusyError(message)
        if code == "service_conflict":
            raise ServiceConflictError(message)
        if code == "recovery_required":
            raise ServiceRecoveryRequiredError(message)
        if code == "approval_conflict":
            raise ApprovalConflictError(message)
        if code == "forbidden":
            raise PermissionError(message)
        raise AgentServiceHTTPError(message, status_code=response.status_code)

    @staticmethod
    def _thread_path(thread_id: str) -> str:
        return "/v1/threads/" + _segment(thread_id)


def _segment(value: str) -> str:
    return quote(value, safe="")


def _error_payload(response: httpx2.Response) -> tuple[str, str | None]:
    try:
        error = msgspec.json.decode(response.content, type=ErrorResponse)
    except (msgspec.DecodeError, TypeError, ValueError):
        return "AgentService request failed", None
    return error.message, error.code


async def _sse_events(lines):
    event_name = "message"
    data: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data:
                yield event_name, "\n".join(data)
            event_name, data = "message", []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data.append(value)
    if data:
        yield event_name, "\n".join(data)
