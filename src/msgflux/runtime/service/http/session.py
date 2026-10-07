"""Remote facade for one thread owned by an AgentService."""

from __future__ import annotations

import asyncio
import math
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from uuid import uuid4

import msgspec

from msgflux.runtime.service import AdmissionReceipt, RunSummary, ServiceThread
from msgflux.runtime.service.http.client import AgentServiceClient, RemoteThreadWatcher
from msgflux.runtime.service.http.records import SnapshotRecord
from msgflux.runtime.service.store import validate_identifier


class AgentSessionClient:
    """Bind an HTTP client to one service-owned Agent thread.

    The service owns the Agent, workspace, stores, and execution lifecycle. This
    facade borrows its :class:`AgentServiceClient` and never closes it.
    """

    __slots__ = ("_client", "_thread")

    def __init__(self, client: AgentServiceClient, thread: ServiceThread) -> None:
        if not isinstance(client, AgentServiceClient):
            raise TypeError("`client` must be an AgentServiceClient")
        thread = msgspec.convert(thread, type=ServiceThread, strict=True)
        validate_identifier(thread.thread_id, "thread_id")
        validate_identifier(thread.agent_id, "agent_id")
        self._client = client
        self._thread = thread

    @classmethod
    async def open(
        cls,
        client: AgentServiceClient,
        *,
        agent_id: str = "main",
        thread_id: str | None = None,
        cwd: str | Path | None = None,
    ) -> AgentSessionClient:
        """Open or retrieve a service thread and bind this facade to it."""
        thread = await client.open_thread(agent_id, thread_id=thread_id, cwd=cwd)
        return cls(client, thread)

    @property
    def thread_id(self) -> str:
        """Stable identity of the bound thread."""
        return self._thread.thread_id

    @property
    def agent_id(self) -> str:
        """Service-side Agent configuration selected for the thread."""
        return self._thread.agent_id

    @property
    def workspace_root(self) -> str | None:
        """Canonical host path bound to the thread, if one was supplied."""
        return self._thread.cwd

    async def prompt(
        self, prompt: str, *, request_id: str | None = None
    ) -> AdmissionReceipt:
        """Admit a prompt without coupling execution to an observer."""
        if request_id is None:
            request_id = uuid4().hex
        return await self._client.prompt(self.thread_id, prompt, request_id=request_id)

    async def receipt(self, request_id: str) -> AdmissionReceipt:
        """Return the current admission receipt for a request identity."""
        return await self._client.receipt(self.thread_id, request_id)

    async def wait(
        self, request_id: str, *, poll_interval: float = 0.05
    ) -> AdmissionReceipt:
        """Poll a receipt until it leaves accepted/running state.

        Cancelling this coroutine only stops polling; the service continues to
        own and execute the admitted run.
        """
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(poll_interval)
            or poll_interval <= 0
        ):
            raise ValueError("poll_interval must be a finite positive number")
        while True:
            current = await self.receipt(request_id)
            if current.status not in {"accepted", "running"}:
                return current
            await asyncio.sleep(poll_interval)

    def watch(self) -> AbstractAsyncContextManager[RemoteThreadWatcher]:
        """Return a caller-owned observer with a fresh snapshot on connect."""
        return self._client.watch(self.thread_id)

    async def snapshot(self) -> SnapshotRecord:
        """Fetch the current portable thread projection."""
        return await self._client.snapshot(self.thread_id)

    async def cancel(self, run_id: str) -> bool:
        """Request cooperative interruption of the explicitly selected run."""
        return await self._client.interrupt(self.thread_id, run_id)

    async def steer(self, run_id: str, content: str):
        """Send a steering message to the explicitly selected live run."""
        return await self._client.steer(self.thread_id, run_id, content)

    async def resume(self, run_id: str) -> AdmissionReceipt:
        """Ask the service to resume a saved run through trusted recovery checks."""
        return await self._client.resume_checkpoint(self.thread_id, run_id)

    async def runs(self) -> tuple[RunSummary, ...]:
        """List service-provided checkpoint summaries for this thread."""
        return await self._client.runs(self.thread_id)

    async def latest_run(self) -> RunSummary | None:
        """Return the first checkpoint summary, or ``None`` when there are none."""
        runs = await self.runs()
        return runs[0] if runs else None

    async def aclose(self) -> None:
        """No-op: this facade borrows its HTTP client and service thread."""
