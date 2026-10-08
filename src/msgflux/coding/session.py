"""A small facade over service-owned foreground Agent executions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, aclosing, asynccontextmanager
from pathlib import Path
from uuid import uuid4

from msgflux.coding.checkpoints import CodingCheckpointExtension
from msgflux.coding.watcher import _SessionWatcher
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope, new_thread_id
from msgflux.runtime.service import (
    AdmissionReceipt,
    AgentService,
    AgentSession,
    EventRecord,
    RunSummary,
    ServiceThread,
    SnapshotRecord,
    SQLiteServiceStore,
)
from msgflux.runtime.service.serialization import snapshot_record


class CodingSession:
    """Keep a durable thread identity while observing service-owned Agent work.

    AgentService owns executions; AgentSession binds their live dependencies.
    This facade supplies the application API for one thread.
    Passing an Agent creates a small embedded service for convenience. Use
    :meth:`from_service` when a host owns the service and its lifetime.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        thread_id: str | None = None,
        checkpoint_store=None,
        task_store=None,
        agent_inbox=None,
        scope_factory: Callable[[ExecutionScope], ExecutionScope] | None = None,
        service_store: SQLiteServiceStore | None = None,
    ) -> None:
        if not isinstance(agent, Agent):
            raise TypeError("`agent` must be an Agent")
        if thread_id is not None and (not isinstance(thread_id, str) or not thread_id):
            raise ValueError("`thread_id` must be a non-empty string or None")
        if not agent.has_extension("coding_checkpoints"):
            agent.register_extension("coding_checkpoints", CodingCheckpointExtension())
        self.agent = agent
        self._thread_id = thread_id if thread_id is not None else new_thread_id()
        self.namespace = agent.get_module_name()
        self._scope_factory = scope_factory
        self._binding = AgentSession(
            agent,
            checkpoint_store=checkpoint_store,
            task_store=task_store,
            agent_inbox=agent_inbox,
            scope_factory=scope_factory,
        )
        self._service_path: Path | None = (
            agent.agent_dir / "runtime" / "service.sqlite3"
            if agent.agent_dir is not None and service_store is None
            else None
        )
        self._owns_service_store = service_store is None
        self._service_store = (
            service_store
            if service_store is not None
            else (
                agent._resources.service_store()
                if self._service_path is not None and self._service_path.exists()
                else SQLiteServiceStore(":memory:")
            )
        )
        self.service = AgentService(store=self._service_store)
        self.service._before_write = self._persist_service
        self._owns_service = True
        self._service_persisted = (
            self._service_path is not None and self._service_path.exists()
        )
        self._close_task: asyncio.Task | None = None
        try:
            self._register_agent()
        except BaseException:
            if self._owns_service_store:
                self._service_store.close()
            raise

    def _register_agent(self) -> None:
        agent_id = self.namespace
        self.service.store.bind_thread(ServiceThread(self._thread_id, agent_id))
        self._binding.bind_resources(self.thread_id)
        self.service.register(agent_id, lambda _thread: self._binding)

    @classmethod
    async def from_service(cls, service: AgentService, thread_id: str) -> CodingSession:
        """Attach a facade to a thread already bound to a shared service."""
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("`thread_id` must be a non-empty string")
        session = await service.session(thread_id)
        agent = session.agent
        self = cls.__new__(cls)
        self.agent = agent
        self.service = service
        self._owns_service = False
        self._service_store = None
        self._owns_service_store = False
        self._thread_id = thread_id
        self.namespace = session.namespace
        self._binding = session
        self._service_path = None
        self._service_persisted = False
        self._scope_factory = session.scope_factory
        self._close_task = None
        return self

    @property
    def checkpoint_store(self):
        return self._binding.checkpoint_store

    @property
    def task_store(self):
        return self._binding.task_store

    @property
    def agent_inbox(self):
        return self._binding.agent_inbox

    def _persist_service(self) -> None:
        if self._service_path is None or self._service_persisted:
            return
        store = self.agent._resources.service_store()
        try:
            for thread in self._service_store.threads():
                store.bind_thread(thread)
        except BaseException:
            store.close()
            raise
        self._service_store.close()
        self._service_store = store
        self.service.store = store
        self._service_persisted = True

    @property
    def thread_id(self) -> str:
        """Stable durable conversation identity for this session."""
        return self._thread_id

    async def workspace_policy(self):
        """Read the effective policy through the generic Agent service."""
        return await self.service.workspace_policy(self.thread_id)

    async def update_workspace_policy(
        self, *, permissions=None, approval_policy=None, expected_revision=None
    ):
        """Persist and apply an override to this thread's future operations."""
        return await self.service.update_workspace_policy(
            self.thread_id,
            permissions=permissions,
            approval_policy=approval_policy,
            expected_revision=expected_revision,
        )

    async def approval_reviews(self, run_id: str):
        return await self.service.approval_reviews(self.thread_id, run_id)

    async def decide_approval(
        self, run_id: str, request_id: str, *, approved: bool, expected_revision: int
    ):
        return await self.service.decide_approval(
            self.thread_id,
            run_id,
            request_id,
            approved=approved,
            expected_revision=expected_revision,
        )

    async def cancel(self, run_id: str) -> bool:
        """Request cooperative cancellation of the explicitly selected run."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("`run_id` must be a non-empty string")
        return await self.service.interrupt(self._thread_id, run_id)

    async def aclose(self) -> None:
        """Close an embedded service without taking ownership of borrowed stores."""
        if not self._owns_service:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._close_task)

    async def _shutdown(self) -> None:
        try:
            await self.service.aclose()
            if self.agent.agent_dir is not None:
                await self.agent._close_thread_resources(self.thread_id)
        finally:
            if self._owns_service_store:
                self._service_store.close()

    async def snapshot(self) -> SnapshotRecord:
        """Return the service's durable and live projection for this thread."""
        return snapshot_record(await self.service.snapshot(self._thread_id))

    def watch(self) -> AbstractAsyncContextManager[_SessionWatcher]:
        """Observe a portable snapshot and future events, as with an HTTP session."""
        return self._watch()

    @asynccontextmanager
    async def _watch(self):
        # Subscribe before projecting the snapshot, preserving the service's
        # atomic attach boundary. Conversion failures still close the observer.
        async with self.service.watch(self.thread_id) as watcher:
            yield _SessionWatcher(watcher)

    async def runs(self) -> tuple[RunSummary, ...]:
        """Return saved run summaries through the service's shared projection."""
        return await self.service.runs(self.thread_id)

    async def latest_run(self) -> RunSummary | None:
        """Return the latest saved run summary, or None for an empty thread."""
        runs = await self.runs()
        return runs[0] if runs else None

    def saved_state(self, run_id: str) -> dict:
        if self.checkpoint_store is None:
            raise ValueError("This session has no checkpoint store")
        state = self.checkpoint_store.load_state(self.namespace, self.thread_id, run_id)
        if state is None:
            raise ValueError(f"Unknown run: {run_id}")
        return state

    async def receipt(self, request_id: str) -> AdmissionReceipt:
        return self.service.receipt(self._thread_id, request_id)

    async def wait(self, request_id: str) -> AdmissionReceipt:
        return await self.service.wait(self._thread_id, request_id)

    async def prompt(
        self, prompt: str, *, request_id: str | None = None
    ) -> AdmissionReceipt:
        """Admit input without attaching an event observer."""
        if request_id is None:
            request_id = uuid4().hex
        return await self.service.prompt(
            self._thread_id,
            prompt,
            request_id=request_id,
        )

    async def stream(
        self, prompt: str, *, request_id: str | None = None
    ) -> AsyncIterator[EventRecord]:
        """Admit a prompt and yield ordered events while the service runs it."""
        async with self.watch() as watcher:
            receipt = await self.prompt(prompt, request_id=request_id)
            async with aclosing(self._observe(watcher, receipt.run_id)) as events:
                async for event in events:
                    yield event

    async def resume(
        self, run_id: str, *, worker_stopped: bool = False
    ) -> AdmissionReceipt:
        """Admit checkpoint recovery; observe execution separately through watch.

        worker_stopped is a trusted host assertion, not a remote client option.
        The service validates checkpoint state and owns the resumed execution.
        """
        return await self.service.resume_checkpoint(
            self.thread_id, run_id, worker_stopped=worker_stopped
        )

    async def steer(self, run_id: str, content: str):
        """Publish a user message through the target run's existing AgentInbox."""
        return await self.service.steer(self._thread_id, run_id, content)

    async def _observe(self, watcher, run_id: str):
        receipt = self.service.receipt_for_run(self._thread_id, run_id)
        related_runs = {run_id}
        settled = asyncio.create_task(
            self.service.wait(self._thread_id, receipt.request_id)
        )
        next_event = asyncio.create_task(anext(watcher))
        try:
            while True:
                done, _ = await asyncio.wait(
                    (next_event, settled), return_when=asyncio.FIRST_COMPLETED
                )
                if settled in done:
                    # Closing the observer preserves its queued events and gives
                    # iteration an explicit end after the producer has settled.
                    await watcher.aclose()
                try:
                    event = await next_event
                except StopAsyncIteration:
                    break
                if self._is_related_event(event, run_id, related_runs):
                    related_runs.add(event.run_id)
                if event.run_id in related_runs:
                    yield event
                next_event = asyncio.create_task(anext(watcher))
            self._raise_receipt(await settled)
        finally:
            if not settled.done():
                settled.cancel()
            if not next_event.done():
                next_event.cancel()
            await asyncio.gather(settled, next_event, return_exceptions=True)

    @staticmethod
    def _is_related_event(event, run_id: str, related_runs: set[str]) -> bool:
        return event.type in {"run.start", "run.resume"} and (
            event.data.get("root_run_id") == run_id
            or event.data.get("parent_run_id") in related_runs
        )

    @staticmethod
    def _raise_receipt(receipt) -> None:
        if receipt.status == "paused":
            raise TaskPauseRequestedError(message=receipt.error or "Run paused")
        if receipt.status == "failed":
            raise RuntimeError(receipt.error or "Agent run failed")
        if receipt.status == "interrupted":
            raise RuntimeError(receipt.error or "Agent run interrupted")
