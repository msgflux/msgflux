"""Embedded Agent execution ownership, independent of observing frontends."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from collections.abc import Callable
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import msgspec

from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.logger import logger
from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.context import ExecutionScope, execution_context, new_thread_id
from msgflux.runtime.events import EventType
from msgflux.runtime.service.records import (
    AdmissionReceipt,
    AdmissionRecord,
    AdmissionStatus,
    RunSummary,
    ServiceBusyError,
    ServiceConflictError,
    ServiceRecoveryRequiredError,
    ServiceThread,
)
from msgflux.runtime.service.store import SQLiteServiceStore, validate_identifier


class AgentSession:
    """Live dependencies supplied by a trusted, per-thread host factory.

    AgentService resolves this dependency binding for one thread. CodingSession
    provides the application-facing conversation API over that service.
    The service borrows stores and Agent dependencies. ``on_close`` releases
    resources owned by the factory, including any delegated work it created.
    No network framework or user-writable configuration chooses these grants.
    """

    def __init__(
        self,
        agent,
        *,
        checkpoint_store=None,
        task_store=None,
        agent_inbox=None,
        scope_factory: Callable[[ExecutionScope], ExecutionScope] | None = None,
        on_close: Callable[[], Any] | None = None,
    ) -> None:
        agent_store = getattr(agent, "checkpoint_store", None)
        if (
            checkpoint_store is not None
            and agent_store is not None
            and checkpoint_store is not agent_store
        ):
            raise ValueError(
                "checkpoint_store conflicts with the Agent's configured store"
            )
        self.agent = agent
        self.checkpoint_store = (
            checkpoint_store
            if checkpoint_store is not None
            else getattr(agent, "checkpoint_store", None)
        )
        self.task_store = task_store
        self.agent_inbox = agent_inbox
        self.scope_factory = scope_factory
        self.on_close = on_close
        self.namespace = agent.get_module_name()
        validate_identifier(self.namespace, "namespace")

    def scope(
        self,
        thread_id: str,
        *,
        run_id: str | None = None,
        signal: AbortSignal | None = None,
    ) -> ExecutionScope:
        scope = ExecutionScope(
            thread_id=thread_id,
            namespace=self.namespace,
            workspace=getattr(self.agent, "workspace", None),
        )
        if self.scope_factory is not None:
            scope = self.scope_factory(scope)
            if not isinstance(scope, ExecutionScope):
                raise TypeError("scope_factory must return ExecutionScope")
            if scope.thread_id != thread_id:
                raise ValueError("scope_factory must preserve thread_id")
        if scope.run_id is not None or scope.abort_signal is not None:
            raise ValueError("scope_factory must preserve service-owned run identity")
        return scope.with_overrides(
            namespace=self.namespace,
            run_id=run_id,
            abort_signal=signal,
        )

    def context(self, scope: ExecutionScope):
        return execution_context(
            scope=scope,
            checkpoint_store=self.checkpoint_store,
            task_store=self.task_store,
            agent_inbox=self.agent_inbox,
        )


class _Worker(msgspec.Struct):
    receipt: AdmissionReceipt
    session: AgentSession
    scope: ExecutionScope
    task: asyncio.Task | None = None


class AgentService:
    """Own foreground executions while any number of clients observe them.

    ``prompt`` commits admission before scheduling work. The service consumes
    Agent.stream_events internally; clients use watch(), which owns no worker.
    Closing or cancelling a wait does not cancel execution. The trusted host
    supplies a fresh AgentSession per thread and explicitly owns persistent
    stores. One service belongs to one async event loop.
    """

    def __init__(self, *, store: SQLiteServiceStore) -> None:
        self.store = store
        self._owner_id = uuid4().hex
        self._factories: dict[str, Callable] = {}
        self._sessions: dict[str, AgentSession] = {}
        self._workers: dict[tuple[str, str], _Worker] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._shutdown_task: asyncio.Task | None = None

    def register(self, agent_id: str, factory: Callable) -> None:
        """Register a sync/async ServiceThread -> AgentSession host factory."""
        self._require_open()
        validate_identifier(agent_id, "agent_id")
        if not callable(factory):
            raise TypeError("factory must be callable")
        if agent_id in self._factories:
            raise ServiceConflictError(f"Agent {agent_id!r} is already registered")
        self._factories[agent_id] = factory

    def agents(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def threads(self) -> tuple[ServiceThread, ...]:
        self._require_open()
        return self.store.threads()

    async def open_thread(
        self,
        agent_id: str,
        *,
        thread_id: str | None = None,
        cwd: str | Path | None = None,
    ) -> ServiceThread:
        """Bind a conversation and optional absolute workspace root lazily.

        Omitting cwd preserves an existing binding. An explicit root is
        canonicalized and immutable; no factory runs or process cwd changes.
        """
        async with self._lock:
            self._require_open()
            if agent_id not in self._factories:
                raise KeyError(agent_id)
            if thread_id is not None:
                validate_identifier(thread_id, "thread_id")
            canonical_cwd = None
            if cwd is not None:
                if not isinstance(cwd, (str, Path)):
                    raise ValueError("`cwd` must be an absolute existing directory")
                requested_path = Path(cwd)
                if not requested_path.is_absolute():
                    raise ValueError("`cwd` must be an absolute existing directory")
                try:
                    canonical_path = requested_path.resolve(strict=True)
                except (OSError, RuntimeError) as error:
                    raise ValueError(
                        "`cwd` must be an absolute existing directory"
                    ) from error
                if not canonical_path.is_dir():
                    raise ValueError("`cwd` must be an absolute existing directory")
                canonical_cwd = str(canonical_path)
            resolved_thread_id = thread_id if thread_id is not None else new_thread_id()
            if canonical_cwd is None:
                try:
                    canonical_cwd = self.store.thread(resolved_thread_id).cwd
                except KeyError:
                    pass
            return self.store.bind_thread(
                ServiceThread(resolved_thread_id, agent_id, canonical_cwd)
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("AgentService is closing or closed")

    async def _session(self, thread_id: str) -> AgentSession:
        self._require_open()
        thread = self.store.thread(thread_id)
        if thread_id not in self._sessions:
            factory = self._factories[thread.agent_id]
            session = factory(thread)
            if inspect.isawaitable(session):
                session = await session
            if not isinstance(session, AgentSession):
                raise TypeError("factory must return AgentSession")
            if any(item.agent is session.agent for item in self._sessions.values()):
                raise ServiceConflictError("Factories must isolate Agents by thread")
            try:
                session.scope(thread_id)
            except BaseException:
                if session.on_close is not None:
                    closed = session.on_close()
                    if inspect.isawaitable(closed):
                        await closed
                raise
            self._sessions[thread_id] = session
        return self._sessions[thread_id]

    async def session(self, thread_id: str) -> AgentSession:
        """Resolve live dependencies for a trusted in-process host.

        Frontends should normally use prompt/watch. This accessor lets a host
        build a domain-specific facade without duplicating factory ownership.
        """
        async with self._lock:
            return await self._session(thread_id)

    async def runs(self, thread_id: str) -> tuple[RunSummary, ...]:
        """Return saved run metadata for a thread, newest first.

        The checkpoint store remains authoritative; this projection never
        exposes saved state or host configuration.
        """
        async with self._lock:
            session = await self._session(thread_id)
            store = session.checkpoint_store
            list_runs = getattr(store, "list_runs", None)
            if not callable(list_runs):
                return ()
            summaries = []
            for item in list_runs(session.namespace, thread_id):
                # msgspec's strict conversion intentionally rejects integer to
                # float coercion, so normalize the one permitted provider form.
                record = dict(item)
                updated_at = record.get("updated_at")
                if isinstance(updated_at, int) and not isinstance(updated_at, bool):
                    record["updated_at"] = float(updated_at)
                summaries.append(msgspec.convert(record, type=RunSummary, strict=True))
            return tuple(summaries)

    @staticmethod
    def _validate_new_input(session: AgentSession, thread_id: str) -> None:
        store = session.checkpoint_store
        if store is None:
            return
        state = store.load_latest_run(session.namespace, thread_id)
        if state is None:
            return
        session.agent._validate_checkpoint_command_receipts(state)
        if state.get("status") not in {"completed", "interrupted"}:
            raise ServiceRecoveryRequiredError(
                "The latest run requires explicit recovery"
            )

    async def prompt(
        self,
        thread_id: str,
        prompt: str,
        *,
        request_id: str,
    ) -> AdmissionReceipt:
        """Admit one input; duplicate identity returns the same run, never another."""
        async with self._lock:
            session = await self._session(thread_id)
            existing = self.store.get(thread_id, request_id)
            if existing is None:
                self._validate_new_input(session, thread_id)
            record = self.store.admit(thread_id, request_id, prompt, session.namespace)
            if record.receipt.status == "accepted":
                self._schedule(record, session)
            return record.receipt

    def _schedule(self, record: AdmissionRecord, session: AgentSession) -> None:
        key = (record.receipt.thread_id, record.receipt.request_id)
        if key in self._workers:
            return
        signal = AbortSignal()
        scope = session.scope(
            record.receipt.thread_id,
            run_id=record.receipt.run_id,
            signal=signal,
        )
        worker = _Worker(record.receipt, session, scope)
        self._workers[key] = worker
        # Clients may be observing another thread or a nested Agent. Never adopt
        # their ContextVars, event sink, task handle, or execution lineage.
        worker.task = asyncio.create_task(
            self._produce(record, worker), context=contextvars.Context()
        )
        worker.task.add_done_callback(self._observe_worker_failure)

    @staticmethod
    def _observe_worker_failure(task: asyncio.Task) -> None:
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "AgentService worker failed to settle its journal", exc_info=error
                )

    async def _produce(self, record: AdmissionRecord, worker: _Worker) -> None:
        receipt = record.receipt
        key = (receipt.thread_id, receipt.request_id)
        status: AdmissionStatus = "completed"
        error = None
        claimed = False
        try:
            claimed = self.store.claim(record, self._owner_id)
            if not claimed:
                return
            claimed_record = self.store.get(receipt.thread_id, receipt.request_id)
            worker.receipt = claimed_record.receipt
            receipt = worker.receipt
            status = await self._execute(record, worker)
        except TaskPauseRequestedError as exc:
            status, error = "paused", str(exc)
        except (AbortRequestedError, TaskInterruptRequestedError) as exc:
            status, error = "interrupted", str(exc)
        except asyncio.CancelledError:
            status, error = "interrupted", "Service worker cancelled"
            raise
        except Exception as exc:
            status, error = "failed", str(exc)
        except BaseException as exc:
            status, error = "failed", str(exc)
            raise
        finally:
            try:
                if claimed:
                    self.store.finish(receipt, self._owner_id, status, error)
            finally:
                self._workers.pop(key, None)

    @staticmethod
    async def _execute(record: AdmissionRecord, worker: _Worker) -> AdmissionStatus:
        status: AdmissionStatus = "completed"
        with worker.session.context(worker.scope):
            async with aclosing(
                worker.session.agent.stream_events(record.prompt, scope=worker.scope)
            ) as events:
                async for event in events:
                    if (
                        event.run_id == worker.receipt.run_id
                        and len(event.source_path) == 1
                        and event.type == EventType.RUN_END
                    ):
                        outcome = event.data.get("outcome")
                        if outcome in {"completed", "interrupted", "failed"}:
                            status = outcome
        return status

    def receipt(self, thread_id: str, request_id: str) -> AdmissionReceipt:
        record = self.store.get(thread_id, request_id)
        if record is None:
            raise KeyError(request_id)
        return record.receipt

    def receipt_for_run(self, thread_id: str, run_id: str) -> AdmissionReceipt:
        """Look up an admission by its durable execution identity."""
        record = self.store.get_for_run(thread_id, run_id)
        if record is None:
            raise KeyError(run_id)
        return record.receipt

    async def wait(self, thread_id: str, request_id: str) -> AdmissionReceipt:
        """Wait for this attempt; cancelling the waiter leaves work running."""
        worker = self._workers.get((thread_id, request_id))
        if worker is not None:
            await asyncio.shield(worker.task)
        receipt = self.receipt(thread_id, request_id)
        if receipt.status in {"accepted", "running"}:
            raise ServiceRecoveryRequiredError("This service does not own that attempt")
        return receipt

    @asynccontextmanager
    async def watch(self, thread_id: str, *, event_buffer_limit: int | None = None):
        """Attach to existing Agent watch; observer disposal has no execution effect."""
        async with self._lock:
            session = await self._session(thread_id)
            with session.context(session.scope(thread_id)):
                watcher = session.agent.watch(
                    thread_id,
                    event_buffer_limit=event_buffer_limit,
                )
                await watcher.__aenter__()
        try:
            yield watcher
        finally:
            await watcher.aclose()

    async def snapshot(self, thread_id: str):
        async with self.watch(thread_id) as watcher:
            return watcher.snapshot

    async def interrupt(self, thread_id: str, run_id: str) -> bool:
        """Explicitly signal only the selected locally owned run."""
        async with self._lock:
            self._require_open()
            worker = self._worker_for_run(thread_id, run_id)
            if worker is None:
                return False
            worker.scope.abort_signal.abort("Service run interrupted")
            return True

    def _worker_for_run(self, thread_id: str, run_id: str) -> _Worker | None:
        return next(
            (
                worker
                for worker in self._workers.values()
                if worker.receipt.thread_id == thread_id
                and worker.receipt.run_id == run_id
            ),
            None,
        )

    async def steer(self, thread_id: str, run_id: str, content: str):
        """Publish a user message through the run's existing AgentInbox."""
        async with self._lock:
            self._require_open()
            worker = self._worker_for_run(thread_id, run_id)
            if worker is None or worker.scope.abort_signal.aborted:
                raise ServiceBusyError("The target run is not active in this service")
            inbox = worker.session.agent_inbox or worker.session.agent.agent_inbox
            return inbox.fork(
                namespace=worker.session.namespace,
                thread_id=thread_id,
                run_id=run_id,
            ).user_message(content)

    async def resume(
        self,
        thread_id: str,
        request_id: str,
        *,
        worker_stopped: bool = False,
    ) -> AdmissionReceipt:
        """Trusted-host recovery, never an implicit retry of a started input.

        ``worker_stopped`` is a host assertion of quiescence, not evidence derived
        from elapsed time. A started run without a checkpoint stays uncertain.
        Workspace and approvals remain subject to Agent's recovery validation.
        """
        if not isinstance(worker_stopped, bool):
            raise TypeError("worker_stopped must be a host-supplied bool")
        async with self._lock:
            session = await self._session(thread_id)
            record = self.store.get(thread_id, request_id)
            if record is None:
                raise KeyError(request_id)
            if record.namespace != session.namespace:
                raise ServiceConflictError("The host changed the checkpoint namespace")
            if (thread_id, request_id) in self._workers:
                raise ServiceBusyError("The attempt still has a local worker")
            if record.receipt.status == "accepted":
                self._schedule(record, session)
                return record.receipt
            if record.receipt.status in {"completed", "interrupted"}:
                return record.receipt
            if record.owner_id != self._owner_id and not worker_stopped:
                raise ServiceRecoveryRequiredError(
                    "Establish old-worker quiescence first"
                )
            state = self._recovery_checkpoint(session, record.receipt)
            record = self.store.prepare_resume(record)
            if state.get("status") in {"completed", "interrupted"}:
                if not self.store.claim(record, self._owner_id):
                    raise ServiceConflictError("Another worker claimed recovery")
                claimed = self.store.get(thread_id, request_id)
                return self.store.finish(
                    claimed.receipt, self._owner_id, state["status"]
                )
            self._schedule(record, session)
            return record.receipt

    async def resume_checkpoint(
        self,
        thread_id: str,
        run_id: str,
        *,
        worker_stopped: bool = False,
    ) -> AdmissionReceipt:
        """Resume by run identity, including checkpoints predating the service.

        Importing a checkpoint without an admission requires a trusted host to
        establish worker quiescence. The saved context remains authoritative;
        no original input is reconstructed or resent as a new run.
        """
        if not isinstance(worker_stopped, bool):
            raise TypeError("worker_stopped must be a host-supplied bool")
        validate_identifier(run_id, "run_id")
        async with self._lock:
            session = await self._session(thread_id)
            record = self.store.get_for_run(thread_id, run_id)
            if record is None:
                if not worker_stopped:
                    raise ServiceRecoveryRequiredError(
                        "Establish old-worker quiescence before importing a checkpoint"
                    )
                receipt = AdmissionReceipt(
                    thread_id, f"checkpoint:{run_id}", run_id, "running"
                )
                self._recovery_checkpoint(session, receipt)
                record = self.store.adopt_checkpoint(
                    thread_id, run_id, session.namespace
                )
            request_id = record.receipt.request_id
        return await self.resume(thread_id, request_id, worker_stopped=worker_stopped)

    @staticmethod
    def _recovery_checkpoint(session: AgentSession, receipt: AdmissionReceipt):
        store = session.checkpoint_store
        state = (
            store.load_state(session.namespace, receipt.thread_id, receipt.run_id)
            if store is not None
            else None
        )
        if state is None:
            raise ServiceRecoveryRequiredError("A started run has no checkpoint")
        latest = store.load_latest_run(session.namespace, receipt.thread_id)
        if (
            state.get("status") not in {"completed", "interrupted"}
            and latest is not None
            and latest.get("scope", {}).get("run_id") != receipt.run_id
        ):
            raise ServiceRecoveryRequiredError(
                "Only the latest unfinished run can resume"
            )
        with session.context(session.scope(receipt.thread_id, run_id=receipt.run_id)):
            session.agent._validate_checkpoint_command_receipts(state)
            session.agent._validate_checkpoint_workspace(state)
        return state

    async def aclose(self) -> None:
        """Stop admissions, join foreground work, then close factory-owned resources.

        Cancellation of this wait leaves shutdown running. Non-cooperative code
        can delay shutdown; this method does not forcibly kill external workers.
        Borrowed stores remain open for host inspection and recovery.
        """
        async with self._lock:
            if self._shutdown_task is None:
                self._closed = True
                self._shutdown_task = asyncio.create_task(
                    self._shutdown(), context=contextvars.Context()
                )
            task = self._shutdown_task
        await asyncio.shield(task)

    async def _shutdown(self) -> None:
        workers = tuple(self._workers.values())
        for worker in workers:
            worker.scope.abort_signal.abort("AgentService shutdown")
        errors = []
        for result in await asyncio.gather(
            *(worker.task for worker in workers),
            return_exceptions=True,
        ):
            if isinstance(result, Exception):
                errors.append(result)
        for session in self._sessions.values():
            if session.on_close is not None:
                try:
                    result = session.on_close()
                    if inspect.isawaitable(result):
                        await result
                except Exception as error:
                    errors.append(error)
        if errors:
            raise ExceptionGroup("AgentService shutdown failed", errors)
