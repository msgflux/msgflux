"""Embedded Agent execution ownership, independent of observing frontends."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from collections.abc import Callable
from contextlib import ExitStack, aclosing, asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import msgspec
from opentelemetry.trace import SpanContext

from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.logger import logger
from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.agent_inbox.base import AgentInboxClosedError
from msgflux.runtime.compaction import CompactionResult
from msgflux.runtime.context import ExecutionScope, execution_context, new_thread_id
from msgflux.runtime.events import EventType, _hub_event_sink
from msgflux.runtime.service.approvals import project, review_record
from msgflux.runtime.service.cache_policy import SessionCachePolicy
from msgflux.runtime.service.inspection import (
    admission_evidence,
    approval_evidence,
    checkpoint_reasons,
)
from msgflux.runtime.service.policies import (
    approval_review_context,
    clipped_policy,
    initial_policy,
    ready_approval_resume,
    requested_permissions,
)
from msgflux.runtime.service.records import (
    AdmissionReceipt,
    AdmissionRecord,
    AdmissionStatus,
    AgentConfiguration,
    AgentInspection,
    ApprovalReview,
    RunInspection,
    RunResult,
    RunSummary,
    ServiceBusyError,
    ServiceConflictError,
    ServiceRecoveryRequiredError,
    ServiceThread,
)
from msgflux.runtime.service.session_cache import (
    SessionLease,
    _SessionCache,
    _SessionLoadCleanupError,
)
from msgflux.runtime.service.store import SQLiteServiceStore, validate_identifier
from msgflux.runtime.workspace.errors import WorkspaceIdentityError
from msgflux.runtime.workspace.policy import WorkspacePolicy, WorkspacePolicyState
from msgflux.telemetry.service import (
    execution_span,
    execution_trigger,
    finish_execution_span,
)
from msgflux.utils.errors import exception_message
from msgflux.utils.validation import is_valid_account_name

_DEFAULT_CACHE_POLICY = SessionCachePolicy()


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
        approval_reviewer: str | None = None,
        on_close: Callable[[], Any] | None = None,
    ) -> None:
        managed = getattr(agent, "agent_dir", None) is not None
        if managed and any(
            item is not None for item in (checkpoint_store, task_store, agent_inbox)
        ):
            raise ValueError(
                "agent_dir cannot be combined with explicit session stores"
            )
        self._managed = managed
        self._thread_id = None
        self._host_on_close = on_close
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
        if approval_reviewer is not None and (
            not isinstance(approval_reviewer, str) or not approval_reviewer.strip()
        ):
            raise ValueError("approval_reviewer must be a non-empty string")
        self.approval_reviewer = approval_reviewer or ("user" if managed else None)
        self.workspace_policy_state = None
        self.on_close = self._close if managed else on_close
        self.namespace = agent.get_module_name()
        validate_identifier(self.namespace, "namespace")

    def bind_resources(self, thread_id: str, *, create: bool = False) -> None:
        """Bind managed per-thread stores; observation never creates a new thread."""
        if not self._managed:
            return
        if self._thread_id is not None and self._thread_id != thread_id:
            raise ValueError("AgentSession must retain its bound thread identity")
        self._thread_id = thread_id
        if self.checkpoint_store is None and (
            create or self.agent._resources.has_thread(thread_id)
        ):
            resources = self.agent._bind_resources(thread_id)
            self.checkpoint_store = resources.checkpoint_store
            self.task_store = resources.task_store
            self.agent_inbox = resources.agent_inbox

    async def _close(self) -> None:
        if self._thread_id is not None:
            await self.agent._close_thread_resources(
                self._thread_id, before_close=self._host_on_close
            )
        elif self._host_on_close is not None:
            result = self._host_on_close()
            if inspect.isawaitable(result):
                await result

    def scope(
        self,
        thread_id: str,
        *,
        run_id: str | None = None,
        signal: AbortSignal | None = None,
        account_name: str | None = None,
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
        self.bind_resources(thread_id)
        resolved_scope = scope.with_overrides(
            namespace=self.namespace,
            run_id=run_id,
            abort_signal=signal,
            principal=scope.principal or self.approval_reviewer,
        )
        return replace(resolved_scope, account_name=account_name)

    def context(self, scope: ExecutionScope):
        self.bind_resources(scope.thread_id, create=scope.run_id is not None)
        return execution_context(
            scope=scope,
            checkpoint_store=self.checkpoint_store,
            task_store=self.task_store,
            agent_inbox=self.agent_inbox,
            workspace_policy=self.workspace_policy_state,
        )


class _Worker(msgspec.Struct):
    receipt: AdmissionReceipt
    session: AgentSession
    scope: ExecutionScope
    task: asyncio.Task | None = None
    trace_trigger: SpanContext | None = None


class AgentService:
    """Own foreground executions while any number of clients observe them.

    ``prompt`` commits admission before scheduling work. The service consumes
    Agent.stream_events internally; clients use watch(), which owns no worker.
    Closing or cancelling a wait does not cancel execution. The trusted host
    supplies a fresh AgentSession per thread and explicitly owns persistent
    stores. One service belongs to one async event loop.
    """

    def __init__(
        self,
        *,
        store: SQLiteServiceStore,
        default_agent: str | None = None,
        default_account: str = "default",
        cache_policy: SessionCachePolicy | None = _DEFAULT_CACHE_POLICY,
    ) -> None:
        if cache_policy is not None and not isinstance(
            cache_policy, SessionCachePolicy
        ):
            raise TypeError("cache_policy must be SessionCachePolicy or None")
        if default_agent is not None:
            validate_identifier(default_agent, "default_agent")
        self.default_account = self._validate_account_name(default_account)
        self.store = store
        self.default_agent = default_agent
        self._owner_id = uuid4().hex
        self._owner_lease = None
        self._factories: dict[str, Callable] = {}
        self._configurations: dict[str, AgentConfiguration] = {}
        self._cache = _SessionCache(
            self._load_session,
            self._release_reason,
            policy=cache_policy,
            busy_reason=lambda thread_id, session: self._release_reason(
                thread_id, session, mark_closing=False
            ),
            validate=self._validate_session_identity,
        )
        self._producer_tasks: dict[asyncio.Task, _Worker] = {}
        self._workers: dict[tuple[str, str], _Worker] = {}
        self._maintenance_tasks: dict[str, asyncio.Task] = {}
        self._watchers = set()
        self._lock = asyncio.Lock()
        self._closed = False
        self._shutdown_task: asyncio.Task | None = None
        self._before_write = None

    def _notify_before_write(self) -> None:
        """Let a facade persist its journal before the service changes it."""
        if self._before_write is not None:
            self._before_write()

    @staticmethod
    def _validate_account_name(account_name: str | None) -> str:
        if not is_valid_account_name(account_name):
            raise ValueError(
                "account_name must be 1-64 characters, start with a letter or digit, "
                "and contain only letters, digits, '.', '_' or '-'; "
                "'.' and '..' are invalid"
            )
        return account_name

    def register(
        self,
        agent_id: str,
        factory: Callable,
        *,
        configuration: AgentConfiguration | None = None,
    ) -> None:
        """Register a lazy host factory and optional safe inspection metadata."""
        self._require_open()
        validate_identifier(agent_id, "agent_id")
        if not callable(factory):
            raise TypeError("factory must be callable")
        if agent_id in self._factories:
            raise ServiceConflictError(f"Agent {agent_id!r} is already registered")
        if configuration is not None:
            if not isinstance(configuration, AgentConfiguration):
                raise TypeError("configuration must be AgentConfiguration")
            configuration = msgspec.convert(
                msgspec.to_builtins(configuration), type=AgentConfiguration, strict=True
            )
        self._factories[agent_id] = factory
        if configuration is not None:
            self._configurations[agent_id] = configuration

    def inspect_agent(self, agent_id: str) -> AgentInspection:
        """Inspect loaded defaults without invoking the registered factory."""
        self._require_open()
        if agent_id not in self._factories:
            raise KeyError(agent_id)
        return AgentInspection(
            agent_id=agent_id,
            is_default=agent_id == self.default_agent,
            account_name=self.default_account,
            configuration=self._configurations.get(agent_id),
        )

    def agents(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def threads(self) -> tuple[ServiceThread, ...]:
        self._require_open()
        return self.store.threads()

    def _select_thread_agent(
        self, agent_id: str | None, thread_id: str | None
    ) -> tuple[str, ServiceThread | None]:
        existing = None
        if thread_id is not None:
            try:
                existing = self.store.thread(thread_id)
            except KeyError:
                pass
        if existing is not None:
            # The persisted binding is authoritative when reopening a
            # thread. An explicit different agent is rejected by bind_thread.
            return (
                agent_id if agent_id is not None else existing.agent_id,
                existing,
            )

        selected_agent = agent_id if agent_id is not None else self.default_agent
        if selected_agent is None:
            raise ValueError(
                "agent_id is required to create a thread when no "
                "default_agent is configured"
            )
        if selected_agent not in self._factories:
            raise KeyError(selected_agent)
        return selected_agent, None

    async def open_thread(
        self,
        agent_id: str | None = None,
        *,
        thread_id: str | None = None,
        cwd: str | Path | None = None,
        interactive: bool | None = None,
    ) -> ServiceThread:
        """Bind a conversation and optional absolute workspace root lazily.

        New threads use the explicit agent or ``default_agent``. Reopening an
        existing thread preserves its stored agent when omitted. Omitting cwd
        preserves an existing binding. No factory runs or process cwd changes.
        """
        async with self._lock:
            self._require_open()
            if agent_id is not None:
                validate_identifier(agent_id, "agent_id")
            if thread_id is not None:
                validate_identifier(thread_id, "thread_id")
            interactive = msgspec.convert(interactive, type=bool | None, strict=True)
            selected_agent, existing = self._select_thread_agent(agent_id, thread_id)
            selected_interactive = (
                interactive
                if interactive is not None
                else existing.interactive
                if existing is not None
                else True
            )
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
                if existing is not None:
                    canonical_cwd = existing.cwd
            return self.store.bind_thread(
                ServiceThread(
                    resolved_thread_id,
                    selected_agent,
                    canonical_cwd,
                    selected_interactive,
                )
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("AgentService is closing or closed")

    async def _load_session(self, thread_id: str) -> AgentSession:
        try:
            return await self._create_session(thread_id)
        except WorkspaceIdentityError as error:
            raise ServiceRecoveryRequiredError(
                f"Cannot load workspace for thread {thread_id!r}: {error}"
            ) from error

    def _agent_is_loaded(self, session: AgentSession) -> bool:
        return any(item.agent is session.agent for item in self._sessions.values())

    def _validate_session_identity(self, session: AgentSession) -> None:
        """Reject a duplicate Agent atomically with cache publication."""
        if self._agent_is_loaded(session):
            raise ServiceConflictError("Factories must isolate Agents by thread")

    async def _create_session(self, thread_id: str) -> AgentSession:
        self._require_open()
        thread = self.store.thread(thread_id)
        factory = self._factories[thread.agent_id]
        session = factory(thread)
        if inspect.isawaitable(session):
            session = await session
        if not isinstance(session, AgentSession):
            raise TypeError("factory must return AgentSession")
        if self._agent_is_loaded(session):
            raise ServiceConflictError("Factories must isolate Agents by thread")
        try:
            scope = session.scope(thread_id)
            stored = self.store.workspace_policy(thread_id)
            selected = stored or initial_policy(thread_id, scope.workspace)
            session.workspace_policy_state = (
                WorkspacePolicyState(
                    scope.workspace, clipped_policy(selected, scope.workspace)
                )
                if scope.workspace is not None
                else None
            )
        except BaseException as initialization_error:
            if session.on_close is not None:
                try:
                    closed = session.on_close()
                    if inspect.isawaitable(closed):
                        await closed
                except BaseException as cleanup_error:
                    failure = BaseExceptionGroup(
                        "Session initialization and cleanup failed",
                        [initialization_error, cleanup_error],
                    )
                    raise _SessionLoadCleanupError(session, failure) from failure
            raise
        return session

    @property
    def _sessions(self):
        return self._cache.sessions

    async def acquire_session(self, thread_id: str) -> SessionLease:
        """Pin trusted live dependencies until the returned lease is closed."""
        self._require_open()
        self.store.thread(thread_id)
        lease = await self._cache.acquire(thread_id)
        try:
            session = lease.session
            state = session.workspace_policy_state
            saved = (
                self.store.workspace_policy(thread_id) if state is not None else None
            )
            if saved is not None and saved.revision > state.current.revision:
                state.current = clipped_policy(saved, state.workspace)
        except BaseException:
            await lease.aclose()
            raise
        return lease

    @asynccontextmanager
    async def _use_session(self, thread_id: str):
        lease = await self.acquire_session(thread_id)
        try:
            yield lease.session
        finally:
            await lease.aclose()

    def _release_reason(
        self, thread_id: str, session: AgentSession, *, mark_closing: bool = True
    ) -> str | None:
        if thread_id in self._maintenance_tasks:
            return "The thread is being compacted"
        if getattr(session, "_retained_by_facade", False):
            return "The embedded CodingSession retains this Agent"
        if any(
            worker.receipt.thread_id == thread_id
            for worker in self._producer_tasks.values()
        ):
            return "A foreground execution or its finalizer is still active"
        if not session._managed:
            reason = session.agent._active_background_future_reason(
                (session.agent.tool_library,)
            )
            if reason is not None:
                return reason
            if session.task_store is not None:
                reason = session.agent._unfinished_task_reason(
                    thread_id, session.task_store
                )
                if reason is not None:
                    return reason
        with session.context(session.scope(thread_id)):
            if mark_closing:
                return session.agent._mark_thread_idle_closing(thread_id)
            return session.agent._thread_release_reason(thread_id)

    async def release_session(self, thread_id: str) -> bool:
        """Release idle owned resources; keep the logical thread and saved state."""
        self._require_open()
        self.store.thread(thread_id)
        return await self._cache.release(thread_id)

    async def workspace_policy(self, thread_id: str) -> WorkspacePolicy:
        """Read this thread's effective policy without creating default rows."""
        async with self._use_session(thread_id) as session:
            async with self._lock:
                state = session.workspace_policy_state
                return (
                    state.current
                    if state is not None
                    else initial_policy(thread_id, None)
                )

    async def update_workspace_policy(
        self,
        thread_id: str,
        *,
        permissions=None,
        approval_policy=None,
        expected_revision: int | None = None,
    ) -> WorkspacePolicy:
        """Persist a host-authorized override before publishing it to live calls.

        Permission presets are bounded by the trusted workspace ceiling.
        Changing approval mode does not itself bypass pending batch validation.
        """
        async with self._use_session(thread_id) as session:
            async with self._lock:
                self._require_no_maintenance(thread_id)
                self._require_open()
                self._notify_before_write()
                state = session.workspace_policy_state
                workspace = state.workspace if state is not None else None
                if workspace is None:
                    raise ValueError("This session has no workspace policy to update")
                current = state.current
                selected = (
                    requested_permissions(permissions, workspace)
                    if permissions is not None
                    else current.permission_set()
                )
                proposed = msgspec.structs.replace(
                    current,
                    permissions=tuple(sorted(selected.grants)),
                    resources=tuple(
                        sorted(selected.resources, key=lambda r: (r.resource, r.action))
                    ),
                    approval_policy=approval_policy
                    if approval_policy is not None
                    else current.approval_policy,
                )
                proposed = clipped_policy(proposed, workspace)
                saved = self.store.update_workspace_policy(
                    proposed, expected_revision=expected_revision
                )
                state.current = saved
                record, error = self._resume_policy_batch(session, thread_id)
                self._publish_policy_update(session, saved, record, error)
                return saved

    def _resume_policy_batch(self, session, thread_id):
        if thread_id in self._maintenance_tasks:
            return None, None
        state = session.workspace_policy_state
        if self._closed or state is None or state.current.approval_policy != "never":
            return None, None
        record, error = ready_approval_resume(self, session, thread_id)
        if record is not None:
            record = self.store.prepare_resume(record)
            self._schedule(record, session)
        return record, error

    @staticmethod
    def _publish_policy_update(session, policy, record, error):
        _hub_event_sink().emit(
            EventType.WORKSPACE_POLICY_UPDATED,
            {
                "policy": msgspec.to_builtins(policy),
                "resumed_run_id": record.receipt.run_id if record is not None else None,
                "resume_error": msgspec.to_builtins(error)
                if error is not None
                else None,
            },
            scope=session.scope(policy.thread_id),
        )

    async def runs(self, thread_id: str) -> tuple[RunSummary, ...]:
        """Return saved run metadata for a thread, newest first.

        The checkpoint store remains authoritative; this projection never
        exposes saved state or host configuration.
        """
        async with self._use_session(thread_id) as session:
            async with self._lock:
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
                    summaries.append(
                        msgspec.convert(record, type=RunSummary, strict=True)
                    )
                return tuple(summaries)

    async def thread_result(self, thread_id: str) -> RunResult:
        """Select the latest admission once and return its current run result."""
        self._require_open()
        admission = self.store.latest_admission(thread_id)
        if admission is None:
            raise KeyError(f"Thread {thread_id} has no admitted runs")
        return await self.run_result(thread_id, admission.receipt.run_id)

    async def run_result(self, thread_id: str, run_id: str) -> RunResult:
        """Return the answer recorded for this run; unfinished runs have no output."""
        from msgflux.runtime.service.serialization import (  # noqa: PLC0415
            checkpoint_answer,
        )

        receipt = self.receipt_for_run(thread_id, run_id)
        if receipt.status != "completed":
            return RunResult(thread_id, run_id, receipt.status, error=receipt.error)
        async with self._use_session(thread_id) as session:
            admission = self.store.get_for_run(thread_id, run_id)
            store = session.checkpoint_store
            checkpoint = (
                store.load_state(admission.namespace, thread_id, run_id)
                if store is not None
                else None
            )
            if checkpoint is None or checkpoint.get("status") != "completed":
                raise ServiceRecoveryRequiredError(
                    "Completed run has no final checkpoint"
                )
            return RunResult(
                thread_id, run_id, "completed", output=checkpoint_answer(checkpoint)
            )

    async def inspect_run(self, thread_id: str, run_id: str) -> RunInspection:
        """Read recorded evidence without changing state or promising safe replay."""
        validate_identifier(thread_id, "thread_id")
        validate_identifier(run_id, "run_id")
        async with self._use_session(thread_id) as session:
            async with self._lock:
                admission = self.store.get_for_run(thread_id, run_id)
                store = session.checkpoint_store
                namespace_matches = (
                    admission is None or admission.namespace == session.namespace
                )
                checkpoint = (
                    store.load_state(session.namespace, thread_id, run_id)
                    if store is not None and namespace_matches
                    else None
                )
                if admission is None and checkpoint is None:
                    raise KeyError(run_id)
                local_worker = self._worker_for_run(thread_id, run_id) is not None
                receipt = admission.receipt if admission is not None else None
                requires_quiescence, reasons = admission_evidence(
                    admission, session.namespace, self._owner_id, local_worker
                )
                checkpoint_status = None
                checkpoint_revision = None
                approval_phase = None
                approval_request_count = 0
                if checkpoint is None:
                    if (
                        namespace_matches
                        and receipt.status in {"running", "paused", "failed"}
                        and not local_worker
                    ):
                        reasons.append("A started run has no checkpoint.")
                else:
                    checkpoint_status = checkpoint.get("status")
                    if not isinstance(checkpoint_status, str):
                        checkpoint_status = None
                        reasons.append("The saved checkpoint status is malformed.")
                    if receipt is not None and receipt.status != checkpoint_status:
                        reasons.append("Admission and checkpoint statuses differ.")
                    revision = checkpoint.get("_checkpoint", {}).get("revision")
                    checkpoint_revision = revision if type(revision) is int else None
                    approval_phase, approval_request_count, approval_reasons = (
                        approval_evidence(checkpoint)
                    )
                    reasons.extend(approval_reasons)
                    if checkpoint_status is not None:
                        reasons.extend(
                            checkpoint_reasons(session, checkpoint, thread_id, run_id)
                        )
                return RunInspection(
                    run_id=run_id,
                    receipt=receipt,
                    checkpoint_status=checkpoint_status,
                    checkpoint_revision=checkpoint_revision,
                    local_worker=local_worker,
                    requires_quiescence=requires_quiescence,
                    approval_phase=approval_phase,
                    approval_request_count=approval_request_count,
                    background_tasks=(
                        tuple(
                            session.task_store.list_summaries(
                                thread_id=thread_id, run_id=run_id
                            )
                        )
                        if session.task_store is not None and namespace_matches
                        else ()
                    ),
                    reasons=tuple(dict.fromkeys(reasons)),
                )

    async def approval_reviews(
        self, thread_id: str, run_id: str
    ) -> tuple[ApprovalReview, ...]:
        """Review checkpoint-bound approvals without returning invocation args."""
        async with self._use_session(thread_id) as session:
            async with self._lock:
                if session.approval_reviewer is None:
                    raise PermissionError(
                        "This session has no configured approval reviewer"
                    )
                with approval_review_context(session, thread_id, run_id) as policy:
                    if policy is None:
                        return ()
                    state = session.agent.inspect_approval_batch(thread_id, run_id)
                    request_ids = (
                        state.get("runtime", {})
                        .get("extensions", {})
                        .get("pending_approvals", {})
                        .get("requests", {})
                        .values()
                    )
                    reviews = [
                        review_record(session, thread_id, run_id, request_id)
                        for request_id in request_ids
                    ]
                    return tuple(reviews)

    async def decide_approval(
        self,
        thread_id: str,
        run_id: str,
        request_id: str,
        *,
        approved: bool,
        expected_revision: int,
    ) -> ApprovalReview:
        """Record one host review; execution still requires explicit resume."""
        if type(expected_revision) is not int or expected_revision <= 0:
            raise ValueError("expected_revision must be a positive integer")
        async with self._use_session(thread_id) as session:
            async with self._lock:
                reviewer = session.approval_reviewer
                if reviewer is None:
                    raise PermissionError(
                        "This session has no configured approval reviewer"
                    )
                with session.context(session.scope(thread_id, run_id=run_id)):
                    # Validate the checkpoint binding before changing the journal.
                    initial_review = review_record(
                        session, thread_id, run_id, request_id
                    )
                    result = session.agent.decide_approval(
                        request_id,
                        approved=approved,
                        decided_by=reviewer,
                        expected_revision=expected_revision,
                    )
                    return project(result, diff=initial_review.diff)

    @staticmethod
    def _validate_command_receipts(session: AgentSession, state) -> None:
        try:
            session.agent._validate_checkpoint_command_receipts(state)
        except TaskPauseRequestedError as exc:
            raise ServiceRecoveryRequiredError(str(exc)) from exc

    @staticmethod
    def _validate_new_input(session: AgentSession, thread_id: str) -> None:
        store = session.checkpoint_store
        if store is None:
            return
        state = store.load_latest_run(session.namespace, thread_id)
        if state is None:
            return
        AgentService._validate_command_receipts(session, state)
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
        account_name: str | None = None,
    ) -> AdmissionReceipt:
        """Admit one input; duplicate identity returns the same run, never another."""
        if account_name is not None:
            account_name = self._validate_account_name(account_name)
        async with self._use_session(thread_id) as session:
            async with self._lock:
                self._require_no_maintenance(thread_id)
                self._require_open()
                self._notify_before_write()
                existing = self.store.get(thread_id, request_id)
                if existing is None:
                    self._validate_new_input(session, thread_id)
                    session.bind_resources(thread_id, create=True)
                    state = session.workspace_policy_state
                    if (
                        state is not None
                        and self.store.workspace_policy(thread_id) is None
                    ):
                        saved = self.store.update_workspace_policy(
                            state.current, expected_revision=0
                        )
                        state.current = saved
                record = self.store.admit(
                    thread_id,
                    request_id,
                    prompt,
                    session.namespace,
                    account_name=account_name or self.default_account,
                    check_account=account_name is not None,
                )
                if record.receipt.status == "accepted":
                    self._schedule(record, session)
                return record.receipt

    def _require_no_maintenance(self, thread_id):
        if thread_id in self._maintenance_tasks:
            raise ServiceBusyError("The thread is being compacted")

    async def compact(
        self, thread_id: str, *, run_id: str | None = None
    ) -> CompactionResult:
        """Compact a paused/failed thread without admitting or resuming a run."""
        from msgflux.runtime.service.compaction import compact  # noqa: PLC0415

        return await compact(self, thread_id, run_id)

    def _schedule(self, record: AdmissionRecord, session: AgentSession) -> None:
        self._require_no_maintenance(record.receipt.thread_id)
        key = (record.receipt.thread_id, record.receipt.request_id)
        if key in self._workers:
            return
        signal = AbortSignal()
        scope = session.scope(
            record.receipt.thread_id,
            run_id=record.receipt.run_id,
            signal=signal,
            account_name=record.account_name,
        )
        worker = _Worker(
            record.receipt, session, scope, trace_trigger=execution_trigger()
        )
        self._workers[key] = worker
        # Clients may be observing another thread or a nested Agent. Never adopt
        # their ContextVars, event sink, task handle, or execution lineage.
        coroutine = self._produce(record, worker)
        try:
            worker.task = asyncio.create_task(coroutine, context=contextvars.Context())
        except BaseException as error:
            if self._workers.get(key) is worker:
                self._workers.pop(key, None)
            try:
                coroutine.close()
            except BaseException as cleanup_error:
                error.add_note(
                    f"Could not close unsubmitted AgentService worker coroutine: "
                    f"{cleanup_error}"
                )
            raise
        self._producer_tasks[worker.task] = worker
        worker.task.add_done_callback(self._producer_finished)

    def _producer_finished(self, task: asyncio.Task) -> None:
        worker = self._producer_tasks.pop(task, None)
        if worker is not None:
            self._cache.activity_finished(worker.receipt.thread_id)
        self._observe_worker_failure(task)

    @staticmethod
    def _observe_worker_failure(task: asyncio.Task) -> None:
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "AgentService worker failed to settle its journal", exc_info=error
                )

    async def _produce(  # noqa: C901 - keep ownership and terminal cleanup together
        self, record: AdmissionRecord, worker: _Worker
    ) -> None:
        with ExitStack() as spans:
            receipt = record.receipt
            key = (receipt.thread_id, receipt.request_id)
            status: AdmissionStatus = "completed"
            error = None
            claimed = False
            span = None
            error_type = None
            execution_error: BaseException | None = None
            try:
                claimed = self.store.claim(record, self._owner_id)
                if not claimed:
                    return
                claimed_record = self.store.get(receipt.thread_id, receipt.request_id)
                worker.receipt = claimed_record.receipt
                receipt = worker.receipt
                span = spans.enter_context(
                    execution_span(
                        receipt,
                        self.store.thread(receipt.thread_id).agent_id,
                        worker.trace_trigger,
                    )
                )
                status = await self._execute(record, worker)
            except TaskPauseRequestedError as exc:
                execution_error = exc
                error_type = type(exc).__name__
                status, error = "paused", exception_message(exc)
            except (AbortRequestedError, TaskInterruptRequestedError) as exc:
                execution_error = exc
                error_type = type(exc).__name__
                status, error = "interrupted", exception_message(exc)
            except asyncio.CancelledError as exc:
                execution_error = exc
                error_type = "CancelledError"
                status, error = "interrupted", "Service worker cancelled"
                raise
            except Exception as exc:
                execution_error = exc
                if not claimed:
                    raise
                error_type = type(exc).__name__
                status, error = "failed", exception_message(exc)
            except BaseException as exc:
                execution_error = exc
                error_type = type(exc).__name__
                status, error = "failed", exception_message(exc)
                raise
            finally:
                try:
                    if claimed:
                        self._finish_worker_run(receipt, status, error, execution_error)
                    if span is not None:
                        finish_execution_span(span, status, error_type)
                finally:
                    self._workers.pop(key, None)
                # A policy can change while the model/approval preparation is still
                # running. Reevaluate only after that worker has finished its Agent
                # context and durably paused; never overlap the two executions.
                if status == "paused":
                    await self._resume_quiescent_policy(worker, key)

    def _finish_worker_run(
        self,
        receipt: AdmissionReceipt,
        status: AdmissionStatus,
        error: str | None,
        execution_error: BaseException | None,
    ) -> None:
        try:
            self.store.finish(receipt, self._owner_id, status, error)
        except BaseException as settlement_error:
            if isinstance(execution_error, asyncio.CancelledError):
                logger.error(
                    "AgentService worker cancellation could not be settled "
                    "in the journal",
                    exc_info=settlement_error,
                )
                raise execution_error from settlement_error
            if execution_error is not None:
                raise BaseExceptionGroup(
                    "Run execution and journal settlement failed",
                    [execution_error, settlement_error],
                ) from None
            raise

    async def _resume_quiescent_policy(self, worker, key):
        if self._closed:
            return
        async with self._lock:
            if key in self._workers:
                return
            resumed, blocked = self._resume_policy_batch(
                worker.session, worker.receipt.thread_id
            )
            if resumed is not None or blocked is not None:
                self._publish_policy_update(
                    worker.session,
                    worker.session.workspace_policy_state.current,
                    resumed,
                    blocked,
                )

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

    def _wait_receipt(self, thread_id: str, request_id: str) -> AdmissionReceipt:
        """Read a receipt only when this service can wait for its live attempt."""
        receipt = self.receipt(thread_id, request_id)
        if (
            receipt.status in {"accepted", "running"}
            and (thread_id, request_id) not in self._workers
        ):
            raise ServiceRecoveryRequiredError("This service does not own that attempt")
        return receipt

    async def wait(self, thread_id: str, request_id: str) -> AdmissionReceipt:
        """Wait for this attempt; cancelling the waiter leaves work running."""
        key = (thread_id, request_id)
        worker = self._workers.get(key)
        while worker is not None:
            # Completion of a cancelled worker is not cancellation of its observer.
            await asyncio.wait((worker.task,))
            if worker.task.cancelled():
                receipt = self.receipt(thread_id, request_id)
                if receipt.status not in {"accepted", "running"}:
                    return receipt
            # Preserve execution/settlement errors when no terminal receipt exists.
            worker.task.result()
            replacement = self._workers.get(key)
            if replacement is worker:
                break
            worker = replacement
        return self._wait_receipt(thread_id, request_id)

    @asynccontextmanager
    async def watch(self, thread_id: str, *, event_buffer_limit: int | None = None):
        """Attach to existing Agent watch; observer disposal has no execution effect."""
        watcher = await self._attach_watcher(thread_id, event_buffer_limit)
        try:
            yield watcher
        finally:
            self._watchers.discard(watcher)
            await watcher.aclose()

    async def _attach_watcher(self, thread_id, event_buffer_limit):
        """Pin dependencies only while subscribing and reading the snapshot."""
        watcher = None
        try:
            async with self._use_session(thread_id) as session:
                async with self._lock:
                    self._require_open()
                    with session.context(session.scope(thread_id)):
                        policy = (
                            session.agent._get_workspace_approvals(force=True)
                            if session._managed and session.checkpoint_store is not None
                            else None
                            if session._managed
                            else session.agent._get_effective_approvals()
                        )
                        watcher = session.agent.watch(
                            thread_id,
                            approvals=policy,
                            event_buffer_limit=event_buffer_limit,
                        )
                        await watcher.__aenter__()
                        self._watchers.add(watcher)
                return watcher
        except BaseException:
            if watcher is not None:
                self._watchers.discard(watcher)
                await watcher.aclose()
            raise

    async def snapshot(self, thread_id: str):
        async with self.watch(thread_id) as watcher:
            return watcher.snapshot

    async def active_runs(self) -> tuple[AdmissionReceipt, ...]:
        """Snapshot root runs executing in this service, without loading sessions.

        Includes scheduled workers and workers draining an interruption request.
        Saved checkpoints and background tasks do not establish live ownership.
        """
        async with self._lock:
            self._require_open()
            return tuple(
                worker.receipt
                for worker in self._workers.values()
                if worker.task is not None and not worker.task.done()
            )

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
            try:
                return inbox.fork(
                    namespace=worker.session.namespace,
                    thread_id=thread_id,
                    run_id=run_id,
                ).user_message(content)
            except AgentInboxClosedError as error:
                raise ServiceBusyError(
                    "The target run no longer accepts steering"
                ) from error

    def _recover_approval_owner(self, session, record):
        """Use local host evidence to adopt only an unexecuted approval batch."""
        from msgflux.runtime.service.recovery import (  # noqa: PLC0415
            validate_approval_recovery,
        )

        if self._owner_lease is None:
            raise ServiceRecoveryRequiredError("Establish old-worker quiescence first")
        with self._owner_lease.reclaim(record.owner_id):
            validate_approval_recovery(self, session, record)
            return self.store.adopt_paused(record, self._owner_id)

    async def resume(
        self,
        thread_id: str,
        request_id: str,
        *,
        worker_stopped: bool = False,
    ) -> AdmissionReceipt:
        """Trusted-host recovery, never an implicit retry of a started input.

        The local daemon may verify its old owner's lifetime lock for a paused,
        unexecuted approval batch. Otherwise ``worker_stopped`` is an explicit
        host assertion of quiescence, never inferred from elapsed time.
        A started run without a checkpoint stays uncertain.
        Workspace and approvals remain subject to Agent's recovery validation.
        """
        if not isinstance(worker_stopped, bool):
            raise TypeError("worker_stopped must be a host-supplied bool")
        async with self._use_session(thread_id) as session:
            async with self._lock:
                self._require_no_maintenance(thread_id)
                record = self.store.get(thread_id, request_id)
                if record is None:
                    raise KeyError(request_id)
                if record.namespace != session.namespace:
                    raise ServiceConflictError(
                        "The host changed the checkpoint namespace"
                    )
                if (thread_id, request_id) in self._workers:
                    raise ServiceBusyError("The attempt still has a local worker")
                self._notify_before_write()
                if record.receipt.status == "accepted":
                    self._schedule(record, session)
                    return record.receipt
                if record.receipt.status in {"completed", "interrupted"}:
                    return record.receipt
                if record.owner_id != self._owner_id and not worker_stopped:
                    record = self._recover_approval_owner(session, record)
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
        async with self._use_session(thread_id) as session:
            async with self._lock:
                self._require_no_maintenance(thread_id)
                record = self.store.get_for_run(thread_id, run_id)
                if record is None:
                    if not worker_stopped:
                        raise ServiceRecoveryRequiredError(
                            "Establish old-worker quiescence before importing "
                            "a checkpoint"
                        )
                    receipt = AdmissionReceipt(
                        thread_id, f"checkpoint:{run_id}", run_id, "running"
                    )
                    state = self._recovery_checkpoint(session, receipt)
                    saved_scope = state.get("scope")
                    saved_account = (
                        saved_scope.get("account_name")
                        if isinstance(saved_scope, dict)
                        else None
                    )
                    if saved_account is not None:
                        saved_account = self._validate_account_name(saved_account)
                    self._notify_before_write()
                    record = self.store.adopt_checkpoint(
                        thread_id, run_id, session.namespace, saved_account
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
        try:
            session.agent._validated_checkpoint_messages(state)
        except ValueError as error:
            raise ServiceRecoveryRequiredError(
                f"The checkpoint conversation history is malformed: {error}"
            ) from error
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
            AgentService._validate_command_receipts(session, state)
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
        maintenance = tuple(self._maintenance_tasks.values())
        for task in maintenance:
            task.cancel()
        if maintenance:
            await asyncio.gather(*maintenance, return_exceptions=True)
        workers = tuple(self._producer_tasks.values())
        for worker in workers:
            worker.scope.abort_signal.abort("AgentService shutdown")
        errors = []
        for result in await asyncio.gather(
            *(worker.task for worker in workers), return_exceptions=True
        ):
            if isinstance(result, Exception):
                errors.append(result)
        for watcher in tuple(self._watchers):
            await watcher.aclose()
        self._watchers.clear()
        errors.extend(await self._cache.close_all())
        if errors:
            raise ExceptionGroup("AgentService shutdown failed", errors)
