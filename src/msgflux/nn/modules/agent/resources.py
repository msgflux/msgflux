"""Lazy Agent persistence and ownership, independent of model/workspace lifetime."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from concurrent.futures import Future
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from threading import RLock

import msgspec

from msgflux.core.dotdict import dotdict
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.runtime.agent_resources import AgentResources, BoundAgentResources
from msgflux.runtime.context import execution_context, get_execution_context


class _OwnedThread(msgspec.Struct):
    resources: BoundAgentResources
    libraries: set = msgspec.field(default_factory=set)
    active: int = 0
    active_calls: int = 0
    detached_finalizers: set = msgspec.field(default_factory=set)
    closing: bool = False
    idle_closing: bool = False
    children_settled: bool = False
    lock: object = msgspec.field(default_factory=RLock)


_CURRENT_RESOURCES = ContextVar("msgflux_owned_agent_resources", default=None)


def _get_tool_result_store(*, create=True, config=None):
    owned = _CURRENT_RESOURCES.get()
    if owned is None:
        if not create:
            return None
        raise RuntimeError(
            "Managed tool offload requires agent_dir or inherited Agent resources"
        )
    return owned.resources.tool_result_store(create=create, config=config)


def resource_call(method):
    """Cover direct forward calls as well as Module's wider lifecycle contexts."""
    if inspect.iscoroutinefunction(method):

        @wraps(method)
        async def async_call(self, *args, **kwargs):
            prepared = self._resource_call_kwargs(args, kwargs)
            with self._resource_context(prepared):
                return await method(self, *args, **prepared)

        return async_call

    @wraps(method)
    def sync_call(self, *args, **kwargs):
        prepared = self._resource_call_kwargs(args, kwargs)
        with self._resource_context(prepared):
            return method(self, *args, **prepared)

    return sync_call


class AgentResourceMixin:
    """Bind managed stores to calls without changing global Agent dependencies."""

    def _resource_call_kwargs(self, args, kwargs):
        # Module creates event identities before input preparation. Extract an
        # envelope's history here so its thread also owns run.start events.
        if self._resources is None and _CURRENT_RESOURCES.get() is None:
            return kwargs
        message = args[0] if args else kwargs.get("message")
        if (
            kwargs.get("messages") is None
            and isinstance(message, dotdict)
            and self.messages is not None
        ):
            messages = self._get_content_from_message(self.messages, message)
            if messages is not None:
                return {**kwargs, "messages": messages}
        return kwargs

    def _call_impl(self, *args, **kwargs):
        return super()._call_impl(*args, **self._resource_call_kwargs(args, kwargs))

    async def _acall_impl(self, *args, **kwargs):
        return await super()._acall_impl(
            *args, **self._resource_call_kwargs(args, kwargs)
        )

    async def stream_events(self, *args, **kwargs):
        prepared = self._resource_call_kwargs(args, kwargs)
        async with aclosing(super().stream_events(*args, **prepared)) as events:
            async for event in events:
                yield event

    def _init_resources(self, agent_dir, *, checkpoint_store, agent_inbox, approvals):
        if agent_dir is not None and any(
            item is not None for item in (checkpoint_store, agent_inbox, approvals)
        ):
            raise ValueError(
                "agent_dir cannot be combined with explicit checkpoint_store, "
                "agent_inbox or approvals; choose managed or host-owned storage"
            )
        self._resources = AgentResources(agent_dir) if agent_dir is not None else None
        self.agent_dir: Path | None = (
            self._resources.agent_dir if self._resources is not None else None
        )
        self._resource_lock = RLock()
        self._owned_threads: dict[str, _OwnedThread] = {}
        self._resources_closed = False
        self._resource_close_tasks = {}
        self._resource_close_callbacks_done: set[str] = set()
        self._resource_close_callback_failures: dict[str, BaseException] = {}
        self._resource_shutdown_task = None

    def _bind_resources(self, thread_id):
        if self._resources is None:
            return None
        with self._resource_lock:
            if self._resources_closed:
                raise RuntimeError("Agent resources are closed")
            owned = self._owned_threads.get(thread_id)
            if owned is None:
                bundle = self._resources.bind(
                    thread_id,
                    namespace=self.get_module_name(),
                    verbose=self.config.get("verbose", False),
                )
                owned = _OwnedThread(bundle)
                self._owned_threads[thread_id] = owned
            if owned.closing:
                raise RuntimeError("Agent thread resources are closing")
            return owned.resources

    @contextmanager
    def _resource_context(self, kwargs):
        inherited = _CURRENT_RESOURCES.get()
        if inherited is None and self._resources is None:
            from msgflux.nn.extensions.tool_output import (  # noqa: PLC0415
                _bind_managed_offload,
            )

            _bind_managed_offload(self, None)
            yield
            return
        scope = self._get_requested_scope(kwargs) or get_execution_context()["scope"]
        thread_id = self._resolve_thread_id(
            messages=kwargs.get("messages"), thread_id=scope.thread_id
        )
        scope = scope.with_overrides(
            thread_id=thread_id, namespace=self.get_module_name()
        )
        kwargs["scope"] = scope
        if inherited is not None:
            if thread_id != inherited.resources.thread_id:
                raise ValueError("Nested Agent cannot replace its durable thread")
            if self.agent_dir is not None and (
                self.agent_dir != inherited.resources.thread_dir.parent.parent
            ):
                raise ValueError("Nested Agent cannot replace its agent_dir")
            if self.checkpoint_store is not None or self.approvals is not None:
                raise ValueError("Nested Agent cannot override managed persistence")
            owned = inherited
        else:
            context = get_execution_context()
            with self._resource_lock:
                existing = self._owned_threads.get(thread_id)
                for key in ("checkpoint_store", "task_store", "agent_inbox"):
                    current = context.get(key)
                    expected = (
                        getattr(existing.resources, key)
                        if existing is not None
                        else None
                    )
                    if current is not None and current is not expected:
                        raise ValueError(f"agent_dir conflicts with inherited {key}")
                self._bind_resources(thread_id)
                owned = self._owned_threads[thread_id]
        from msgflux.nn.extensions.tool_output import (  # noqa: PLC0415
            _bind_managed_offload,
        )

        _bind_managed_offload(self, owned, inherit=inherited is not None)
        # The GIL does not make a read-modify-write lifecycle operation atomic.
        with owned.lock:
            if owned.closing:
                raise RuntimeError("Agent thread resources are closing")
            owned.active += 1
            owned.active_calls += 1
            owned.libraries.add(self.tool_library)
        token = _CURRENT_RESOURCES.set(owned)
        try:
            with execution_context(
                scope=scope,
                checkpoint_store=owned.resources.checkpoint_store,
                task_store=owned.resources.task_store,
                agent_inbox=owned.resources.agent_inbox,
            ):
                yield
        finally:
            _CURRENT_RESOURCES.reset(token)
            with owned.lock:
                owned.active -= 1
                owned.active_calls -= 1

    def _retain_detached_resources(self):
        """Keep owned thread resources open through detached stream finalizers."""
        owned = _CURRENT_RESOURCES.get()
        if owned is None:
            return lambda: None
        finished = Future()
        with owned.lock:
            if owned.closing and owned.active_calls <= 0:
                raise RuntimeError("Agent thread resources are closing")
            owned.active += 1
            owned.detached_finalizers.add(finished)

        def release() -> None:
            with owned.lock:
                if finished not in owned.detached_finalizers:
                    return
                owned.detached_finalizers.remove(finished)
                owned.active -= 1
                if not finished.done():
                    finished.set_result(None)

        return release

    async def _close_thread_resources(self, thread_id, *, before_close=None):
        with self._resource_lock:
            callback_failure = self._resource_close_callback_failures.get(thread_id)
            if callback_failure is not None:
                raise callback_failure
            owned = self._owned_threads.get(thread_id)
            if owned is not None:
                task = self._resource_close_tasks.get(thread_id)
                if task is None or (task.done() and task.exception() is not None):
                    task = asyncio.create_task(
                        self._settle_thread_resources(thread_id, owned, before_close),
                        context=contextvars.Context(),
                    )
                    self._resource_close_tasks[thread_id] = task
        if owned is None:
            await self._run_thread_close_callback(thread_id, before_close)
            return
        await asyncio.shield(task)

    def _thread_release_reason(self, thread_id) -> str | None:
        """Explain why this managed thread cannot be safely released as idle."""
        with self._resource_lock:
            owned = self._owned_threads.get(thread_id)
            if owned is None:
                return None
            with owned.lock:
                return self._thread_release_reason_locked(thread_id, owned)

    def _thread_release_reason_locked(self, thread_id, owned) -> str | None:
        if owned.closing:
            return "Thread resources are already closing."
        if owned.active:
            return "An Agent call is still using thread resources."

        reason = self._active_background_future_reason(tuple(owned.libraries))
        if reason is not None:
            return reason
        reason = self._unfinished_task_reason(thread_id, owned.resources.task_store)
        if reason is not None:
            return reason
        return self._checkpoint_release_reason(thread_id, owned)

    @staticmethod
    def _active_background_future_reason(libraries) -> str | None:
        for library in libraries:
            dispatcher = getattr(library, "_background_dispatcher", None)
            if dispatcher is None:
                continue
            with dispatcher._task_futures_lock:
                if any(
                    not future.done() for future in dispatcher._task_futures.values()
                ):
                    return "A background task future is still active."
        return None

    @staticmethod
    def _unfinished_task_reason(thread_id, task_store) -> str | None:
        unfinished_query = getattr(task_store, "has_unfinished_for_thread", None)
        if unfinished_query is None:
            return "The task store cannot confirm thread quiescence."
        try:
            if unfinished_query(thread_id=thread_id):
                return "A queued, running, or paused background task remains."
        except Exception as error:
            return f"Background task state could not be checked: {error}"
        return None

    def _checkpoint_release_reason(self, thread_id, owned) -> str | None:
        checkpoints = owned.resources.checkpoint_store
        namespace = self.get_module_name()
        try:
            for status in ("running", "paused", "failed"):
                for run in checkpoints.list_runs(namespace, thread_id, status=status):
                    checkpoint = checkpoints.load_state(
                        namespace, thread_id, run["run_id"]
                    )
                    if checkpoint is None:
                        return "A nonterminal checkpoint is unavailable."
                    if status == "running":
                        return "A checkpoint still records a running attempt."
                    try:
                        self._validate_checkpoint_command_receipts(checkpoint)
                    except TaskPauseRequestedError as error:
                        return str(error)
                    reason = self._uncertain_checkpoint_approval_reason(checkpoint)
                    if reason is not None:
                        return reason
        except Exception as error:
            return f"Checkpoint quiescence could not be checked: {error}"
        return None

    @staticmethod
    def _uncertain_checkpoint_approval_reason(checkpoint) -> str | None:
        extensions = checkpoint.get("runtime", {}).get("extensions", {})
        if not isinstance(extensions, dict):
            return "Checkpoint approval state is malformed."
        pending = extensions.get("pending_approvals")
        if pending is None:
            return None
        if not isinstance(pending, dict) or pending.get("schema_version") != 1:
            return "Checkpoint approval state is unsupported."
        requests = pending.get("requests")
        if not isinstance(requests, dict):
            return "Checkpoint approval requests are malformed."
        phase = pending.get("phase")
        if phase is None and requests:
            return None  # Legacy awaiting-decision checkpoints are quiescent.
        if phase in {"awaiting_decision", "awaiting-decision", "approved"}:
            return None
        return "Pending approval execution requires host reconciliation."

    def _mark_thread_idle_closing(self, thread_id) -> str | None:
        """Atomically refuse activity or fence new calls before idle cleanup."""
        with self._resource_lock:
            owned = self._owned_threads.get(thread_id)
            if owned is None:
                return None
            with owned.lock:
                reason = self._thread_release_reason_locked(thread_id, owned)
                if reason is not None:
                    return reason
                owned.closing = True
                owned.idle_closing = True
                return None

    async def _settle_thread_resources(self, thread_id, owned, before_close):
        with owned.lock:
            owned.closing = True
            idle_closing = owned.idle_closing
            children_settled = owned.children_settled
        if not idle_closing and not children_settled:
            # Shutdown may cooperatively interrupt child work. Idle release
            # already proved quiescence and must not use close to discover it.
            while True:
                futures = set()
                with owned.lock:
                    libraries = tuple(owned.libraries)
                for task in owned.resources.task_store.list():
                    for library in libraries:
                        future = library.get_background_dispatcher().get_task_future(
                            task.task_id
                        )
                        if future is not None and not future.done():
                            owned.resources.task_store.request_interrupt(task.task_id)
                            futures.add(future)
                if not futures:
                    break
                await asyncio.gather(
                    *(
                        asyncio.wrap_future(f) if isinstance(f, Future) else f
                        for f in futures
                    ),
                    return_exceptions=True,
                )
        with owned.lock:
            if owned.active_calls:
                raise RuntimeError("Stop active Agent calls before closing resources")
            owned.children_settled = True
        await self._wait_for_detached_finalizers(owned)
        with owned.lock:
            if owned.active:
                raise RuntimeError("Stop active Agent calls before closing resources")
        await self._run_thread_close_callback(thread_id, before_close)
        with owned.lock:
            owned.resources.close()

    @staticmethod
    async def _wait_for_detached_finalizers(owned):
        while True:
            with owned.lock:
                if owned.active_calls:
                    raise RuntimeError(
                        "Stop active Agent calls before closing resources"
                    )
                if not owned.detached_finalizers:
                    return
                pending = tuple(owned.detached_finalizers)
            for future in pending:
                await asyncio.shield(asyncio.wrap_future(future))

    async def _run_thread_close_callback(self, thread_id, callback):
        """Run host cleanup once; failed callbacks remain quarantined.

        Host callbacks can have arbitrary side effects, so retrying one after an
        exception could duplicate cleanup that actually completed. An operator
        must resolve that failure explicitly before resource shutdown can retry.
        """
        if callback is None:
            return
        with self._resource_lock:
            failure = self._resource_close_callback_failures.get(thread_id)
            if failure is not None:
                raise failure
            if thread_id in self._resource_close_callbacks_done:
                return
        try:
            result = callback()
            if inspect.isawaitable(result):
                await result
        except BaseException as error:
            with self._resource_lock:
                self._resource_close_callback_failures[thread_id] = error
            raise
        with self._resource_lock:
            self._resource_close_callbacks_done.add(thread_id)

    async def aclose(self):
        """Close owned persistence after stopping calls and delegated work.

        Models, workspaces and explicit adapters remain owned by their caller.
        The service stops foreground runs before invoking this lifecycle.
        Cancelling this wait leaves cleanup running.
        """
        with self._resource_lock:
            if self._resource_shutdown_task is None or (
                self._resource_shutdown_task.done()
                and self._resource_shutdown_task.exception() is not None
            ):
                self._resources_closed = True
                self._resource_shutdown_task = asyncio.create_task(
                    self._shutdown_resources(), context=contextvars.Context()
                )
            task = self._resource_shutdown_task
        await asyncio.shield(task)

    async def _shutdown_resources(self):
        errors = []
        for thread_id in tuple(self._owned_threads):
            try:
                await self._close_thread_resources(thread_id)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Agent resource shutdown failed", errors)
