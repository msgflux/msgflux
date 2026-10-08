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
from msgflux.runtime.agent_resources import AgentResources, BoundAgentResources
from msgflux.runtime.context import execution_context, get_execution_context


class _OwnedThread(msgspec.Struct):
    resources: BoundAgentResources
    libraries: set = msgspec.field(default_factory=set)
    active: int = 0
    closing: bool = False
    lock: object = msgspec.field(default_factory=RLock)


_CURRENT_RESOURCES = ContextVar("msgflux_owned_agent_resources", default=None)


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
        # The GIL does not make a read-modify-write lifecycle operation atomic.
        with owned.lock:
            if owned.closing:
                raise RuntimeError("Agent thread resources are closing")
            owned.active += 1
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

    async def _close_thread_resources(self, thread_id, *, before_close=None):
        with self._resource_lock:
            owned = self._owned_threads.get(thread_id)
            if owned is not None:
                task = self._resource_close_tasks.get(thread_id)
                if task is None or (task.done() and task.exception() is not None):
                    task = asyncio.create_task(
                        self._settle_thread_resources(owned, before_close),
                        context=contextvars.Context(),
                    )
                    self._resource_close_tasks[thread_id] = task
        if owned is None:
            if before_close is not None:
                result = before_close()
                if inspect.isawaitable(result):
                    await result
            return
        await asyncio.shield(task)

    async def _settle_thread_resources(self, owned, before_close):
        with owned.lock:
            owned.closing = True
        # Child libraries register in the same lifetime, so their futures also
        # settle before the stores close. Cooperative interruption is not kill.
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
            if owned.active:
                raise RuntimeError("Stop active Agent calls before closing resources")
        if before_close is not None:
            result = before_close()
            if inspect.isawaitable(result):
                await result
        with owned.lock:
            owned.resources.close()

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
