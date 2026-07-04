"""Execution context propagated through msgFlux runtimes.

This module owns the shared ContextVars used by components that need execution
identity and inherited runtime configuration. The first durable features mainly
use `thread_id`, `run_id`, and `namespace`, but the extra fields are already
present so nested runtimes can propagate lineage consistently later.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

from msgflux.runtime.abort import AbortSignal

DEFAULT_NAMESPACE = "default_namespace"


def new_thread_id() -> str:
    return f"thd_{uuid4().hex}"


def new_run_id() -> str:
    return f"run_{uuid4().hex}"


@dataclass(frozen=True)
class ExecutionScope:
    """Durable execution identity propagated through runtime components."""

    thread_id: str | None = None
    namespace: str = DEFAULT_NAMESPACE
    run_id: str | None = None
    parent_run_id: str | None = None
    root_run_id: str | None = None
    abort_signal: AbortSignal | None = None

    def with_overrides(
        self,
        *,
        thread_id: str | None = None,
        namespace: str | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> ExecutionScope:
        resolved_run_id = run_id if run_id is not None else self.run_id
        return ExecutionScope(
            thread_id=thread_id if thread_id is not None else self.thread_id,
            namespace=namespace if namespace is not None else self.namespace,
            run_id=resolved_run_id,
            parent_run_id=(
                parent_run_id if parent_run_id is not None else self.parent_run_id
            ),
            root_run_id=(
                root_run_id
                if root_run_id is not None
                else self.root_run_id or resolved_run_id
            ),
            abort_signal=(
                abort_signal if abort_signal is not None else self.abort_signal
            ),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "thread_id": self.thread_id,
            "namespace": self.namespace,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "root_run_id": self.root_run_id,
        }


_CURRENT_SCOPE: contextvars.ContextVar[ExecutionScope | None] = contextvars.ContextVar(
    "msgflux_execution_scope",
    default=None,
)
_CURRENT_THREAD_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "msgflux_thread_id",
    default=None,
)
_CURRENT_NAMESPACE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "msgflux_namespace",
    default=DEFAULT_NAMESPACE,
)
_CURRENT_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "msgflux_run_id",
    default=None,
)
_CURRENT_PARENT_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "msgflux_parent_run_id",
    default=None,
)
_CURRENT_ROOT_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "msgflux_root_run_id",
    default=None,
)
_CURRENT_CHECKPOINT_STORE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "msgflux_checkpoint_store",
    default=None,
)
_CURRENT_TASK_STORE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "msgflux_task_store",
    default=None,
)
_CURRENT_AGENT_INBOX: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "msgflux_agent_inbox",
    default=None,
)
_CURRENT_TASK_HANDLE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "msgflux_task_handle",
    default=None,
)
_CURRENT_TASK_ACTIVITY_RECORDER: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "msgflux_task_activity_recorder",
    default=None,
)
_CURRENT_ABORT_SIGNAL: contextvars.ContextVar[AbortSignal | None] = (
    contextvars.ContextVar(
        "msgflux_abort_signal",
        default=None,
    )
)


@contextmanager
def execution_context(
    *,
    scope: ExecutionScope | None = None,
    thread_id: str | None = None,
    namespace: str | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    root_run_id: str | None = None,
    checkpoint_store: Any = None,
    task_store: Any = None,
    agent_inbox: Any = None,
    task_handle: Any = None,
    task_activity_recorder: Any = None,
    abort_signal: AbortSignal | None = None,
):
    """Set execution identity for the enclosed scope.

    Resolution rules intentionally favor explicit inputs and otherwise inherit
    from the current context. Call boundaries that require durable identities
    should materialize missing thread/run ids explicitly.
    """
    if scope is not None and not isinstance(scope, ExecutionScope):
        raise TypeError(
            f"`scope` must be an ExecutionScope or None, given `{type(scope)}`"
        )

    current_scope = get_execution_scope()
    base_scope = scope or current_scope

    current_thread_id = _CURRENT_THREAD_ID.get()
    resolved_thread_id = (
        thread_id
        if thread_id is not None
        else base_scope.thread_id or current_thread_id
    )
    current_namespace = _CURRENT_NAMESPACE.get()
    resolved_namespace = (
        namespace
        if namespace is not None
        else base_scope.namespace or current_namespace or DEFAULT_NAMESPACE
    )

    current_run_id = _CURRENT_RUN_ID.get()
    resolved_run_id = run_id if run_id is not None else base_scope.run_id
    if resolved_run_id is None:
        resolved_run_id = current_run_id

    current_parent_run_id = _CURRENT_PARENT_RUN_ID.get()
    resolved_parent_run_id = (
        parent_run_id
        if parent_run_id is not None
        else base_scope.parent_run_id or current_parent_run_id
    )

    current_root_run_id = _CURRENT_ROOT_RUN_ID.get()
    if root_run_id is not None:
        resolved_root_run_id = root_run_id
    elif base_scope.root_run_id is not None:
        resolved_root_run_id = base_scope.root_run_id
    elif current_root_run_id is not None:
        resolved_root_run_id = current_root_run_id
    else:
        resolved_root_run_id = resolved_run_id

    current_abort_signal = _CURRENT_ABORT_SIGNAL.get()
    resolved_abort_signal = (
        abort_signal
        if abort_signal is not None
        else base_scope.abort_signal or current_abort_signal
    )

    resolved_scope = ExecutionScope(
        thread_id=resolved_thread_id,
        namespace=resolved_namespace,
        run_id=resolved_run_id,
        parent_run_id=resolved_parent_run_id,
        root_run_id=resolved_root_run_id,
        abort_signal=resolved_abort_signal,
    )

    current_checkpoint_store = _CURRENT_CHECKPOINT_STORE.get()
    resolved_checkpoint_store = (
        checkpoint_store if checkpoint_store is not None else current_checkpoint_store
    )
    current_task_store = _CURRENT_TASK_STORE.get()
    resolved_task_store = task_store if task_store is not None else current_task_store
    current_agent_inbox = _CURRENT_AGENT_INBOX.get()
    resolved_agent_inbox = (
        agent_inbox if agent_inbox is not None else current_agent_inbox
    )
    current_task_handle = _CURRENT_TASK_HANDLE.get()
    resolved_task_handle = (
        task_handle if task_handle is not None else current_task_handle
    )
    current_task_activity_recorder = _CURRENT_TASK_ACTIVITY_RECORDER.get()
    resolved_task_activity_recorder = (
        task_activity_recorder
        if task_activity_recorder is not None
        else current_task_activity_recorder
    )

    scope_token = _CURRENT_SCOPE.set(resolved_scope)
    thread_token = _CURRENT_THREAD_ID.set(resolved_scope.thread_id)
    namespace_token = _CURRENT_NAMESPACE.set(resolved_scope.namespace)
    run_token = _CURRENT_RUN_ID.set(resolved_scope.run_id)
    parent_run_token = _CURRENT_PARENT_RUN_ID.set(resolved_scope.parent_run_id)
    root_run_token = _CURRENT_ROOT_RUN_ID.set(resolved_scope.root_run_id)
    checkpoint_token = _CURRENT_CHECKPOINT_STORE.set(resolved_checkpoint_store)
    task_store_token = _CURRENT_TASK_STORE.set(resolved_task_store)
    inbox_token = _CURRENT_AGENT_INBOX.set(resolved_agent_inbox)
    task_handle_token = _CURRENT_TASK_HANDLE.set(resolved_task_handle)
    activity_token = _CURRENT_TASK_ACTIVITY_RECORDER.set(
        resolved_task_activity_recorder
    )
    abort_token = _CURRENT_ABORT_SIGNAL.set(resolved_abort_signal)
    try:
        yield resolved_scope
    finally:
        _CURRENT_SCOPE.reset(scope_token)
        _CURRENT_THREAD_ID.reset(thread_token)
        _CURRENT_NAMESPACE.reset(namespace_token)
        _CURRENT_RUN_ID.reset(run_token)
        _CURRENT_PARENT_RUN_ID.reset(parent_run_token)
        _CURRENT_ROOT_RUN_ID.reset(root_run_token)
        _CURRENT_CHECKPOINT_STORE.reset(checkpoint_token)
        _CURRENT_TASK_STORE.reset(task_store_token)
        _CURRENT_AGENT_INBOX.reset(inbox_token)
        _CURRENT_TASK_HANDLE.reset(task_handle_token)
        _CURRENT_TASK_ACTIVITY_RECORDER.reset(activity_token)
        _CURRENT_ABORT_SIGNAL.reset(abort_token)


@contextmanager
def thread_context(
    *,
    thread_id: str | None = None,
    namespace: str | None = None,
):
    """Compatibility helper for callers that only care about thread scope."""
    with execution_context(thread_id=thread_id, namespace=namespace):
        yield


def get_execution_scope() -> ExecutionScope:
    """Return the active execution scope."""
    return _CURRENT_SCOPE.get() or ExecutionScope()


def get_execution_context() -> Mapping[str, Any | None]:
    """Return the current execution context."""
    scope = get_execution_scope()
    return {
        "scope": scope,
        "thread_id": scope.thread_id,
        "namespace": scope.namespace,
        "run_id": scope.run_id,
        "parent_run_id": scope.parent_run_id,
        "root_run_id": scope.root_run_id,
        "checkpoint_store": _CURRENT_CHECKPOINT_STORE.get(),
        "task_store": _CURRENT_TASK_STORE.get(),
        "agent_inbox": _CURRENT_AGENT_INBOX.get(),
        "task_handle": _CURRENT_TASK_HANDLE.get(),
        "task_activity_recorder": _CURRENT_TASK_ACTIVITY_RECORDER.get(),
        "abort_signal": _CURRENT_ABORT_SIGNAL.get(),
    }


def get_thread_context() -> Mapping[str, str | None]:
    """Return the current thread-scoped context."""
    scope = get_execution_scope()
    return {
        "thread_id": scope.thread_id,
        "namespace": scope.namespace,
    }


def get_thread_id() -> str | None:
    """Return the active execution thread id."""
    return get_execution_scope().thread_id
