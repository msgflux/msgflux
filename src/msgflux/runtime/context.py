"""Execution context propagated through msgFlux runtimes.

This module owns the shared ContextVars used by components that need execution
identity and inherited runtime configuration. The first durable features mainly
use `session_id`, `run_id`, and `namespace`, but the extra fields are already
present so nested runtimes can propagate lineage consistently later.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_SESSION_ID = "default"
DEFAULT_NAMESPACE = "default"


@dataclass(frozen=True)
class ExecutionScope:
    """Durable execution identity propagated through runtime components."""

    session_id: str = DEFAULT_SESSION_ID
    namespace: str = DEFAULT_NAMESPACE
    run_id: str | None = None
    parent_run_id: str | None = None
    root_run_id: str | None = None

    def with_overrides(
        self,
        *,
        session_id: str | None = None,
        namespace: str | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
    ) -> ExecutionScope:
        resolved_run_id = run_id if run_id is not None else self.run_id
        return ExecutionScope(
            session_id=session_id if session_id is not None else self.session_id,
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
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "session_id": self.session_id,
            "namespace": self.namespace,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "root_run_id": self.root_run_id,
        }


_CURRENT_SCOPE: contextvars.ContextVar[ExecutionScope | None] = contextvars.ContextVar(
    "msgflux_execution_scope",
    default=None,
)
_CURRENT_SESSION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "msgflux_session_id",
    default=DEFAULT_SESSION_ID,
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


@contextmanager
def execution_context(
    *,
    scope: ExecutionScope | None = None,
    session_id: str | None = None,
    namespace: str | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    root_run_id: str | None = None,
    checkpoint_store: Any = None,
    agent_inbox: Any = None,
    task_handle: Any = None,
    task_activity_recorder: Any = None,
):
    """Set execution identity for the enclosed scope.

    Resolution rules intentionally favor explicit inputs and otherwise inherit
    from the current context. When no session exists yet, the stable default
    session is used so components can always rely on an active session id.
    """
    if scope is not None and not isinstance(scope, ExecutionScope):
        raise TypeError(
            f"`scope` must be an ExecutionScope or None, given `{type(scope)}`"
        )

    current_scope = get_execution_scope()
    base_scope = scope or current_scope

    current_session_id = _CURRENT_SESSION_ID.get()
    resolved_session_id = (
        session_id
        if session_id is not None
        else base_scope.session_id or current_session_id
    )
    if resolved_session_id is None:
        resolved_session_id = DEFAULT_SESSION_ID

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

    resolved_scope = ExecutionScope(
        session_id=resolved_session_id,
        namespace=resolved_namespace,
        run_id=resolved_run_id,
        parent_run_id=resolved_parent_run_id,
        root_run_id=resolved_root_run_id,
    )

    current_checkpoint_store = _CURRENT_CHECKPOINT_STORE.get()
    resolved_checkpoint_store = (
        checkpoint_store if checkpoint_store is not None else current_checkpoint_store
    )
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
    session_token = _CURRENT_SESSION_ID.set(resolved_scope.session_id)
    namespace_token = _CURRENT_NAMESPACE.set(resolved_scope.namespace)
    run_token = _CURRENT_RUN_ID.set(resolved_scope.run_id)
    parent_run_token = _CURRENT_PARENT_RUN_ID.set(resolved_scope.parent_run_id)
    root_run_token = _CURRENT_ROOT_RUN_ID.set(resolved_scope.root_run_id)
    checkpoint_token = _CURRENT_CHECKPOINT_STORE.set(resolved_checkpoint_store)
    inbox_token = _CURRENT_AGENT_INBOX.set(resolved_agent_inbox)
    task_handle_token = _CURRENT_TASK_HANDLE.set(resolved_task_handle)
    activity_token = _CURRENT_TASK_ACTIVITY_RECORDER.set(
        resolved_task_activity_recorder
    )
    try:
        yield resolved_scope
    finally:
        _CURRENT_SCOPE.reset(scope_token)
        _CURRENT_SESSION_ID.reset(session_token)
        _CURRENT_NAMESPACE.reset(namespace_token)
        _CURRENT_RUN_ID.reset(run_token)
        _CURRENT_PARENT_RUN_ID.reset(parent_run_token)
        _CURRENT_ROOT_RUN_ID.reset(root_run_token)
        _CURRENT_CHECKPOINT_STORE.reset(checkpoint_token)
        _CURRENT_AGENT_INBOX.reset(inbox_token)
        _CURRENT_TASK_HANDLE.reset(task_handle_token)
        _CURRENT_TASK_ACTIVITY_RECORDER.reset(activity_token)


@contextmanager
def session_context(
    *,
    session_id: str | None = None,
    namespace: str | None = None,
):
    """Compatibility helper for callers that only care about session scope."""
    with execution_context(session_id=session_id, namespace=namespace):
        yield


def get_execution_scope() -> ExecutionScope:
    """Return the active execution scope."""
    return _CURRENT_SCOPE.get() or ExecutionScope()


def get_execution_context() -> Mapping[str, Any | None]:
    """Return the current execution context."""
    scope = get_execution_scope()
    return {
        "scope": scope,
        "session_id": scope.session_id,
        "namespace": scope.namespace,
        "run_id": scope.run_id,
        "parent_run_id": scope.parent_run_id,
        "root_run_id": scope.root_run_id,
        "checkpoint_store": _CURRENT_CHECKPOINT_STORE.get(),
        "agent_inbox": _CURRENT_AGENT_INBOX.get(),
        "task_handle": _CURRENT_TASK_HANDLE.get(),
        "task_activity_recorder": _CURRENT_TASK_ACTIVITY_RECORDER.get(),
    }


def get_session_context() -> Mapping[str, str | None]:
    """Return the current session-scoped context."""
    scope = get_execution_scope()
    return {
        "session_id": scope.session_id,
        "namespace": scope.namespace,
    }


def get_session_id() -> str:
    """Return the active execution session id."""
    return get_execution_scope().session_id
