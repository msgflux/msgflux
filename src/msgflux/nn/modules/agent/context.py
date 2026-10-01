# ruff: noqa: A002

"""Shared Agent execution context and compatibility helpers."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Union

from msgflux.chat_messages import ChatMessages
from msgflux.nn.hooks.events import BeforeResume
from msgflux.runtime.agent_run import AgentRun, agent_run_context
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.workspace.receipts import (
    bind_command_receipt_persist,
    retain_command_receipts,
)

if TYPE_CHECKING:
    from msgflux.nn.modules.agent.core import Agent


def _apply_before_resume(
    resumed: Mapping[str, Any],
    event: Any,
    *,
    vars: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and apply a transformed durable-resume payload."""
    if not isinstance(event, BeforeResume):
        raise TypeError("Agent `before_resume` hooks must return BeforeResume")
    if not isinstance(event.messages, ChatMessages):
        raise TypeError("BeforeResume.messages must be ChatMessages")
    if not isinstance(event.scope, ExecutionScope):
        raise TypeError("BeforeResume.scope must be an ExecutionScope")
    if event.model_preference is not None and not isinstance(
        event.model_preference, str
    ):
        raise TypeError("BeforeResume.model_preference must be a string or None")

    restored_scope = resumed["scope"]
    identity = ("thread_id", "namespace", "run_id")
    changed = [
        field
        for field in identity
        if getattr(event.scope, field) != getattr(restored_scope, field)
    ]
    if changed:
        fields = ", ".join(changed)
        raise ValueError(
            f"BeforeResume cannot change restored checkpoint identity fields: {fields}."
        )

    return {
        **resumed,
        "messages": event.messages,
        "model_preference": event.model_preference,
        "scope": event.scope,
        "vars": vars,
    }


def _apply_explicit_model_preference(
    inputs: dict[str, Any], kwargs: Mapping[str, Any]
) -> None:
    """Honor the caller's selection over the saved or hook-provided preference."""
    preference = kwargs.get("model_preference")
    if preference is not None:
        inputs["model_preference"] = preference


def _require_lifecycle_payload(event: str, payload: Any, expected_type: type):
    if not isinstance(payload, expected_type):
        raise TypeError(
            f"Agent `{event}` hooks must return {expected_type.__name__} or None"
        )
    return payload


# Reserved kwargs that should not be treated as task inputs
_RESERVED_KWARGS = {
    "task",
    "vars",
    "messages",
    "task_multimodal",
    "task_context",
    "model_preference",
    "tool_filter",
    "scope",
    "tool_call_id",
    "approvals",
}

_UNSET = object()
_DEFAULT_AGENT_ANNOTATIONS = {"task": str, "return": str}
ToolFilterValue = Union[str, List[str]]
ToolFilter = Dict[str, ToolFilterValue]


class _BeforeRunEndHookError(Exception):
    """Identify a failed pre-commit hook so it is not executed twice."""

    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error = error


_CURRENT_AGENT_CONTEXT = contextvars.ContextVar(
    "msgflux_current_agent_context",
    default=None,
)


@contextmanager
def _agent_context(agent: Agent, *, scope, vars):
    current = _CURRENT_AGENT_CONTEXT.get() or {}
    agent_id = id(agent)
    if agent_id in current:
        with agent_run_context(current[agent_id]["run"]):
            yield current[agent_id]
        return
    run = AgentRun(
        namespace=agent.get_module_name(),
        thread_id=scope.thread_id,
        run_id=scope.run_id,
    )
    state = {"scope": scope, "vars": vars or {}, "run": run, "agent": agent}
    updated = dict(current)
    updated[agent_id] = state
    token = _CURRENT_AGENT_CONTEXT.set(updated)
    receipt_context = (
        bind_command_receipt_persist(
            lambda receipt: _persist_command_receipt(state, receipt)
        )
        if agent._get_effective_checkpoint_store() is not None
        else nullcontext()
    )
    try:
        with agent_run_context(run), receipt_context:
            yield state
    finally:
        _CURRENT_AGENT_CONTEXT.reset(token)


def _persist_command_receipt(state, receipt) -> None:
    """Atomically checkpoint a command update without running lifecycle hooks."""
    agent = state["agent"]
    run = state["run"]
    messages = state.get("messages")
    store = agent._get_effective_checkpoint_store()
    if not isinstance(messages, ChatMessages) or store is None:
        raise RuntimeError(
            "Cannot persist command receipt without Agent checkpoint state"
        )
    if not getattr(store, "supports_atomic_commit", False):
        raise RuntimeError("Command receipts require atomic checkpoint commits")
    thread_id = messages.thread_id or run.thread_id
    run_id = run.run_id
    namespace = agent.get_module_name()
    if not all(
        isinstance(value, str) and value for value in (thread_id, run_id, namespace)
    ):
        raise RuntimeError("Command receipt requires a durable Agent run identity")
    with run._command_receipt_lock:
        state_snapshot = agent._build_checkpoint_state(messages, status="running")
        extensions = dict(run.extension_state)
        extensions["command_receipts"] = retain_command_receipts(
            extensions.get("command_receipts", []), receipt
        )
        committed = store.commit_state(
            namespace,
            thread_id,
            run_id,
            state_snapshot,
            expected_revision=run.revision,
            extension_state=extensions,
            branch_id=run.branch_id,
            head_item_id=run.head_item_id,
            event={"event_type": "command_receipt", "state": receipt.state},
        )
        run.extension_state = extensions
        run.revision = committed.revision


def _prepare_agent_guard_input(model_execution_params):
    """Extract user content from ChatML messages for guard validation."""
    messages = model_execution_params.get("messages")
    if not messages:
        return model_execution_params
    last_message = messages[-1]
    if isinstance(last_message.get("content"), list):
        if last_message.get("content")[0]["type"] == "image_url":
            return [last_message]
        else:
            return last_message.get("content")[-1]
    else:
        return last_message.get("content")


def _prepare_agent_guard_output(model_response):
    """Convert model response to string for guard validation."""
    if isinstance(model_response, str):
        return model_response
    return str(model_response)


__all__ = ["ToolFilter", "ToolFilterValue"]
