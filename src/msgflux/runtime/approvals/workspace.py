"""Workspace-derived approval policy for durable managed Agent threads."""

from __future__ import annotations

import hashlib
import inspect
import marshal
from typing import TYPE_CHECKING

import msgspec

from msgflux.runtime.approvals.agent import AgentApprovals
from msgflux.runtime.context import get_execution_context
from msgflux.tools.workspace_changes import workspace_change_tool

if TYPE_CHECKING:
    from msgflux.nn.modules.agent.core import Agent

_APPROVED_CAPABILITIES = frozenset(
    {
        "filesystem.write",
        "filesystem.delete",
        "filesystem.mkdir",
        "process.execute",
    }
)


def get_workspace_approvals(
    agent: Agent,
    *,
    force: bool = False,
    _policy_version: str | None = None,
) -> AgentApprovals | None:
    """Build the current managed workspace approval policy, if enabled.

    The current durable override wins over the Workspace default. A stored
    pending batch keeps a resolver available across a change to ``never`` so
    replay is subject to the approval package's binding and reconciliation
    checks instead of dropping the policy and executing the old call.
    """
    if type(force) is not bool:
        raise TypeError("force must be a boolean")
    context = get_execution_context()
    scope = context["scope"]
    mode, revision = _approval_mode(context)

    from msgflux.runtime.agent_run import get_agent_run  # noqa: PLC0415

    run = get_agent_run()
    pending = run.get_extension("pending_approvals") if run is not None else None
    if mode == "never" and not force and pending is None:
        return None

    selected = _selected_tool_revisions(agent)
    if not selected:
        return None
    owned = _current_owned_thread(agent, scope.thread_id)
    if owned is None:
        if mode == "on-request" or force:
            raise ValueError(
                "Workspace approval policy requires a managed approval store"
            )
        return None

    policy_version = _policy_version
    if policy_version is None and pending is not None:
        policy_version = pending.get("policy_version")
    if policy_version is None:
        policy_version = f"workspace-{mode}-r{revision}"
    if not isinstance(policy_version, str) or not policy_version:
        raise ValueError("Pending approval policy version is invalid")
    return AgentApprovals(owned.resources.approval_store, selected, policy_version)


def _approval_mode(context):
    scope = context["scope"]
    state = context.get("workspace_policy")
    if state is not None:
        current = state.current
        if current.thread_id != scope.thread_id:
            raise ValueError("Workspace policy does not belong to this thread")
        return current.approval_policy, current.revision
    workspace = scope.workspace
    return getattr(workspace, "approval_policy", "never"), 0


def _selected_tool_revisions(agent):
    selected = {}
    for definition in agent.tool_library.registry.definitions():
        protected = bool(
            _APPROVED_CAPABILITIES.intersection(definition.required_permissions)
        )
        protected = protected or any(
            item.action in _APPROVED_CAPABILITIES
            for item in definition.required_resources
        )
        protected = protected or workspace_change_tool(definition) is not None
        if not protected:
            continue
        if (
            definition.dispatch.name != "foreground"
            or definition.feedback.name == "call_as_response"
        ):
            raise ValueError(
                "Workspace approval policy cannot protect non-foreground tool "
                f"{definition.name!r}"
            )
        selected[definition.name] = _definition_revision(definition)
    return selected


def _current_owned_thread(agent, thread_id):
    if not isinstance(thread_id, str) or not thread_id:
        return None
    # Nested Agents borrow the active parent's bundle; do not provision stores
    # from a read-only review or observation path.
    from msgflux.nn.modules.agent.resources import _CURRENT_RESOURCES  # noqa: PLC0415

    active = _CURRENT_RESOURCES.get()
    if active is not None and active.resources.thread_id == thread_id:
        return active
    return getattr(agent, "_owned_threads", {}).get(thread_id)


def _definition_revision(definition) -> str:
    """Fingerprint implementation source and its logical public schema."""
    executor = definition.executor
    implementation = getattr(executor, "impl", executor)
    source = _implementation_source(implementation)
    schema = msgspec.json.encode(msgspec.to_builtins(definition.input_schema))
    explicit = definition.metadata.get("tool_revision")
    if explicit is not None and not isinstance(explicit, str):
        raise TypeError("metadata.tool_revision must be a string when supplied")
    payload = b"\0".join(
        (
            definition.name.encode("utf-8"),
            source,
            schema,
            (explicit or "").encode("utf-8"),
        )
    )
    return hashlib.sha256(payload).hexdigest()


def _implementation_source(implementation) -> bytes:
    if inspect.isfunction(implementation):
        targets = (implementation,)
    else:
        targets = tuple(
            item
            for cls in reversed(type(implementation).__mro__)
            if cls is not object
            for item in (
                cls,
                cls.__dict__.get("__call__"),
                cls.__dict__.get("acall"),
                cls.__dict__.get("prepare_workspace_change"),
            )
            if item is not None
        )
    parts = []
    for target in targets:
        try:
            parts.append(inspect.getsource(target).encode("utf-8"))
        except (OSError, TypeError):
            function = getattr(target, "__func__", target)
            code = getattr(function, "__code__", None)
            if code is not None:
                parts.append(marshal.dumps(code))
    if parts:
        return b"\0".join(parts)
    identity = f"{type(implementation).__module__}.{type(implementation).__qualname__}"
    return identity.encode("utf-8")


__all__ = ["get_workspace_approvals"]
