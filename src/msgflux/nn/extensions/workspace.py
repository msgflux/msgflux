"""Request-local workspace guidance; presentation only, never authorization."""

from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.extensions.prompt import _append_section
from msgflux.nn.hooks import Hook, ModelContext
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.permissions import PermissionSet
from msgflux.runtime.workspace import workspace_path
from msgflux.runtime.workspace_contracts import (
    WorkspacePromptInfo,
    WorkspaceWriteCapabilities,
)
from msgflux.utils.msgspec import msgspec_dumps


def _resources(filesystem, permissions):
    grouped = {}
    prefix = f"workspace:{filesystem.workspace_id}:"
    for grant in permissions.resources:
        if not grant.resource.startswith(prefix) or grant.action not in {
            "filesystem.read",
            "filesystem.write",
            "filesystem.delete",
            "filesystem.list",
            "filesystem.mkdir",
        }:
            continue
        path = grant.resource[len(prefix) :]
        try:
            if workspace_path(path) != path:
                continue
        except ValueError:
            continue
        grouped.setdefault(path, set()).add(grant.action.removeprefix("filesystem."))
    return [
        {"path": path, "actions": sorted(actions)}
        for path, actions in sorted(grouped.items())
    ]


def _json(value):
    # Keep host-supplied path/description text inside its JSON data boundary.
    return (
        msgspec_dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


class WorkspacePromptExtension(AgentExtension):
    """Append bounded guidance from the current live scope on each model request.

    No file discovery, remote calls, cached permissions or durable state. Backend
    descriptions are host-trusted data. An omitted grant is not a denied grant.
    """

    def __init__(self, *, max_resources: int = 20, max_chars: int = 6000):
        super().__init__("workspace_prompt")
        if type(max_resources) is not int or max_resources < 0:
            raise ValueError("max_resources must be a non-negative integer")
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        self.max_resources = max_resources
        self.max_chars = max_chars

    def hooks(self):
        return (Hook(event="transform_system_prompt", handler=self._add_workspace),)

    def _add_workspace(self, ctx: ModelContext) -> ModelContext:
        scope = get_execution_scope()
        environment = scope.environment
        if environment is None:
            return ctx
        environment.require_active()
        filesystem = environment.filesystem
        info, capabilities = filesystem.prompt_info, filesystem.write_capabilities
        if not isinstance(info, WorkspacePromptInfo):
            raise TypeError("Backend prompt_info must be WorkspacePromptInfo")
        if not isinstance(capabilities, WorkspaceWriteCapabilities):
            raise TypeError(
                "Backend write_capabilities must be WorkspaceWriteCapabilities"
            )
        permissions = scope.permissions or PermissionSet()
        resources = _resources(filesystem, permissions)
        executor = environment.process_executor
        payload = {
            "storage": info.storage,
            "guidance": info.guidance,
            "paths": (
                "Virtual absolute POSIX paths rooted at /. "
                "Relative paths use each tool's configured cwd."
            ),
            "write_guarantee": environment.write_guarantee,
            "write_capabilities": capabilities,
            "resources": resources[: self.max_resources],
            "omitted_resources": max(0, len(resources) - self.max_resources),
            "process_executor": "not configured"
            if executor is None
            else {
                "execution_granted": permissions.allows(("process.execute",)),
                "required_isolation": sorted(environment.requirements.mechanisms),
                "declared_isolation": sorted(executor.capabilities.mechanisms),
            },
            "notice": (
                "Resources are exact path grants, not subtree grants. "
                "Omitted resources are not necessarily denied. "
                "Runtime checks and configured approvals still apply. "
                "Declarations are not proof of isolation. "
                "Network access policy is not described here, "
                "including host/model traffic."
            ),
        }
        while True:
            section = f"<workspace_context>\n{_json(payload)}\n</workspace_context>"
            if len(section) <= self.max_chars:
                return _append_section(ctx, section)
            if not payload["resources"]:
                raise ValueError(
                    "Workspace description exceeds max_chars without resource entries"
                )
            payload["resources"].pop()
            payload["omitted_resources"] += 1
