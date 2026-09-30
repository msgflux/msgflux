"""Shared execution contract for provider-neutral workspace mutation tools."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

from msgflux.runtime.workspace_api import AgentWorkspace, resolve_workspace
from msgflux.runtime.workspace_changes import PreparedFileChange

_PREPARED_CHANGE = ContextVar("msgflux_prepared_workspace_change", default=None)


class WorkspaceChangeTool(ABC):
    """Prepare without effects; execute the exact proposal approved by the host."""

    def __init__(self):
        self.tool_config = deepcopy(self.tool_config)

    @abstractmethod
    def prepare_workspace_change(
        self, arguments, workspace: AgentWorkspace
    ) -> PreparedFileChange:
        """Return a read-only preview using only authorized workspace operations."""

    def _apply(self, arguments, workspace):
        workspace = resolve_workspace(workspace)
        prepared = _PREPARED_CHANGE.get()
        change = (
            prepared[1]
            if prepared is not None and prepared[0] is self
            else self.prepare_workspace_change(arguments, workspace)
        )
        workspace._apply_prepared(change)
        return {"status": "completed"}


def workspace_change_tool(definition):
    impl = getattr(definition.executor, "impl", None)
    return impl if isinstance(impl, WorkspaceChangeTool) else None


@contextmanager
def workspace_change_execution(impl, change):
    token = _PREPARED_CHANGE.set((impl, change) if change is not None else None)
    try:
        yield
    finally:
        _PREPARED_CHANGE.reset(token)
