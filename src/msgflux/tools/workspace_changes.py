"""Shared execution contract for provider-neutral workspace mutation tools."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.workspace import workspace_path
from msgflux.runtime.workspace_changes import PreparedFileChange, WorkspaceEditor

_PREPARED_CHANGE = ContextVar("msgflux_prepared_workspace_change", default=None)


class WorkspaceChangeTool(ABC):
    """Prepare without effects; execute the exact proposal approved by the host."""

    def __init__(self, *, cwd: str = "/"):
        self.cwd = workspace_path(cwd)
        self.tool_config = deepcopy(self.tool_config)

    @abstractmethod
    def prepare_workspace_change(self, arguments, filesystem) -> PreparedFileChange:
        """Return a read-only preview using only authorized workspace operations."""

    @staticmethod
    def _editor(filesystem, *, require_approval=True) -> WorkspaceEditor:
        environment = get_execution_scope().environment
        if environment is None or environment.filesystem is not filesystem:
            raise PermissionError("Filesystem is not bound to the live environment")
        return environment.workspace_editor(require_approval=require_approval)

    def _apply(self, arguments, filesystem):
        prepared = _PREPARED_CHANGE.get()
        change = (
            prepared[1]
            if prepared is not None and prepared[0] is self
            else self.prepare_workspace_change(arguments, filesystem)
        )
        # The Agent guard already consumed the decision. Outside a protected
        # Agent invocation, confirmation is the host's policy, just as for Bash.
        self._editor(filesystem, require_approval=False).apply(change)
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
