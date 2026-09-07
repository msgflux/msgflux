"""Durable execution-local state shared by Agent extensions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


_CURRENT_AGENT_RUN: ContextVar[AgentRun | None] = ContextVar(
    "msgflux_agent_run", default=None
)


@dataclass
class AgentRun:
    """Mutable in-memory view of one durable Agent run.

    The run identity remains the caller's execution scope. Branch and budget
    changes are persisted as extension-neutral metadata and do not create a new
    thread, run, or budget.
    """

    namespace: str | None = None
    thread_id: str | None = None
    run_id: str | None = None
    parent_run_id: str | None = None
    root_run_id: str | None = None
    revision: int = 0
    branch_id: str = "root"
    head_item_id: str | None = None
    budgets: dict[str, Any] = field(default_factory=dict)
    extension_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("`revision` must be non-negative")
        if not self.branch_id or not isinstance(self.branch_id, str):
            raise ValueError("`branch_id` must be a non-empty string")
        self.budgets = deepcopy(dict(self.budgets))
        self.extension_state = deepcopy(dict(self.extension_state))

    @property
    def extensions(self) -> dict[str, Any]:
        return self.extension_state

    def get_extension(self, name: str, default: Any = None) -> Any:
        return self.extension_state.get(name, default)

    def set_extension(self, name: str, value: Any) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Extension names must be non-empty strings")
        self.extension_state[name] = deepcopy(value)

    def durable_state(self) -> dict[str, Any]:
        """Return a JSON/msgpack-friendly snapshot for checkpoint state."""
        return {
            "schema_version": 1,
            "namespace": self.namespace,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "root_run_id": self.root_run_id,
            "revision": self.revision,
            "branch_id": self.branch_id,
            "head_item_id": self.head_item_id,
            "budgets": deepcopy(self.budgets),
            "extensions": deepcopy(self.extension_state),
        }

    @classmethod
    def from_durable_state(
        cls,
        state: Mapping[str, Any] | None,
        *,
        namespace: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentRun:
        if not isinstance(state, Mapping):
            return cls(namespace=namespace, thread_id=thread_id, run_id=run_id)
        return cls(
            namespace=state.get("namespace", namespace),
            thread_id=state.get("thread_id", thread_id),
            run_id=state.get("run_id", run_id),
            parent_run_id=state.get("parent_run_id"),
            root_run_id=state.get("root_run_id"),
            revision=int(state.get("revision", 0)),
            branch_id=state.get("branch_id", "root"),
            head_item_id=state.get("head_item_id"),
            budgets=state.get("budgets", {}),
            extension_state=state.get("extensions", {}),
        )



def get_agent_run() -> AgentRun | None:
    return _CURRENT_AGENT_RUN.get()


def get_current_agent_run() -> AgentRun | None:
    """Compatibility alias for extensions that need the active run."""
    return get_agent_run()


@contextmanager
def agent_run_context(run: AgentRun) -> Iterator[AgentRun]:
    if not isinstance(run, AgentRun):
        raise TypeError("`run` must be an AgentRun")
    token = _CURRENT_AGENT_RUN.set(run)
    try:
        yield run
    finally:
        _CURRENT_AGENT_RUN.reset(token)


__all__ = ["AgentRun", "agent_run_context", "get_agent_run", "get_current_agent_run"]
