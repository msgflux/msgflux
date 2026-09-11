"""Base classes and runtime ownership for Agent extensions."""

from __future__ import annotations

import contextvars
import weakref
from collections.abc import Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from msgflux.nn.extensions.tool_library import ToolLibraryExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.module import Module
from msgflux.runtime.agent_run import get_agent_run

if TYPE_CHECKING:
    from msgflux.nn.modules.agent import Agent

__all__ = ["AgentExtension", "AgentExtensionHandle"]

_CURRENT_EXTENSION_SNAPSHOTS = contextvars.ContextVar(
    "msgflux_agent_extension_snapshots",
    default=None,
)
_CURRENT_EXTENSION_STATE = contextvars.ContextVar(
    "msgflux_agent_extension_state",
    default=None,
)


@contextmanager
def _extension_snapshot(agent: Agent, names: frozenset[str]):
    current = _CURRENT_EXTENSION_SNAPSHOTS.get() or {}
    agent_id = id(agent)
    if agent_id in current:
        yield current[agent_id]
        return
    updated = dict(current)
    updated[agent_id] = names
    token = _CURRENT_EXTENSION_SNAPSHOTS.set(updated)
    state_token = None
    if _CURRENT_EXTENSION_STATE.get() is None:
        state_token = _CURRENT_EXTENSION_STATE.set({})
    try:
        yield names
    finally:
        if state_token is not None:
            _CURRENT_EXTENSION_STATE.reset(state_token)
        _CURRENT_EXTENSION_SNAPSHOTS.reset(token)


def _get_extension_snapshot(agent: Agent) -> frozenset[str] | None:
    return (_CURRENT_EXTENSION_SNAPSHOTS.get() or {}).get(id(agent))


class AgentExtension(Module):
    """A named package of hooks and tools installed on an Agent.

    Extensions keep composition policy out of :class:`Agent`. They may expose
    lifecycle hooks, public tools, or both. Network work belongs in async hook
    handlers; registration itself is intentionally synchronous and local.
    """

    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("`name` must be a non-empty string")
        self.name = name
        self._agent_ref = None

    def state(self) -> dict[str, Any]:
        """Return mutable state isolated to this extension and active run."""
        if self._agent_ref is None or self._agent_ref() is None:
            raise RuntimeError("The extension is not registered on an Agent")
        states = _CURRENT_EXTENSION_STATE.get()
        if states is None:
            raise RuntimeError("Extension state is only available during an Agent run")
        key = (id(self._agent_ref()), self.name)
        return states.setdefault(key, {})

    def durable_state(self) -> dict[str, Any]:
        """Return this extension's checkpointed, execution-local state.

        Values must be serializable by the checkpoint store. New runs start
        empty; resuming a run restores its state before ``before_resume`` hooks.
        """
        agent = self.agent
        run = get_agent_run()
        if run is None or run.namespace not in {None, agent.get_module_name()}:
            raise RuntimeError("Durable extension state requires its active Agent run")
        return run.extension_state.setdefault(self.name, {})

    @property
    def agent(self) -> Agent:
        """Return the Agent that currently owns this extension."""
        agent = self._agent_ref() if self._agent_ref is not None else None
        if agent is None:
            raise RuntimeError("The extension is not registered on an Agent")
        return agent

    def _bind_agent(self, agent: Agent) -> None:
        self._agent_ref = weakref.ref(agent)

    def _unbind_agent(self) -> None:
        self._agent_ref = None

    def hooks(self) -> Iterable[Hook]:
        """Return lifecycle or module hooks contributed by this extension."""
        return ()

    def tools(self) -> Iterable[Any]:
        """Return tools contributed by this extension."""
        return ()

    def capabilities(self) -> Iterable[str]:
        """Return stable runtime capabilities enabled by this extension."""
        return ()

    def on_register(self, agent: Agent) -> None:
        """Run synchronous local setup after all contributions are installed."""

    def on_remove(self, agent: Agent) -> None:
        """Run synchronous cleanup after active runs release the extension."""

    async def aon_remove(self, agent: Agent) -> None:
        """Run optional asynchronous cleanup.

        The default delegates to :meth:`on_remove`. Override this method when
        cleanup needs async I/O and call :meth:`AgentExtensionHandle.aremove`.
        """
        self.on_remove(agent)


class AgentExtensionHandle:
    """Ownership handle returned by ``Agent.register_extension``."""

    def __init__(self, agent: Agent, name: str) -> None:
        self._agent_ref = weakref.ref(agent)
        self.name = name

    @property
    def active(self) -> bool:
        agent = self._agent_ref()
        return agent is not None and agent.has_extension(self.name)

    def remove(self) -> None:
        """Detach the extension for new runs and clean it up when safe."""
        agent = self._agent_ref()
        if agent is not None:
            agent.remove_extension(self.name)

    async def aremove(self) -> None:
        """Async counterpart that supports asynchronous extension cleanup."""
        agent = self._agent_ref()
        if agent is not None:
            await agent.aremove_extension(self.name)


class _ExtensionHook(Hook):
    """Keep a contributed hook available to runs that captured its extension."""

    def __init__(self, agent: Agent, extension_name: str, hook: Hook) -> None:
        super().__init__(
            event=hook.event,
            on=hook.on,
            target=hook.target,
            method=hook.method,
        )
        self._agent_ref = weakref.ref(agent)
        self.extension_name = extension_name
        self.hook = hook

    @property
    def processor_key(self):
        return self.hook.processor_key

    def __call__(self, module, args, kwargs, output=None):
        agent = self._agent_ref()
        if agent is None or not agent._extension_is_visible(self.extension_name):
            return output if self.on == "post" else None
        return self.hook(module, args, kwargs, output)

    async def acall(self, module, args, kwargs, output=None):
        agent = self._agent_ref()
        if agent is None or not agent._extension_is_visible(self.extension_name):
            return output if self.on == "post" else None
        return await self.hook.acall(module, args, kwargs, output)

    def handle(self, payload: Any) -> Any:
        agent = self._agent_ref()
        if agent is None or not agent._extension_is_visible(self.extension_name):
            return payload
        return self.hook.handle(payload)

    async def ahandle(self, payload: Any) -> Any:
        agent = self._agent_ref()
        if agent is None or not agent._extension_is_visible(self.extension_name):
            return payload
        return await self.hook.ahandle(payload)


class _AgentToolsExtension(ToolLibraryExtension):
    """Internal ownership bridge for tools contributed by an Agent extension."""

    def __init__(self, name: str, tools: Iterable[Any]) -> None:
        super().__init__(f"agent:{name}")
        self._tools = tuple(tools)

    def tools(self):
        return self._tools
