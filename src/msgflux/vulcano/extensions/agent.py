from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Mapping, Protocol

from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_scope,
)
from msgflux.vulcano.events import EventDraft, EventType

__all__ = [
    "AgentAdapter",
    "AgentApi",
    "AgentRunResult",
    "MsgfluxAgentAdapter",
    "ToolLibraryApi",
    "ToolRegistration",
]


class _RegistrationTracker(Protocol):
    def __call__(self, registration: ToolRegistration) -> None: ...


class AgentAdapter(Protocol):
    """Adapter between Vulcano's stable API and one Agent implementation."""

    async def run(
        self,
        agent: object,
        message: object | None = None,
        **kwargs: object,
    ) -> object: ...

    def stream_events(
        self,
        agent: object,
        message: object | None = None,
        **kwargs: object,
    ) -> AsyncIterator[EventDraft]: ...


@dataclass(frozen=True)
class AgentRunResult:
    """Terminal result produced while forwarding Agent events to a client."""

    content: str
    status: str


class MsgfluxAgentAdapter:
    """Compatibility adapter for Agent.acall and ModelStreamResponse.

    This adapter is intentionally isolated from the extension API. When msgflux's
    native ``Agent.stream_events()`` lands, only this adapter needs to translate
    those events into Vulcano's domain event contract.
    """

    async def run(
        self,
        agent: object,
        message: object | None = None,
        **kwargs: object,
    ) -> object:
        acall = getattr(agent, "acall", None)
        if callable(acall):
            if message is None:
                result = acall(**kwargs)
            else:
                result = acall(message, **kwargs)
            if not inspect.isawaitable(result):
                raise TypeError("Agent.acall() must return an awaitable")
            return await result

        if not callable(agent):
            raise TypeError("The bound main agent must be callable or define acall()")
        if message is None:
            return await asyncio.to_thread(agent, **kwargs)
        return await asyncio.to_thread(agent, message, **kwargs)

    async def stream_events(
        self,
        agent: object,
        message: object | None = None,
        **kwargs: object,
    ) -> AsyncIterator[EventDraft]:
        yield EventDraft(EventType.ASSISTANT_STARTED)
        chunks: list[str] = []
        try:
            response = await self.run(agent, message, **kwargs)
            consumer = getattr(response, "consume", None)
            consumed = consumer() if callable(consumer) else None
            if consumed is not None and hasattr(consumed, "__aiter__"):
                async for chunk in consumed:
                    delta = _response_text(chunk)
                    chunks.append(delta)
                    yield EventDraft(
                        EventType.ASSISTANT_DELTA,
                        {"delta": delta},
                    )
                content = _response_text(getattr(response, "data", None))
                if not content:
                    content = "".join(chunks)
            else:
                content = _response_text(response)
                if content:
                    chunks.append(content)
                    yield EventDraft(
                        EventType.ASSISTANT_DELTA,
                        {"delta": content},
                    )
        except Exception:
            yield EventDraft(
                EventType.ASSISTANT_COMPLETED,
                {"content": "".join(chunks), "status": "failed"},
            )
            raise

        yield EventDraft(
            EventType.ASSISTANT_COMPLETED,
            {"content": content, "status": "completed"},
        )


class _AgentBinding:
    def __init__(
        self,
        agent: object | None = None,
        adapter: AgentAdapter | None = None,
    ) -> None:
        self._agent = agent
        self._adapter: AgentAdapter = adapter or MsgfluxAgentAdapter()
        if agent is not None:
            self._validate_agent(agent)

    @property
    def is_bound(self) -> bool:
        return self._agent is not None

    def bind(
        self,
        agent: object,
        adapter: AgentAdapter | None = None,
    ) -> None:
        if self._agent is not None:
            raise RuntimeError("Vulcano already has a bound main Agent")
        self._validate_agent(agent)
        self._agent = agent
        if adapter is not None:
            self._adapter = adapter

    def agent(self) -> object:
        if self._agent is None:
            raise RuntimeError(
                "No main Agent is bound to Vulcano. Pass agent=... to "
                "VulcanoRuntime or call bind_agent() before start()."
            )
        return self._agent

    def library(self) -> object:
        library = getattr(self.agent(), "tool_library", None)
        required = ("add", "remove", "get_tool_names")
        if library is None or any(
            not callable(getattr(library, method, None)) for method in required
        ):
            raise TypeError(
                "The bound main Agent must expose a ToolLibrary-compatible tool_library"
            )
        return library

    @property
    def adapter(self) -> AgentAdapter:
        return self._adapter

    @staticmethod
    def _validate_agent(agent: object) -> None:
        if not callable(agent) and not callable(getattr(agent, "acall", None)):
            raise TypeError("The main Agent must be callable or define acall()")
        library = getattr(agent, "tool_library", None)
        required = ("add", "remove", "get_tool_names")
        if library is None or any(
            not callable(getattr(library, method, None)) for method in required
        ):
            raise TypeError(
                "The main Agent must expose a ToolLibrary-compatible tool_library"
            )


class ToolRegistration:
    """Ownership handle for one tool added to the main Agent library."""

    def __init__(self, library: object, name: str) -> None:
        self._library = library
        self.name = name
        self._active = True
        self._cleanups: list[Callable[[], None]] = []

    @property
    def active(self) -> bool:
        return self._active

    def remove(self) -> None:
        if not self._active:
            return
        cleanup_error: Exception | None = None
        for cleanup in reversed(self._cleanups):
            try:
                cleanup()
            except Exception as error:
                cleanup_error = cleanup_error or error
        self._cleanups.clear()
        names = _all_tool_names(self._library)
        if self.name in names:
            self._library.remove(self.name)
        self._active = False
        if cleanup_error is not None:
            raise cleanup_error

    def _add_cleanup(self, cleanup: Callable[[], None]) -> None:
        if not self._active:
            cleanup()
            return
        self._cleanups.append(cleanup)

    def __enter__(self) -> ToolRegistration:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.remove()


class ToolLibraryApi:
    """Owner-aware high-level facade over the main Agent ToolLibrary."""

    def __init__(
        self,
        binding: _AgentBinding,
        *,
        assert_active: Callable[[], None],
        track: _RegistrationTracker,
    ) -> None:
        self._binding = binding
        self._assert_active = assert_active
        self._track = track

    @property
    def names(self) -> tuple[str, ...]:
        self._assert_active()
        return tuple(self._binding.library().get_tool_names())

    def register(self, tool: Callable[..., object]) -> ToolRegistration:
        self._assert_active()
        library = self._binding.library()
        before = set(library.get_tool_names())
        registered_name = library.add(tool)
        after = set(library.get_tool_names())
        name = _registered_tool_name(registered_name, before, after)
        registration = ToolRegistration(library, name)
        self._track(registration)
        return registration

    def tool(self, tool: Callable[..., object]) -> Callable[..., object]:
        """Register a callable while preserving it for decorator use."""
        self.register(tool)
        return tool

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, object],
        **kwargs: object,
    ) -> object:
        """Execute a tool through ToolLibrary's public execution pipeline."""
        self._assert_active()
        library = self._binding.library()
        aexecute = getattr(library, "aexecute", None)
        if callable(aexecute):
            result = aexecute(name, arguments, **kwargs)
            if not inspect.isawaitable(result):
                raise TypeError("ToolLibrary.aexecute() must return an awaitable")
            return await result

        execute = getattr(library, "execute", None)
        if callable(execute):
            return await asyncio.to_thread(execute, name, arguments, **kwargs)
        raise RuntimeError(
            "Tool execution requires the ToolLibrary.execute/aexecute API from "
            "the runtime-stack release"
        )

    def __contains__(self, name: str) -> bool:
        return name in self.names


class AgentApi:
    """High-level facade over Vulcano's main Agent and its ToolLibrary."""

    def __init__(
        self,
        binding: _AgentBinding,
        *,
        assert_active: Callable[[], None],
        track: _RegistrationTracker,
    ) -> None:
        self._binding = binding
        self._assert_active = assert_active
        self.tools = ToolLibraryApi(
            binding,
            assert_active=assert_active,
            track=track,
        )

    @property
    def is_bound(self) -> bool:
        self._assert_active()
        return self._binding.is_bound

    @property
    def name(self) -> str | None:
        self._assert_active()
        if not self._binding.is_bound:
            return None
        agent = self._binding.agent()
        name = getattr(agent, "name", None)
        if isinstance(name, str):
            return name
        get_name = getattr(agent, "get_module_name", None)
        return str(get_name()) if callable(get_name) else type(agent).__name__

    async def run(
        self,
        message: object | None = None,
        *,
        scope: ExecutionScope | None = None,
        **kwargs: object,
    ) -> object:
        """Run the main Agent without projecting its response to a client."""
        self._assert_active()
        resolved_scope = _resolve_scope(scope)
        with execution_context(scope=resolved_scope):
            return await self._binding.adapter.run(
                self._binding.agent(),
                message,
                **kwargs,
            )

    async def stream_events(
        self,
        message: object | None = None,
        *,
        scope: ExecutionScope | None = None,
        **kwargs: object,
    ) -> AsyncIterator[EventDraft]:
        """Yield stable Vulcano events for one main-Agent execution."""
        self._assert_active()
        resolved_scope = _resolve_scope(scope)
        with execution_context(scope=resolved_scope):
            stream = self._binding.adapter.stream_events(
                self._binding.agent(),
                message,
                **kwargs,
            )
            async for event in stream:
                self._assert_active()
                if not isinstance(event, EventDraft):
                    raise TypeError(
                        "AgentAdapter.stream_events() must yield EventDraft"
                    )
                yield event

    async def respond(
        self,
        message: object | None = None,
        *,
        emit: Callable[[EventDraft], Awaitable[None]],
        scope: ExecutionScope | None = None,
        **kwargs: object,
    ) -> AgentRunResult:
        """Run the Agent and forward its event stream to a runtime emitter."""
        content = ""
        status = "completed"
        async for event in self.stream_events(message, scope=scope, **kwargs):
            await emit(event)
            if event.type == EventType.ASSISTANT_DELTA:
                content += str(event.payload.get("delta", ""))
            elif event.type == EventType.ASSISTANT_COMPLETED:
                content = str(event.payload.get("content", content))
                status = str(event.payload.get("status", status))
        return AgentRunResult(content=content, status=status)


def _resolve_scope(scope: ExecutionScope | None) -> ExecutionScope:
    if scope is not None and not isinstance(scope, ExecutionScope):
        raise TypeError("scope must be an ExecutionScope or None")
    return scope or get_execution_scope()


def _registered_tool_name(
    result: object,
    before: set[str],
    after: set[str],
) -> str:
    if isinstance(result, str) and result:
        return result
    added = after - before
    if len(added) == 1:
        return added.pop()
    raise RuntimeError("ToolLibrary.add() did not identify exactly one registered tool")


def _all_tool_names(library: object) -> set[str]:
    get_tools = getattr(library, "get_tools", None)
    if callable(get_tools):
        return {str(name) for name, _tool in get_tools()}
    return {str(name) for name in library.get_tool_names()}


def _response_text(response: object) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, bytes):
        return response.decode(errors="replace")
    if isinstance(response, Mapping):
        for key in ("response", "answer", "text"):
            value = response.get(key)
            if isinstance(value, str):
                return value
    return str(response)
