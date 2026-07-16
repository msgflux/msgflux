from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import AsyncIterator, Mapping, Protocol, Sequence, cast

from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_scope,
    new_run_id,
    new_thread_id,
)
from msgflux.vulcano.actions import RuntimeAction, StopRuntime, SubmitInput
from msgflux.vulcano.commands import (
    CommandContext,
    CommandOptions,
    CommandRegistry,
    CommandResult,
)
from msgflux.vulcano.events import (
    DomainEvent,
    EventDraft,
    EventStream,
    EventSubscription,
    EventType,
)
from msgflux.vulcano.extensions import (
    AgentAdapter,
    ExtensionControl,
    ExtensionManager,
    ExtensionSettings,
)

__all__ = ["MockResponder", "Responder", "RuntimeProtocol", "VulcanoRuntime"]


class Responder(Protocol):
    """Streaming response source used by a Vulcano runtime."""

    def stream(self, prompt: str) -> AsyncIterator[str]: ...


class RuntimeProtocol(Protocol):
    """Client-facing runtime contract implemented independently of Textual."""

    commands: CommandRegistry

    def subscribe(self) -> EventSubscription: ...

    async def start(self) -> None: ...

    async def dispatch(self, action: RuntimeAction) -> None: ...


class MockResponder:
    """Deterministic streaming responder used until the Agent adapter lands."""

    def __init__(self, delay: float = 0.01) -> None:
        if delay < 0:
            raise ValueError("Mock stream delay cannot be negative")
        self.delay = delay

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        response = f"Mock runtime received: {prompt}"
        for chunk in re.findall(r"\S+\s*", response):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


class VulcanoRuntime:
    """Headless action/event runtime with a replaceable response source."""

    def __init__(
        self,
        *,
        responder: Responder | None = None,
        agent: object | None = None,
        agent_adapter: AgentAdapter | None = None,
        scope: ExecutionScope | None = None,
        stream_delay: float = 0.01,
        services: Mapping[str, object] | None = None,
        cwd: str | Path | None = None,
        extension_paths: Sequence[str | Path] = (),
        extensions_enabled: bool = True,
        discover_extensions: bool = True,
        trust_project_extensions: bool = False,
        extension_user_directory: str | Path | None = None,
    ) -> None:
        if responder is not None and agent is not None:
            raise ValueError("Configure either responder or agent, not both")
        if scope is not None and not isinstance(scope, ExecutionScope):
            raise TypeError("scope must be an ExecutionScope or None")
        requested_scope = scope or ExecutionScope()
        thread_id = requested_scope.thread_id or new_thread_id()
        self._thread_scope = ExecutionScope(
            thread_id=thread_id,
            namespace=requested_scope.namespace,
            abort_signal=requested_scope.abort_signal,
        )
        self._pending_scope = (
            ExecutionScope(
                thread_id=thread_id,
                namespace=requested_scope.namespace,
                run_id=requested_scope.run_id,
                parent_run_id=requested_scope.parent_run_id,
                root_run_id=requested_scope.root_run_id or requested_scope.run_id,
                abort_signal=requested_scope.abort_signal,
            )
            if requested_scope.run_id is not None
            else None
        )
        self.commands = CommandRegistry()
        self._events = EventStream()
        self._history: list[DomainEvent] = []
        self._sequence = 0
        self._started = False
        self._stopped = False
        self._lifecycle_lock = asyncio.Lock()
        self._dispatch_lock = asyncio.Lock()
        self._custom_responder = responder is not None
        self._responder = responder or MockResponder(stream_delay)
        resolved_cwd = Path(cwd or Path.cwd()).expanduser().resolve()
        resolved_extension_paths = tuple(
            (
                path_value if path_value.is_absolute() else resolved_cwd / path_value
            ).resolve()
            for path in extension_paths
            for path_value in (Path(path).expanduser(),)
        )
        user_directory = (
            Path(extension_user_directory).expanduser().resolve()
            if extension_user_directory is not None
            else None
        )
        extension_settings = ExtensionSettings(
            cwd=resolved_cwd,
            explicit_paths=resolved_extension_paths,
            enabled=extensions_enabled,
            auto_discover=discover_extensions,
            trust_project=trust_project_extensions,
            user_directory=user_directory,
        )
        self.extensions = ExtensionManager(
            self.commands,
            extension_settings,
            services=services,
            agent=agent,
            agent_adapter=agent_adapter,
        )
        self._install_builtin_commands()

    @property
    def history(self) -> tuple[DomainEvent, ...]:
        return tuple(self._history)

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped

    def subscribe(self) -> EventSubscription:
        return self._events.subscribe()

    def bind_agent(
        self,
        agent: object,
        *,
        adapter: AgentAdapter | None = None,
    ) -> None:
        """Bind the main Agent before extensions are loaded."""
        if self._started:
            raise RuntimeError("The main Agent must be bound before runtime.start()")
        if self._stopped:
            raise RuntimeError("Vulcano runtime is stopped")
        if self._custom_responder:
            raise RuntimeError("Cannot bind an Agent when a custom responder is set")
        self.extensions.bind_agent(agent, adapter)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            if self._stopped:
                raise RuntimeError("Vulcano runtime cannot be restarted")
            extension_report = await self.extensions.load_all()
            for info in extension_report.loaded:
                await self._emit(EventType.EXTENSION_LOADED, info.to_dict())
            for info in extension_report.failed:
                await self._emit(EventType.EXTENSION_FAILED, info.to_dict())
            self._started = True
            await self._emit(
                EventType.RUNTIME_STARTED,
                {
                    "runtime": (
                        "agent" if self.extensions.api.agent.is_bound else "mock"
                    ),
                    "agent": self.extensions.api.agent.name,
                    "commands": len(self.commands),
                    "extensions": len(extension_report.loaded),
                },
            )

    async def stop(
        self,
        *,
        reason: str = "requested",
        correlation_id: str | None = None,
    ) -> None:
        async with self._lifecycle_lock:
            if self._stopped:
                return
            self._stopped = True
            await self._emit(
                EventType.RUNTIME_STOPPED,
                {"reason": reason},
                correlation_id=correlation_id,
            )
            await self.extensions.unload_all()
            await self._events.close()

    async def dispatch(self, action: RuntimeAction) -> None:
        if self._stopped:
            raise RuntimeError("Vulcano runtime is stopped")
        if not self._started:
            await self.start()
        async with self._dispatch_lock:
            if self._stopped:
                raise RuntimeError("Vulcano runtime is stopped")
            if isinstance(action, SubmitInput):
                scope = self._next_submission_scope()
                with execution_context(scope=scope):
                    await self._handle_input(action)
                return
            if isinstance(action, StopRuntime):
                await self.stop(
                    reason=action.reason,
                    correlation_id=action.correlation_id,
                )
                return
            raise TypeError(f"Unsupported Vulcano action: {type(action)!r}")

    async def _handle_input(self, action: SubmitInput) -> None:
        text = action.text.strip()
        if not text:
            return
        if text.startswith("/"):
            await self._execute_command(text, action.correlation_id)
            return

        await self._emit(
            EventType.MESSAGE_USER,
            {"content": text},
            correlation_id=action.correlation_id,
        )
        if self.extensions.api.agent.is_bound:
            try:
                await self.extensions.api.agent.respond(
                    text,
                    emit=self._draft_emitter(action.correlation_id),
                )
            except Exception as error:
                await self._emit(
                    EventType.RUNTIME_ERROR,
                    {"message": str(error)},
                    correlation_id=action.correlation_id,
                )
            return

        await self._emit(
            EventType.ASSISTANT_STARTED,
            {},
            correlation_id=action.correlation_id,
        )

        chunks: list[str] = []
        try:
            async for delta in self._responder.stream(text):
                chunks.append(delta)
                await self._emit(
                    EventType.ASSISTANT_DELTA,
                    {"delta": delta},
                    correlation_id=action.correlation_id,
                )
        except Exception as error:
            await self._emit(
                EventType.RUNTIME_ERROR,
                {"message": str(error)},
                correlation_id=action.correlation_id,
            )
            await self._emit(
                EventType.ASSISTANT_COMPLETED,
                {"content": "".join(chunks), "status": "failed"},
                correlation_id=action.correlation_id,
            )
            return

        await self._emit(
            EventType.ASSISTANT_COMPLETED,
            {"content": "".join(chunks), "status": "completed"},
            correlation_id=action.correlation_id,
        )

    async def _execute_command(self, text: str, correlation_id: str) -> None:
        try:
            invocation = self.commands.parse(text)
            command = self.commands.resolve(invocation.name)
        except (ValueError, LookupError) as error:
            await self._emit(
                EventType.COMMAND_ERROR,
                {"message": str(error), "raw": text},
                correlation_id=correlation_id,
            )
            return

        await self._emit(
            EventType.COMMAND_STARTED,
            {
                "name": command.name,
                "arguments": list(invocation.arguments),
                "raw": invocation.raw,
            },
            correlation_id=correlation_id,
        )
        context = CommandContext(
            commands=self.commands,
            api=self.extensions.api,
            scope=get_execution_scope(),
            correlation_id=correlation_id,
            _event_emitter=self._draft_emitter(correlation_id),
        )
        try:
            result = await self.commands.invoke(invocation, context)
        except Exception as error:
            await self._emit(
                EventType.COMMAND_ERROR,
                {"message": str(error), "name": command.name},
                correlation_id=correlation_id,
            )
            await self._emit(
                EventType.COMMAND_COMPLETED,
                {"name": command.name, "status": "failed"},
                correlation_id=correlation_id,
            )
            return

        for event in result.events:
            await self._emit(
                event.type,
                event.payload,
                correlation_id=correlation_id,
            )
        await self._emit(
            EventType.COMMAND_COMPLETED,
            {"name": command.name, "status": "completed"},
            correlation_id=correlation_id,
        )
        if result.stop_runtime:
            await self.stop(
                reason=f"command:/{command.name}",
                correlation_id=correlation_id,
            )

    def _next_submission_scope(self) -> ExecutionScope:
        if self._pending_scope is not None:
            scope = self._pending_scope
            self._pending_scope = None
            return scope
        run_id = new_run_id()
        return self._thread_scope.with_overrides(
            run_id=run_id,
            root_run_id=run_id,
        )

    def _draft_emitter(self, correlation_id: str | None):
        async def emit(event: EventDraft) -> None:
            if not isinstance(event, EventDraft):
                raise TypeError("Command and Agent emitters require EventDraft")
            await self._emit(
                event.type,
                event.payload,
                correlation_id=correlation_id,
            )

        return emit

    async def _emit(
        self,
        event_type: str,
        payload: Mapping[str, object] | None = None,
        *,
        correlation_id: str | None = None,
    ) -> DomainEvent:
        self._sequence += 1
        event = DomainEvent(
            type=event_type,
            sequence=self._sequence,
            payload=dict(payload or {}),
            correlation_id=correlation_id,
        )
        self._history.append(event)
        await self._events.publish(event)
        diagnostics = await self.extensions.notify(event)
        for diagnostic in diagnostics:
            await self._emit_extension_failure(
                diagnostic.to_dict(),
                correlation_id=correlation_id,
            )
        return event

    async def _emit_extension_failure(
        self,
        payload: Mapping[str, object],
        *,
        correlation_id: str | None,
    ) -> None:
        self._sequence += 1
        event = DomainEvent(
            type=EventType.EXTENSION_FAILED,
            sequence=self._sequence,
            payload=dict(payload),
            correlation_id=correlation_id,
        )
        self._history.append(event)
        await self._events.publish(event)

    def _install_builtin_commands(self) -> None:
        builtins = (
            (
                "help",
                CommandOptions(
                    aliases=("commands",),
                    description="List runtime commands.",
                    usage="/help",
                    handler=_help_command,
                    category="runtime",
                ),
            ),
            (
                "echo",
                CommandOptions(
                    description="Echo arguments from the runtime.",
                    usage="/echo <text>",
                    handler=_echo_command,
                    category="runtime",
                ),
            ),
            (
                "clear",
                CommandOptions(
                    description="Clear the client transcript.",
                    usage="/clear",
                    handler=_clear_command,
                    category="client",
                ),
            ),
            (
                "about",
                CommandOptions(
                    description="Describe the active runtime.",
                    usage="/about",
                    handler=_about_command,
                    category="runtime",
                ),
            ),
            (
                "extensions",
                CommandOptions(
                    description="List loaded extensions and diagnostics.",
                    usage="/extensions",
                    handler=_extensions_command,
                    category="runtime",
                ),
            ),
            (
                "reload",
                CommandOptions(
                    description="Reload extensions from their configured sources.",
                    usage="/reload",
                    handler=_reload_command,
                    category="runtime",
                ),
            ),
            (
                "quit",
                CommandOptions(
                    aliases=("exit",),
                    description="Stop the runtime and close clients.",
                    usage="/quit",
                    handler=_quit_command,
                    category="runtime",
                ),
            ),
        )
        for name, options in builtins:
            self.extensions.api.register_command(name, options)


def _help_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments
    lines = ["## Commands", ""]
    for command in context.commands:
        aliases = ""
        if command.aliases:
            aliases = f" (aliases: {', '.join('/' + a for a in command.aliases)})"
        lines.append(
            f"- **{command.usage or '/' + command.name}** — "
            f"{command.description}{aliases}"
        )
    return CommandResult(
        events=(EventDraft(EventType.COMMAND_OUTPUT, {"text": "\n".join(lines)}),)
    )


def _echo_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del context
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": arguments},
            ),
        )
    )


def _clear_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments, context
    return CommandResult(
        events=(
            EventDraft(
                EventType.CLIENT_ACTION,
                {"action": "transcript.clear"},
            ),
        )
    )


def _about_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments
    if context.api.agent.is_bound:
        runtime_description = (
            f"the main Agent `{context.api.agent.name}` and its ToolLibrary"
        )
    else:
        runtime_description = "the mock response adapter"
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {
                    "text": (
                        f"**Vulcano** is running with {runtime_description}. "
                        "The Textual client has no agent logic."
                    )
                },
            ),
        )
    )


def _quit_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments, context
    return CommandResult(stop_runtime=True)


def _extension_control(context: CommandContext) -> ExtensionControl:
    return cast(ExtensionControl, context.api.services["extensions"])


def _extensions_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments
    control = _extension_control(context)
    if not control.enabled:
        text = "Extensions are disabled for this runtime."
    elif not control.records:
        text = "No extensions are currently loaded."
    else:
        lines = ["## Extensions", ""]
        for info in control.records:
            detail = f"{info.source.kind}, generation {info.generation}"
            if info.error:
                detail = f"{detail}: {info.error}"
            lines.append(f"- **{info.name}** — {info.state} ({detail})")
        if control.diagnostics:
            lines.extend(("", "### Diagnostics", ""))
            lines.extend(
                f"- **{diagnostic.extension}/{diagnostic.phase}** — "
                f"{diagnostic.message}"
                for diagnostic in control.diagnostics
            )
        text = "\n".join(lines)
    return CommandResult(events=(EventDraft(EventType.COMMAND_OUTPUT, {"text": text}),))


async def _reload_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments
    control = _extension_control(context)
    if not control.enabled:
        return CommandResult(
            events=(
                EventDraft(
                    EventType.COMMAND_OUTPUT,
                    {"text": "Extensions are disabled for this runtime."},
                ),
            )
        )

    report = await control.reload()
    events: list[EventDraft] = []
    events.extend(
        EventDraft(EventType.EXTENSION_UNLOADED, info.to_dict())
        for info in report.unloaded
    )
    events.extend(
        EventDraft(EventType.EXTENSION_LOADED, info.to_dict()) for info in report.loaded
    )
    events.extend(
        EventDraft(EventType.EXTENSION_FAILED, info.to_dict()) for info in report.failed
    )
    events.extend(
        EventDraft(EventType.EXTENSION_FAILED, diagnostic.to_dict())
        for diagnostic in report.diagnostics
        if diagnostic.phase == "unload"
    )
    events.append(
        EventDraft(
            EventType.COMMAND_OUTPUT,
            {
                "text": (
                    f"Reloaded extensions: {len(report.loaded)} loaded, "
                    f"{len(report.failed)} failed."
                )
            },
        )
    )
    return CommandResult(events=tuple(events))
