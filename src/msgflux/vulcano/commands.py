from __future__ import annotations

import inspect
import re
import shlex
import weakref
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Iterator, Mapping, Protocol

from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    new_run_id,
    new_thread_id,
)
from msgflux.vulcano.events import EventDraft, custom_message_type

if TYPE_CHECKING:
    from msgflux.vulcano.extensions.agent import AgentApi
    from msgflux.vulcano.ui import ExtensionUiApi

__all__ = [
    "CommandContext",
    "CommandEventEmitter",
    "CommandHandler",
    "CommandInvocation",
    "CommandOptions",
    "CommandParseError",
    "CommandRegistration",
    "CommandRegistry",
    "CommandResult",
    "CommandSpec",
    "UnknownCommandError",
]


class CommandParseError(ValueError):
    """Raised when a slash command cannot be parsed."""


class UnknownCommandError(LookupError):
    """Raised when no slash command owns a requested name."""


@dataclass(frozen=True)
class CommandInvocation:
    name: str
    arguments: tuple[str, ...]
    raw: str


@dataclass(frozen=True)
class CommandResult:
    events: tuple[EventDraft, ...] = ()
    stop_runtime: bool = False


class CommandApi(Protocol):
    @property
    def agent(self) -> AgentApi: ...

    @property
    def ui(self) -> ExtensionUiApi: ...

    @property
    def services(self) -> Mapping[str, object]: ...


CommandEventEmitter = Callable[[EventDraft], Awaitable[None]]


@dataclass(frozen=True)
class CommandContext:
    commands: CommandRegistry
    api: CommandApi
    scope: ExecutionScope = field(default_factory=ExecutionScope)
    correlation_id: str | None = None
    _event_emitter: CommandEventEmitter | None = field(default=None, repr=False)

    @property
    def services(self) -> Mapping[str, object]:
        return self.api.services

    @property
    def ui(self) -> ExtensionUiApi:
        return self.api.ui

    async def emit(self, event: EventDraft) -> None:
        """Publish an event immediately during a slash-command flow."""
        if self._event_emitter is None:
            raise RuntimeError("This command context is not bound to a runtime emitter")
        await self._event_emitter(event)

    async def send_message(
        self,
        custom_type: str,
        content: object,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Publish an extension-owned message for a registered UI renderer."""
        await self.emit(
            EventDraft(
                custom_message_type(custom_type),
                {
                    "custom_type": custom_type,
                    "content": content,
                    "details": dict(details or {}),
                },
            )
        )

    def child_scope(
        self,
        *,
        namespace: str | None = None,
        run_id: str | None = None,
    ) -> ExecutionScope:
        """Derive a child execution while preserving durable lineage."""
        child_run_id = run_id or new_run_id()
        parent_run_id = self.scope.run_id
        return ExecutionScope(
            thread_id=self.scope.thread_id or new_thread_id(),
            namespace=namespace or self.scope.namespace,
            run_id=child_run_id,
            parent_run_id=parent_run_id,
            root_run_id=self.scope.root_run_id or parent_run_id or child_run_id,
            abort_signal=self.scope.abort_signal,
        )

    def use_scope(
        self,
        scope: ExecutionScope | None = None,
    ) -> AbstractContextManager[ExecutionScope]:
        """Activate this command's scope, or an explicitly derived scope."""
        active_scope = scope if scope is not None else self.scope
        if not isinstance(active_scope, ExecutionScope):
            raise TypeError("scope must be an ExecutionScope")
        return execution_context(scope=active_scope)

    def with_scope(self, scope: ExecutionScope) -> CommandContext:
        """Copy the context with a different durable execution scope."""
        if not isinstance(scope, ExecutionScope):
            raise TypeError("scope must be an ExecutionScope")
        return CommandContext(
            commands=self.commands,
            api=self.api,
            scope=scope,
            correlation_id=self.correlation_id,
            _event_emitter=self._event_emitter,
        )

    def with_api(self, api: CommandApi) -> CommandContext:
        """Copy runtime capabilities while rebinding extension ownership."""
        return CommandContext(
            commands=self.commands,
            api=api,
            scope=self.scope,
            correlation_id=self.correlation_id,
            _event_emitter=self._event_emitter,
        )


CommandHandler = Callable[
    [str, CommandContext],
    None | CommandResult | Awaitable[None | CommandResult],
]
ArgumentCompletionProvider = Callable[
    [str],
    object | Awaitable[object],
]
RuntimeCommandHandler = Callable[
    [CommandContext, CommandInvocation],
    CommandResult | Awaitable[CommandResult],
]


@dataclass(frozen=True)
class CommandOptions:
    """Pi-shaped options passed to ExtensionApi.register_command()."""

    handler: CommandHandler
    description: str = ""
    get_argument_completions: ArgumentCompletionProvider | None = None
    usage: str = ""
    aliases: tuple[str, ...] = ()
    category: str = "extension"


_VALID_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    handler: RuntimeCommandHandler
    get_argument_completions: ArgumentCompletionProvider | None = None
    usage: str = ""
    aliases: tuple[str, ...] = ()
    category: str = "general"
    owner: str = "registry"

    def __post_init__(self) -> None:
        names = (self.name, *self.aliases)
        invalid = [name for name in names if not _VALID_NAME.fullmatch(name)]
        if invalid:
            raise ValueError(
                "Command names must be lowercase identifiers: " + ", ".join(invalid)
            )
        if len(set(names)) != len(names):
            raise ValueError(f"Command {self.name!r} repeats a name or alias")


class CommandRegistration:
    """Handle that removes exactly the command registered by this call."""

    def __init__(self, registry: CommandRegistry, command: CommandSpec) -> None:
        self._registry_ref = weakref.ref(registry)
        self._command = command

    def remove(self) -> None:
        registry = self._registry_ref()
        if registry is not None:
            registry._remove_if_current(self._command)

    def __enter__(self) -> CommandRegistration:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.remove()


class CommandRegistry:
    """Runtime-owned registry for built-in and extension slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, CommandSpec] = {}
        self._names: dict[str, str] = {}

    def _register(self, command: CommandSpec) -> CommandRegistration:
        names = (command.name, *command.aliases)
        collisions = [name for name in names if name in self._names]
        if collisions:
            raise ValueError(
                "Command names are already registered: " + ", ".join(collisions)
            )

        self._commands[command.name] = command
        for name in names:
            self._names[name] = command.name
        return CommandRegistration(self, command)

    def parse(self, text: str) -> CommandInvocation:
        stripped = text.strip()
        if not stripped.startswith("/"):
            raise CommandParseError("Slash commands must start with '/'")
        try:
            parts = shlex.split(stripped[1:])
        except ValueError as error:
            raise CommandParseError(str(error)) from error
        if not parts:
            raise CommandParseError("A command name is required after '/'")
        return CommandInvocation(
            name=parts[0].lower(),
            arguments=tuple(parts[1:]),
            raw=stripped,
        )

    def resolve(self, name: str) -> CommandSpec:
        normalized = name.lower().lstrip("/")
        canonical = self._names.get(normalized)
        if canonical is None:
            raise UnknownCommandError(f"Unknown command: /{normalized}")
        return self._commands[canonical]

    async def invoke(
        self,
        invocation: CommandInvocation,
        context: CommandContext,
    ) -> CommandResult:
        result = self.resolve(invocation.name).handler(context, invocation)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, CommandResult):
            raise TypeError("Command handlers must return CommandResult")
        return result

    def _unregister(self, name: str) -> CommandSpec:
        command = self.resolve(name)
        self._remove(command)
        return command

    def _remove_if_current(self, command: CommandSpec) -> None:
        if self._commands.get(command.name) is command:
            self._remove(command)

    def _remove(self, command: CommandSpec) -> None:
        self._commands.pop(command.name, None)
        for name in (command.name, *command.aliases):
            if self._names.get(name) == command.name:
                self._names.pop(name)

    def __contains__(self, name: str) -> bool:
        return name.lower().lstrip("/") in self._names

    def __iter__(self) -> Iterator[CommandSpec]:
        return iter(self._commands.values())

    def __len__(self) -> int:
        return len(self._commands)
