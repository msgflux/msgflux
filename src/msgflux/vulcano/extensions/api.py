from __future__ import annotations

import inspect
from typing import Callable, Mapping, Protocol

from msgflux.vulcano.commands import (
    CommandContext,
    CommandHandler,
    CommandInvocation,
    CommandOptions,
    CommandRegistration,
    CommandRegistry,
    CommandResult,
    CommandSpec,
    RuntimeCommandHandler,
)
from msgflux.vulcano.extensions.agent import (
    AgentApi,
    ToolLibraryApi,
    ToolRegistration,
    _AgentBinding,
)
from msgflux.vulcano.extensions.types import (
    ExtensionCleanup,
    ExtensionContext,
    ExtensionObserver,
    ExtensionSource,
)

__all__ = ["ExtensionApi"]


class _Registration(Protocol):
    def remove(self) -> None: ...


class _ExtensionHost(Protocol):
    commands: CommandRegistry

    def register_observer(
        self,
        owner: str,
        event_type: str,
        observer: ExtensionObserver,
        context: ExtensionContext,
    ) -> _Registration: ...


class ExtensionApi:
    """Capability surface passed to one extension generation."""

    def __init__(
        self,
        host: _ExtensionHost,
        *,
        owner: str,
        context: ExtensionContext,
        agent_binding: _AgentBinding,
    ) -> None:
        self._host = host
        self._owner = owner
        self._context = context
        self._active = True
        self._registrations: list[_Registration] = []
        self._cleanups: list[ExtensionCleanup] = []
        self.agent = AgentApi(
            agent_binding,
            assert_active=self._assert_active,
            track=self._track_registration,
        )

    @property
    def source(self) -> ExtensionSource:
        self._assert_active()
        return self._context.source

    @property
    def generation(self) -> int:
        self._assert_active()
        return self._context.generation

    @property
    def context(self) -> ExtensionContext:
        self._assert_active()
        return self._context

    @property
    def services(self) -> Mapping[str, object]:
        self._assert_active()
        return self._context.services

    @property
    def tools(self) -> ToolLibraryApi:
        """Convenience alias for the main Agent's ToolLibrary facade."""
        self._assert_active()
        return self.agent.tools

    def register_tool(self, tool: Callable[..., object]) -> ToolRegistration:
        """Register a tool in the main Agent with extension ownership."""
        return self.tools.register(tool)

    def tool(self, tool: Callable[..., object]) -> Callable[..., object]:
        """Decorator form of register_tool()."""
        self.register_tool(tool)
        return tool

    def register_command(
        self,
        name: str,
        options: CommandOptions,
    ) -> CommandRegistration:
        """Register a slash command using Pi's name/options shape."""
        self._assert_active()
        if not isinstance(options, CommandOptions):
            raise TypeError("register_command() options must be CommandOptions")
        handler = self._guard_handler(options.handler)
        owned_command = CommandSpec(
            name=name,
            description=options.description,
            handler=handler,
            get_argument_completions=options.get_argument_completions,
            usage=options.usage,
            aliases=options.aliases,
            category=options.category,
            owner=self._owner,
        )
        registration = self._host.commands._register(owned_command)
        self._registrations.append(registration)
        return registration

    def command(
        self,
        name: str,
        description: str,
        *,
        usage: str = "",
        aliases: tuple[str, ...] = (),
        category: str = "extension",
    ) -> Callable[[CommandHandler], CommandHandler]:
        self._assert_active()

        def decorator(handler: CommandHandler) -> CommandHandler:
            self.register_command(
                name,
                CommandOptions(
                    description=description,
                    handler=handler,
                    usage=usage,
                    aliases=aliases,
                    category=category,
                ),
            )
            return handler

        return decorator

    def register_observer(
        self,
        event_type: str,
        observer: ExtensionObserver,
    ) -> _Registration:
        self._assert_active()
        registration = self._host.register_observer(
            self._owner,
            event_type,
            observer,
            self._context,
        )
        self._registrations.append(registration)
        return registration

    def on(
        self,
        event_type: str,
        observer: ExtensionObserver,
    ) -> _Registration:
        """Subscribe using Pi's canonical event-handler shape."""
        return self.register_observer(event_type, observer)

    def observe(
        self,
        event_type: str,
    ) -> Callable[[ExtensionObserver], ExtensionObserver]:
        self._assert_active()

        def decorator(observer: ExtensionObserver) -> ExtensionObserver:
            self.on(event_type, observer)
            return observer

        return decorator

    def on_cleanup(self, cleanup: ExtensionCleanup) -> ExtensionCleanup:
        self._assert_active()
        self._cleanups.append(cleanup)
        return cleanup

    def _track_registration(self, registration: _Registration) -> None:
        self._assert_active()
        self._registrations.append(registration)

    async def _deactivate(self) -> tuple[str, ...]:
        if not self._active:
            return ()
        self._active = False
        errors: list[str] = []
        for registration in reversed(self._registrations):
            try:
                registration.remove()
            except Exception as error:
                errors.append(f"registration cleanup failed: {error}")
        self._registrations.clear()

        for cleanup in reversed(self._cleanups):
            try:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                errors.append(f"extension cleanup failed: {error}")
        self._cleanups.clear()
        return tuple(errors)

    def _guard_handler(self, handler: CommandHandler) -> RuntimeCommandHandler:
        async def guarded(
            context: CommandContext,
            invocation: CommandInvocation,
        ) -> CommandResult:
            self._assert_active()
            owned_context = context.with_api(self)
            raw_parts = invocation.raw[1:].split(maxsplit=1)
            arguments = raw_parts[1] if len(raw_parts) == 2 else ""
            result = handler(arguments, owned_context)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                return CommandResult()
            if not isinstance(result, CommandResult):
                raise TypeError("Command handlers must return CommandResult or None")
            return result

        return guarded

    def _assert_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                f"Extension {self._owner!r} belongs to a stale runtime generation"
            )
