from __future__ import annotations

import asyncio
import re
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Mapping, Protocol, Sequence, cast

from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_scope,
    new_run_id,
    new_thread_id,
)
from msgflux.vulcano.actions import (
    ActivateSessionTab,
    CancelExecution,
    CloseSessionTab,
    InputMode,
    ResolvePermission,
    RuntimeAction,
    StopRuntime,
    SubmitInput,
    ToggleSessionPin,
)
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
from msgflux.vulcano.permissions import PermissionManager
from msgflux.vulcano.sessions import SessionController, SessionStore, SessionTransition
from msgflux.vulcano.ui import UiManager

__all__ = ["MockResponder", "Responder", "RuntimeProtocol", "VulcanoRuntime"]


class Responder(Protocol):
    """Streaming response source used by a Vulcano runtime."""

    def stream(self, prompt: str) -> AsyncIterator[str]: ...


class RuntimeProtocol(Protocol):
    """Client-facing runtime contract implemented independently of Textual."""

    commands: CommandRegistry
    permissions: PermissionManager
    ui: UiManager

    @property
    def is_busy(self) -> bool: ...

    @property
    def queued_inputs(self) -> tuple[SubmitInput, ...]: ...

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
        session_store: SessionStore | None = None,
        session_directory: str | Path | None = None,
        export_directory: str | Path | None = None,
        extension_paths: Sequence[str | Path] = (),
        extensions_enabled: bool = True,
        discover_extensions: bool = True,
        trust_project_extensions: bool = False,
        extension_user_directory: str | Path | None = None,
        max_session_tabs: int = 5,
    ) -> None:
        if responder is not None and agent is not None:
            raise ValueError("Configure either responder or agent, not both")
        if session_store is not None and session_directory is not None:
            raise ValueError("Configure either session_store or session_directory")
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
        resolved_cwd = Path(cwd or Path.cwd()).expanduser().resolve()
        resolved_store = session_store or (
            SessionStore(session_directory) if session_directory is not None else None
        )
        if resolved_store is not None:
            resolved_store.ensure(thread_id)
            stored_events = resolved_store.load(thread_id)
        else:
            stored_events = ()
        self.sessions = SessionController(
            resolved_store,
            thread_id,
            export_directory=export_directory or resolved_cwd,
            max_tabs=max_session_tabs,
        )
        self._session_replay = (
            resolved_store.replay(thread_id) if resolved_store is not None else ()
        )
        self._history: list[DomainEvent] = list(stored_events)
        self._sequence = max(
            (event.sequence for event in stored_events),
            default=0,
        )
        self._started = False
        self._stopped = False
        self._lifecycle_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._submission_lock = asyncio.Lock()
        self._active_submission: asyncio.Task[None] | None = None
        self._active_input: SubmitInput | None = None
        self._steering_queue: deque[SubmitInput] = deque()
        self._follow_up_queue: deque[SubmitInput] = deque()
        self._custom_responder = responder is not None
        self._responder = responder or MockResponder(stream_delay)
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
        service_values = dict(services or {})
        reserved_services = {"permissions", "sessions"}.intersection(service_values)
        if reserved_services:
            names = ", ".join(sorted(reserved_services))
            raise ValueError(f"Reserved runtime service names: {names}")
        self.permissions = PermissionManager(
            self._emit_permission_event,
            interactive=lambda: self.ui.available,
        )
        service_values["sessions"] = self.sessions
        service_values["permissions"] = self.permissions
        self.extensions = ExtensionManager(
            self.commands,
            extension_settings,
            services=service_values,
            agent=agent,
            agent_adapter=agent_adapter,
            permission_manager=self.permissions,
        )
        self.ui = self.extensions.ui
        self._install_builtin_commands()

    @property
    def history(self) -> tuple[DomainEvent, ...]:
        return tuple(self._history)

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped

    @property
    def is_busy(self) -> bool:
        return self._active_submission is not None

    @property
    def queued_inputs(self) -> tuple[SubmitInput, ...]:
        return (*self._steering_queue, *self._follow_up_queue)

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
            for event in self._session_replay:
                await self._events.publish(event)
            self._session_replay = ()
            await self._emit(
                EventType.RUNTIME_STARTED,
                {
                    "runtime": (
                        "agent" if self.extensions.api.agent.is_bound else "mock"
                    ),
                    "agent": self.extensions.api.agent.name,
                    "commands": len(self.commands),
                    "extensions": len(extension_report.loaded),
                    "thread_id": self._thread_scope.thread_id,
                },
            )
            await self._emit_session_tabs()

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
            self.permissions.cancel_all()
            self.sessions.terminate_tabs()
            await self._emit_session_tabs()
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
        if self._stopped:
            raise RuntimeError("Vulcano runtime is stopped")
        if isinstance(action, SubmitInput):
            await self._submit_input(action)
            return
        if await self._dispatch_session_action(action):
            return
        if isinstance(action, ResolvePermission):
            self.permissions.resolve(action.request_id, action.decision)
            return
        if isinstance(action, CancelExecution):
            await self._cancel_execution(action)
            return
        if isinstance(action, StopRuntime):
            await self._cancel_execution(
                CancelExecution(
                    reason=action.reason,
                    correlation_id=action.correlation_id,
                ),
                emit_if_idle=False,
            )
            await self.stop(
                reason=action.reason,
                correlation_id=action.correlation_id,
            )
            return
        raise TypeError(f"Unsupported Vulcano action: {type(action)!r}")

    async def _dispatch_session_action(self, action: RuntimeAction) -> bool:
        if isinstance(action, ActivateSessionTab):
            await self._activate_session_tab(action)
        elif isinstance(action, CloseSessionTab):
            await self._close_session_tab(action)
        elif isinstance(action, ToggleSessionPin):
            await self._toggle_session_pin(action)
        else:
            return False
        return True

    async def _submit_input(self, action: SubmitInput) -> None:
        if self.sessions.ensure_current_tab():
            await self._emit_session_tabs()
        queued: SubmitInput | None = None
        queue_position = 0
        task: asyncio.Task[None] | None = None
        async with self._submission_lock:
            if self._active_submission is not None:
                mode: InputMode = "steer" if action.mode == "auto" else action.mode
                queued = SubmitInput(
                    action.text,
                    mode=mode,
                    correlation_id=action.correlation_id,
                )
                if mode == "follow_up":
                    self._follow_up_queue.append(queued)
                else:
                    self._steering_queue.append(queued)
                queue_position = len(self._steering_queue) + len(self._follow_up_queue)
            else:
                task = asyncio.create_task(
                    self._run_submission_chain(action),
                    name=f"vulcano-input-{action.correlation_id}",
                )
                self._active_submission = task
                self._active_input = action

        if queued is not None:
            await self._emit(
                EventType.INPUT_QUEUED,
                {
                    "content": queued.text.strip(),
                    "mode": queued.mode,
                    "position": queue_position,
                },
                correlation_id=queued.correlation_id,
            )
            return
        if task is not None:
            await task

    async def _run_submission_chain(self, first: SubmitInput) -> None:
        chain_task = asyncio.current_task()
        current = first
        while True:
            scope = self._next_submission_scope()
            try:
                with execution_context(scope=scope):
                    await self._handle_input(current)
            except asyncio.CancelledError:
                async with self._submission_lock:
                    if self._active_submission is chain_task:
                        self._active_submission = None
                        self._active_input = None
                return

            async with self._submission_lock:
                if self._stopped:
                    self._steering_queue.clear()
                    self._follow_up_queue.clear()
                    next_input = None
                elif self._steering_queue:
                    next_input = self._steering_queue.popleft()
                elif self._follow_up_queue:
                    next_input = self._follow_up_queue.popleft()
                else:
                    next_input = None

                if next_input is None:
                    if self._active_submission is chain_task:
                        self._active_submission = None
                        self._active_input = None
                    return
                self._active_input = next_input
                remaining = len(self._steering_queue) + len(self._follow_up_queue)

            await self._emit(
                EventType.INPUT_DEQUEUED,
                {
                    "content": next_input.text.strip(),
                    "mode": next_input.mode,
                    "remaining": remaining,
                },
                correlation_id=next_input.correlation_id,
            )
            current = next_input

    async def _cancel_execution(
        self,
        action: CancelExecution,
        *,
        emit_if_idle: bool = True,
    ) -> None:
        async with self._submission_lock:
            task = self._active_submission
            active_input = self._active_input
            queued = (*self._steering_queue, *self._follow_up_queue)
            self._steering_queue.clear()
            self._follow_up_queue.clear()
            if task is not None:
                self._active_submission = None
                self._active_input = None
                task.cancel()

        if queued:
            await self._emit(
                EventType.INPUT_QUEUE_CLEARED,
                {
                    "reason": action.reason,
                    "items": [
                        {
                            "content": item.text.strip(),
                            "mode": item.mode,
                            "correlation_id": item.correlation_id,
                        }
                        for item in queued
                    ],
                },
                correlation_id=action.correlation_id,
            )
        if task is not None or emit_if_idle:
            await self._emit(
                EventType.EXECUTION_CANCELLED,
                {
                    "reason": action.reason,
                    "active": task is not None,
                    "target_correlation_id": (
                        active_input.correlation_id
                        if active_input is not None
                        else None
                    ),
                    "queued_cleared": len(queued),
                },
                correlation_id=action.correlation_id,
            )
        if task is not None and task is not asyncio.current_task():
            await task

    async def _handle_input(self, action: SubmitInput) -> None:
        text = action.text.strip()
        if not text:
            return
        if text.startswith("/"):
            await self._execute_command(text, action.correlation_id)
            return

        await self._emit(
            EventType.MESSAGE_USER,
            {
                "content": text,
                "scope": get_execution_scope().to_dict(),
            },
            correlation_id=action.correlation_id,
        )
        if self.extensions.api.agent.is_bound:
            try:
                await self.extensions.api.agent.respond(
                    text,
                    emit=self._draft_emitter(action.correlation_id),
                )
            except asyncio.CancelledError:
                await self._emit(
                    EventType.ASSISTANT_COMPLETED,
                    {"status": "aborted"},
                    correlation_id=action.correlation_id,
                )
                raise
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
        except asyncio.CancelledError:
            await self._emit(
                EventType.ASSISTANT_COMPLETED,
                {"content": "".join(chunks), "status": "aborted"},
                correlation_id=action.correlation_id,
            )
            raise
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
                "scope": get_execution_scope().to_dict(),
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
        except asyncio.CancelledError:
            await self._emit(
                EventType.COMMAND_COMPLETED,
                {"name": command.name, "status": "aborted"},
                correlation_id=correlation_id,
            )
            raise
        except Exception as error:
            self.sessions.consume_transition()
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
        transition = self.sessions.consume_transition()
        if transition is not None:
            await self._apply_session_transition(transition)
        if result.stop_runtime:
            await self.stop(
                reason=f"command:/{command.name}",
                correlation_id=correlation_id,
            )

    async def _apply_session_transition(self, transition: SessionTransition) -> None:
        store = self.sessions.store
        if store is None:
            raise RuntimeError("Vulcano session persistence is disabled")
        replay = store.replay(transition.thread_id)
        self.sessions.activate(transition.thread_id)
        self._thread_scope = ExecutionScope(
            thread_id=transition.thread_id,
            namespace=self._thread_scope.namespace,
            abort_signal=self._thread_scope.abort_signal,
        )
        self._pending_scope = None
        await self._emit(
            EventType.SESSION_SWITCHED,
            {
                "kind": transition.kind,
                "thread_id": transition.thread_id,
                "events": [event.to_dict() for event in replay],
            },
        )
        await self._emit_session_tabs()

    async def _activate_session_tab(self, action: ActivateSessionTab) -> None:
        async with self._session_lock:
            if self.is_busy:
                await self._emit_session_tab_error(
                    "Cannot switch sessions while an execution is active",
                    action.correlation_id,
                )
                return
            if action.thread_id == self.sessions.current_thread_id:
                self.sessions.ensure_current_tab()
                await self._emit_session_tabs()
                return
            try:
                self.sessions.request_resume(action.thread_id)
                transition = self.sessions.consume_transition()
                if transition is not None:
                    await self._apply_session_transition(transition)
            except (LookupError, RuntimeError, ValueError) as error:
                await self._emit_session_tab_error(
                    str(error),
                    action.correlation_id,
                )

    async def _close_session_tab(self, action: CloseSessionTab) -> None:
        async with self._session_lock:
            if self.is_busy:
                await self._emit_session_tab_error(
                    "Cannot close sessions while an execution is active",
                    action.correlation_id,
                )
                return
            try:
                closed, next_thread_id = self.sessions.close_tab(action.thread_id)
            except (LookupError, ValueError) as error:
                await self._emit_session_tab_error(
                    str(error),
                    action.correlation_id,
                )
                return
            if (
                action.thread_id == self.sessions.current_thread_id
                and next_thread_id is not None
            ):
                await self._apply_session_transition(
                    SessionTransition("resume", next_thread_id)
                )
                return
            await self._emit_session_tabs(closed=closed.to_dict())

    async def _toggle_session_pin(self, action: ToggleSessionPin) -> None:
        async with self._session_lock:
            try:
                self.sessions.toggle_tab_pin(action.thread_id)
            except (LookupError, ValueError) as error:
                await self._emit_session_tab_error(
                    str(error),
                    action.correlation_id,
                )
                return
            await self._emit_session_tabs()

    async def _emit_session_tabs(
        self,
        *,
        closed: Mapping[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "active_thread_id": self.sessions.active_tab_thread_id,
            "tabs": [tab.to_dict() for tab in self.sessions.tabs],
            "persistence": self.sessions.enabled,
            "max_tabs": self.sessions.max_tabs,
        }
        if closed is not None:
            payload["closed"] = dict(closed)
        await self._emit(EventType.SESSION_TABS_UPDATED, payload)

    async def _emit_session_tab_error(
        self,
        message: str,
        correlation_id: str,
    ) -> None:
        await self._emit(
            EventType.RUNTIME_ERROR,
            {"message": message},
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

    async def _emit_permission_event(
        self,
        event: EventDraft,
        correlation_id: str | None,
    ) -> None:
        await self._emit(
            event.type,
            event.payload,
            correlation_id=correlation_id,
        )

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
        if self.sessions.store is not None:
            self.sessions.store.append(self.sessions.current_thread_id, event)
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
        if self.sessions.store is not None:
            self.sessions.store.append(self.sessions.current_thread_id, event)
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
                "view",
                CommandOptions(
                    description="Set transcript detail mode.",
                    usage="/view <full|compact>",
                    handler=_view_command,
                    get_argument_completions=lambda _value: ("full", "compact"),
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
                "session",
                CommandOptions(
                    description="Show the active durable session.",
                    usage="/session",
                    handler=_session_command,
                    category="session",
                ),
            ),
            (
                "sessions",
                CommandOptions(
                    description="List durable sessions.",
                    usage="/sessions",
                    handler=_sessions_command,
                    category="session",
                ),
            ),
            (
                "resume",
                CommandOptions(
                    description="Resume another durable session.",
                    usage="/resume <thread-id>",
                    handler=_resume_command,
                    get_argument_completions=lambda _value: (
                        tuple(info.thread_id for info in self.sessions.list())
                        if self.sessions.enabled
                        else ()
                    ),
                    category="session",
                ),
            ),
            (
                "new",
                CommandOptions(
                    description="Start a new empty durable session.",
                    usage="/new",
                    handler=_new_session_command,
                    category="session",
                ),
            ),
            (
                "fork",
                CommandOptions(
                    description="Fork this session and continue on the new thread.",
                    usage="/fork [event-sequence]",
                    handler=_fork_command,
                    category="session",
                ),
            ),
            (
                "export",
                CommandOptions(
                    description="Export this session as Markdown.",
                    usage="/export [path]",
                    handler=_export_command,
                    category="session",
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


def _view_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del context
    mode = arguments.strip().lower()
    if mode not in {"full", "compact"}:
        raise ValueError("Usage: /view <full|compact>")
    return CommandResult(
        events=(
            EventDraft(
                EventType.CLIENT_ACTION,
                {"action": "transcript.view", "mode": mode},
            ),
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Transcript view changed to **{mode}**."},
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


def _session_control(context: CommandContext) -> SessionController:
    return cast(SessionController, context.services["sessions"])


def _session_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments
    sessions = _session_control(context)
    state = "enabled" if sessions.enabled else "disabled"
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {
                    "text": (
                        "## Session\n\n"
                        f"- Thread: `{sessions.current_thread_id}`\n"
                        f"- Persistence: **{state}**"
                    )
                },
            ),
        )
    )


def _sessions_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments
    sessions = _session_control(context)
    records = sessions.list()
    lines = ["## Sessions", "", "| Thread | Events | Parent |", "|---|---:|---|"]
    lines.extend(
        (
            f"| `{record.thread_id}` | {record.event_count} | "
            f"`{record.parent_thread_id or '-'}` |"
        )
        for record in records
    )
    return CommandResult(
        events=(EventDraft(EventType.COMMAND_OUTPUT, {"text": "\n".join(lines)}),)
    )


def _resume_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    thread_id = arguments.strip()
    if not thread_id:
        raise ValueError("Usage: /resume <thread-id>")
    info = _session_control(context).request_resume(thread_id)
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Resuming session `{info.thread_id}`."},
            ),
        )
    )


def _new_session_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    if arguments.strip():
        raise ValueError("Usage: /new")
    info = _session_control(context).request_new()
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Started new session `{info.thread_id}`."},
            ),
        )
    )


def _fork_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    value = arguments.strip()
    try:
        sequence = int(value) if value else None
    except ValueError as error:
        raise ValueError("Usage: /fork [event-sequence]") from error
    if sequence is not None and sequence < 1:
        raise ValueError("Fork event sequence must be greater than zero")
    info = _session_control(context).request_fork(sequence)
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Forked session as `{info.thread_id}`."},
            ),
        )
    )


def _export_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    destination = arguments.strip() or None
    path = _session_control(context).export(destination)
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Exported session to `{path}`."},
            ),
        )
    )


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
