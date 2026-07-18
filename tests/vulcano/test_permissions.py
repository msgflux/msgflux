import asyncio

import pytest

from msgflux.runtime import ExecutionScope
from msgflux.vulcano import (
    CancelExecution,
    CommandOptions,
    CommandResult,
    EventDraft,
    EventType,
    PermissionManager,
    SubmitInput,
    VulcanoRuntime,
)


class _AvailabilityDriver:
    def apply_state(self, state) -> None:
        del state


@pytest.mark.asyncio
async def test_permission_manager_remembers_grant_within_thread_session():
    events: list[tuple[EventDraft, str | None]] = []
    requested = asyncio.Event()

    async def emit(event: EventDraft, correlation_id: str | None) -> None:
        events.append((event, correlation_id))
        if event.type == EventType.PERMISSION_REQUESTED:
            requested.set()

    manager = PermissionManager(emit, interactive=lambda: True)
    scope = ExecutionScope(thread_id="thd_permission", run_id="run_one")
    first = asyncio.create_task(
        manager.request(
            "extension",
            "shell",
            "Run tests?",
            resource="pytest tests/vulcano",
            remember_key="test-suite",
            scope=scope,
            correlation_id="command-one",
        )
    )
    await requested.wait()
    pending = manager.pending[0]

    assert manager.resolve(pending.request_id, "allow_session")
    first_result = await first
    second_result = await manager.request(
        "extension",
        "shell",
        "Run tests again?",
        resource="pytest tests/vulcano",
        remember_key="test-suite",
        scope=scope,
        correlation_id="command-two",
    )

    assert first_result.allowed
    assert first_result.remembered
    assert first_result.source == "user"
    assert second_result.allowed
    assert second_result.decision == "allow_session"
    assert second_result.source == "session"
    assert [event.type for event, _correlation in events] == [
        EventType.PERMISSION_REQUESTED,
        EventType.PERMISSION_RESOLVED,
        EventType.PERMISSION_REQUESTED,
        EventType.PERMISSION_RESOLVED,
    ]
    assert events[2][0].payload["requires_confirmation"] is False
    assert manager.pending == ()


@pytest.mark.asyncio
async def test_permission_manager_cancellation_resolves_audit_event():
    events: list[EventDraft] = []
    requested = asyncio.Event()

    async def emit(event: EventDraft, correlation_id: str | None) -> None:
        del correlation_id
        events.append(event)
        if event.type == EventType.PERMISSION_REQUESTED:
            requested.set()

    manager = PermissionManager(emit, interactive=lambda: True)
    task = asyncio.create_task(
        manager.request("extension", "network", "Call the remote API?")
    )
    await requested.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert events[-1].type == EventType.PERMISSION_RESOLVED
    assert events[-1].payload["decision"] == "cancelled"
    assert events[-1].payload["allowed"] is False
    assert manager.pending == ()


@pytest.mark.asyncio
async def test_command_permission_denies_safely_without_frontend():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    async def protected(arguments, context):
        result = await context.request_permission(
            "write",
            "Modify the requested file?",
            resource=arguments,
        )
        return CommandResult(
            events=(
                EventDraft(
                    EventType.COMMAND_OUTPUT,
                    {"text": f"allowed={result.allowed} source={result.source}"},
                ),
            )
        )

    runtime.extensions.api.register_command(
        "protected",
        CommandOptions(
            description="Exercise the permission broker.",
            handler=protected,
        ),
    )

    await runtime.dispatch(
        SubmitInput("/protected src/example.py", correlation_id="permission-command")
    )

    requested = next(
        event
        for event in runtime.history
        if event.type == EventType.PERMISSION_REQUESTED
    )
    resolved = next(
        event
        for event in runtime.history
        if event.type == EventType.PERMISSION_RESOLVED
    )
    output = next(
        event for event in runtime.history if event.type == EventType.COMMAND_OUTPUT
    )
    assert requested.payload["requires_confirmation"] is False
    assert requested.payload["scope"]["thread_id"] is not None
    assert resolved.payload["decision"] == "deny"
    assert resolved.payload["source"] == "headless"
    assert output.payload["text"] == "allowed=False source=headless"
    assert requested.correlation_id == "permission-command"
    assert resolved.correlation_id == "permission-command"


@pytest.mark.asyncio
async def test_cancelling_command_clears_pending_permission():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    driver = _AvailabilityDriver()
    runtime.ui.bind(driver)

    async def protected(arguments, context):
        del arguments
        await context.request_permission("shell", "Run the command?")
        return CommandResult()

    runtime.extensions.api.register_command(
        "protected",
        CommandOptions(description="Wait for permission.", handler=protected),
    )
    subscription = runtime.subscribe()
    active = asyncio.create_task(
        runtime.dispatch(SubmitInput("/protected", correlation_id="protected"))
    )
    while True:
        event = await subscription.__anext__()
        if event.type == EventType.PERMISSION_REQUESTED:
            break

    await runtime.dispatch(
        CancelExecution(reason="test", correlation_id="cancel-permission")
    )
    await active
    await subscription.aclose()
    runtime.ui.unbind(driver)

    resolved = next(
        event
        for event in runtime.history
        if event.type == EventType.PERMISSION_RESOLVED
    )
    completed = next(
        event for event in runtime.history if event.type == EventType.COMMAND_COMPLETED
    )
    assert resolved.payload["decision"] == "cancelled"
    assert completed.payload["status"] == "aborted"
    assert runtime.permissions.pending == ()
    assert not runtime.is_busy
