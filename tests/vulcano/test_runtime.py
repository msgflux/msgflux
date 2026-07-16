import json

import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime import ExecutionScope, get_execution_scope
from msgflux.vulcano import (
    CommandOptions,
    CommandResult,
    EventDraft,
    EventType,
    SubmitInput,
    VulcanoRuntime,
)


class _FakeAgent:
    name = "runtime_agent"

    def __init__(self):
        self.calls = []
        self.scopes = []
        self.tool_library = ToolLibrary(self.name, [])

    async def acall(self, message, **kwargs):
        self.calls.append((message, kwargs))
        self.scopes.append(get_execution_scope())
        return f"Agent response: {message}"


def test_unbound_agent_facade_reports_how_to_bind_main_agent():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    def example_tool(value: str) -> str:
        """Return a value."""
        return value

    assert not runtime.extensions.api.agent.is_bound
    with pytest.raises(RuntimeError, match="No main Agent is bound"):
        runtime.extensions.api.register_tool(example_tool)


def test_internal_commands_are_owned_by_core_extension_api():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    assert {command.owner for command in runtime.commands} == {"vulcano"}


@pytest.mark.asyncio
async def test_subscriber_receives_runtime_start_event():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    subscription = runtime.subscribe()

    await runtime.start()
    event = await subscription.__anext__()
    await subscription.aclose()

    assert event.type == EventType.RUNTIME_STARTED
    assert event.payload["runtime"] == "mock"
    assert event.sequence == 1


@pytest.mark.asyncio
async def test_mock_runtime_streams_ordered_assistant_events():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    await runtime.dispatch(SubmitInput("explain this diff", correlation_id="run-1"))

    events = runtime.history
    event_types = [event.type for event in events]
    deltas = [
        str(event.payload["delta"])
        for event in events
        if event.type == EventType.ASSISTANT_DELTA
    ]

    assert event_types[0] == EventType.RUNTIME_STARTED
    assert EventType.MESSAGE_USER in event_types
    assert EventType.ASSISTANT_STARTED in event_types
    assert EventType.ASSISTANT_COMPLETED == event_types[-1]
    assert "".join(deltas) == "Mock runtime received: explain this diff"
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert all(
        event.correlation_id == "run-1"
        for event in events
        if event.type != EventType.RUNTIME_STARTED
    )


@pytest.mark.asyncio
async def test_extension_command_executes_inside_runtime():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    def greet(arguments, context):
        assert "runtime" not in context.api.services
        assert "extensions" in context.api.services
        assert context.api.source.kind == "core"
        return CommandResult(
            events=(
                EventDraft(
                    EventType.COMMAND_OUTPUT,
                    {"text": f"Hello, {arguments}"},
                ),
            )
        )

    runtime.extensions.api.register_command(
        "greet",
        CommandOptions(
            description="Greet a user.",
            handler=greet,
        ),
    )

    await runtime.dispatch(SubmitInput("/greet Ada", correlation_id="command-1"))

    output = next(
        event for event in runtime.history if event.type == EventType.COMMAND_OUTPUT
    )
    assert output.payload["text"] == "Hello, Ada"
    assert output.correlation_id == "command-1"


@pytest.mark.asyncio
async def test_runtime_manages_durable_scope_for_each_command_submission():
    initial_scope = ExecutionScope(
        thread_id="thd_saved",
        namespace="vulcano",
        run_id="run_resumed",
    )
    runtime = VulcanoRuntime(
        scope=initial_scope,
        stream_delay=0,
        extensions_enabled=False,
    )
    captured_scopes = []

    def capture(arguments, context):
        del arguments
        assert get_execution_scope() == context.scope
        captured_scopes.append(context.scope)
        return CommandResult()

    runtime.extensions.api.register_command(
        "capture-scope",
        CommandOptions(description="Capture the execution scope.", handler=capture),
    )

    await runtime.dispatch(SubmitInput("/capture-scope"))
    await runtime.dispatch(SubmitInput("/capture-scope"))

    resumed, fresh = captured_scopes
    assert resumed.thread_id == "thd_saved"
    assert resumed.run_id == "run_resumed"
    assert resumed.root_run_id == "run_resumed"
    assert fresh.thread_id == resumed.thread_id
    assert fresh.run_id != resumed.run_id
    assert fresh.root_run_id == fresh.run_id
    assert fresh.parent_run_id is None


@pytest.mark.asyncio
async def test_clear_and_quit_are_runtime_owned_commands():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    await runtime.dispatch(SubmitInput("/clear"))
    await runtime.dispatch(SubmitInput("/quit"))

    clear_event = next(
        event for event in runtime.history if event.type == EventType.CLIENT_ACTION
    )
    assert clear_event.payload == {"action": "transcript.clear"}
    assert runtime.history[-1].type == EventType.RUNTIME_STOPPED
    assert not runtime.is_running

    with pytest.raises(RuntimeError, match="stopped"):
        await runtime.dispatch(SubmitInput("after shutdown"))


@pytest.mark.asyncio
async def test_unknown_command_becomes_event_instead_of_client_exception():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    await runtime.dispatch(SubmitInput("/does-not-exist"))

    assert runtime.history[-1].type == EventType.COMMAND_ERROR
    assert "Unknown command" in str(runtime.history[-1].payload["message"])


@pytest.mark.asyncio
async def test_domain_event_is_transport_ready():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    await runtime.dispatch(SubmitInput("/echo ready", correlation_id="event-1"))

    event = next(
        item for item in runtime.history if item.type == EventType.COMMAND_OUTPUT
    )
    encoded = json.dumps(event.to_dict())
    decoded = json.loads(encoded)

    assert decoded["type"] == EventType.COMMAND_OUTPUT
    assert decoded["correlation_id"] == "event-1"
    assert decoded["payload"] == {"text": "ready"}


@pytest.mark.asyncio
async def test_bound_main_agent_drives_the_runtime_event_stream():
    agent = _FakeAgent()
    runtime = VulcanoRuntime(agent=agent, extensions_enabled=False)

    await runtime.dispatch(SubmitInput("use the agent", correlation_id="agent-1"))

    started = next(
        event for event in runtime.history if event.type == EventType.RUNTIME_STARTED
    )
    assert started.payload["runtime"] == "agent"
    assert started.payload["agent"] == "runtime_agent"
    assert agent.calls == [("use the agent", {})]
    agent_scope = agent.scopes[0]
    assert agent_scope.thread_id is not None
    assert agent_scope.run_id is not None
    assert agent_scope.root_run_id == agent_scope.run_id
    assert agent_scope.parent_run_id is None
    assert [
        event.type for event in runtime.history if event.correlation_id == "agent-1"
    ] == [
        EventType.MESSAGE_USER,
        EventType.ASSISTANT_STARTED,
        EventType.ASSISTANT_DELTA,
        EventType.ASSISTANT_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_main_agent_cannot_be_rebound_after_extensions_load():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    await runtime.start()

    with pytest.raises(RuntimeError, match="before runtime.start"):
        runtime.bind_agent(_FakeAgent())
