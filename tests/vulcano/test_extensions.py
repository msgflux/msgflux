from textwrap import dedent

import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime import get_execution_scope
from msgflux.vulcano import CommandResult, EventType, SubmitInput, VulcanoRuntime
from msgflux.vulcano.extensions import loader as extension_loader


class _FakeAgent:
    name = "main_agent"

    def __init__(self):
        self.tool_library = ToolLibrary(self.name, [])
        self.calls = []
        self.scopes = []

    async def acall(self, message, **kwargs):
        self.calls.append((message, kwargs))
        self.scopes.append(get_execution_scope())
        return f"Agent completed: {message}"


def _write_extension(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_explicit_extension_registers_command_with_owned_api(tmp_path):
    extension = _write_extension(
        tmp_path / "greetings.py",
        """
        import asyncio

        from msgflux.vulcano import (
            CommandOptions,
            CommandResult,
            EventDraft,
            EventType,
        )

        EXTENSION_NAME = "greetings"
        EXTENSION_API_VERSION = 1

        async def setup(api):
            await asyncio.sleep(0)

            def greet(arguments, context):
                source = context.api.source.kind
                generation = context.api.generation
                has_control = "extensions" in context.api.services
                return CommandResult(events=(
                    EventDraft(
                        EventType.COMMAND_OUTPUT,
                        {
                            "text": (
                                f"Hello, {arguments} "
                                f"[{source}:{generation}:{has_control}]"
                            )
                        },
                    ),
                ))

            api.register_command(
                "greet",
                CommandOptions(
                    description="Greet a person.",
                    usage="/greet <name>",
                    handler=greet,
                ),
            )
        """,
    )
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[extension],
        discover_extensions=False,
    )

    await runtime.start()
    await runtime.dispatch(SubmitInput("/greet Ada"))
    await runtime.dispatch(SubmitInput("/extensions"))

    outputs = [
        str(event.payload["text"])
        for event in runtime.history
        if event.type == EventType.COMMAND_OUTPUT
    ]
    assert outputs[0] == "Hello, Ada [cli:1:True]"
    assert "greetings" in outputs[1]
    assert runtime.extensions.records[0].state == "loaded"


@pytest.mark.asyncio
async def test_failed_setup_rolls_back_partial_registrations(tmp_path):
    extension = _write_extension(
        tmp_path / "broken.py",
        """
        from msgflux.vulcano import CommandResult

        EXTENSION_NAME = "broken"

        def setup(api):
            @api.command("partial", "Must be rolled back.")
            def partial(arguments, context):
                return CommandResult()

            raise RuntimeError("setup exploded")
        """,
    )
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[extension],
        discover_extensions=False,
    )

    await runtime.start()

    assert "partial" not in runtime.commands
    assert runtime.extensions.records[0].state == "failed"
    failure = next(
        event for event in runtime.history if event.type == EventType.EXTENSION_FAILED
    )
    assert "setup exploded" in str(failure.payload["error"])


@pytest.mark.asyncio
async def test_observer_failure_is_reported_without_stopping_stream(tmp_path):
    extension = _write_extension(
        tmp_path / "observer.py",
        """
        from msgflux.vulcano import EventType

        EXTENSION_NAME = "observer"

        def setup(api):
            def observe_user(event, context):
                raise RuntimeError("observer exploded")

            api.on(EventType.MESSAGE_USER, observe_user)
        """,
    )
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[extension],
        discover_extensions=False,
    )

    await runtime.dispatch(SubmitInput("keep streaming"))

    assert any(
        event.type == EventType.EXTENSION_FAILED
        and event.payload.get("phase") == "observe"
        and event.payload.get("message") == "observer exploded"
        for event in runtime.history
    )
    assert runtime.history[-1].type == EventType.ASSISTANT_COMPLETED


@pytest.mark.asyncio
async def test_reload_cleans_old_generation_and_loads_new_source(tmp_path):
    marker = tmp_path / "cleanup.txt"
    extension = tmp_path / "versioned.py"

    def write_version(value):
        _write_extension(
            extension,
            f"""
            from pathlib import Path

            from msgflux.vulcano import CommandResult, EventDraft, EventType

            EXTENSION_NAME = "versioned"

            def setup(api):
                @api.command("version", "Show extension version.")
                def version(arguments, context):
                    return CommandResult(events=(
                        EventDraft(EventType.COMMAND_OUTPUT, {{"text": "{value}"}}),
                    ))

                @api.on_cleanup
                def cleanup():
                    Path({str(marker)!r}).write_text("cleaned", encoding="utf-8")
            """,
        )

    write_version("one")
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[extension],
        discover_extensions=False,
    )
    await runtime.start()
    await runtime.dispatch(SubmitInput("/version"))

    write_version("version-two")
    await runtime.dispatch(SubmitInput("/reload"))
    await runtime.dispatch(SubmitInput("/version"))

    version_outputs = [
        event.payload["text"]
        for event in runtime.history
        if event.type == EventType.COMMAND_OUTPUT
        and event.payload.get("text") in {"one", "version-two"}
    ]
    assert version_outputs == ["one", "version-two"]
    assert marker.read_text(encoding="utf-8") == "cleaned"
    assert runtime.extensions.records[0].generation == 2
    assert any(event.type == EventType.EXTENSION_UNLOADED for event in runtime.history)


@pytest.mark.asyncio
async def test_project_extension_requires_explicit_trust(tmp_path):
    _write_extension(
        tmp_path / ".vulcano" / "extensions" / "project.py",
        """
        from msgflux.vulcano import CommandResult

        EXTENSION_NAME = "project"

        def setup(api):
            @api.command("project-command", "Project command.")
            def project_command(arguments, context):
                return CommandResult()
        """,
    )
    user_directory = tmp_path / "user-extensions"
    untrusted = VulcanoRuntime(
        cwd=tmp_path,
        stream_delay=0,
        extension_user_directory=user_directory,
        trust_project_extensions=False,
    )
    trusted = VulcanoRuntime(
        cwd=tmp_path,
        stream_delay=0,
        extension_user_directory=user_directory,
        trust_project_extensions=True,
    )

    await untrusted.start()
    await trusted.start()

    assert "project-command" not in untrusted.commands
    assert "project-command" in trusted.commands


@pytest.mark.asyncio
async def test_installed_entry_point_loads_and_old_api_becomes_stale(
    tmp_path,
    monkeypatch,
):
    captured_apis = []

    def setup(api):
        captured_apis.append(api)

        @api.command("installed-command", "Installed command.")
        def installed_command(arguments, context):
            return CommandResult()

    class FakeDistribution:
        name = "example-distribution"

    class FakeEntryPoint:
        name = "example"
        dist = FakeDistribution()

        def load(self):
            return setup

    monkeypatch.setattr(
        extension_loader.metadata,
        "entry_points",
        lambda *, group: [FakeEntryPoint()],
    )
    runtime = VulcanoRuntime(
        cwd=tmp_path,
        stream_delay=0,
        extension_user_directory=tmp_path / "user-extensions",
    )

    await runtime.start()
    assert "installed-command" in runtime.commands
    assert captured_apis[0].source.kind == "installed"

    await runtime.dispatch(SubmitInput("/reload"))

    assert len(captured_apis) == 2
    assert captured_apis[1].generation == 2
    with pytest.raises(RuntimeError, match="stale runtime generation"):
        _ = captured_apis[0].generation


@pytest.mark.asyncio
async def test_extension_owns_main_agent_tool_and_streaming_command_flow(tmp_path):
    extension = _write_extension(
        tmp_path / "goal.py",
        """
        from msgflux.vulcano import CommandResult, EventDraft, EventType

        EXTENSION_NAME = "goal"

        def setup(api):
            @api.tool
            def inspect_goal(goal: str) -> str:
                \"\"\"Inspect one goal before execution.\"\"\"
                return goal

            @api.command("goal", "Run a custom flow over the main Agent.")
            async def goal(arguments, context):
                flow_scope = context.child_scope(namespace="goal")
                result = await context.api.agent.respond(
                    "Plan and execute: " + arguments,
                    emit=context.emit,
                    scope=flow_scope,
                    vars={"flow": "goal"},
                )
                return CommandResult(events=(
                    EventDraft(
                        EventType.COMMAND_OUTPUT,
                        {"text": f"flow={result.status}"},
                    ),
                ))
        """,
    )
    agent = _FakeAgent()
    runtime = VulcanoRuntime(
        agent=agent,
        extension_paths=[extension],
        discover_extensions=False,
    )

    await runtime.start()
    assert runtime.extensions.api.agent.name == "main_agent"
    assert runtime.extensions.api.tools.names == ("inspect_goal",)
    assert runtime.commands.resolve("goal").owner == "goal"

    await runtime.dispatch(SubmitInput("/goal ship it", correlation_id="goal-1"))

    flow_events = [
        event for event in runtime.history if event.correlation_id == "goal-1"
    ]
    assert [event.type for event in flow_events] == [
        EventType.COMMAND_STARTED,
        EventType.ASSISTANT_STARTED,
        EventType.ASSISTANT_DELTA,
        EventType.ASSISTANT_COMPLETED,
        EventType.COMMAND_OUTPUT,
        EventType.COMMAND_COMPLETED,
    ]
    assert flow_events[2].payload["delta"] == (
        "Agent completed: Plan and execute: ship it"
    )
    assert agent.calls == [("Plan and execute: ship it", {"vars": {"flow": "goal"}})]
    flow_scope = agent.scopes[0]
    assert flow_scope.thread_id is not None
    assert flow_scope.namespace == "goal"
    assert flow_scope.run_id is not None
    assert flow_scope.parent_run_id is not None
    assert flow_scope.root_run_id == flow_scope.parent_run_id

    await runtime.stop()
    assert agent.tool_library.get_tool_names() == []


@pytest.mark.asyncio
async def test_failed_extension_setup_removes_registered_agent_tool(tmp_path):
    extension = _write_extension(
        tmp_path / "broken_tool.py",
        """
        EXTENSION_NAME = "broken-tool"

        def setup(api):
            @api.tool
            def temporary_tool(value: str) -> str:
                \"\"\"Return a temporary value.\"\"\"
                return value

            raise RuntimeError("tool setup exploded")
        """,
    )
    agent = _FakeAgent()
    runtime = VulcanoRuntime(
        agent=agent,
        extension_paths=[extension],
        discover_extensions=False,
    )

    await runtime.start()

    assert agent.tool_library.get_tool_names() == []
    assert runtime.extensions.records[0].state == "failed"
    assert "tool setup exploded" in runtime.extensions.records[0].error
