"""End-to-end navigation flow through the Agent runtime."""

from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    execution_context,
)
from msgflux.tools.builtin import DeleteTool
from msgflux.tools.builtin.workspace_query import GlobTool, GrepTool, LsTool
from msgflux.utils.msgspec import msgspec_dumps, msgspec_loads


def _tool_response(name, arguments, call_id):
    response = ModelResponse()
    calls = ToolCallAggregator()
    calls.process(0, call_id, name, msgspec_dumps(arguments))
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


@pytest.mark.asyncio
async def test_agent_navigates_searches_and_deletes_with_compact_checkpointed_outputs():
    files = {
        "/.gitignore": b"src/ignored/\n",
        "/src/main.py": b"needle = 1\n",
        "/src/readme.txt": b"needle documentation\n",
        "/src/ignored/secret.py": b"needle must not be returned\n",
    }
    filesystem = InMemoryWorkspace("navigation", files)
    checkpoint_store = InMemoryCheckpointStore()
    scope = ExecutionScope(
        namespace="navigation",
        thread_id="thread",
        run_id="run",
        environment=ExecutionEnvironment(filesystem),
        permissions=PermissionSet(
            resources=[
                filesystem.permission("/", "filesystem.list"),
                filesystem.permission("/src", "filesystem.list"),
                filesystem.permission("/.gitignore", "filesystem.read"),
                filesystem.permission("/src/main.py", "filesystem.read"),
                filesystem.permission("/src/readme.txt", "filesystem.read"),
                filesystem.permission("/src/main.py", "filesystem.delete"),
            ]
        ),
    )
    agent = Agent(
        name="navigator",
        model=Mock(model_type="chat_completion"),
        tools=[LsTool(), GlobTool(), GrepTool(), DeleteTool()],
        checkpoint_store=checkpoint_store,
    )
    agent.generator.aforward = AsyncMock(
        side_effect=[
            _tool_response("ls", {"path": "/src"}, "ls-1"),
            _tool_response("glob", {"pattern": "**/*.py", "path": "/"}, "glob-1"),
            _tool_response("grep", {"pattern": "needle", "path": "/"}, "grep-1"),
            _tool_response("delete", {"path": "/src/main.py"}, "delete-1"),
            _text_response("done"),
        ]
    )

    events = [
        event
        async for event in agent.stream_events("Find and remove the match", scope=scope)
    ]

    assert events
    assert any(event.type == "tool.end" for event in events)
    assert events[-1].type == "run.end"
    assert agent.generator.aforward.await_count == 5
    with pytest.raises(FileNotFoundError), execution_context(scope=scope):
        filesystem.read_bytes("/src/main.py")

    state = checkpoint_store.load_state("navigator", "thread", "run")
    outputs = [
        msgspec_loads(item["output"])
        for item in state["messages"]["items"]
        if item.get("type") == "function_call_output"
    ]
    assert len(outputs) == 4
    assert all(len(msgspec_dumps(output)) < 2_000 for output in outputs)
    assert outputs[0]["path"] == "/src"
    assert [entry["name"] for entry in outputs[0]["entries"]] == [
        "ignored",
        "main.py",
        "readme.txt",
    ]
    assert [match["path"] for match in outputs[1]["matches"]] == ["/src/main.py"]
    assert [match["path"] for match in outputs[2]["matches"]] == [
        "/src/main.py",
        "/src/readme.txt",
    ]
    assert outputs[3] == {"status": "completed"}
    assert "must not be returned" not in msgspec_dumps(outputs)
