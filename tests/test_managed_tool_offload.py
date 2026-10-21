"""Managed offload uses the active thread across real Agent tool invocations."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.extensions import ManagedToolOutputOffloadExtension
from msgflux.runtime import ExecutionScope, ToolOutputOffloadConfig
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.tools.builtin import BashTool, ReadFileTool
from msgflux.utils.msgspec import msgspec_dumps


def _response(tool=None, arguments=None):
    response = ModelResponse()
    if tool:
        calls = ToolCallAggregator()
        calls.process(0, "call-output", tool, msgspec_dumps(arguments or {}))
        response.set_response_type("tool_call")
        response.add(calls)
    else:
        response.set_response_type("text_generation")
        response.add("done")
    return response


def _agent(tmp_path, tools, *, config=None, agent_dir=True):
    model = Mock()
    model.model_type = "chat_completion"
    workspace = AgentWorkspace.local(tmp_path)
    agent = Agent(
        "main",
        model,
        tools=tools,
        workspace=workspace,
        agent_dir=tmp_path / "state" if agent_dir else None,
        extensions=[
            ManagedToolOutputOffloadExtension(
                config
                or ToolOutputOffloadConfig(max_inline_bytes=128, preview_bytes=16)
            )
        ],
    )
    return agent, workspace


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["hello\n" * 100, {"payload": "olá" * 300}])
async def test_agent_feedback_contains_published_path_and_read_roundtrip(
    tmp_path, value
):
    def produce() -> Any:
        """Produce a report."""
        return value

    agent, workspace = _agent(tmp_path, [produce, ReadFileTool()])
    seen = {}

    async def model_response(**kwargs):
        messages = kwargs["messages"].to_chatml()
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            return _response("produce")
        descriptor = msgspec.json.decode(tool_messages[-1]["content"])
        seen.update(descriptor)
        assert Path(descriptor["path"]).is_file()
        return _response()

    agent.generator.aforward = AsyncMock(side_effect=model_response)
    try:
        await agent.acall("produce", scope=ExecutionScope(thread_id="first"))
        with agent._resource_context({"scope": ExecutionScope(thread_id="first")}):
            read = await agent.tool_library.acall(
                [("read-output", "read", {"path": seen["path"]})]
            )
        content = read.tool_calls[0].result
        assert ("hello" in content) if isinstance(value, str) else ("olá" in content)
        store = agent._owned_threads["first"].resources.tool_result_store()
        reference = store.get(seen["reference"]["result_id"])
        raw = b"".join(store.iter_bytes(reference))
        assert (
            raw.decode() if isinstance(value, str) else msgspec.json.decode(raw)
        ) == value
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_small_output_does_not_create_artifact_directory(tmp_path):
    def tiny() -> str:
        """Return a small result."""
        return "ok"

    agent, workspace = _agent(tmp_path, [tiny])
    agent.generator.aforward = AsyncMock(side_effect=[_response("tiny"), _response()])
    try:
        await agent.acall("tiny", scope=ExecutionScope(thread_id="small"))
        assert not (tmp_path / "state/threads/small/tool-results").exists()
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_parallel_threads_publish_separate_paths(tmp_path):
    def produce() -> str:
        """Return a large report."""
        return "report\n" * 100

    agent, workspace = _agent(tmp_path, [produce])
    try:

        async def call(thread):
            with agent._resource_context({"scope": ExecutionScope(thread_id=thread)}):
                result = await agent.tool_library.acall(
                    [("call-" + thread, "produce", {})]
                )
                return result.tool_calls[0].result

        first, second = await asyncio.gather(call("one"), call("two"))
        assert "/threads/one/" in first["path"]
        assert "/threads/two/" in second["path"]
        assert Path(first["path"]).read_bytes() == Path(second["path"]).read_bytes()
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_shell_capture_is_managed_and_readable(tmp_path):
    agent, workspace = _agent(tmp_path, [BashTool(), ReadFileTool()])
    try:
        with agent._resource_context({"scope": ExecutionScope(thread_id="shell")}):
            response = await agent.tool_library.acall(
                [
                    (
                        "shell-call",
                        "bash",
                        {"command": ['printf "line\\n%.0s" $(seq 1 100)']},
                    )
                ]
            )
            result = response.tool_calls[0].result
            reference = result.output_reference
            assert reference is not None
            store = agent._owned_threads["shell"].resources.tool_result_store()
            path = str(store.root / reference.result_id / "content")
            read = await agent.tool_library.acall(
                [("read-call", "read", {"path": path})]
            )
            assert "line" in read.tool_calls[0].result
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_missing_managed_resources_fails_before_model_or_tools(tmp_path):
    agent, workspace = _agent(tmp_path, [], agent_dir=False)
    agent.generator.aforward = AsyncMock()
    try:
        with pytest.raises(ValueError, match="requires agent_dir"):
            await agent.acall("hello")
        agent.generator.aforward.assert_not_called()
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_child_inherits_offload_and_cannot_replace_limits(tmp_path):
    def produce() -> str:
        """Produce a large nested report."""
        return "nested\n" * 100

    parent, workspace = _agent(tmp_path, [])
    model = Mock()
    model.model_type = "chat_completion"
    child = Agent("child", model, tools=[produce])
    conflicting = Agent(
        "different", model, extensions=[ManagedToolOutputOffloadExtension()]
    )
    try:
        with parent._resource_context({"scope": ExecutionScope(thread_id="shared")}):
            with child._resource_context({}):
                result = await child.tool_library.acall(
                    [("nested-call", "produce", {})]
                )
                assert "/threads/shared/" in result.tool_calls[0].result["path"]
            with pytest.raises(ValueError, match="cannot replace tool output"):
                with conflicting._resource_context({}):
                    pytest.fail("conflicting configuration entered")
        assert parent._owned_threads["shared"].active == 0
        assert not child._owned_threads
    finally:
        await child.aclose()
        await conflicting.aclose()
        await parent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_quota_error_never_publishes_reference_or_retries_tool(tmp_path):
    effects = []

    def produce() -> str:
        """Perform an action and return a large result."""
        effects.append("executed")
        return "x" * 1000

    config = ToolOutputOffloadConfig(
        max_inline_bytes=128, preview_bytes=16, max_result_bytes=256
    )
    agent, workspace = _agent(tmp_path, [produce], config=config)
    try:
        with agent._resource_context({"scope": ExecutionScope(thread_id="quota")}):
            response = await agent.tool_library.acall([("quota-call", "produce", {})])
        outcome = response.tool_calls[0]
        assert outcome.result is None
        assert "ToolResultTooLargeError" in str(outcome.error)
        assert "do not retry automatically" in str(outcome.error)
        assert effects == ["executed"]
        assert not list((tmp_path / "state/threads/quota/tool-results").glob("res_*"))
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_inline_bytes": True},
        {"preview_bytes": -1},
        {"max_store_bytes": 0},
        {"max_capture_bytes": 0},
    ],
)
def test_offload_limits_are_strict(kwargs):
    with pytest.raises(ValueError):
        ToolOutputOffloadConfig(**kwargs)


@pytest.mark.asyncio
async def test_background_result_keeps_original_reference(tmp_path):
    from msgflux.tools.config import tool_config

    @tool_config(allow_background=True)
    def produce() -> str:
        """Produce a report in background."""
        return "background\n" * 100

    agent, workspace = _agent(tmp_path, [produce])
    try:
        with agent._resource_context({"scope": ExecutionScope(thread_id="background")}):
            dispatched = await agent.tool_library.acall(
                [("dispatch", "produce", {"run_in_background": True})]
            )
            task_id = (
                dispatched.tool_calls[0].result.split("task_id='")[1].split("'")[0]
            )
            waited = await agent.tool_library.acall(
                [("wait", "task_wait", {"task_id": task_id, "timeout": 3.0})]
            )
            result = waited.tool_calls[0].result
            assert result["type"] == "tool_result_reference"
            assert Path(result["path"]).read_text() == "background\n" * 100
            assert (
                len(
                    list(
                        (tmp_path / "state/threads/background/tool-results").glob(
                            "res_*"
                        )
                    )
                )
                == 1
            )
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_removing_agent_extension_removes_capture_and_transform(tmp_path):
    def produce() -> str:
        """Return a large report."""
        return "report\n" * 100

    agent, workspace = _agent(tmp_path, [produce])
    try:
        with agent._resource_context({"scope": ExecutionScope(thread_id="removal")}):
            before = await agent.tool_library.acall([("before", "produce", {})])
            assert before.tool_calls[0].result["type"] == "tool_result_reference"
        agent.remove_extension("tool_output_offload")
        assert not agent.tool_library.has_extension("tool_output_offload")
        with agent._resource_context({"scope": ExecutionScope(thread_id="removal")}):
            after = await agent.tool_library.acall([("after", "produce", {})])
            assert after.tool_calls[0].result == "report\n" * 100
        assert not agent.has_extension("tool_output_offload")
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_read_budget_instruction_reaches_model_without_new_offload(tmp_path):
    (tmp_path / "large.txt").write_text("PRIVATE_FILE_CONTENT" * 100)
    agent, workspace = _agent(tmp_path, [ReadFileTool(max_text_bytes=64)])
    feedback = []
    calls = 0

    async def respond(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response("read", {"path": "large.txt"})
        feedback.extend(kwargs["messages"].to_chatml())
        return _response()

    agent.generator.aforward = AsyncMock(side_effect=respond)
    try:
        await agent.acall(
            "read large.txt", scope=ExecutionScope(thread_id="read-budget")
        )
        text = "\n".join(str(item.get("content", "")) for item in feedback)
        assert "No content was returned" in text
        assert "Bash" in text and "read cannot retrieve it" in text
        assert "smaller limit" not in text
        assert "PRIVATE_FILE_CONTENT" not in text
        assert not (tmp_path / "state/threads/read-budget/tool-results").exists()
        assert calls == 2
    finally:
        await agent.aclose()
        await workspace.aclose()
