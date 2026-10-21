"""Agent idle-release fences preserve active work and uncertain state."""

from __future__ import annotations

import asyncio
from threading import Event
from unittest.mock import AsyncMock, Mock

import msgflux as mf
import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime import AgentWorkspace
from msgflux.tools.builtin import AgentTool, BashTool


def _text(content="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _tool_call(name, arguments):
    calls = ToolCallAggregator()
    calls.process(0, "call-1", name, arguments)
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _agent(name, *, agent_dir=None, workspace=None, tools=()):
    model = Mock(model_type="chat_completion")
    agent = Agent(
        name=name,
        model=model,
        agent_dir=agent_dir,
        workspace=workspace,
        tools=list(tools),
    )
    return agent


@pytest.mark.asyncio
async def test_idle_release_refuses_a_live_foreground_call(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _text()

    agent = _agent("foreground-release", agent_dir=tmp_path / "agent")
    agent.generator.aforward = AsyncMock(side_effect=answer)
    thread_id = "foreground-release"
    call = asyncio.create_task(
        agent.acall("wait", scope=ExecutionScope(thread_id=thread_id, run_id="run"))
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        reason = agent._thread_release_reason(thread_id)
        assert reason == "An Agent call is still using thread resources."
        assert agent._mark_thread_idle_closing(thread_id) == reason
        assert not agent._owned_threads[thread_id].closing

        release.set()
        await asyncio.wait_for(call, timeout=3)
        assert agent._thread_release_reason(thread_id) is None
        assert agent._mark_thread_idle_closing(thread_id) is None
        assert agent._owned_threads[thread_id].idle_closing
    finally:
        release.set()
        if not call.done():
            await asyncio.gather(call, return_exceptions=True)
        await agent.aclose()


@pytest.mark.asyncio
async def test_idle_release_refuses_real_agenttool_bash_child_until_terminal(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = AgentWorkspace.local(workspace_root)
    agent_dir = tmp_path / "managed-agent"
    marker = workspace_root / "bash-started"

    child = _agent("bash-child", workspace=workspace, tools=[BashTool()])
    child.generator.aforward = AsyncMock(
        side_effect=[
            _tool_call("bash", '{"command":"touch bash-started && sleep 0.5"}'),
            _text("bash complete"),
        ]
    )

    root = _agent("background-root", agent_dir=agent_dir, workspace=workspace)
    root.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
    root.tool_library.add(child)
    root.generator.aforward = AsyncMock(
        side_effect=[
            _tool_call(
                "agent",
                '{"name":"bash-child","message":"run bash","run_in_background":true}',
            ),
            _text("delegated"),
        ]
    )
    thread_id = "background-bash-release"
    try:
        await asyncio.wait_for(
            root.acall(
                "delegate",
                scope=ExecutionScope(thread_id=thread_id, run_id="root-run"),
            ),
            timeout=5,
        )
        deadline = asyncio.get_running_loop().time() + 5
        while not marker.exists() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert marker.exists(), "the child Bash process never reached its barrier"

        reason = root._thread_release_reason(thread_id)
        assert reason is not None
        assert "Agent call" in reason or "task" in reason
        assert root._mark_thread_idle_closing(thread_id) == reason
        assert not root._owned_threads[thread_id].closing

        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if not root._thread_release_reason(thread_id):
                break
            await asyncio.sleep(0.02)
        assert root._thread_release_reason(thread_id) is None
        assert root._mark_thread_idle_closing(thread_id) is None
    finally:
        await root.aclose()
        await child.aclose()
