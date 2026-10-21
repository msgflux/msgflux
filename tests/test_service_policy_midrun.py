"""A policy change during approval preparation takes effect after a safe pause."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.coding import CodingSession
from msgflux.nn import Agent
from msgflux.runtime.agent_run import get_agent_run
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.tools.builtin import ApplyPatchTool
from tests.test_service_policy_integration import _response


@pytest.mark.asyncio
async def test_never_during_approval_preparation_resumes_after_worker_is_quiescent(
    tmp_path,
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "note.txt").write_text("old")
    workspace = AgentWorkspace.local(project, approval_policy="on-request")
    agent = Agent(
        name="main",
        model=Mock(model_type="chat_completion"),
        workspace=workspace,
        agent_dir=tmp_path / "state",
        tools=[ApplyPatchTool()],
    )
    agent.generator.aforward = AsyncMock(
        side_effect=[_response(tool_call=True), _response()]
    )
    preparing = asyncio.Event()
    release = asyncio.Event()
    original = agent._adrain_inbox_into_messages

    async def gated_drain(*args, **kwargs):
        run = get_agent_run()
        pending = run.get_extension("pending_approvals") if run is not None else None
        if pending is not None and not pending["requests"]:
            preparing.set()
            await release.wait()
        return await original(*args, **kwargs)

    agent._adrain_inbox_into_messages = gated_drain
    session = CodingSession(agent, thread_id="midrun-thread")
    try:
        receipt = await session.prompt("update", request_id="midrun")
        await asyncio.wait_for(preparing.wait(), 3)
        await session.update_workspace_policy(approval_policy="never")
        release.set()
        # A paused receipt can be observed before the resumption acquires the
        # host lock. Watch the durable receipt until the resumed run settles.
        async with asyncio.timeout(5):
            while (await session.receipt("midrun")).status != "completed":
                await asyncio.sleep(0.01)
        assert (await session.receipt("midrun")).run_id == receipt.run_id
        assert (project / "note.txt").read_text() == "new"
        assert agent.generator.aforward.await_count == 2
    finally:
        release.set()
        await session.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_generic_service_policy_update_uses_embedded_persistence(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    agent = Agent(
        name="main",
        model=Mock(model_type="chat_completion"),
        workspace=workspace,
        agent_dir=tmp_path / "state",
    )
    agent.generator.aforward = AsyncMock(return_value=_response())
    session = CodingSession(agent, thread_id="generic-thread")
    try:
        policy = await session.service.update_workspace_policy(
            session.thread_id, permissions="read-only"
        )
        assert policy.revision == 1
        receipt = await session.service.prompt(
            session.thread_id, "hello", request_id="generic-prompt"
        )
        assert (await session.wait(receipt.request_id)).status == "completed"
        assert (await session.workspace_policy()).permissions == policy.permissions
    finally:
        await session.aclose()
        await workspace.aclose()
