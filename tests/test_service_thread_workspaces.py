"""Real tools keep per-thread working directories across frontend reconnects."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.tools.builtin import BashTool, ReadFileTool


def _text(content):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _tools(marker):
    calls = ToolCallAggregator()
    calls.process(
        0,
        "inspect",
        "read",
        json.dumps({"path": "fixture.txt", "offset": None, "limit": None}),
    )
    calls.process(
        1,
        "execute",
        "bash",
        json.dumps(
            {
                "command": f"pwd; cat fixture.txt; printf '{marker}' > generated.txt",
                "timeout_ms": 2000,
            }
        ),
    )
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _host(journal_path, storage, *, barrier=None):
    service = AgentService(store=SQLiteServiceStore(journal_path))
    created = {}

    def factory(thread):
        workspace = AgentWorkspace.local(thread.cwd)
        checkpoints = SQLiteCheckpointStore(storage / f"{thread.thread_id}.sqlite3")
        model = Mock()
        model.model_type = "chat_completion"
        agent = Agent(
            name="main",
            model=model,
            workspace=workspace,
            tools=[ReadFileTool(), BashTool()],
            checkpoint_store=checkpoints,
        )
        round_index = 0

        async def answer(**_kwargs):
            nonlocal round_index
            round_index += 1
            if round_index % 2 == 1:
                if barrier is not None:
                    barrier[thread.thread_id].set()
                    await asyncio.wait_for(
                        asyncio.gather(*(event.wait() for event in barrier.values())), 5
                    )
                return _tools(thread.thread_id)
            return _text("done")

        agent.generator.aforward = AsyncMock(side_effect=answer)
        created[thread.thread_id] = agent

        async def close():
            await workspace.aclose()
            checkpoints.close()

        return AgentSession(agent, on_close=close)

    service.register("coding", factory)
    return service, created


@pytest.mark.asyncio
async def test_two_workspace_threads_execute_real_tools_and_restore_saved_roots(
    tmp_path, monkeypatch
):
    directories = {name: tmp_path / name for name in ("documents", "downloads")}
    for name, root in directories.items():
        root.mkdir()
        (root / "fixture.txt").write_text(f"fixture-{name}")
    journal = tmp_path / "service.sqlite3"
    barrier = {name: asyncio.Event() for name in directories}
    service, agents = _host(journal, tmp_path, barrier=barrier)
    original_cwd = Path.cwd()
    receipts = {}
    try:
        for name, root in directories.items():
            binding = await service.open_thread("coding", thread_id=name, cwd=root)
            assert binding.cwd == str(root.resolve())
            receipts[name] = await service.prompt(name, "inspect", request_id="first")
        settled = await asyncio.wait_for(
            asyncio.gather(*(service.wait(name, "first") for name in directories)), 10
        )
        assert all(receipt.status == "completed" for receipt in settled)
        assert Path.cwd() == original_cwd
        for name, root in directories.items():
            assert (root / "generated.txt").read_text() == name
            snapshot = await service.snapshot(name)
            history = str(snapshot.messages.to_chatml())
            assert str(root) in history
            assert f"fixture-{name}" in history
            other_name = next(other for other in directories if other != name)
            assert f"fixture-{other_name}" not in history
            assert agents[name].generator.aforward.await_count == 2
    finally:
        await service.aclose()
        service.store.close()

    frontend = tmp_path / "third-frontend"
    frontend.mkdir()
    monkeypatch.chdir(frontend)
    restarted, new_agents = _host(journal, tmp_path)
    try:
        for name, root in directories.items():
            (root / "generated.txt").unlink()
            binding = await restarted.open_thread("coding", thread_id=name)
            assert binding.cwd == str(root.resolve())
            duplicate = await restarted.prompt(name, "inspect", request_id="first")
            assert duplicate.run_id == receipts[name].run_id
            assert duplicate.status == "completed"
            assert new_agents[name].generator.aforward.await_count == 0
            await restarted.prompt(name, "inspect again", request_id="second")
            completed = await asyncio.wait_for(restarted.wait(name, "second"), 10)
            assert completed.status == "completed", completed.error
            assert (root / "generated.txt").read_text() == name
            assert not (frontend / "generated.txt").exists()
            assert Path.cwd() == frontend
    finally:
        await restarted.aclose()
        restarted.store.close()
