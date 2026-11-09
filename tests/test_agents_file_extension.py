"""AGENTS instructions are request-local and use the authorized workspace."""

import asyncio
import json
import os
import os
from unittest.mock import Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent, AgentsFileExtension
from msgflux.nn.extensions import AgentsFileExtension as ExportedExtension
from msgflux.nn.extensions.discovery import AgentExtensionCatalog, ExtensionConfig
from msgflux.runtime import (
    AgentWorkspace,
    ExecutionScope,
    PermissionSet,
    execution_context,
)
from tests.test_workspace_prompt_extension import RecordingModel


@pytest.mark.parametrize("filename", ["AGENTS.md", "Agents.md", "agents.md"])
@pytest.mark.asyncio
async def test_variants_preserve_contents_and_base_prompt(tmp_path, filename):
    content = "# Instructions\n\nUse msgspec.\nKeep the public API small.\n"
    (tmp_path / filename).write_text(content)
    model = RecordingModel()
    workspace = AgentWorkspace.local(tmp_path, read_only=True)
    agent = Agent(
        name="reader",
        model=model,
        system_prompt="Base.",
        workspace=workspace,
        extensions=[AgentsFileExtension()],
    )
    try:
        assert await agent.acall("hello") == "ok"
        prompt = model.calls[-1]["system_prompt"]
        assert (
            prompt
            == f'Base.\n\n<agents_file name="{filename}">\n{content}\n</agents_file>'
        )
        assert agent.system_prompt.data == "Base."
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.parametrize("winner", ["AGENTS.md", "Agents.md", "agents.md"])
def test_priority_is_deterministic(tmp_path, winner):
    candidates = ("AGENTS.md", "Agents.md", "agents.md")
    for filename in candidates[candidates.index(winner) :]:
        (tmp_path / filename).write_text(f"Rules from {filename}")
    model = RecordingModel()
    workspace = AgentWorkspace.local(tmp_path, read_only=True)
    agent = Agent(
        name="reader",
        model=model,
        workspace=workspace,
        extensions=[AgentsFileExtension()],
    )
    try:
        assert agent("hello") == "ok"
        assert f"Rules from {winner}" in model.calls[-1]["system_prompt"]
        assert model.calls[-1]["system_prompt"].count("Rules from") == 1
    finally:
        asyncio.run(agent.aclose())
        asyncio.run(workspace.aclose())


@pytest.mark.parametrize("mode", ["missing", "blank", "no-workspace", "opt-out"])
@pytest.mark.asyncio
async def test_optional_file_does_not_change_unrelated_prompts(
    tmp_path, monkeypatch, mode
):
    monkeypatch.chdir(tmp_path)
    if mode != "missing":
        (tmp_path / "AGENTS.md").write_text(
            " \n" if mode == "blank" else "MUST NOT LOAD"
        )
    if mode == "blank":
        (tmp_path / "agents.md").write_text("MUST NOT LOAD")
    workspace = (
        None
        if mode == "no-workspace"
        else AgentWorkspace.local(tmp_path, read_only=True)
    )
    model = RecordingModel()
    agent = Agent(
        name="reader",
        model=model,
        workspace=workspace,
        system_prompt="Base.",
        extensions=[] if mode == "opt-out" else [AgentsFileExtension()],
    )
    try:
        assert await agent.acall("hello") == "ok"
        assert model.calls[-1]["system_prompt"] == "Base."
    finally:
        await agent.aclose()
        if workspace is not None:
            await workspace.aclose()


@pytest.mark.asyncio
async def test_reloads_on_each_request_without_persisting_or_accumulating_in_history(
    tmp_path,
):
    file = tmp_path / "AGENTS.md"
    file.write_text("FIRST_INSTRUCTION")
    workspace = AgentWorkspace.local(tmp_path, read_only=True)
    model, checkpoints = RecordingModel(), InMemoryCheckpointStore()
    agent = Agent(
        name="reader",
        model=model,
        workspace=workspace,
        system_prompt="Base.",
        checkpoint_store=checkpoints,
        extensions=[AgentsFileExtension()],
    )
    try:
        await agent.acall("one", scope=ExecutionScope(thread_id="t", run_id="one"))
        file.write_text("SECOND_INSTRUCTION")
        await agent.acall("two", scope=ExecutionScope(thread_id="t", run_id="two"))
        assert "FIRST_INSTRUCTION" in model.calls[0]["system_prompt"]
        prompt = model.calls[1]["system_prompt"]
        assert "FIRST_INSTRUCTION" not in prompt
        assert prompt.count("SECOND_INSTRUCTION") == 1
        assert prompt.count("<agents_file") == 1
        history = json.dumps(checkpoints.load_state("reader", "t", "two")["messages"])
        assert (
            "FIRST_INSTRUCTION" not in history and "SECOND_INSTRUCTION" not in history
        )
        assert agent.system_prompt.data == "Base."
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_workspace_cwd_and_concurrent_threads_do_not_use_process_directory(
    tmp_path, monkeypatch
):
    process = tmp_path / "process"
    process.mkdir()
    (process / "AGENTS.md").write_text("WRONG_PROCESS_INSTRUCTIONS")
    monkeypatch.chdir(process)
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "AGENTS.md").write_text("ROOT_INSTRUCTIONS")
    (root / "sub").mkdir()
    (root / "sub" / "agents.md").write_text("SUB_INSTRUCTIONS")
    owner = AgentWorkspace.local(root, read_only=True)
    model = RecordingModel()
    agent = Agent(name="reader", model=model, extensions=[AgentsFileExtension()])
    try:
        await asyncio.gather(
            agent.acall(
                "root", scope=ExecutionScope(thread_id="root", workspace=owner)
            ),
            agent.acall(
                "sub",
                scope=ExecutionScope(thread_id="sub", workspace=owner.with_cwd("sub")),
            ),
        )
        prompts = [call["system_prompt"] for call in model.calls]
        assert len(prompts) == 2
        assert sum("ROOT_INSTRUCTIONS" in prompt for prompt in prompts) == 1
        assert sum("SUB_INSTRUCTIONS" in prompt for prompt in prompts) == 1
        assert all("WRONG_PROCESS_INSTRUCTIONS" not in prompt for prompt in prompts)
    finally:
        await agent.aclose()
        await owner.aclose()


@pytest.mark.asyncio
async def test_warmup_and_read_only_grant_work_without_listing_permission(tmp_path):
    (tmp_path / "Agents.md").write_bytes(b"\xef\xbb\xbfWARMUP_INSTRUCTIONS")
    workspace = AgentWorkspace.local(
        tmp_path, permissions=PermissionSet(("filesystem.read",))
    )
    model = RecordingModel()
    agent = Agent(name="reader", model=model, extensions=[AgentsFileExtension()])
    try:
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            await agent.awarmup_system_prompt()
            agent.warmup_system_prompt()
        assert len(model.calls) == 2
        assert all(
            "WARMUP_INSTRUCTIONS" in call["system_prompt"] for call in model.calls
        )
        assert all("\ufeff" not in call["system_prompt"] for call in model.calls)
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.parametrize("max_bytes", [0, -1, True, 1.5, "10"])
def test_invalid_limits_are_rejected(max_bytes):
    with pytest.raises(ValueError, match="max_bytes"):
        AgentsFileExtension(max_bytes=max_bytes)


@pytest.mark.parametrize(
    "content,limit,error",
    [(b"x" * 11, 10, ValueError), (b"\xff", 10, UnicodeDecodeError)],
)
@pytest.mark.asyncio
async def test_invalid_or_oversized_file_does_not_generate(
    tmp_path, content, limit, error
):
    (tmp_path / "AGENTS.md").write_bytes(content)
    workspace = AgentWorkspace.local(tmp_path, read_only=True)
    model = RecordingModel()
    agent = Agent(
        name="reader",
        model=model,
        workspace=workspace,
        extensions=[AgentsFileExtension(max_bytes=limit)],
    )
    try:
        with pytest.raises(error):
            await agent.acall("hello")
        assert model.calls == []
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_read_permission_is_not_bypassed(tmp_path):
    (tmp_path / "AGENTS.md").write_text("PRIVATE_INSTRUCTIONS")
    workspace = AgentWorkspace.local(tmp_path, permissions=PermissionSet())
    model = RecordingModel()
    agent = Agent(
        name="reader",
        model=model,
        workspace=workspace,
        extensions=[AgentsFileExtension()],
    )
    try:
        with pytest.raises(PermissionError):
            await agent.acall("hello")
        assert not model.calls
    finally:
        await agent.aclose()
        await workspace.aclose()


def test_exports_and_catalog_configuration():
    assert AgentsFileExtension is ExportedExtension
    catalog = AgentExtensionCatalog(
        {
            "project_rules": ExtensionConfig(
                factory="msgflux.nn.extensions:AgentsFileExtension",
                options={"max_bytes": 1000},
            )
        }
    )
    catalog.validate(["project_rules"])
    extension = catalog.create("project_rules")
    assert extension.name == "project_rules"
    assert extension.max_bytes == 1000


@pytest.mark.asyncio
async def test_does_not_search_parent_or_nested_directories(tmp_path):
    (tmp_path / "AGENTS.md").write_text("PARENT_INSTRUCTIONS")
    (tmp_path / "sub" / "nested").mkdir(parents=True)
    (tmp_path / "sub" / "nested" / "AGENTS.md").write_text("NESTED_INSTRUCTIONS")
    owner = AgentWorkspace.local(tmp_path, read_only=True)
    model = RecordingModel()
    agent = Agent(
        name="reader",
        model=model,
        workspace=owner.with_cwd("sub"),
        system_prompt="Base.",
        extensions=[AgentsFileExtension()],
    )
    try:
        await agent.acall("hello")
        assert model.calls[-1]["system_prompt"] == "Base."
    finally:
        await agent.aclose()
        await owner.aclose()


@pytest.mark.parametrize("kind", ["symlink", "directory"])
@pytest.mark.asyncio
async def test_backend_file_constraints_are_preserved(tmp_path, kind):
    target = tmp_path / "AGENTS.md"
    if kind == "directory":
        target.mkdir()
    else:
        outside = tmp_path.parent / (tmp_path.name + "-external.md")
        outside.write_text("OUTSIDE_INSTRUCTIONS")
        target.symlink_to(outside)
    workspace = AgentWorkspace.local(tmp_path, read_only=True)
    model = RecordingModel()
    agent = Agent(
        name="reader",
        model=model,
        workspace=workspace,
        extensions=[AgentsFileExtension()],
    )
    try:
        with pytest.raises((PermissionError, IsADirectoryError, OSError)):
            await agent.acall("hello")
        assert model.calls == []
    finally:
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires FIFO support")
def test_fifo_does_not_block_the_model_request(tmp_path):
    import subprocess
    import sys

    os.mkfifo(tmp_path / "AGENTS.md")
    code = """
import asyncio, sys
from msgflux.nn import Agent, AgentsFileExtension
from msgflux.runtime import AgentWorkspace
from tests.test_workspace_prompt_extension import RecordingModel
async def main():
    model = RecordingModel()
    workspace = AgentWorkspace.local(sys.argv[1], read_only=True)
    agent = Agent(name='reader', model=model, workspace=workspace, extensions=[AgentsFileExtension()])
    try:
        try:
            await agent.acall('hello')
        except PermissionError:
            assert not model.calls
        else:
            raise AssertionError('FIFO must not be read')
    finally:
        await agent.aclose()
        await workspace.aclose()
asyncio.run(main())
"""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and local regression code
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_in_memory_workspace_uses_the_same_extension_without_host_file_access():
    from msgflux.runtime import ExecutionEnvironment, InMemoryWorkspace

    filesystem = InMemoryWorkspace(
        "virtual", {"/project/agents.md": b"VIRTUAL_INSTRUCTIONS"}
    )
    workspace = AgentWorkspace.from_environment(
        ExecutionEnvironment(filesystem),
        cwd="/project",
        permissions=PermissionSet(("filesystem.read",)),
    )
    model = RecordingModel()
    agent = Agent(
        name="reader",
        model=model,
        workspace=workspace,
        extensions=[AgentsFileExtension()],
    )
    try:
        await agent.acall("hello")
        assert "VIRTUAL_INSTRUCTIONS" in model.calls[-1]["system_prompt"]
    finally:
        await agent.aclose()
        await workspace.aclose()
