import os
from contextlib import contextmanager

import pytest

from msgflux.nn.modules.agent.resources import (
    _CURRENT_RESOURCES,
    _OwnedThread,
    _get_tool_result_store,
)
from msgflux.runtime.agent_resources import AgentResources
from msgflux.runtime import (
    AgentWorkspace,
    InMemoryWorkspace,
    DockerWorkspaceBackend,
    ExecutionEnvironment,
    ExecutionScope,
    PermissionSet,
    ToolOutputOffloadConfig,
    execution_context,
)
from msgflux.tools.builtin.workspace_tools import ReadFileTool


@contextmanager
def _active_thread(tmp_path, thread_id="thread-a"):
    resources = AgentResources(tmp_path / "agent")
    bound = resources.bind(thread_id, namespace="test-agent")
    owned = _OwnedThread(bound)
    token = _CURRENT_RESOURCES.set(owned)
    try:
        yield bound
    finally:
        _CURRENT_RESOURCES.reset(token)
        bound.close()


def _workspace(tmp_path, *, read=True):
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    permissions = PermissionSet({"filesystem.read"} if read else set())
    workspace = AgentWorkspace.local(project, read_only=True, permissions=permissions)
    scope = ExecutionScope(
        thread_id="thread-a",
        run_id="run-a",
        workspace=workspace,
        permissions=permissions,
    )
    return workspace, scope


def _write(store, text):
    return store.put([text.encode("utf-8")], media_type="text/plain; charset=utf-8")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_read_tool_reads_active_thread_result_with_line_paging(
    tmp_path, asynchronous
):
    workspace, scope = _workspace(tmp_path)
    read = ReadFileTool()

    with _active_thread(tmp_path) as bound:
        assert not (bound.thread_dir / "tool-results").exists()
        assert _get_tool_result_store(create=False) is None
        store = _get_tool_result_store(create=True, config=ToolOutputOffloadConfig())
        ref = _write(store, "alpha\nbeta\ngamma\n")
        path = str(store.root / ref.result_id / "content")
        with execution_context(scope=scope):
            if asynchronous:
                assert (
                    await read.acall(path, offset=2, limit=1, workspace=workspace)
                    == "beta\n"
                )
            else:
                assert read(path, offset=2, limit=1, workspace=workspace) == "beta\n"


@pytest.mark.parametrize(
    ("content", "offset", "limit", "expected"),
    [
        ("one\ntwo\nthree", 1, 2, "one\ntwo\n"),
        ("one\ntwo\nthree", 3, 2, "three"),
        ("", 1, 1, ""),
        ("one\n", 2, 1, None),
    ],
)
def test_result_read_line_semantics(tmp_path, content, offset, limit, expected):
    workspace, scope = _workspace(tmp_path)
    read = ReadFileTool()

    with _active_thread(tmp_path) as bound:
        store = _get_tool_result_store(create=True)
        ref = _write(store, content)
        path = str(store.root / ref.result_id / "content")
        with execution_context(scope=scope):
            if expected is None:
                with pytest.raises(ValueError, match="offset exceeds"):
                    read(path, offset=offset, limit=limit, workspace=workspace)
            else:
                assert (
                    read(path, offset=offset, limit=limit, workspace=workspace)
                    == expected
                )


def test_result_read_requires_live_filesystem_read_permission(tmp_path):
    workspace, scope = _workspace(tmp_path, read=False)
    read = ReadFileTool()

    with _active_thread(tmp_path):
        store = _get_tool_result_store(create=True)
        ref = _write(store, "secret\n")
        path = str(store.root / ref.result_id / "content")
        with execution_context(scope=scope), pytest.raises(PermissionError):
            read(path, workspace=workspace)


def test_regular_workspace_read_falls_through_without_managed_store(tmp_path):
    workspace, scope = _workspace(tmp_path)
    (tmp_path / "project" / "source.txt").write_text("ordinary file\n")
    with execution_context(scope=scope):
        assert ReadFileTool()("source.txt", workspace=workspace) == "ordinary file\n"


def test_result_read_only_routes_active_store_content(tmp_path):
    workspace, scope = _workspace(tmp_path)
    other_resources = AgentResources(tmp_path / "other-agent")
    other_bound = other_resources.bind("thread-b", namespace="other-agent")
    read = ReadFileTool()

    try:
        with _active_thread(tmp_path) as bound:
            store = _get_tool_result_store(create=True)
            other_store = other_bound.tool_result_store(create=True)
            ref = _write(store, "own\n")
            other_ref = _write(other_store, "other\n")
            own_path = str(store.root / ref.result_id / "content")
            other_path = str(other_store.root / other_ref.result_id / "content")
            with execution_context(scope=scope):
                assert read(own_path, workspace=workspace) == "own\n"
                with pytest.raises((FileNotFoundError, PermissionError)):
                    read(other_path, workspace=workspace)
    finally:
        other_bound.close()


def test_result_store_reopens_for_same_agent_thread(tmp_path):
    with _active_thread(tmp_path) as bound:
        store = _get_tool_result_store(create=True)
        ref = _write(store, "durable\n")
        root = store.root
        path = root / ref.result_id / "content"

    with _active_thread(tmp_path) as reopened:
        assert reopened.thread_dir / "tool-results" == root
        store = _get_tool_result_store(create=False)
        assert store.get(ref.result_id) == ref
        assert path.read_bytes() == b"durable\n"


def test_result_read_integrity_and_single_long_line_limit(tmp_path):
    workspace, scope = _workspace(tmp_path)
    read = ReadFileTool()

    with _active_thread(tmp_path):
        store = _get_tool_result_store(create=True)
        ref = _write(store, "valid\n")
        path = store.root / ref.result_id / "content"
        path.write_bytes(b"evil!\n")
        with execution_context(scope=scope), pytest.raises(ValueError):
            read(str(path), workspace=workspace)

        large_ref = _write(store, "x" * 1_000_001)
        large_path = str(store.root / large_ref.result_id / "content")
        with (
            execution_context(scope=scope),
            pytest.raises(ValueError, match="byte limit"),
        ):
            read(large_path, workspace=workspace)


@pytest.mark.asyncio
async def test_offloaded_path_reads_through_docker_workspace(tmp_path):
    if os.environ.get("MSGFLUX_TEST_DOCKER") != "1":
        pytest.skip("Set MSGFLUX_TEST_DOCKER=1 for real isolated process tests")
    root = tmp_path / "project"
    root.mkdir()
    (root / "project.txt").write_text("ordinary workspace file\n")
    (root / "large.txt").write_text("row\n" * 100)
    backend = DockerWorkspaceBackend(root, image="python:3.12-slim")
    async with await backend.open("read-results") as binding:
        environment = ExecutionEnvironment.from_binding(
            binding, write_guarantee="cooperative_compare"
        )
        permissions = PermissionSet({"filesystem.read"})
        workspace = AgentWorkspace.from_environment(
            environment, permissions=permissions
        )
        scope = ExecutionScope(
            thread_id="thread-a",
            run_id="run-a",
            workspace=workspace,
            permissions=permissions,
        )
        with _active_thread(tmp_path):
            store = _get_tool_result_store(create=True)
            ref = _write(store, "host artifact\nsecond line\n")
            artifact_path = str(store.root / ref.result_id / "content")
            with execution_context(scope=scope):
                read = ReadFileTool()
                assert (
                    await read.acall(
                        artifact_path, offset=2, limit=1, workspace=workspace
                    )
                    == "second line\n"
                )
                assert (
                    await read.acall("project.txt", workspace=workspace)
                    == "ordinary workspace file\n"
                )
                bounded = ReadFileTool(max_text_bytes=8)
                with pytest.raises(ValueError, match="Bash"):
                    await bounded.acall(artifact_path, workspace=workspace)
                with pytest.raises(ValueError, match="smaller limit"):
                    await bounded.acall("large.txt", workspace=workspace)
                assert (
                    await bounded.acall("large.txt", limit=1, workspace=workspace)
                    == "row\n"
                )


@pytest.mark.parametrize("artifact", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_large_read_rejects_without_content_and_can_be_paged(
    tmp_path, artifact, asynchronous
):
    workspace, scope = _workspace(tmp_path)
    read = ReadFileTool(max_text_bytes=24)
    content = "á\n" * 10  # 30 UTF-8 bytes, despite only 20 characters.
    with _active_thread(tmp_path):
        if artifact:
            store = _get_tool_result_store()
            ref = _write(store, content)
            path = str(store.root / ref.result_id / "content")
            before = set(store.root.iterdir())
        else:
            path = "large.txt"
            (tmp_path / "project" / path).write_text(content)
        with execution_context(scope=scope):
            with pytest.raises(ValueError) as rejected:
                if asynchronous:
                    await read.acall(path, workspace=workspace)
                else:
                    read(path, workspace=workspace)
            message = str(rejected.value)
            assert "24 bytes" in message
            assert "No content was returned" in message
            assert "offset" in message and "smaller limit" in message
            assert "Bash" not in message and "single line" not in message
            if asynchronous:
                page = await read.acall(path, offset=2, limit=4, workspace=workspace)
            else:
                page = read(path, offset=2, limit=4, workspace=workspace)
            assert page == "á\n" * 4
        if artifact:
            assert set(store.root.iterdir()) == before


@pytest.mark.parametrize("artifact", [False, True])
@pytest.mark.parametrize("prefix", ["", "ok\n"])
def test_single_long_line_recommends_another_tool(tmp_path, artifact, prefix):
    workspace, scope = _workspace(tmp_path)
    read = ReadFileTool(max_text_bytes=8)
    with _active_thread(tmp_path):
        if artifact:
            store = _get_tool_result_store()
            ref = _write(store, prefix + "🌍" * 4)
            path = str(store.root / ref.result_id / "content")
        else:
            path = "long.txt"
            (tmp_path / "project" / path).write_text(prefix + "🌍" * 4)
        with execution_context(scope=scope), pytest.raises(ValueError) as rejected:
            read(path, limit=2, workspace=workspace)
        assert "single line" in str(rejected.value)
        assert "read cannot retrieve it" in str(rejected.value)
        assert "Bash" in str(rejected.value)
        assert "paginate" not in str(rejected.value)
        assert "smaller limit" not in str(rejected.value)
        assert "🌍" not in str(rejected.value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_text_read_budget_requires_positive_integer(value):
    with pytest.raises(ValueError, match="max_text_bytes"):
        ReadFileTool(max_text_bytes=value)


@pytest.mark.parametrize(
    "content,single_line", [("row\n" * 4, False), ("ok\n" + "x" * 13, True)]
)
def test_memory_workspace_distinguishes_page_and_single_line_overflow(
    content, single_line
):
    filesystem = InMemoryWorkspace("budget", {"/text": content.encode()})
    workspace = AgentWorkspace.from_environment(
        ExecutionEnvironment(filesystem), permissions=PermissionSet({"filesystem.read"})
    )
    with (
        execution_context(scope=ExecutionScope(workspace=workspace)),
        pytest.raises(ValueError) as rejected,
    ):
        ReadFileTool(max_text_bytes=12)("/text", workspace=workspace)
    message = str(rejected.value)
    assert ("Bash" in message) is single_line
    assert ("smaller limit" in message) is not single_line
    assert ("single line" in message) is single_line
