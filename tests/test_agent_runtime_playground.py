"""Regression tests for the offline Agent runtime playground harness."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_agent_runtime.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("validate_agent_runtime", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_approved_full_access_run_reports_effects_events_and_image(harness):
    summary = asyncio.run(harness.run_demo())

    assert summary.runs == 1
    assert summary.approved_changes == 3
    assert summary.denied_changes == 0
    assert summary.image_notifications == 1
    assert summary.stream_events > 0
    assert summary.tools_checked == ("read", "write", "edit", "apply_patch", "bash")
    assert {"message.delta", "tool.start", "tool.approval_required"} <= set(
        summary.event_types
    )
    assert summary.image_provenance is True


def test_denied_approval_keeps_denied_write_but_resumes_other_tools(harness):
    summary = asyncio.run(harness.run_demo(deny=True))

    assert summary.runs == 1
    assert summary.approved_changes == 0
    assert summary.denied_changes == 3
    assert summary.image_notifications == 1
    assert summary.stream_events > 0
    assert summary.tools_checked == ("read", "write", "edit", "apply_patch", "bash")
    assert summary.image_provenance is True


def test_resource_denial_is_checked_independently_of_approval(harness):
    assert asyncio.run(harness.run_permission_denial()) is True


@pytest.mark.parametrize("profile", ["baseline", "workspace", "combined"])
@pytest.mark.parametrize("deny", [False, True])
def test_extension_profiles_resume_without_prompt_accumulation(harness, profile, deny):
    summary = asyncio.run(harness.run_demo(profile=profile, deny=deny))
    assert summary.approved_changes == (0 if deny else 3)
    assert summary.denied_changes == (3 if deny else 0)
    assert summary.image_provenance
    assert "run.end" in summary.event_types


def test_unknown_extension_profile_fails_before_running(harness):
    with pytest.raises(ValueError, match="Unknown extension profile"):
        asyncio.run(harness.run_demo(profile="typo"))


@pytest.mark.asyncio
async def test_slow_bounded_watcher_does_not_abort_agent_and_can_reconnect(harness):
    from msgflux.exceptions import EventBufferOverflowError

    store = harness.InMemoryCheckpointStore()
    agent = harness.Agent(
        name="bounded",
        model=harness.ScriptedModel([harness._text("done", streamed=True)]),
        checkpoint_store=store,
        config={"stream": True},
    )
    scope = harness.ExecutionScope(namespace="bounded", thread_id="t", run_id="r")
    async with agent.watch("t", event_buffer_limit=1) as slow:
        events = [event async for event in agent.stream_events("go", scope=scope)]
        assert events[-1].type == "run.end"
        with pytest.raises(EventBufferOverflowError):
            await anext(slow)
    assert store.load_state("bounded", "t", "r")["status"] == "completed"
    async with agent.watch("t", event_buffer_limit=10) as fresh:
        assert fresh.snapshot.messages.to_chatml()[-1]["content"] == "done"


@pytest.mark.asyncio
async def test_direct_buffer_overflow_interrupts_agent_and_settles_checkpoint(harness):
    from msgflux.exceptions import EventBufferOverflowError
    from msgflux.runtime.events import emit_event, EventType

    cleaned = asyncio.Event()

    class BurstingModel(harness.ScriptedModel):
        async def acall(self, **kwargs):
            try:
                for index in range(50):
                    emit_event(EventType.TOOL_UPDATE, {"index": index})
                await asyncio.Event().wait()
            finally:
                cleaned.set()

    store = harness.InMemoryCheckpointStore()
    agent = harness.Agent(
        name="bounded", model=BurstingModel([]), checkpoint_store=store
    )
    scope = harness.ExecutionScope(namespace="bounded", thread_id="t", run_id="r")
    with pytest.raises(EventBufferOverflowError):
        async with asyncio.timeout(5):
            async for _ in agent.stream_events("go", scope=scope, event_buffer_limit=8):
                pass
    assert cleaned.is_set()
    assert store.load_state("bounded", "t", "r")["status"] == "interrupted"


@pytest.mark.asyncio
@pytest.mark.parametrize("approved_thread", ["left", "right"])
async def test_concurrent_workspace_runs_isolate_approvals_and_effects(
    harness, approved_thread
):
    from msgflux.runtime import get_execution_scope

    barrier = asyncio.Barrier(2)

    class ConcurrentModel:
        model_type = "chat_completion"

        def __init__(self):
            self.prompts = {}

        async def acall(self, **kwargs):
            thread = get_execution_scope().thread_id
            assert thread not in self.prompts  # One round per independent budget.
            self.prompts[thread] = kwargs["system_prompt"]
            await barrier.wait()
            return harness._tool(
                "write",
                {"path": "/note.txt", "content": f"updated-{thread}"},
                "shared-call-id",
            )

    model = ConcurrentModel()
    store, journal = harness.InMemoryCheckpointStore(), harness.InMemoryApprovalStore()
    agent = harness.Agent(
        name="offline-demo",
        model=model,
        tools=[harness.WriteTool()],
        checkpoint_store=store,
        approvals=harness.AgentApprovals(journal, {"write": "v1"}, "concurrent-v1"),
        extensions=[
            harness.WorkspacePromptExtension(),
            harness.ToolTurnLimitExtension(1),
        ],
        config={"stream": True},
    )
    scopes, filesystems, previews = {}, {}, {}
    for thread in ("left", "right"):
        fs = harness.InMemoryWorkspace(
            thread, {"/note.txt": f"original-{thread}".encode()}
        )
        filesystems[thread] = fs
        scopes[thread] = replace(
            harness._scope(fs, thread=thread, run="shared-run"),
            principal=thread,
            permissions=harness.PermissionSet(
                resources=[
                    fs.permission("/note.txt", "filesystem.read"),
                    fs.permission("/note.txt", "filesystem.write"),
                    fs.permission(f"/{thread}-only", "filesystem.read"),
                ]
            ),
        )

    async def run(thread):
        def decide(record, preview):
            assert record.binding.thread_id == thread
            assert record.binding.principal == thread
            assert preview.before == f"original-{thread}"
            assert preview.after == f"updated-{thread}"
            previews[thread] = record.request_id
            return thread == approved_thread

        return await harness.drive_agent(agent, scopes[thread], decider=decide)

    tasks = [asyncio.create_task(run(thread)) for thread in ("left", "right")]
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        assert len(set(previews.values())) == 2
        for thread, events in zip(("left", "right"), results, strict=True):
            other = "right" if thread == "left" else "left"
            assert f"/{thread}-only" in model.prompts[thread]
            assert f"/{other}-only" not in model.prompts[thread]
            assert sum(e.type == "tool.approval_required" for e in events) == 1
            assert sum(e.type == "run.end" for e in events) == 1
            with harness.execution_context(scope=scopes[thread]):
                expected = (
                    f"updated-{thread}"
                    if thread == approved_thread
                    else f"original-{thread}"
                )
                assert filesystems[thread].read_text("/note.txt") == expected
            state = store.load_state("offline-demo", thread, "shared-run")
            assert state["status"] == "completed"
            assert f"updated-{other}" not in str(state["messages"])
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("deny", [False, True])
async def test_compaction_scope_and_approval_preserve_budget(harness, deny):
    from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction
    from msgflux.nn.extensions import CompactionExtension, CompactionPolicy
    from msgflux.tools.builtin import close_context_scope, open_context_scope

    class CompactingModel(harness.ScriptedModel):
        def __init__(self, responses):
            super().__init__(responses)
            self.compacted = []

        async def acount_context_tokens(self, **kwargs):
            return ContextTokenEstimate(input_tokens=100, source="heuristic")

        async def acompact_context(self, messages, **kwargs):
            self.compacted.append(deepcopy(messages))
            return ModelCompaction(
                format="messages",
                items=[{"role": "user", "content": "Summary of prior work"}],
            )

    model = CompactingModel(
        [
            harness._tool(
                "open_context_scope", {"name": "work", "summary": "Start work"}
            ),
            harness._tool("write", {"path": "/note.txt", "content": "changed"}),
            harness._tool("close_context_scope", {"summary": "Work finished"}),
        ]
    )
    store = harness.InMemoryCheckpointStore()
    journal = harness.InMemoryApprovalStore()
    fs = harness.InMemoryWorkspace("compact", {"/note.txt": b"original"})
    scope = harness._scope(fs, thread="t", run="r")
    agent = harness.Agent(
        name="offline-demo",
        model=model,
        tools=[open_context_scope, harness.WriteTool(), close_context_scope],
        extensions=[
            harness.WorkspacePromptExtension(),
            harness.ToolTurnLimitExtension(3),
            CompactionExtension(
                CompactionPolicy(
                    context_capacity=100,
                    reserved_output_tokens=0,
                    safety_margin_tokens=0,
                )
            ),
        ],
        approvals=harness.AgentApprovals(journal, {"write": "v1"}, "compact-v1"),
        checkpoint_store=store,
        config={"stream": True},
    )
    messages = harness.ChatMessages()
    messages.begin_turn(turn_id="prior")
    messages.add_user("Prior work details")
    messages.add_assistant("Prior answer")
    messages.end_turn()
    events = await harness.drive_agent(
        agent, scope, decider=lambda *_: not deny, messages=messages
    )
    assert model.calls == 3
    assert len(model.compacted) == 1
    assert all("function_call" not in str(context) for context in model.compacted)
    types = [event.type for event in events]
    assert types.count("compaction.start") == types.count("compaction.end") == 1
    assert types.index("compaction.end") < types.index("tool.approval_required")
    assert types.count("tool.approval_required") == 1
    assert types.count("run.end") == 1
    state = store.load_state(scope.namespace, "t", "r")
    assert state["status"] == "completed"
    assert state["runtime"]["branch_id"] == "root"
    branches = state["messages"]["metadata"]["runtime"]["context_scopes"]["branches"]
    assert branches["work"]["closed"]
    assert "Work finished" in str(state["messages"]["items"])
    assert "Prior work details" in str(state["messages"])
    assert "Tool budget: 1 round(s) remaining" in model.prompts[-1]
    with harness.execution_context(scope=scope):
        assert fs.read_text("/note.txt") == ("original" if deny else "changed")


@pytest.mark.asyncio
@pytest.mark.parametrize("deny", [False, True])
@pytest.mark.parametrize("width", [1, 7])
async def test_artifact_projection_after_approval_preserves_next_turn(
    harness, deny, width
):
    from msgflux.nn import ArtifactExtension, ArtifactRegistry

    registry = ArtifactRegistry()
    registry.register("Expanded report 🐍", artifact_id="report")
    canonical = (
        r"Result: {{artifact:report}} / {{artifact:missing}} / \{{artifact:report}}"
    )
    expected = "Result: Expanded report 🐍 / {{artifact:missing}} / {{artifact:report}}"
    response = harness.ModelStreamResponse(mode="async")
    response.set_response_type("text_generation")
    for offset in range(0, len(canonical), width):
        response.add(canonical[offset : offset + width])
    response.finish()
    model = harness.ScriptedModel(
        [
            harness._tool("write", {"path": "/note.txt", "content": "changed"}),
            response,
            harness._text("next turn", streamed=True),
        ]
    )
    store = harness.InMemoryCheckpointStore()
    journal = harness.InMemoryApprovalStore()
    fs = harness.InMemoryWorkspace("artifacts", {"/note.txt": b"original"})
    scope = harness._scope(fs, thread="t", run="r")
    agent = harness.Agent(
        name="offline-demo",
        model=model,
        tools=[harness.WriteTool()],
        extensions=[ArtifactExtension(registry), harness.WorkspacePromptExtension()],
        approvals=harness.AgentApprovals(journal, {"write": "v1"}, "artifact-v1"),
        checkpoint_store=store,
        config={"stream": True},
    )
    events = await harness.drive_agent(agent, scope, decider=lambda *_: not deny)
    assert sum(e.type == "tool.approval_required" for e in events) == 1
    assert (
        "".join(e.data["delta"] for e in events if e.type == "message.delta")
        == expected
    )
    assert (
        next(e for e in events if e.type == "message.end").data["content"] == expected
    )
    state = store.load_state(scope.namespace, "t", "r")
    assert any(item.get("content") == canonical for item in state["messages"]["items"])
    assert "Expanded report" not in str(state["messages"])
    with harness.execution_context(scope=scope):
        assert fs.read_text("/note.txt") == ("original" if deny else "changed")
    following = [
        event
        async for event in agent.stream_events(
            "continue",
            scope=replace(scope, run_id="next"),
            messages=harness.ChatMessages(),
        )
    ]
    assert model.calls == 3
    assert any(item.get("content") == canonical for item in model.inputs[-1])
    assert "Expanded report" not in str(model.inputs[-1])
    assert (
        "".join(e.data["delta"] for e in following if e.type == "message.delta")
        == "next turn"
    )
    assert model.prompts[-1].count("<workspace_context>") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("image", [False, True])
async def test_inbox_delivery_survives_checkpoint_outage(harness, image):
    from msgflux.utils.chat import ChatBlock

    class UnavailableStore(harness.InMemoryCheckpointStore):
        unavailable = True
        rejected = 0

        def commit_state(self, namespace, thread_id, run_id, state, **kwargs):
            if self.unavailable and state["messages"]["metadata"].get("inbox_receipts"):
                self.rejected += 1
                raise OSError("checkpoint outage after inbox delivery")
            return super().commit_state(namespace, thread_id, run_id, state, **kwargs)

    store = UnavailableStore()
    inbox = harness.AgentInbox(store=harness.InMemoryAgentInboxStore())
    model = harness.ScriptedModel(
        [
            harness._text("uncommitted", streamed=True),
            harness._text("delivered", streamed=True),
        ]
    )
    agent = harness.Agent(
        name="offline-demo",
        model=model,
        agent_inbox=inbox,
        checkpoint_store=store,
        extensions=[harness.WorkspacePromptExtension()],
        config={"stream": True},
    )
    fs = harness.InMemoryWorkspace("inbox", {})
    scope = harness._scope(fs, thread="t", run="r")
    scoped = agent._get_scoped_agent_inbox(scope)
    content = (
        [ChatBlock.image("data:image/png;base64,AA==")] if image else "retained message"
    )
    notification = scoped.user_message(content)
    with pytest.raises(OSError, match="checkpoint outage"):
        async for _ in agent.stream_events("receive", scope=scope):
            pass
    assert store.rejected > 0
    assert [item.notification_id for item in scoped.peek()] == [
        notification.notification_id
    ]
    store.unavailable = False
    events = [event async for event in agent.stream_events("receive", scope=scope)]
    state = store.load_state(scope.namespace, "t", "r")
    assert state["status"] == "completed"
    delivered = [
        item
        for item in state["messages"]["items"]
        if notification.notification_id
        in item.get("metadata", {}).get("inbox_receipts", [])
    ]
    assert len(delivered) == 1
    assert ("image_url" in str(delivered[0])) == image
    assert scoped.peek() == []
    assert model.calls == 2
    assert "uncommitted" not in str(state["messages"])
    assert "delivered" in str(state["messages"])
    assert sum(event.type == "run.end" for event in events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["permissions", "principal", "inbox", "file"])
async def test_approved_write_cannot_bypass_changed_scope_on_resume(harness, change):
    from msgflux.exceptions import TaskInterruptRequestedError, TaskPauseRequestedError

    fs = harness.InMemoryWorkspace("resume", {"/note.txt": b"original"})
    journal = harness.InMemoryApprovalStore()
    checkpoints = harness.InMemoryCheckpointStore()
    model = harness.ScriptedModel(
        [
            harness._tool("write", {"path": "/note.txt", "content": "changed"}),
        ]
    )
    agent = harness.Agent(
        name="offline-demo",
        model=model,
        tools=[harness.WriteTool()],
        extensions=[
            harness.WorkspacePromptExtension(),
            harness.ToolTurnLimitExtension(1),
        ],
        approvals=harness.AgentApprovals(journal, {"write": "v1"}, "resume-v1"),
        checkpoint_store=checkpoints,
        config={"stream": True},
    )
    scope = harness._scope(fs, thread="t", run="r")
    events = []
    with pytest.raises(TaskPauseRequestedError):
        async for event in agent.stream_events("write", scope=scope):
            events.append(event)
    (record,) = journal.pending(scope.namespace, "t", "r")
    agent.decide_approval(record.request_id, approved=True, decided_by="host")
    resumed = scope
    if change == "permissions":
        resumed = replace(scope, permissions=harness.PermissionSet())
    elif change == "principal":
        resumed = replace(scope, principal="another-user")
    elif change == "file":
        with harness.execution_context(scope=scope):
            fs.write_text("/note.txt", "host edit")
    else:
        agent._get_scoped_agent_inbox(scope).interrupt(reason="operator cancelled")
    expected = (
        TaskInterruptRequestedError if change == "inbox" else TaskPauseRequestedError
    )
    with pytest.raises(expected):
        async for event in agent.stream_events(None, scope=resumed):
            events.append(event)
    assert model.calls == 1
    assert journal.get(scope.namespace, record.request_id).status == "approved"
    with harness.execution_context(scope=scope):
        assert fs.read_text("/note.txt") == (
            "host edit" if change == "file" else "original"
        )
    assert not any(event.type == "tool.start" for event in events)
    assert not any(event.type == "run.end" for event in events)
    state = checkpoints.load_state(scope.namespace, "t", "r")
    assert state["status"] == ("interrupted" if change == "inbox" else "paused")


@pytest.mark.asyncio
@pytest.mark.parametrize("deny", [False, True])
@pytest.mark.parametrize("local", [False, True])
async def test_terminal_budget_survives_approval_resume(harness, tmp_path, deny, local):
    from msgflux.runtime import LocalWorkspaceBackend

    binding = None
    if local:
        (tmp_path / "note.txt").write_text("original")
        binding = await LocalWorkspaceBackend(tmp_path).open("terminal")
        fs = binding.filesystem
        environment = harness.ExecutionEnvironment.from_binding(
            binding, write_guarantee="cooperative_compare"
        )
    else:
        fs = harness.InMemoryWorkspace("terminal", {"/note.txt": b"original"})
        environment = harness.ExecutionEnvironment(fs)
    try:
        journal = harness.InMemoryApprovalStore()
        checkpoints = harness.InMemoryCheckpointStore()
        model = harness.ScriptedModel(
            [
                harness._tool("write", {"path": "/note.txt", "content": "changed"}),
            ]
        )
        agent = harness.Agent(
            name="terminal",
            model=model,
            tools=[harness.WriteTool()],
            extensions=[
                harness.WorkspacePromptExtension(),
                harness.ToolTurnLimitExtension(1),
            ],
            approvals=harness.AgentApprovals(journal, {"write": "v1"}, "terminal-v1"),
            checkpoint_store=checkpoints,
            config={"stream": True},
        )
        scope = harness.ExecutionScope(
            namespace="terminal",
            thread_id="t",
            run_id="r",
            principal="host",
            environment=environment,
            permissions=harness.PermissionSet(
                resources=[
                    fs.permission("/note.txt", action)
                    for action in ("filesystem.read", "filesystem.write")
                ]
            ),
        )
        previews = []

        def decide(record, preview):
            previews.append(preview)
            assert preview.before == "original" and preview.after == "changed"
            assert "-original" in preview.diff and "+changed" in preview.diff
            return not deny

        events = await harness.drive_agent(agent, scope, decider=decide)
        assert len(previews) == 1
        assert model.calls == 1  # Resume must not issue a final model request.
        assert "Tool budget: 1 round(s) remaining" in model.prompts[0]
        assert model.prompts[0].count("<workspace_context>") == 1
        types = [event.type for event in events]
        assert types.count("tool.approval_required") == 1
        assert types.count("run.end") == 1
        assert "run.paused" in types
        state = checkpoints.load_state("terminal", "t", "r")
        assert state["status"] == "completed"
        with harness.execution_context(scope=scope):
            assert fs.read_text("/note.txt") == ("original" if deny else "changed")
        if local:
            assert (tmp_path / "note.txt").read_text() == (
                "original" if deny else "changed"
            )
    finally:
        if binding is not None:
            await binding.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["model", "tool"])
@pytest.mark.parametrize("same_thread", [False, True])
async def test_external_abort_cleans_pending_operation_and_checkpoints(
    harness, phase, same_thread
):
    from msgflux.exceptions import TaskInterruptRequestedError
    from msgflux.runtime import AbortSignal

    started, cleaned = asyncio.Event(), asyncio.Event()
    effects = []
    recovering = False

    async def pending() -> str:
        """Wait for an external operation."""
        if recovering:
            effects.append("completed")
            return "done"
        started.set()
        try:
            await asyncio.Event().wait()
            effects.append("completed")
            return "done"
        finally:
            cleaned.set()

    class BlockingModel(harness.ScriptedModel):
        async def acall(self, **kwargs):
            if phase == "model" and not recovering:
                self.calls += 1
                return await pending()
            return await super().acall(**kwargs)

    model = BlockingModel([harness._tool("pending", {})])
    checkpoints = harness.InMemoryCheckpointStore()
    signal = AbortSignal()
    fs = harness.InMemoryWorkspace("abort", {})
    scope = replace(harness._scope(fs, thread="t", run="r"), abort_signal=signal)
    agent = harness.Agent(
        name="offline-demo",
        model=model,
        tools=[pending],
        extensions=[
            harness.WorkspacePromptExtension(),
            harness.ToolTurnLimitExtension(1),
        ],
        checkpoint_store=checkpoints,
        config={"stream": True},
    )
    events = []

    async def consume():
        async for event in agent.stream_events("start", scope=scope):
            events.append(event)

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        signal.abort("operator stopped run")
        with pytest.raises(TaskInterruptRequestedError, match="operator stopped run"):
            await asyncio.wait_for(task, timeout=5)
        assert cleaned.is_set()
        assert effects == []
        assert model.calls == 1
        state = checkpoints.load_state(scope.namespace, "t", "r")
        assert state["status"] == "interrupted"
        assert not any(event.type == "run.end" for event in events)
        assert any(event.type == "tool.start" for event in events) == (phase == "tool")

        # A fresh run is not a replay of the interrupted operation.
        recovering = True
        if phase == "tool":
            model.responses.append(harness._tool("pending", {}, "recovery-call"))
        fresh = replace(
            scope,
            run_id="recovery",
            thread_id="t" if same_thread else "new-thread",
            abort_signal=AbortSignal(),
        )
        recovered = [
            event async for event in agent.stream_events("try again", scope=fresh)
        ]
        assert effects == ["completed"]
        assert model.calls == 2
        assert "Tool budget: 1 round(s) remaining" in model.prompts[-1]
        assert model.prompts[-1].count("<workspace_context>") == 1
        assert signal.aborted and not fresh.abort_signal.aborted
        assert sum(event.type == "tool.start" for event in recovered) == 1
        assert sum(event.type == "run.end" for event in recovered) == 1
        assert checkpoints.load_state(scope.namespace, "t", "r") == state
        assert (
            checkpoints.load_state(fresh.namespace, fresh.thread_id, fresh.run_id)[
                "status"
            ]
            == "completed"
        )
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_matrix_cli_reports_each_profile_and_decision():
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--matrix"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    results = json.loads(completed.stdout)
    assert {(r["profile"], r["decision"]) for r in results} == {
        (profile, decision)
        for profile in ("baseline", "workspace", "combined")
        for decision in ("approve", "deny")
    }
    assert all(r["summary"]["image_provenance"] for r in results)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["model", "tool"])
async def test_closing_event_consumer_cleans_active_operation(harness, phase):
    from msgflux.runtime import AbortSignal

    started, cleaned = asyncio.Event(), asyncio.Event()

    async def waiting() -> str:
        """Wait until the consumer closes."""
        started.set()
        try:
            await asyncio.Event().wait()
            return "unexpected completion"
        finally:
            cleaned.set()

    class WaitingModel(harness.ScriptedModel):
        async def acall(self, **kwargs):
            if phase == "model":
                return await waiting()
            return await super().acall(**kwargs)

    store = harness.InMemoryCheckpointStore()
    signal = AbortSignal()
    scope = replace(
        harness._scope(harness.InMemoryWorkspace("close", {}), thread="t", run="r"),
        abort_signal=signal,
    )
    agent = harness.Agent(
        name="offline-demo",
        model=WaitingModel([harness._tool("waiting", {})]),
        tools=[waiting],
        checkpoint_store=store,
        extensions=[harness.WorkspacePromptExtension()],
        config={"stream": True},
    )
    stream = agent.stream_events("start", scope=scope)
    try:
        await asyncio.wait_for(anext(stream), timeout=5)
        await asyncio.wait_for(started.wait(), timeout=5)
        await asyncio.wait_for(stream.aclose(), timeout=5)
        assert cleaned.is_set()
        assert signal.aborted
        assert store.load_state(scope.namespace, "t", "r")["status"] == "interrupted"
        # Repeated closure is harmless and cannot restart the operation.
        await stream.aclose()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_paused_event_consumer_receives_all_buffered_deltas_in_order(harness):
    committed = asyncio.Event()

    class ObservedStore(harness.InMemoryCheckpointStore):
        def commit_state(self, namespace, thread_id, run_id, state, **kwargs):
            result = super().commit_state(namespace, thread_id, run_id, state, **kwargs)
            if state["status"] == "completed":
                committed.set()
            return result

    chunks = [f"{index}:ação 🐍\n" for index in range(32)]
    response = harness.ModelStreamResponse(mode="async")
    response.set_response_type("text_generation")
    for chunk in chunks:
        response.add(chunk)
    response.finish()
    store = ObservedStore()
    agent = harness.Agent(
        name="offline-demo",
        model=harness.ScriptedModel([response]),
        checkpoint_store=store,
        extensions=[harness.WorkspacePromptExtension()],
        config={"stream": True},
    )
    scope = harness._scope(harness.InMemoryWorkspace("slow", {}), thread="t", run="r")
    stream = agent.stream_events("generate", scope=scope)
    try:
        first = await asyncio.wait_for(anext(stream), timeout=5)
        # Do not read further events until the producer has committed its result.
        await asyncio.wait_for(committed.wait(), timeout=5)
        events = [first, *[event async for event in stream]]
        assert [e.data["delta"] for e in events if e.type == "message.delta"] == chunks
        types = [e.type for e in events]
        assert types.index("message.start") < types.index("message.delta")
        assert max(
            i for i, kind in enumerate(types) if kind == "message.delta"
        ) < types.index("message.end")
        assert types.count("run.end") == 1 and types[-1] == "run.end"
        state = store.load_state(scope.namespace, "t", "r")
        assert any(
            item.get("content") == "".join(chunks)
            for item in state["messages"]["items"]
        )
    finally:
        await stream.aclose()


@pytest.mark.parametrize("args", [["--matrix", "--live"], ["--repeat", "0"]])
def test_invalid_matrix_cli_options_fail_before_execution(args):
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 2


@pytest.mark.asyncio
async def test_mixed_decisions_and_safe_event_rendering(harness, capsys):
    async def choose(record, preview):
        return preview is not None and preview.path == "/edit.txt"

    summary = await harness.run_demo(
        approval_decider=choose, on_event=harness.print_event
    )
    assert summary.approved_changes == 1 and summary.denied_changes == 2
    assert summary.image_provenance
    output = capsys.readouterr().err
    assert "[message.delta]" in output and "base64" not in output
    assert "\x1b" not in harness.safe_text("\x1b[31muntrusted")


@pytest.mark.asyncio
async def test_live_oversized_image_rejected_before_model_creation(harness, tmp_path):
    image = tmp_path / "large.png"
    image.write_bytes(b"x" * 1_000_001)

    def factory(*args, **kwargs):
        pytest.fail("must reject image before constructing provider")

    with pytest.raises(ValueError, match="1 MB"):
        await harness.run_live("test-model", image, model_factory=factory)


def test_repeated_run_accumulates_event_deltas_and_image_provenance(harness):
    summary = asyncio.run(harness.run_demo(repeat=2))

    assert summary.runs == 2
    assert summary.approved_changes == 6
    assert summary.denied_changes == 0
    assert summary.image_notifications == 2
    assert summary.stream_events >= 2
    assert summary.image_provenance is True


@pytest.mark.parametrize(
    "args, expected",
    [
        ([], {"runs": 1, "approved_changes": 3, "denied_changes": 0}),
        (["--deny"], {"runs": 1, "approved_changes": 0, "denied_changes": 3}),
        (["--repeat", "2"], {"runs": 2, "approved_changes": 6}),
    ],
)
def test_cli_emits_machine_readable_semantic_summary(args, expected):
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    result = json.loads(completed.stdout[completed.stdout.find("{") :])
    for key, value in expected.items():
        assert result[key] == value
    assert result["tools_checked"]


def test_interactive_cli_accepts_repeated_approval_decisions():
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--interactive", "--repeat", "2"],
        input="y\n" * 8,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    result = json.loads(completed.stdout[completed.stdout.find("{") :])
    assert result["runs"] == 2
    assert result["approved_changes"] == 6
    assert result["denied_changes"] == 0
    assert result["image_notifications"] == 2


def test_interactive_cli_blank_answers_deny_by_default():
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--interactive", "--repeat", "2"],
        input="\n\n",
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    result = json.loads(completed.stdout[completed.stdout.find("{") :])
    assert result["runs"] == 2
    assert result["approved_changes"] == 0
    assert result["denied_changes"] == 6


@pytest.mark.parametrize("model_name", ["test-model", "openai/test-model"])
@pytest.mark.asyncio
async def test_live_responses_image_patch_and_continuation_offline(
    harness, tmp_path, monkeypatch, model_name
):
    from msgflux.models.providers.openai import OpenAIChatCompletion

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    image = tmp_path / "fixture.png"
    image.write_bytes(harness._PNG)
    requests = []
    model = None

    def factory(path, **kwargs):
        nonlocal model
        assert path == "openai/test-model"
        assert kwargs["api_mode"] == "responses" and kwargs["store"] is False
        model = OpenAIChatCompletion(model_id="test-model", **kwargs)

        async def execute(**request):
            requests.append(deepcopy(request))
            if len(requests) == 1:
                events = [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "function_call",
                            "id": "read-item",
                            "call_id": "read-image",
                            "name": "read",
                            "arguments": '{"path":"/image.png"}',
                        },
                    }
                ]
            elif len(requests) == 2:
                events = [
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "apply_patch_call",
                            "id": "patch-item",
                            "call_id": "patch-1",
                            "status": "completed",
                            "operation": {
                                "type": "update_file",
                                "path": "/patch.txt",
                                "diff": "@@\n-original\n+changed",
                            },
                        },
                    }
                ]
            else:
                events = [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": f"message-{len(requests)}",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                    {
                        "type": "response.output_text.delta",
                        "delta": "Observed image.",
                        "output_index": 0,
                        "content_index": 0,
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": f"message-{len(requests)}",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "Observed image."}
                            ],
                        },
                    },
                ]

            async def stream():
                for event in events:
                    yield event

            return stream()

        model._aexecute_model = execute
        model.aclose = AsyncMock(wraps=model.aclose)
        return model

    decisions = []

    def approve(record, preview):
        decisions.append(preview)
        assert preview.before == "original" and preview.after == "changed"
        return True

    replies = iter(["What did you observe before?", "/quit"])
    events = []
    assert (
        await harness.run_live(
            model_name,
            image,
            interactive=True,
            model_factory=factory,
            decider=approve,
            read_input=lambda _: next(replies),
            on_event=events.append,
        )
        == 2
    )
    assert len(decisions) == 1 and len(requests) == 4
    assert any(tool["type"] == "apply_patch" for tool in requests[0]["tools"])
    assert not any(
        tool["type"] == "shell" or tool.get("name") == "bash"
        for tool in requests[0]["tools"]
    )
    assert "input_image" not in str(requests[0]["input"])
    assert "input_image" in str(requests[1]["input"])
    assert "apply_patch_call_output" in str(requests[2]["input"])
    assert "Observed image." in str(requests[3]["input"])
    assert sum(event.type == "run.end" for event in events) == 2
    model.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_closes_model_on_failure(harness, tmp_path):
    image = tmp_path / "fixture.png"
    image.write_bytes(harness._PNG)
    model = harness.ScriptedModel([])
    model.aclose = AsyncMock()
    with pytest.raises(AssertionError, match="exhausted"):
        await harness.run_live(
            "test-model",
            image,
            model_factory=lambda *args, **kwargs: model,
            on_event=lambda _: None,
        )
    model.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_explicit_function_tool_transport(harness, tmp_path):
    image = tmp_path / "fixture.png"
    image.write_bytes(harness._PNG)
    model = harness.ScriptedModel([harness._text("done", streamed=True)])
    model.aclose = AsyncMock()

    def factory(path, **kwargs):
        assert path == "openai/gpt-4.1-mini"
        assert kwargs["native_tools"] is False
        return model

    assert (
        await harness.run_live(
            "openai/gpt-4.1-mini",
            image,
            native_tools=False,
            model_factory=factory,
            on_event=lambda _: None,
        )
        == 1
    )
    model.aclose.assert_awaited_once()


def test_cli_live_is_explicit_opt_in():
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--model", "test-model"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 2
    assert "require explicit --live" in completed.stderr
