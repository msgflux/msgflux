"""A control signal must preserve already completed tool outcomes in history."""

import asyncio
import json
from threading import Barrier, Lock
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope
from msgflux.tools.config import tool_config


def _response(kind, data):
    response = ModelResponse()
    response.set_response_type(kind)
    response.add(data)
    return response


def _native_batch(calls):
    aggregate = ToolCallAggregator(api_mode="responses")
    history = []
    for index, (call_id, name, arguments) in enumerate(calls):
        serialized = json.dumps(arguments, separators=(",", ":"))
        aggregate.process(index, call_id, name, serialized)
        history.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": serialized,
            }
        )
    response = _response("tool_call", aggregate)
    response.history_items = history
    return response


def _output_text(item):
    value = item["output"]
    try:
        return json.dumps(json.loads(value), sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("control_type", "status"),
    [
        (TaskPauseRequestedError, "paused"),
        (TaskInterruptRequestedError, "interrupted"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("control_first", [False, True])
@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "sqlite"])
async def test_controlled_batch_checkpoints_completed_sibling_outcomes(  # noqa: C901
    tmp_path, control_type, status, asynchronous, control_first, persistent
):
    effects_file = tmp_path / "effects.log"
    effect_barrier = Barrier(2)
    effect_lock = Lock()
    control_error = control_type("run", "post-effect control requested")
    control_state = {"raised": False}

    def write_line(label):
        with effect_lock, effects_file.open("a", encoding="utf-8") as stream:
            stream.write(f"{label}\n")
        effect_barrier.wait(timeout=3)

    @tool_config(return_direct=True)
    def controlled_effect(value: str) -> str:
        """Perform a side effect before the host asks the run to pause."""
        write_line(f"controlled:{value}")
        return "controlled-result"

    def sibling_effect(value: str) -> str:
        """Perform a sibling side effect and return its result."""
        write_line(f"sibling:{value}")
        return "sibling-result"

    def after_tool(context):
        if context.tool_name == "controlled_effect" and not control_state["raised"]:
            control_state["raised"] = True
            raise control_error

    controlled = ("call-controlled", "controlled_effect", {"value": "A"})
    sibling = ("call-sibling", "sibling_effect", {"value": "B"})
    calls = [controlled, sibling] if control_first else [sibling, controlled]
    first_response = _native_batch(calls)
    model = Mock(model_type="chat_completion")
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    checkpoints = (
        SQLiteCheckpointStore(checkpoint_path)
        if persistent
        else InMemoryCheckpointStore()
    )

    def make_agent(agent_model, store):
        return Agent(
            name="agent",
            model=agent_model,
            tools=[controlled_effect, sibling_effect],
            checkpoint_store=store,
            hooks=[Hook(event="after_tool", handler=after_tool)],
        )

    agent = make_agent(model, checkpoints)
    if asynchronous:
        agent.generator.aforward = AsyncMock(return_value=first_response)
    else:
        agent.generator.forward = Mock(return_value=first_response)
    scope = ExecutionScope(thread_id="thread", run_id="run")
    resumed_agent = None

    async def invoke(target, message, call_scope):
        if asynchronous:
            return await target.acall(message, scope=call_scope)
        return target(message, scope=call_scope)

    try:
        with pytest.raises(control_type) as raised:
            await invoke(agent, "start", scope)
        assert raised.value is control_error
        assert sorted(effects_file.read_text().splitlines()) == [
            "controlled:A",
            "sibling:B",
        ]

        paused = checkpoints.load_state("agent", "thread", "run")
        assert paused["status"] == status
        items = paused["messages"]["items"]
        calls_by_id = {
            item["call_id"]: item
            for item in items
            if item.get("type") == "function_call"
        }
        outputs_by_id = {
            item["call_id"]: item
            for item in items
            if item.get("type") == "function_call_output"
        }
        assert set(calls_by_id) == {"call-controlled", "call-sibling"}
        assert set(outputs_by_id) == set(calls_by_id)
        assert "sibling-result" in _output_text(outputs_by_id["call-sibling"])

        controlled_output = _output_text(outputs_by_id["call-controlled"]).lower()
        assert "interrupted" in controlled_output or "uncertain" in controlled_output
        assert "post-effect control requested" in controlled_output
        assert not any(
            claim in controlled_output
            for claim in ("rolled back", "rollback complete", "effect undone")
        )

        saved_before_resume = paused
        resume_run_id = "run" if status == "paused" else "continued-run"
        resume_scope = ExecutionScope(thread_id="thread", run_id=resume_run_id)

        # A pause resumes the same run; an interruption continues in a new run.
        # Reopening SQLite ensures the partial outcomes survive process restart.
        await agent.aclose()
        agent = None
        if persistent:
            checkpoints.close()
            checkpoints = SQLiteCheckpointStore(checkpoint_path)
        resumed_model = Mock(model_type="chat_completion")
        if asynchronous:
            resumed_model_response = AsyncMock(
                return_value=_response("text_generation", "resumed")
            )
        else:
            resumed_model_response = Mock(
                return_value=_response("text_generation", "resumed")
            )
        resumed_agent = make_agent(resumed_model, checkpoints)
        if asynchronous:
            resumed_agent.generator.aforward = resumed_model_response
        else:
            resumed_agent.generator.forward = resumed_model_response

        assert await invoke(resumed_agent, "resume", resume_scope) == "resumed"
        assert sorted(effects_file.read_text().splitlines()) == [
            "controlled:A",
            "sibling:B",
        ]
        if status == "interrupted":
            interrupted = checkpoints.load_state("agent", "thread", "run")
            assert interrupted == saved_before_resume
        completed = checkpoints.load_state("agent", "thread", resume_run_id)
        assert completed["status"] == "completed"
        completed_ids = {
            item["call_id"]
            for item in completed["messages"]["items"]
            if item.get("type") == "function_call_output"
        }
        assert completed_ids == {"call-controlled", "call-sibling"}
    finally:
        if agent is not None:
            await agent.aclose()
        if resumed_agent is not None:
            await resumed_agent.aclose()
        if persistent:
            checkpoints.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_tool_library_preserves_control_exception_identity(asynchronous):
    from msgflux.nn.modules.tool import ToolLibrary

    failure = TaskPauseRequestedError("run", "stop in after_tool")

    def effect() -> str:
        """Complete one effect."""
        return "done"

    def after_tool(_context):
        raise failure

    library = ToolLibrary("tools", [effect])
    library.register_lifecycle_hook(
        "after_tool", Hook(event="after_tool", handler=after_tool)
    )
    with pytest.raises(TaskPauseRequestedError) as raised:
        if asynchronous:
            await library.acall([("call-effect", "effect", {})])
        else:
            library([("call-effect", "effect", {})])
    assert raised.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_type", "abort_cause"),
    [
        (AbortRequestedError("tool aborted"), TaskInterruptRequestedError, True),
        (asyncio.CancelledError("tool cancelled"), asyncio.CancelledError, False),
    ],
    ids=["abort", "per-tool-cancellation"],
)
async def test_gathered_tool_control_keeps_completed_sibling_checkpoint(
    tmp_path, failure, expected_type, abort_cause
):
    effects_file = tmp_path / "async-effects.log"
    barrier = asyncio.Barrier(2)

    def record(label):
        with effects_file.open("a", encoding="utf-8") as stream:
            stream.write(f"{label}\n")

    async def controlled_tool() -> str:
        """Finish an effect, then raise a per-tool control signal."""
        record("controlled")
        await barrier.wait()
        raise failure

    async def sibling_tool() -> str:
        """Finish a sibling effect that should remain in the checkpoint."""
        record("sibling")
        await barrier.wait()
        return "sibling-result"

    model = Mock(model_type="chat_completion")
    checkpoints = InMemoryCheckpointStore()
    agent = Agent(
        name="gathered-control",
        model=model,
        tools=[controlled_tool, sibling_tool],
        checkpoint_store=checkpoints,
    )
    agent.generator.aforward = AsyncMock(
        return_value=_native_batch(
            [
                ("call-controlled", "controlled_tool", {}),
                ("call-sibling", "sibling_tool", {}),
            ]
        )
    )

    try:
        with pytest.raises(expected_type) as raised:
            await agent.acall(
                "start", scope=ExecutionScope(thread_id="thread", run_id="run")
            )

        if abort_cause:
            assert isinstance(raised.value, TaskInterruptRequestedError)
            assert raised.value.__cause__ is failure
        else:
            assert isinstance(raised.value, asyncio.CancelledError)
        assert sorted(effects_file.read_text().splitlines()) == [
            "controlled",
            "sibling",
        ]

        interrupted = checkpoints.load_state("gathered-control", "thread", "run")
        assert interrupted["status"] == "interrupted"
        items = interrupted["messages"]["items"]
        outputs = {
            item["call_id"]: item
            for item in items
            if item.get("type") == "function_call_output"
        }
        assert set(outputs) == {"call-controlled", "call-sibling"}
        assert "sibling-result" in _output_text(outputs["call-sibling"])
    finally:
        await agent.aclose()
