"""Shell offload, structured retrieval and native history compatibility."""

from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.models.tool_adapters.openai_shell import OpenAIShellAdapter, shell_output
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import ToolLibrary
from msgflux.nn.extensions import ToolOutputOffloadExtension
from msgflux.nn.hooks.events import AfterTool
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    LocalToolResultStore,
    PermissionSet,
    ProcessExecutor,
    ProcessResult,
    RuntimeResources,
    SandboxCapabilities,
    ToolResultRef,
    execution_context,
    get_tool_result_reference,
)
from msgflux.tools.builtin import BashTool
from msgflux.tools.shell import ShellCommandResult, ShellResult
from msgflux.tools.runtime import ToolOutcome


def original_result():
    return ShellResult(
        results=(
            ShellCommandResult(
                status="exited",
                returncode=7,
                stdout="🌍" * 2000,
                stderr="error\n" * 2000,
            ),
            ShellCommandResult(status="timed_out", stdout="partial" * 2000),
            ShellCommandResult(status="not_executed", stderr="not started"),
        )
    )


def offload(store, result, *, preview_bytes=48):
    extension = ToolOutputOffloadExtension(
        store, max_inline_bytes=1024, preview_bytes=preview_bytes
    )
    original = AfterTool(tool_call_id="call", tool_name="bash", result=result)
    return extension.hooks()[0].handle(original)


@pytest.mark.parametrize("preview_bytes", [0, 1, 48])
def test_complete_shell_batch_is_json_with_shared_preview_budget(
    tmp_path, preview_bytes
):
    store = LocalToolResultStore(tmp_path)
    original = original_result()
    result = offload(store, original, preview_bytes=preview_bytes).result
    assert isinstance(result, ShellResult)
    ref = get_tool_result_reference(result)
    assert isinstance(ref, ToolResultRef)
    assert ref.media_type == "application/json"
    assert (
        sum(
            len(part.stdout.encode()) + len(part.stderr.encode())
            for part in result.results
        )
        <= preview_bytes
    )
    assert [(p.status, p.returncode) for p in result.results] == [
        (p.status, p.returncode) for p in original.results
    ]
    restored = msgspec.json.decode(
        b"".join(store.iter_bytes(ref, chunk_size=17)), type=ShellResult
    )
    assert restored == original
    store.verify(ref)
    decoded = msgspec.json.decode(msgspec.json.encode(result))
    assert get_tool_result_reference(decoded) == ref
    assert offload(store, result).result is result
    assert len(list(tmp_path.iterdir())) == 1


def test_small_shell_results_keep_existing_wire_shape(tmp_path):
    original = ShellResult(
        results=(ShellCommandResult(status="exited", returncode=0, stdout="ok"),)
    )
    assert offload(LocalToolResultStore(tmp_path), original).result is original
    assert get_tool_result_reference(original) is None
    assert "output_reference" not in msgspec.to_builtins(original)
    assert "metadata" not in shell_output("call", original)


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
@pytest.mark.parametrize("shell", [True, False])
def test_function_feedback_preserves_structured_reference_in_history(
    tmp_path, api_mode, shell
):
    store = LocalToolResultStore(tmp_path)
    result = offload(store, original_result()).result
    reference = result.output_reference
    if not shell:
        result = {"type": "tool_result_reference", "reference": reference.to_dict()}
    aggregator = ToolCallAggregator(api_mode=api_mode)
    aggregator.process(0, "call", "produce", "{}")
    rendered = aggregator.render_outcomes(
        [
            ToolOutcome(
                intent_id="call", tool_name="produce", status="completed", result=result
            )
        ]
    )
    assert get_tool_result_reference(rendered[-1]) == reference
    messages = ChatMessages(rendered)
    assert get_tool_result_reference(list(messages)[-1]) == reference
    assert all("metadata" not in item for item in messages.to_chatml())
    assert all("metadata" not in item for item in messages.to_responses_input())
    if api_mode == "chat_completions":
        assert get_tool_result_reference(aggregator.get_messages()[-1]) == reference


def test_reference_helper_has_no_free_text_parsing(tmp_path):
    ref = LocalToolResultStore(tmp_path).put([b"data"])
    envelope = {"type": "tool_result_reference", "reference": ref.to_dict()}
    assert get_tool_result_reference(envelope) == ref
    assert get_tool_result_reference({"output_reference": ref.to_dict()}) == ref
    assert get_tool_result_reference({"results": ["ordinary data"]}) is None
    assert get_tool_result_reference(msgspec.json.encode(envelope).decode()) is None
    assert get_tool_result_reference({"normal": "output"}) is None
    for metadata in (None, [], "not a mapping"):
        with pytest.raises(ValueError, match="metadata"):
            get_tool_result_reference(
                {"type": "shell_call_output", "metadata": metadata}
            )
    with pytest.raises(ValueError, match="missing"):
        get_tool_result_reference({"type": "tool_result_reference"})
    with pytest.raises(msgspec.ValidationError):
        get_tool_result_reference(
            {"results": [], "output_reference": {"result_id": "../escape"}}
        )


def test_native_history_preserves_reference_without_extra_wire_fields(tmp_path):
    store = LocalToolResultStore(tmp_path / "tool-results")
    result = offload(store, original_result()).result
    native = shell_output("call", result, command_count=3, max_output_length=1000)
    ref = get_tool_result_reference(native)
    assert ref == result.output_reference
    assert native["output"][0]["outcome"] == {"type": "exit", "exit_code": 7}
    assert native["output"][1]["outcome"] == {"type": "timeout"}
    assert ref.result_id in native["output"][0]["stdout"]
    assert "🌍" * 100 not in native["output"][0]["stdout"]
    assert "error\n" * 100 not in native["output"][0]["stderr"]
    history = ChatMessages([native])
    resources = RuntimeResources(tmp_path)
    checkpoints = resources.checkpoint_store("thd_test")
    try:
        checkpoints.save_state(
            "main", "thd_test", "run", {"messages": history._to_state()}
        )
        saved = checkpoints.load_state("main", "thd_test", "run")
    finally:
        checkpoints.close()
    restored = ChatMessages()
    restored._hydrate_state(saved["messages"])
    wire = restored.to_responses_input(provider="openai")[0]
    assert set(wire) == {"type", "call_id", "output", "max_output_length"}
    assert all(set(part) == {"stdout", "stderr", "outcome"} for part in wire["output"])
    stored = next(
        item
        for item in saved["messages"]["items"]
        if item["type"] == "shell_call_output"
    )
    assert get_tool_result_reference(stored) == ref
    projected = OpenAIShellAdapter().project_history(deepcopy(stored))
    assert get_tool_result_reference(projected["output"]) == ref
    assert result.output_reference == ref  # rendering did not mutate the result


@pytest.mark.asyncio
async def test_builtin_bash_event_contains_preview_and_retrievable_reference(tmp_path):
    executor = Mock(spec=ProcessExecutor)
    executor.capabilities = SandboxCapabilities(
        {"filesystem", "network", "process", "resource_limits"}
    )
    executor.supports_workspace.return_value = True
    executor.execute = AsyncMock(
        return_value=ProcessResult(3, b"x" * 20000, b"error" * 2000)
    )
    executor.execute_stream = ProcessExecutor.execute_stream.__get__(executor)
    environment = ExecutionEnvironment(InMemoryWorkspace("test"), executor)
    store = LocalToolResultStore(tmp_path)
    library = ToolLibrary(
        "shell",
        [BashTool()],
        extensions=[
            ToolOutputOffloadExtension(store, max_inline_bytes=1024, preview_bytes=32)
        ],
    )
    with execution_context(
        scope=ExecutionScope(
            environment=environment, permissions=PermissionSet(["process.execute"])
        )
    ):
        events = [
            event
            async for event in library.stream_events(
                [("call", "bash", {"command": "ignored by fake executor"})]
            )
        ]
    event = next(event for event in events if event.type == "tool.end")
    ref = get_tool_result_reference(event.data["result"])
    assert ref is not None
    encoded = msgspec.json.encode(event)
    assert b"x" * 1000 not in encoded
    assert b"error" * 1000 not in encoded
    assert b"x" * 1000 not in msgspec.json.encode(events)
    assert b"error" * 1000 not in msgspec.json.encode(events)
    decoded = msgspec.json.decode(encoded)
    assert get_tool_result_reference(decoded["data"]["result"]) == ref
    complete = msgspec.json.decode(b"".join(store.iter_bytes(ref, chunk_size=1024)))
    assert complete["results"][0]["stdout"] == "x" * 20000
    assert complete["results"][0]["stderr"] == "error" * 2000
    assert complete["results"][0]["returncode"] == 3
