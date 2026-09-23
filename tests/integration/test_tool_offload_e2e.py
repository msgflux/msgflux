"""Opt-in provider checks for bounded shell output and durable references.

These tests make paid model requests only when ``MSGFLUX_LIVE_OFFLOAD=1`` is
set.  The shell executor is deliberately fake: no model-generated command is
ever run on the host.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import msgspec
import pytest

import msgflux as mf
from msgflux.chat_messages import ChatMessages
from msgflux.nn import Agent
from msgflux.nn.extensions import ToolOutputOffloadExtension
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    ProcessExecutor,
    ProcessResult,
    RuntimeResources,
    SandboxCapabilities,
    get_tool_result_reference,
)
from msgflux.tools.builtin import BashTool


@dataclass(frozen=True)
class _ProviderCase:
    name: str
    model_path: str
    api_mode: str
    key: str


CASES = (
    _ProviderCase(
        "openai-responses",
        "openai/gpt-5.6-luna",
        "responses",
        "OPENAI_API_KEY",
    ),
    _ProviderCase(
        "openai-chat-completions",
        "openai/gpt-5.6-luna",
        "chat_completions",
        "OPENAI_API_KEY",
    ),
    _ProviderCase(
        "openrouter-chat-completions",
        "openrouter/openai/gpt-oss-120b",
        "chat_completions",
        "OPENROUTER_API_KEY",
    ),
)


class _FakeExecutor(ProcessExecutor):
    """Return synthetic output while recording the requested argv."""

    def __init__(self) -> None:
        self.requests = []

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            {"filesystem", "network", "process", "resource_limits"}
        )

    def supports_workspace(self, filesystem) -> bool:
        return isinstance(filesystem, InMemoryWorkspace)

    async def execute_stream(self, request, *, on_output, **kwargs) -> ProcessResult:
        del kwargs
        self.requests.append(request)
        await on_output("stdout", b"OFFLOAD_SYNTHETIC_STDOUT\n" * 400)
        await on_output("stderr", b"OFFLOAD_SYNTHETIC_STDERR\n" * 400)
        return ProcessResult(0)


def _skip_reason(case: _ProviderCase) -> str | None:
    if os.getenv("MSGFLUX_LIVE_OFFLOAD") != "1":
        return "set MSGFLUX_LIVE_OFFLOAD=1 to enable paid provider checks"
    if not os.getenv(case.key):
        return f"{case.key} is required for this live provider check"
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_live_provider_shell_output_offload_and_followup(
    tmp_path: Path, case: _ProviderCase
):
    reason = _skip_reason(case)
    if reason:
        pytest.skip(reason)

    resources = RuntimeResources(tmp_path).initialize()
    result_store = resources.tool_result_store()
    checkpoints = resources.checkpoint_store("offload-live")
    executor = _FakeExecutor()
    environment = ExecutionEnvironment(InMemoryWorkspace("live-offload"), executor)
    model_kwargs = {
        "api_mode": case.api_mode,
        "max_tokens": 1024,
        "retry": False,
        "reasoning_effort": "low",
    }
    if case.name == "openai-chat-completions":
        model_kwargs["reasoning_effort"] = "none"
    if case.api_mode == "responses":
        model_kwargs["store"] = False
    model = mf.Model.chat_completion(case.model_path, **model_kwargs)
    agent = Agent(
        name=f"offload_{case.name}",
        model=model,
        tools=[BashTool()],
        checkpoint_store=checkpoints,
        system_prompt=(
            "Call bash exactly once with one harmless command, then acknowledge "
            "the command result briefly. Never call bash more than once."
        ),
        config={"stream": True, "max_tool_turns": 2},
    )
    agent.tool_library.register_extension(
        "tool_output_offload",
        ToolOutputOffloadExtension(
            result_store, max_inline_bytes=128, preview_bytes=32
        ),
    )
    scope = ExecutionScope(
        namespace=agent.name,
        thread_id="offload-live",
        run_id="initial",
        environment=environment,
        permissions=PermissionSet(["process.execute"]),
    )

    try:
        async with asyncio.timeout(120):
            events = [
                event
                async for event in agent.stream_events(
                    "Run the harmless shell check and report that it completed.",
                    scope=scope,
                )
            ]
        tool_end = next(event for event in events if event.type == "tool.end")
        result = tool_end.data["result"]
        reference = get_tool_result_reference(result)
        assert reference is not None
        assert "OFFLOAD_SYNTHETIC_STDOUT" not in str(tool_end.data)
        assert "OFFLOAD_SYNTHETIC_STDERR" not in str(tool_end.data)
        assert (
            sum(
                len(part.stdout.encode()) + len(part.stderr.encode())
                for part in result.results
            )
            <= 32
        )
        restored = msgspec.json.decode(result_store.read(reference), type=dict)
        assert "OFFLOAD_SYNTHETIC_STDOUT" in restored["results"][0]["stdout"]
        assert "OFFLOAD_SYNTHETIC_STDERR" in restored["results"][0]["stderr"]
        result_store.verify(reference)

        state = checkpoints.load_state(agent.name, scope.thread_id, scope.run_id)
        encoded = msgspec.json.encode(state)
        assert reference.result_id.encode() in encoded
        assert b"OFFLOAD_SYNTHETIC_STDOUT" not in encoded

        async with asyncio.timeout(60):
            followup = [
                event
                async for event in agent.stream_events(
                    "Do not call tools. Confirm the previous shell check in one sentence.",
                    scope=ExecutionScope(
                        namespace=agent.name,
                        thread_id="offload-live",
                        run_id="followup",
                        environment=environment,
                        permissions=PermissionSet(["process.execute"]),
                    ),
                )
            ]
        assert followup
        assert len(executor.requests) == 1
        assert (
            checkpoints.load_state(agent.name, "offload-live", "followup")["status"]
            == "completed"
        )

        restored_messages = ChatMessages()
        restored_messages._hydrate_state(state["messages"])
        if case.api_mode == "responses":
            assert any(
                item.get("type") == "shell_call_output" for item in restored_messages
            )
        assert any(
            get_tool_result_reference(item) is not None
            or get_tool_result_reference(item.get("output")) is not None
            for item in restored_messages
        )
    finally:
        checkpoints.close()
        await model.aclose()
