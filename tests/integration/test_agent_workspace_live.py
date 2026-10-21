"""Opt-in live Agent check for the public workspace dependency.

Enable with ``MSGFLUX_LIVE_AGENT_WORKSPACE=1`` and the selected provider's
credentials available to ``Model``. The test uses a temporary project and makes
a bounded, multiround billable request. Override the model with
``MSGFLUX_LIVE_WORKSPACE_MODEL`` and the provider with
``MSGFLUX_LIVE_WORKSPACE_PROVIDER`` (``openai`` or ``openrouter``).
Select a real Docker workspace with ``MSGFLUX_LIVE_WORKSPACE_BACKEND=docker``;
the image must already exist locally (``MSGFLUX_LIVE_WORKSPACE_IMAGE`` defaults
to ``python:3.12-slim``). The default backend is local host execution.
"""

from __future__ import annotations

import asyncio
import os

import pytest

import msgflux as mf
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.nn import Agent
from msgflux.nn.extensions import ToolTurnLimitExtension
from msgflux.runtime.events import EventType
from msgflux.runtime import DockerWorkspaceBackend, PermissionSet
from msgflux.tools.builtin import ApplyPatchTool, BashTool, ReadFileTool


_OPT_IN = "MSGFLUX_LIVE_AGENT_WORKSPACE"


@pytest.mark.skipif(
    os.getenv(_OPT_IN) != "1",
    reason=f"Set {_OPT_IN}=1 to enable the live workspace Agent test",
)
@pytest.mark.asyncio
async def test_live_agent_reads_edits_and_executes_in_temporary_workspace(tmp_path):
    mf.load_dotenv(os.getenv("MSGFLUX_TEST_DOTENV", ".env"))
    provider = os.getenv("MSGFLUX_LIVE_WORKSPACE_PROVIDER", "openai").lower()
    key_env = os.getenv(
        "MSGFLUX_LIVE_WORKSPACE_KEY_ENV",
        {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(
            provider, "OPENAI_API_KEY"
        ),
    )
    if not os.getenv(key_env):
        pytest.skip(f"{key_env} is not configured")
    defaults = {
        "openai": "openai/gpt-6-luna",
        "openrouter": "openrouter/openai/gpt-6-luna",
    }
    (tmp_path / "status.txt").write_text("status: BLUE-ORBIT-42\n")
    backend = os.getenv("MSGFLUX_LIVE_WORKSPACE_BACKEND", "local")
    if backend == "local":
        workspace = mf.AgentWorkspace.local(tmp_path)
    elif backend == "docker":
        workspace = await mf.AgentWorkspace.open(
            DockerWorkspaceBackend(
                tmp_path,
                image=os.getenv("MSGFLUX_LIVE_WORKSPACE_IMAGE", "python:3.12-slim"),
            ),
            "live-project",
            permissions=PermissionSet(
                [
                    "filesystem.read",
                    "filesystem.write",
                    "process.execute",
                    "process.workspace",
                ]
            ),
            write_guarantee="cooperative_compare",
        )
    else:
        raise ValueError("MSGFLUX_LIVE_WORKSPACE_BACKEND must be local or docker")

    model = mf.Model.chat_completion(
        os.getenv(_MODEL_OVERRIDE, defaults.get(provider, defaults["openai"])),
        api_key_env=key_env,
        api_mode="responses",
        max_tokens=1536,
        reasoning_effort="medium",
        chat_transport=HTTPChatTransport(timeout=60, max_retries=0),
        retry=False,
    )
    agent = Agent(
        name="live_workspace_agent",
        model=model,
        workspace=workspace,
        tools=[ReadFileTool(), ApplyPatchTool(), BashTool()],
        extensions=[ToolTurnLimitExtension(6, warn_remaining=0)],
        config={"stream": True},
        system_prompt=(
            "Complete these required workspace tool calls one at a time and in order: "
            "(1) call read on status.txt; (2) call apply_patch to replace "
            "BLUE-ORBIT-42 with GREEN-ORBIT-42; (3) call read on status.txt again; "
            "(4) call bash with exactly `cat status.txt && printf "
            "'COMMAND_SENTINEL_OK\\n'`. Do not give a final answer until all four "
            "calls have succeeded and you have seen the Bash result. If you have not "
            "called Bash, the task is still incomplete. Then report the updated status "
            "and exact command sentinel."
        ),
    )

    try:
        async with asyncio.timeout(180):
            events = [
                event
                async for event in agent.stream_events(
                    "Update the project status and verify it using all requested tools."
                )
            ]
    finally:
        try:
            await model.aclose()
        finally:
            await workspace.aclose()

    starts = [
        event.data["tool_name"]
        for event in events
        if event.type == EventType.TOOL_START
    ]
    completed = [
        event
        for event in events
        if event.type == EventType.TOOL_END and event.data["error"] is None
    ]
    answer = "".join(
        event.data["delta"] for event in events if event.type == EventType.MESSAGE_DELTA
    )
    diagnostic = {
        "tool_starts": starts,
        "tool_ends": [
            {
                "name": event.data["tool_name"],
                "error": event.data["error"],
                "result": str(event.data["result"])[:500],
            }
            for event in events
            if event.type == EventType.TOOL_END
        ],
        "answer": answer[:1000],
        "sentinel_file": (tmp_path / "status.txt").read_text()[:500],
        "run_end": dict(events[-1].data),
    }
    assert starts[:4] == ["read", "apply_patch", "read", "bash"], diagnostic
    assert {"read", "apply_patch", "bash"} <= {
        event.data["tool_name"] for event in completed
    }, diagnostic
    assert events[-1].type == EventType.RUN_END, diagnostic
    assert "GREEN-ORBIT-42" in answer, diagnostic
    assert "COMMAND_SENTINEL_OK" in answer, diagnostic
    assert (tmp_path / "status.txt").read_text() == "status: GREEN-ORBIT-42\n", (
        diagnostic
    )


_MODEL_OVERRIDE = "MSGFLUX_LIVE_WORKSPACE_MODEL"
