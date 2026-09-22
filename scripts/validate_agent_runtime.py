"""Iterative end-to-end validation of the msgflux Agent runtime.

Offline by default: scripted model and an explicit fake process executor.
The opt-in --live mode makes paid API calls and transmits a selected image.
Neither mode invokes a host shell.

Run from the repository root::

    uv run python scripts/validate_agent_runtime.py
    uv run python scripts/validate_agent_runtime.py --interactive --repeat 2
"""

# ruff: noqa: S101, T201

from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import logging
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import msgspec

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.extensions import (
    CurrentDateExtension,
    FewShotExamplesExtension,
    ToolTurnLimitExtension,
    WorkspacePromptExtension,
)
from msgflux.runtime import (
    AgentApprovals,
    AgentInbox,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryAgentInboxStore,
    InMemoryApprovalStore,
    InMemoryWorkspace,
    PermissionSet,
    ProcessExecutor,
    ProcessResult,
    SandboxCapabilities,
    execution_context,
)
from msgflux.runtime.isolation import SandboxRequirements
from msgflux.runtime.workspace import WorkspaceFilesystem
from msgflux.tools.builtin import (
    ApplyPatchTool,
    BashTool,
    EditTool,
    ReadFileTool,
    WriteTool,
)
from msgflux.utils.msgspec import msgspec_dumps


class DemoSummary(msgspec.Struct, frozen=True, kw_only=True):
    """Stable, machine-readable result returned by :func:`run_demo`."""

    runs: int
    approved_changes: int
    denied_changes: int
    stream_events: int
    image_notifications: int
    tools_checked: tuple[str, ...]
    event_types: tuple[str, ...]
    image_provenance: bool


class ScriptedModel:
    model_type = "chat_completion"

    def __init__(self, responses: Sequence[ModelResponse | ModelStreamResponse]):
        self.responses = list(responses)
        self.calls = 0
        self.inputs = []
        self.prompts = []

    def __call__(self, **_kwargs):
        self.calls += 1
        self.inputs.append(deepcopy(_kwargs.get("messages")))
        self.prompts.append(_kwargs.get("system_prompt") or "")
        if not self.responses:
            raise AssertionError("offline scripted model exhausted")
        return self.responses.pop(0)

    async def acall(self, **kwargs):
        return self(**kwargs)


class FakeProcessExecutor(ProcessExecutor):
    """Deterministic process simulation; intentionally no subprocess fallback."""

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            {"filesystem", "network", "process", "resource_limits"}
        )

    def supports_workspace(self, filesystem: WorkspaceFilesystem) -> bool:
        return isinstance(filesystem, InMemoryWorkspace)

    async def execute_stream(
        self,
        request,
        *,
        filesystem,
        permissions,
        requirements: SandboxRequirements,
        abort_signal,
        on_output,
    ) -> ProcessResult:
        del filesystem, permissions, requirements, abort_signal
        command = request.argv[-1]
        await on_output("stdout", f"simulated:{command}".encode())
        return ProcessResult(0)


def _text(text: str, *, streamed: bool = False):
    if streamed:
        response = ModelStreamResponse(mode="async")
        response.set_response_type("text_generation")
        response.add(text[: len(text) // 2])
        response.add(text[len(text) // 2 :])
        response.finish()
        return response
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _tool(name: str, arguments: dict, call_id: str | None = None) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("tool_call")
    calls = ToolCallAggregator()
    calls.process(0, call_id or uuid4().hex, name, msgspec_dumps(arguments))
    response.add(calls)
    return response


def _scope(fs, *, thread: str, run: str, full: bool = True) -> ExecutionScope:
    resources = []
    if full:
        resources = [
            fs.permission("/note.txt", "filesystem.read"),
            fs.permission("/note.txt", "filesystem.write"),
            fs.permission("/image.png", "filesystem.read"),
        ]
        resources.extend(
            fs.permission(path, operation)
            for path in ("/write.txt", "/edit.txt", "/patch.txt")
            for operation in ("filesystem.read", "filesystem.write")
        )
    return ExecutionScope(
        namespace="offline-demo",
        thread_id=thread,
        run_id=run,
        principal="offline-harness",
        environment=ExecutionEnvironment(fs, FakeProcessExecutor()),
        permissions=PermissionSet(
            ["process.execute"] if full else [], resources=resources
        ),
    )


async def run_permission_denial() -> bool:
    fs = InMemoryWorkspace("denial", {"/note.txt": b"original"})
    model = ScriptedModel(
        [
            _tool("write", {"path": "/note.txt", "content": "blocked"}),
            _text("Permission denied.", streamed=True),
        ]
    )
    agent = Agent(
        name="offline-demo",
        model=model,
        tools=[WriteTool()],
        approvals=None,
        config={"stream": True},
    )
    events = [
        event
        async for event in agent.stream_events(
            "Try writing without permissions",
            approvals=None,
            scope=_scope(fs, thread="deny", run="deny", full=False),
        )
    ]
    assert not any(event.type == "tool.approval_required" for event in events)
    assert "Missing tool resource permissions" in str(model.inputs[-1])
    with execution_context(scope=_scope(fs, thread="verify", run="verify")):
        assert fs.read_text("/note.txt") == "original"
    return True


async def drive_agent(
    agent,
    scope,
    *,
    decider,
    prompt="Run demo",
    on_event=None,
    approval_store=None,
    **run_kwargs,
):
    """Drive stream_events through approval pauses while preserving the run."""
    events = []
    while True:
        try:
            async for event in agent.stream_events(prompt, scope=scope, **run_kwargs):
                events.append(event)
                if on_event:
                    on_event(event)
            return events
        except TaskPauseRequestedError:
            store = approval_store or agent.approvals.store
            pending = store.pending(scope.namespace, scope.thread_id, scope.run_id)
            if not pending:
                raise  # Other pauses/reconciliation need host intervention, not retry.
            for record in pending:
                preview = await agent.ainspect_approval_preview(
                    scope.thread_id, scope.run_id, record.request_id
                )
                choice = decider(record, preview)
                approved = await choice if inspect.isawaitable(choice) else choice
                if type(approved) is not bool:
                    raise TypeError("Approval decider must return bool") from None
                agent.decide_approval(
                    record.request_id, approved=approved, decided_by="runtime-harness"
                )
            prompt = (
                None  # Resume checkpoint; do not append the initial user task again.
            )
            run_kwargs.pop("messages", None)


EXTENSION_PROFILES = ("baseline", "workspace", "combined")


def _extensions(profile):
    if profile not in EXTENSION_PROFILES:
        raise ValueError(f"Unknown extension profile: {profile}")
    if profile == "baseline":
        return []
    extensions = [WorkspacePromptExtension()]
    if profile == "combined":
        extensions.extend(
            [
                CurrentDateExtension(date_factory=lambda: "2030-01-02"),
                FewShotExamplesExtension(
                    "Respect the workspace and approval decisions."
                ),
                ToolTurnLimitExtension(7, warn_remaining=7),
            ]
        )
    return extensions


async def _one_run(decider, run: int, on_event=None, *, profile="baseline"):
    fs = InMemoryWorkspace(
        "demo",
        {
            "/note.txt": b"first\nsecond\nthird\n",
            "/image.png": _PNG,
            "/write.txt": b"original",
            "/edit.txt": b"original",
            "/patch.txt": b"original",
        },
    )
    approvals = InMemoryApprovalStore()
    checkpoints = InMemoryCheckpointStore()
    model = ScriptedModel(
        [
            _tool("read", {"path": "/note.txt", "offset": 2, "limit": 1}),
            _tool("write", {"path": "/write.txt", "content": "written"}),
            _tool("edit", {"path": "/edit.txt", "old": "original", "new": "edited"}),
            _tool(
                "apply_patch",
                {
                    "operation": "update",
                    "path": "/patch.txt",
                    "diff": "@@\n-original\n+patched",
                },
            ),
            _tool("bash", {"command": "printf offline"}),
            _tool("read", {"path": "/image.png"}, "image-read"),
            _text(
                "Image delivered; offline model does not interpret pixels.",
                streamed=True,
            ),
        ]
    )
    agent = Agent(
        name="offline-demo",
        model=model,
        tools=[
            WriteTool(),
            ReadFileTool(supports_vision=True),
            EditTool(),
            ApplyPatchTool(),
            BashTool(),
        ],
        agent_inbox=AgentInbox(store=InMemoryAgentInboxStore()),
        approvals=AgentApprovals(
            approvals,
            {"write": "v1", "edit": "v1", "apply_patch": "v1", "bash": "v1"},
            "offline-policy",
        ),
        checkpoint_store=checkpoints,
        config={"stream": True},
        extensions=_extensions(profile),
        system_prompt="Runtime validation fixture.",
    )
    scope = _scope(fs, thread=f"thread-{run}", run=f"run-{run}")

    choices = {}

    async def choose(record, preview):
        result = decider(record, preview) if decider else True
        result = await result if inspect.isawaitable(result) else result
        if preview is not None:
            assert preview.before == "original" and "-original" in preview.diff
            choices[preview.path] = (result, preview.after)
        return result

    events = await drive_agent(
        agent, scope, decider=choose, approval_store=approvals, on_event=on_event
    )
    types = {event.type for event in events}
    for prompt in model.prompts:
        assert prompt.count("Runtime validation fixture.") == 1
        assert prompt.count("<workspace_context>") == (profile != "baseline")
        if profile == "combined":
            assert prompt.count("The current date is: 2030-01-02") == 1
            assert prompt.count("<examples>") == 1
            assert prompt.count("Tool budget:") == 1
    assert {"message.delta", "tool.start", "tool.approval_required", "run.end"} <= types
    with execution_context(scope=scope):
        for path, (approved, after) in choices.items():
            assert fs.read_text(path) == (after if approved else "original")
    history = checkpoints.load_state(scope.namespace, scope.thread_id, scope.run_id)[
        "messages"
    ]["items"]
    output_index = next(
        i
        for i, item in enumerate(history)
        if item.get("type") == "function_call_output"
        and item.get("call_id") == "image-read"
    )
    image_index = next(
        i
        for i, item in enumerate(history)
        if item.get("metadata", {}).get("inbox_ref") == "image-read"
    )
    assert image_index > output_index and history[image_index]["role"] == "user"
    image_block = next(
        block
        for block in history[image_index]["content"]
        if block.get("type") == "image_url"
    )
    assert base64.b64decode(image_block["image_url"]["url"].split(",", 1)[1]) == _PNG
    assert "base64" not in str(history[output_index])
    assert "image_url" in str(model.inputs[-1])
    assert "offline model does not interpret pixels" in str(history)
    assert any(item.get("output") == "second\n" for item in history)
    assert len(choices) == 3
    assert {"read", "write", "edit", "apply_patch", "bash"} <= {
        item.get("name") for item in history if item.get("type") == "function_call"
    }
    approved = sum(choice[0] for choice in choices.values())
    return approved, len(choices) - approved, events


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


async def run_demo(
    *,
    approval_decider=None,
    repeat: int = 1,
    deny: bool = False,
    on_event=None,
    profile: str = "baseline",
) -> DemoSummary:
    """Run the offline E2E scenario; ``approval_decider`` may be sync or async."""
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be a positive integer")
    if profile not in EXTENSION_PROFILES:
        raise ValueError(f"Unknown extension profile: {profile}")
    approved = denied = events = images = 0
    types = set()
    for index in range(repeat):
        decision = (lambda _record, _preview: False) if deny else approval_decider
        a, d, collected = await _one_run(decision, index, on_event, profile=profile)
        approved += a
        denied += d
        events += len(collected)
        types.update(event.type for event in collected)
        images += 1
    assert await run_permission_denial()
    return DemoSummary(
        runs=repeat,
        approved_changes=approved,
        denied_changes=denied,
        stream_events=events,
        image_notifications=images,
        tools_checked=("read", "write", "edit", "apply_patch", "bash"),
        event_types=tuple(sorted(types)),
        image_provenance=True,
    )


async def run_matrix(*, repeat=1, on_event=None):
    """Fresh agents/stores per profile and decision; no live requests."""
    results = []
    for profile in EXTENSION_PROFILES:
        for deny in (False, True):
            summary = await run_demo(
                profile=profile, deny=deny, repeat=repeat, on_event=on_event
            )
            results.append(
                {
                    "profile": profile,
                    "decision": "deny" if deny else "approve",
                    "summary": msgspec.to_builtins(summary),
                }
            )
    return results


def main() -> None:
    # Keep CLI summaries parseable without changing library logging policy.
    for handler in logging.getLogger("msgflux").handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            handler.setStream(sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactive", action="store_true", help="prompt for each approval"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--profile", choices=EXTENSION_PROFILES, default="baseline")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="run all offline profiles with approval and denial",
    )
    parser.add_argument("--deny", action="store_true", help="deny the protected write")
    parser.add_argument(
        "--events",
        action="store_true",
        help="show event types and text deltas on stderr",
    )
    parser.add_argument(
        "--live", action="store_true", help="opt in to paid OpenAI Responses requests"
    )
    parser.add_argument(
        "--model", help="OpenAI model ID or openai/MODEL (required with --live)"
    )
    parser.add_argument(
        "--native-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="native shell/patch transport; disable for function-only models",
    )
    parser.add_argument(
        "--image", type=Path, help="local image to transmit in live mode (max 1 MB)"
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.matrix and (
        args.live or args.interactive or args.deny or args.profile != "baseline"
    ):
        parser.error(
            "--matrix cannot combine with --live/--interactive/--deny/--profile"
        )
    if args.live and args.profile != "baseline":
        parser.error("--profile is offline only")
    if args.live:
        if not args.model or not args.image or args.deny or args.repeat != 1:
            parser.error(
                "--live requires --model and --image; "
                "do not combine with --deny/--repeat"
            )
        asyncio.run(
            run_live(
                args.model,
                args.image,
                interactive=args.interactive,
                native_tools=args.native_tools,
            )
        )
        return
    if args.model or args.image:
        parser.error("--model/--image require explicit --live")
    if args.matrix:
        print(
            msgspec_dumps(
                asyncio.run(
                    run_matrix(
                        repeat=args.repeat,
                        on_event=print_event if args.events else None,
                    )
                )
            )
        )
        return
    print(
        "OFFLINE: scripted model, in-memory state, simulated Bash; no host commands.",
        file=sys.stderr,
    )
    summary = asyncio.run(
        run_demo(
            approval_decider=prompt_approval if args.interactive else None,
            repeat=args.repeat,
            deny=args.deny,
            on_event=print_event if args.events else None,
            profile=args.profile,
        )
    )
    print(msgspec.json.encode(summary).decode())


def safe_text(value) -> str:
    """Escape terminal controls; never use this to dump entire event payloads."""
    return "".join(
        char if char.isprintable() or char in "\n\t" else repr(char)[1:-1]
        for char in str(value)
    )


def print_event(event) -> None:
    text = (
        event.data.get("delta", "")
        if event.type
        in {
            "message.delta",
            "reasoning.delta",
            "reasoning_summary.delta",
        }
        else event.data.get("tool_name", "")
    )
    print(f"[{safe_text(event.type)}] {safe_text(text)}", file=sys.stderr)


async def prompt_approval(record, preview) -> bool:
    print(f"Approval: {safe_text(record.binding.tool_name)}", file=sys.stderr)
    if preview is not None:
        print(safe_text(preview.diff), file=sys.stderr)
    else:
        print(
            "No file diff available (simulated Bash in offline mode).", file=sys.stderr
        )
    try:
        answer = await asyncio.to_thread(input, "Approve? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


async def run_live(
    model_id: str,
    image_path: Path,
    *,
    interactive: bool = False,
    native_tools: bool = True,
    model_factory=None,
    decider=prompt_approval,
    read_input=input,
    on_event=print_event,
):
    """Opt-in vision playground; real model, virtual files, no process executor.

    Factories/callbacks allow offline wiring tests. Each conversational turn gets
    a new run ID in the same thread, continued from the in-memory checkpoint.
    """
    import msgflux as mf  # noqa: PLC0415

    model_id = model_id.strip().removeprefix("openai/")
    if not model_id or "/" in model_id:
        raise ValueError("Use an OpenAI model ID or openai/MODEL")
    with image_path.open("rb") as image_file:
        image_data = image_file.read(1_000_001)
    if len(image_data) > 1_000_000:
        raise ValueError("Image exceeds the Read tool's 1 MB limit")
    fs = InMemoryWorkspace(
        "live-demo",
        {
            "/image.png": image_data,
            "/note.txt": b"first\nsecond\nthird\n",
            "/write.txt": b"original",
            "/edit.txt": b"original",
            "/patch.txt": b"original",
        },
    )
    print(
        "LIVE: paid requests; selected image and conversation are sent to OpenAI. "
        "Files are virtual; Bash is not exposed without a process executor.",
        file=sys.stderr,
    )
    factory = model_factory or mf.Model.chat_completion
    model = factory(
        f"openai/{model_id}",
        api_mode="responses",
        store=False,
        max_tokens=1500,
        retry=False,
        native_tools=native_tools,
    )
    try:
        journal = InMemoryApprovalStore()
        checkpoints = InMemoryCheckpointStore()
        agent = Agent(
            name="live-demo",
            model=model,
            tools=[
                ReadFileTool(supports_vision=True),
                WriteTool(),
                EditTool(),
                ApplyPatchTool(),
            ],
            checkpoint_store=checkpoints,
            approvals=AgentApprovals(
                journal,
                dict.fromkeys(("write", "edit", "apply_patch"), "demo-v1"),
                "live-demo-v1",
            ),
            config={"stream": True, "max_tool_turns": 12},
            system_prompt=(
                "Work only with the provided virtual files. Read supports images: "
                "an image read publishes a subsequent user-role image via AgentInbox. "
                "Read /image.png before describing its contents; do not infer them "
                "from the filename. Bash is unavailable: there is no process executor. "
                "Editable files are /write.txt, /edit.txt, /patch.txt and /note.txt. "
                "Respect denied tool requests. Keep answers concise."
            ),
        )
        resources = list(
            _scope(fs, thread="unused", run="unused").permissions.resources
        )
        thread = uuid4().hex
        prompt = (
            "Read line 2 of /note.txt. Inspect /image.png using read and describe "
            "what you see. Propose writing that description to /write.txt, then "
            "use edit on /edit.txt and apply_patch on /patch.txt "
            "to add short summaries."
        )
        turns = 0
        while True:
            scope = ExecutionScope(
                namespace="live-demo",
                thread_id=thread,
                run_id=uuid4().hex,
                principal="local-user",
                environment=ExecutionEnvironment(fs),
                permissions=PermissionSet(resources=resources),
            )
            await drive_agent(
                agent,
                scope,
                prompt=prompt,
                decider=decider,
                on_event=on_event,
                approval_store=journal,
                messages=ChatMessages(),
            )
            turns += 1
            if not interactive:
                return turns
            try:
                prompt = await asyncio.to_thread(
                    read_input, "\nNext message (/quit to exit): "
                )
            except EOFError:
                return turns
            if not prompt.strip() or prompt.strip() == "/quit":
                return turns
    finally:
        await model.aclose()


if __name__ == "__main__":
    main()
