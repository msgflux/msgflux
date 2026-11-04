# Quickstart

Build an agent that investigates a codebase, reads the relevant files, and
returns a structured review. Keep the conversation on disk, continue the work
in another process, and delegate a focused investigation to a specialist.

This guide uses direct `Agent(...)` construction. The same components can be
embedded in your application or hosted through Agent Service.

## Install and connect a model

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --prerelease=allow 'msgflux[coding,service]'
export OPENAI_API_KEY="your-api-key"
```

The extras install the workspace query dependencies and service transport used
in this guide. The Python examples use OpenAI; choose another supported provider
through [Chat Completion](learn/models/chat_completion.md).

Run the example from a small project's root directory. Create `review.py` and
add the following sections in order.

## Give the agent a workspace and tools

```python
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

import msgflux as mf
import msgflux.nn as nn
from msgflux.runtime import SQLiteAgentBackend
from msgflux.tools.builtin import GlobTool, GrepTool, LsTool, ReadFileTool


model = mf.Model.chat_completion("openai/gpt-4.1-mini")
workspace = mf.AgentWorkspace.local(Path.cwd())
backend = SQLiteAgentBackend(Path(".msgflux/review-state"))
```

The workspace resolves file paths relative to this project. The backend stores
runtime state on disk so a later process can continue the same conversation.
Keep `.msgflux/review-state/` out of version control.

## Package review behavior in an extension

An `AgentExtension` groups tools and hooks into a reusable capability. Package
the workspace tools with review guidelines that are added to each model request:

```python
from dataclasses import replace

from msgflux.nn.hooks import Hook, ModelContext


class ProjectReviewExtension(nn.AgentExtension):
    def __init__(self, guidelines: str):
        super().__init__("project_review")
        self.guidelines = guidelines

    def tools(self):
        return (LsTool(), GlobTool(), GrepTool(), ReadFileTool())

    def hooks(self):
        return (
            Hook(event="transform_system_prompt", handler=self.add_guidelines),
        )

    def add_guidelines(self, context: ModelContext):
        return replace(
            context,
            system_prompt=(
                f"{context.system_prompt}\n\nReview guidelines:\n{self.guidelines}"
            ),
        )
```

The hook changes the prompt sent to the model while preserving the Agent's
configured `system_prompt`. This extension holds only its configuration;
per-run values belong in `self.state()` or `self.durable_state()`. See
[Agent Extensions](learn/nn/agent/extensions.md) for state and lifecycle details.

## Configure the agent

Attach the extension when constructing the reviewer:

```python
reviewer = nn.Agent(
    name="project_reviewer",
    model=model,
    workspace=workspace,
    backend=backend,
    extensions=[
        ProjectReviewExtension(
            guidelines=(
                "Prioritize correctness, recovery behavior, and missing tests. "
                "Separate confirmed bugs from risks that need investigation."
            ),
        ),
    ],
    system_prompt=(
        "Investigate this project using its source files and tests. "
        "Start by listing the root and finding the files relevant to the task. "
        "Read targeted sections rather than the entire repository. "
        "Do not read .env files or .msgflux runtime data. "
        "Cite file paths in your findings, distinguish evidence from hypotheses, "
        "and finish with concrete next steps."
    ),
    signature="request -> summary: str, findings: list[str], next_steps: list[str]",
)
```

The agent can list directories, find files, search text, and read file sections.
Its tool set provides read operations; it has no shell or editing tool.
The system prompt defines how to investigate, while the **signature** defines
what the application receives: a summary, findings, and next steps. msgFlux
generates the input template and output schema from that signature. Its input
field is named `request`, which is passed as a keyword in the call below.

The extension registers its tools in the Agent's tool library. Your own typed
Python functions can be added alongside it through `tools=`. See
[Tools](learn/nn/agent/tools/index.md).

## Run a review and keep the thread

Append this entry point:

```python
async def main():
    task = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Map the project's entry points and identify one area worth reviewing."
    )
    scope = mf.ExecutionScope(
        thread_id="project-review",
        run_id=uuid4().hex,
    )
    print(f"thread_id={scope.thread_id} run_id={scope.run_id}", file=sys.stderr)
    try:
        report = await reviewer.acall(request=task, scope=scope)
        print(report["summary"])
        for finding in report["findings"]:
            print(f"- {finding}")
        print("\nNext steps:")
        for step in report["next_steps"]:
            print(f"- {step}")
    finally:
        await reviewer.aclose()
        backend.close()
        model.close()


asyncio.run(main())
```

Run the first investigation:

```bash
python review.py
```

The model selects tools, msgFlux executes them against the workspace, and the
agent continues until it produces the structured report. `acall()` lets your
application await the result without blocking its event loop.

Continue the conversation in a new process:

```bash
python review.py "Inspect the area you identified. Which tests cover it, and what is missing?"
```

The stable **thread ID** preserves the conversation across invocations. Each
invocation creates a new **run ID** for a new turn. To recover an interrupted run,
retain and reuse that run's ID rather than generating a new one; see
[Runtime](learn/nn/agent/runtime.md#checkpointing) for the recovery contract.
The same backend directory and agent name must be used when reopening this thread.

## Observe the investigation

To see tool activity while the agent works, replace the call and report-printing
block inside `try` with:

```python
async for event in reviewer.stream_events(request=task, scope=scope):
    if event.type == "tool.start":
        print(f"Using {event.data['tool_name']}", flush=True)
    elif event.type == "message.end" and len(event.source_path) == 1:
        print(event.data["content"])
```

This executes the same task while exposing lifecycle events. The root
`message.end` contains the final report; the source-path check keeps a
specialist's answer from being presented as the coordinator's final output.

Lifecycle observation also works without token streaming. For incremental model
text, enable `config={"stream": True}` on the Agent and consume `message.delta`
events. See [Event Streaming](learn/nn/agent/event-streaming.md) for event payloads
and error handling.

## Delegate a focused review

A specialist is another Agent. Before the `main()` definition, create one that
reuses the review extension with its own guidelines and instructions:

```python
from msgflux.tools.builtin import AgentTool


test_reviewer = nn.Agent(
    name="test_reviewer",
    description="Inspect tests for a specific component and report coverage gaps.",
    model=model,
    workspace=workspace,
    extensions=[
        ProjectReviewExtension(
            guidelines="Focus on test coverage and reproducible missing cases.",
        ),
    ],
    system_prompt=(
        "Review only the component and tests named in the delegated task. "
        "Do not read .env files or .msgflux runtime data. "
        "Return concise findings with file paths and concrete missing cases."
    ),
)
```

Register it in the coordinator's tool library, immediately after its definition:

```python
reviewer.tool_library.add(AgentTool())
reviewer.tool_library.add(test_reviewer)
```

You can also include `AgentTool()` and `test_reviewer` directly in `tools=` when
constructing the coordinator.

Then request the delegation:

```bash
python review.py "Ask test_reviewer to inspect tests for the component we discussed, then prioritize its findings."
```

`AgentTool` presents the registered specialist to the coordinator. The specialist
investigates in its own conversation and returns its result to the coordinator,
which produces the structured report. When adding this specialist, also call
`await test_reviewer.aclose()` during cleanup, before closing the shared backend
and model. See [AgentTool](learn/nn/agent/tools/agent-tool.md) and
[Background Tasks](learn/nn/agent/tools/background-tasks.md) for delegation that
can continue while the coordinator does other work.

## Run a coding agent from the CLI

For a ready-to-use coding harness, configure an account and model:

```bash
msgflux accounts login openai --account-name default
msgflux init --model openai/gpt-4.1-mini
msgflux -p "Map the project's entry points and suggest a focused review."
```

The CLI uses Agent Service and prints the final answer. For work that should
continue after the submitting command exits:

```bash
msgflux -p "Review the project and its tests. Report concrete issues with file paths." --detach --json
msgflux threads result THREAD_ID --wait
```

Use the thread ID returned by the first command in the second. This creates a
CLI-managed coding thread, separate from the Python review thread above.
[CLI](learn/nn/agent/coding-cli.md) covers continuation, model selection, and
monitoring. [Agent Service](learn/nn/agent/service.md) explains how to host your
own Agent factory and separate execution from connected clients.

## Extend this workflow

| Need | Feature |
|---|---|
| A larger tool catalog without sending every full schema initially | [Tool Search](learn/nn/agent/tools/tool-search.md) |
| Review and approve edits or commands | [Workspace and approvals](learn/nn/agent/runtime.md#workspace) |
| Manage model context as a thread grows | [Compaction](learn/nn/agent/compaction.md) |
| Add reusable workflows and lifecycle behavior | [Skills](learn/nn/agent/skills.md) and [Extensions](learn/nn/agent/extensions.md) |
| Combine agents with deterministic application logic | [Custom Modules](learn/nn/module/index.md) |
| Refine typed inputs and outputs | [Signatures](learn/nn/agent/signatures.md) |

Continue in [Learn](learn/index.md) for component details, or
[Guides](guides/index.md) for operating persistent agents.
