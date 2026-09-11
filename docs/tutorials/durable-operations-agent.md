# Durable Operations Coordinator

<span class="tag tag-orange">Advanced</span><span class="tag tag-gray">Responses</span><span class="tag tag-gray">Tool Search</span><span class="tag tag-gray">Background Tasks</span>

Build a warehouse-operations coordinator that can discover a runbook on demand,
delegate analysis to a specialist, launch a reconciliation in the background,
report progress, and resume safely after interruption.

This tutorial combines the runtime features that become important when an agent
does more than one provider call:

- OpenAI Responses behind the normal chat-completion frontend;
- hosted tool search without shared-library mutation;
- several specialist agents behind one `AgentTool`;
- `ExecutionScope`, checkpoints, and an abort signal;
- background tasks, task progress, passive notifications, and `AgentInbox`.

## Architecture

```text
operator request
      |
      v
operations_coordinator (OpenAI Responses)
      |-- tool_search ----> deferred runbook lookup
      |-- agent ----------> inventory specialist
      `-- reconcile_inventory (background)
                              |
                              |-- TaskStore: status, progress, output
                              `-- AgentInbox: completion notification

ExecutionScope + CheckpointStore surround the whole flow.
```

The example uses a small in-process data set so every tool result is
reproducible. Only the model calls require network access.

## 1. Configure Responses And Durable Stores

Install the OpenAI extra, put `OPENAI_API_KEY` in `.env`, and create one model
plus the three storage boundaries used by the workflow:

```python
# pip install "msgflux[openai]"
import time

import msgflux as mf
import msgflux.nn as nn
from msgflux.chat_messages import ChatMessages
from msgflux.tasks import TaskStore
from msgflux.tools.builtin import AgentTool


mf.load_dotenv()

model = mf.Model.chat_completion(
    "openai/gpt-5.6-luna",
    api_mode="responses",
    reasoning_effort="low",
    store=False,
)

checkpoint_store = mf.Store.checkpoint(
    "sqlite",
    path=".msgflux/operations-checkpoints.sqlite3",
)
task_store = TaskStore.sqlite(path=".msgflux/operations-tasks.sqlite3")
inbox_store = mf.Store.agent_inbox(
    "sqlite",
    path=".msgflux/operations-inbox.sqlite3",
)
agent_inbox = mf.AgentInbox(store=inbox_store)
```

`Model.chat_completion(...)` remains the common frontend. `api_mode="responses"`
makes the OpenAI provider convert messages, tools, output limits, and reasoning
parameters at the model boundary.

`store=False` explicitly disables Responses application-state storage for the
request. It does not by itself enable OpenAI Zero Data Retention: ZDR is an
organization or project policy, and individual hosted capabilities have their
own eligibility rules. See OpenAI's
[data controls](https://platform.openai.com/docs/models/default-usage-policies-by-endpoint)
when retention requirements are part of your deployment.

Each local store has one responsibility:

| Store | Persists |
|-------|----------|
| `CheckpointStore` | Agent history and resumable run state. |
| `TaskStore` | Background status, progress, activity, and final output. |
| `AgentInboxStore` | Notifications waiting for an agent provider boundary. |

## 2. Add A Deferred Runbook Tool

The coordinator should not pay the schema cost for every runbook on every
turn. `defer_loading=True` keeps this tool in the logical catalog until it is
needed:

```python
RUNBOOKS = {
    "overlapping_reservations": (
        "Pause new reservations, compare scanner events with order reservations, "
        "reconcile physical stock, then release only verified inventory."
    ),
    "scanner_lag": (
        "Record the last accepted sequence, restart the scanner, and replay "
        "queued updates idempotently before reopening reservations."
    ),
}


@mf.tool_config(defer_loading=True)
def query_runbook(topic: str) -> str:
    """Return the warehouse runbook for an incident topic."""
    return RUNBOOKS.get(topic, f"No runbook found for {topic!r}.")
```

With OpenAI Responses, msgFlux compiles this catalog to hosted tool search.
OpenAI performs selection inside the Responses request and msgFlux retains the
native search call/output items in canonical history. Other providers use the
portable local `tool_search` fallback, whose activated names are stored in
`ChatMessages`. Neither path mutates the shared library, so two conversations
can use different tool subsets while sharing the coordinator instance.

## 3. Add A Specialist Behind AgentTool

`AgentTool` captures agents registered directly in the same tool library. The
specialist remains a normal `nn.Agent`; no constructor-specific list is needed:

```python
inventory_specialist = nn.Agent(
    name="inventory_specialist",
    model=model,
    description=(
        "Analyzes scanner timelines, reservation conflicts, and customer impact."
    ),
    instructions=(
        "Separate observed facts from hypotheses. Return the likely failure "
        "sequence, affected orders, and the next safe action."
    ),
)
```

Later, the coordinator will register both `AgentTool()` and
`inventory_specialist`. Because every agent has `tool_kind="agent"`, the library
captures the specialist and exposes only `agent(name, message)` to the root
model.

## 4. Add A Background Tool With Progress

The reconciliation has deterministic stages and deliberately takes long enough
to observe. `background=True` immediately returns a dispatch acknowledgement
containing a `task_id`. The injected handle updates the task record and checks
for cooperative interruption:

```python
@mf.tool_config(
    background=True,
    inject_handle=True,
    background_capabilities=["activity"],
)
def reconcile_inventory(sku: str, handle: mf.Hidden) -> dict:
    """Reconcile reservations and physical stock for one SKU."""
    stages = (
        ("scanner_events", "Loaded scanner events"),
        ("reservations", "Compared active reservations"),
        ("physical_stock", "Verified physical stock"),
    )

    handle.set_running(stage="prepare", message=f"Starting reconciliation for {sku}")

    for current, (stage, message) in enumerate(stages, 1):
        handle.raise_if_interrupted()
        time.sleep(0.25)
        handle.update_progress(
            stage=stage,
            message=message,
            current=current,
            total=len(stages),
        )

    return {
        "sku": sku,
        "physical_units": 7,
        "reserved_units": 6,
        "overlapping_order_ids": ["ORD-1042", "ORD-1048"],
        "safe_to_reopen": False,
    }
```

The library adds the common task controls automatically. The `activity`
capability additionally exposes `task_activity`; an ordinary Python function
does not support `message`, which is reserved for background agent
continuation.

The function's final dict is stored once in `TaskStore`. Completion generates a
small inbox notification containing the `task_id`; the result itself is read
with `task_output` or `task_wait`, avoiding a duplicate copy in conversation
history.

## 5. Build The Coordinator

Register every component through the coordinator's tool list. The ordering is
not significant, but placing `AgentTool()` first makes the intended public
surface easy to see:

```python
coordinator = nn.Agent(
    name="operations_coordinator",
    model=model,
    checkpoint_store=checkpoint_store,
    agent_inbox=agent_inbox,
    instructions=(
        "For reservation incidents, search for and use the matching runbook. "
        "Delegate timeline analysis through agent(name='inventory_specialist', ...). "
        "Start reconcile_inventory in the background. Do not invent its result: "
        "when completion is reported, call task_output with the notification ref."
    ),
    tools=[
        AgentTool(),
        inventory_specialist,
        query_runbook,
        reconcile_inventory,
    ],
)
```

The initial callable surface now has one agent dispatcher, one background
operation with its applicable task controls, and hosted tool search. The full
runbook schema remains deferred.

## 6. Run With An Execution Scope

Keep conversation history in `ChatMessages`, and give the run a stable
`thread_id` and `run_id`:

```python
messages = ChatMessages(thread_id="warehouse_incident_42")
abort_signal = mf.AbortSignal()
initial_scope = mf.ExecutionScope(
    thread_id="warehouse_incident_42",
    run_id="initial_response",
    abort_signal=abort_signal,
)

incident = """
09:02 - Scanner A stopped sending inventory updates.
09:07 - Orders continued to reserve stock from the last known snapshot.
09:18 - Scanner A restarted and queued updates began arriving.
09:23 - ORD-1042 and ORD-1048 overlapped on SKU-1842.
09:31 - New reservations for SKU-1842 were paused.
"""

with mf.execution_context(scope=initial_scope, task_store=task_store):
    response = coordinator(
        "Use the runbook and specialist, then start reconciliation for "
        f"SKU-1842 in the background. Incident log:\n{incident}",
        messages=messages,
        scope=initial_scope,
    )

print(response)
```

`thread_id` identifies the conversation and scopes portable tool activation.
`run_id` identifies this resumable execution. The checkpoint is saved under the
coordinator namespace plus those IDs. If the run fails or pauses, repeat the
call with the same scope to resume it; use a new `run_id` for a new turn in the
same conversation.

## 7. Inspect Progress And Wait For Output

The model can call task tools itself. An application can use the same tools
directly for a status panel or test harness:

```python
with mf.execution_context(scope=initial_scope, task_store=task_store):
    task_records = coordinator.tool_library(
        [("inspect_1", "task_list", {"status": None})]
    ).tool_calls[0].result

    reconciliation = next(
        task for task in task_records if task["tool_name"] == "reconcile_inventory"
    )
    task_id = reconciliation["task_id"]

    status = coordinator.tool_library(
        [("inspect_2", "task_status", {"task_id": task_id})]
    ).tool_calls[0].result
    print(status["status"], status["progress"])

    output = coordinator.tool_library(
        [("inspect_3", "task_wait", {"task_id": task_id, "timeout": 5.0})]
    ).tool_calls[0].result
    print(output)
```

`task_status` returns the latest progress snapshot. `task_wait` blocks only the
caller using it and returns the same final value as `task_output` when the task
finishes.

## 8. Let AgentInbox Deliver Completion

The background dispatcher also publishes a passive notification. Use a new
run in the same thread for the follow-up turn:

```python
follow_up_scope = mf.ExecutionScope(
    thread_id="warehouse_incident_42",
    run_id="report_reconciliation",
)

with mf.execution_context(scope=follow_up_scope, task_store=task_store):
    final_response = coordinator(
        "Report the reconciliation result and the next safe action.",
        messages=messages,
        scope=follow_up_scope,
    )

print(final_response)
```

Before the next provider call, `AgentInbox` drains the completion into a
synthetic system notification. Its `ref` is the `task_id`; following the
coordinator instructions, the model uses that value with `task_output`. The
notification is then part of canonical history and is not injected a second
time.

## 9. Abort An Active Run

`AbortSignal` is a local control for the active process. A UI, CLI handler, or
supervising thread can request cancellation through the same object:

```python
# Called from the control path while the run is still active.
abort_signal.abort("Operator cancelled the incident workflow.")
```

The runtime observes the request at safe boundaries: before provider output is
committed, before tool execution, or before a later provider call. Open tool
calls are closed as interrupted and the checkpoint records the terminal state.
Long-running Python work should also call `handle.raise_if_interrupted()` at
cooperative checkpoints, as the reconciliation tool does.

An abort signal is process-local and is not a durable cross-process control.
Use agent-inbox control messages or `task_interrupt` when an external process
must request interruption.

## What Each Primitive Owns

| Primitive | Use it for | Do not use it for |
|-----------|------------|-------------------|
| `ChatMessages` | Canonical trajectory and thread-local loaded tools. | Background task output storage. |
| `ExecutionScope` | Thread, run, namespace, lineage, and local abort identity. | Holding stores. |
| `CheckpointStore` | Resuming agent execution. | Background progress polling. |
| `TaskStore` | Background records, progress, activity, and output. | Chat history. |
| `AgentInbox` | Pending notifications and control messages. | Duplicating full task results. |
| `ToolBucket` | Grouping implementations behind one public dispatcher. | Managing background-control lifecycles. |

## Further Reading

- [Runtime](../learn/nn/agent/runtime.md)
- [Tool Search](../learn/nn/agent/tools/tool-search.md)
- [AgentTool](../learn/nn/agent/tools/agent-tool.md)
- [ToolBucket](../learn/nn/agent/tools/tool-bucket.md)
- [Background Tasks](../learn/nn/agent/tools/background-tasks.md)
