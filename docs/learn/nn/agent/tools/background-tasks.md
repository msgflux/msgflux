# Background Tasks

Background tasks let tools start work immediately while the agent continues with a
`task_id` handle. The result can be checked later through task tools, delivered
back through notifications, or continued through `task_message` when the task
declares the corresponding capability.

Use background tasks when a tool may be slow, long-running, interruptible, or
useful to monitor separately from the current model turn.

The main pieces are:

| Piece | Purpose |
|-------|---------|
| `background=True` | Always dispatch this tool in the background. |
| `allow_background=True` | Let the model choose background execution with `run_in_background`. |
| `background_capabilities` | Optional task controls supported by this background tool. |
| `TaskStore` | Stores task state, progress, output, activity, and routing metadata. |
| Task tools | `task_status`, `task_wait`, `task_output`, `task_list`, and `task_interrupt`. |
| Notifications | Completed, failed, and progress updates can be delivered back to the agent inbox. |
| Optional controls | `task_activity` and `task_message` appear only when a registered tool supports them. |

## Mental Model

```text
tool call
  -> immediate dispatch response with task_id
  -> background execution continues
  -> task state lives in TaskStore
  -> result is consumed later through task tools or notifications
```

## Execution Scope And Task Store

Pass a `TaskStore` through runtime context when task state should be shared by
the active execution scope. The scope supplies the thread/run identity; the task
store supplies the persistence boundary for background task records.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tasks import TaskStore

task_store = TaskStore.sqlite(path=".msgflux/tasks.sqlite3")
scope = mf.ExecutionScope(
    thread_id="customer_42",
    run_id="ticket_9001",
)

agent = nn.Agent(
    name="assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=[long_sum],
)

with mf.execution_context(scope=scope, task_store=task_store):
    dispatch = agent.tool_library([("call_1", "long_sum", {"a": 20, "b": 22})])
```

When the tool library runs inside this context, it reads the active
`task_store` and uses it for task creation, status, activity, output, and
interrupt requests. If no task store is provided, msgFlux creates an in-memory
store when background tools are used.

Task control tools are installed automatically while the library contains
background-capable tools. Removing the last `background=True` or
`allow_background=True` tool removes those task tools as well. You can still
remove an individual task tool manually; msgFlux will not reinstall that tool
while the current background tool set remains active.

## Background Capabilities

All background-capable tools install the common task controls. Optional controls
are installed from `background_capabilities`:

| Capability | Control | Meaning |
|------------|---------|---------|
| `activity` | `task_activity` | Read compact task activity. |
| `message` | `task_message` | Deliver a message or continue an agent task. |

`activity` is available to any background source. `message` is currently
reserved for agent sources because it requires an inbox and can continue an
agent through its checkpoint. For example, a future shell tool can expose
activity without accepting agent messages:

```python
@mf.tool_config(background=True, background_capabilities=["activity"])
def monitored_job(command: str) -> str:
    """Run a monitored job."""
    return command
```

`Agent` and `AgentTool` receive `activity` and `message` by default when they
run in the background. Another source kind can add message only together with
an equivalent runtime implementation.

## Basic Background Tool

```python
import time
import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()


@mf.tool_config(background=True)
def long_sum(a: int, b: int) -> int:
    """Compute a sum in the background."""
    time.sleep(2)
    return a + b


agent = nn.Agent(
    name="math_assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    instructions="Use tools when needed.",
    tools=[long_sum],
)

dispatch = agent.tool_library([("call_1", "long_sum", {"a": 20, "b": 22})])
print(dispatch.tool_calls[0].result)
# task_id=... status=running
task_id = dispatch.tool_calls[0].result.split()[0].split("=", 1)[1]

state = agent.tool_library([("call_2", "task_status", {"task_id": task_id})])
print(state.tool_calls[0].result)

result = agent.tool_library([("call_3", "task_output", {"task_id": task_id})])
print(result.tool_calls[0].result)
```

## Waiting For A Task

Sometimes the agent has nothing useful to do until the task finishes.

```python
wait_result = agent.tool_library(
    [("call_5", "task_wait", {"task_id": task_id, "timeout": 5.0})]
)
print(wait_result.tool_calls[0].result)
```

When the task completes, `task_wait` returns the same result as
`task_output(task_id)`. Other states use compact text such as
`status=timeout task_status=running progress=50%` or
`status=failed error=...`.

## Model-Chosen Background Execution

Use `allow_background=True` when a tool is useful both inline and in the
background. msgFlux exposes a reserved boolean argument named
`run_in_background` to the model.

```python
@mf.tool_config(allow_background=True)
def search_archive(query: str) -> list[str]:
    """Search the archive."""
    return expensive_archive_search(query)
```

If the model calls the tool with `run_in_background=true`, msgFlux strips that
argument before calling the Python function and dispatches the work as a
background task. If the model sets it to `false` or `null`, the tool runs
normally and returns its result inline. Manual callers may also omit the
argument, which is treated the same as `false`.

`background=True` still means the developer has forced every call to run in the
background. Use `allow_background=True` only when the model should decide.

## Interrupting A Task

`task_interrupt(task_id)` requests a cooperative interrupt.

```python
interrupt_result = agent.tool_library([("call_6", "task_interrupt", {"task_id": task_id})])
print(interrupt_result.tool_calls[0].result)
```

If the task has not started yet, msgFlux may interrupt it immediately. If it is
already running, the interrupt is observed at the next cooperative checkpoint. For
background subagents, that means before the next provider call.

## Reading Task Activity

`task_activity(task_id)` returns compact newline-separated activity for tasks
that declare the `activity` capability.

```python
activity = agent.tool_library([("call_6", "task_activity", {"task_id": task_id})])
print(activity.tool_calls[0].result)
```

For background agents it can include compact tool call entries such as:

```text
Status: Task queued.
Status: Task running.
ToolCall: search_docs({'query': 'task runtime'})
```

## Task Messaging And Subagent Continuation

For normal background tools, the task id is an operational handle. Background
agents declare the `message` capability by default. The built-in `AgentTool`
uses `task_message` to let the root model send another message to the same
running subagent or continue it from its checkpoint:

```python
task_message(
    task_id="ab12cd34",
    message="The user clarified that the payment already cleared.",
)
```

The task metadata records enough routing information to reconstruct the call:

- which tool owns the task
- which child agent or tool target was selected
- the checkpoint namespace for that child execution
- the parent/root run lineage
- the `thread_id` shared with the root conversation
- the task id used as the child `run_id`

For an agent task, `task_message` re-dispatches the same tool with the saved
routing parameters and a scope like:

```text
thread_id = original root thread
run_id = task_id
parent_run_id = root run that launched the task
root_run_id = root run of the whole execution tree
```

That means the child agent can recover through the normal checkpoint rule:
same `(namespace, thread_id, run_id)` resumes a non-terminal checkpoint. If the
child had completed and you want a new independent subagent conversation, call
the `agent` tool again so a new task id/run id is created. If you want to keep
talking to the same subagent task, use `task_message` with the existing
`task_id`.

## Reporting Progress

Grant `task.progress` when the tool should update its own progress. The handle
is hidden from the model schema and receives only the configured operations.

```python
import time
import msgflux as mf


@mf.tool_config(background=True, handle={"task": ["progress"]})
def process_items(
    items: list[str],
    handle: mf.Hidden,
) -> int:
    """Process items and report progress."""
    handle.task.progress(stage="prepare", message="Preparing work")

    total = len(items)
    for index, item in enumerate(items, 1):
        time.sleep(0.2)
        handle.task.progress(
            stage="process",
            message=f"Processed {item}",
            current=index,
            total=total,
        )

    return total
```

While the task is running, `task_status(task_id)` returns something like:

```python
{
    "task_id": "9b8e2f1a",
    "tool_name": "process_items",
    "status": "running",
    "progress": {
        "stage": "process",
        "message": "Processed b.txt",
        "current": 2,
        "total": 3,
        "percent": 66.67,
    },
}
```

It also includes timing helpers such as:

```python
{
    "started_at": "2026-04-14T14:00:00.000000+00:00",
    "running_for_seconds": 1.243,
    "last_activity_summary": "Progress: Processed b.txt",
}
```

## Passive Notifications Back Into The Agent

Completed and failed tasks are injected back into the next provider call as a
synthetic system message:

```xml
<notifications>
task_id=abcd1234 status=completed tool=long_sum
</notifications>
```

That tells the model about an unexpected state change without repeating tool
instructions. It can call `task_output`, `task_status`, or `task_activity` when
it needs details.

```python
agent = nn.Agent(
    name="assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    instructions=(
        "If you receive a task notification with status=completed, "
        "call task_output for that task before answering."
    ),
    tools=[long_sum],
)
```

## Progress Notifications

The same injected handle can publish lightweight agent-visible updates.

```python
@mf.tool_config(
    background=True,
    handle={"notifications": ["publish"], "task": ["read"]},
)
def process_items(
    items: list[str],
    handle: mf.Hidden,
) -> int:
    """Process items and publish progress notifications."""
    total = len(items)
    for index, item in enumerate(items, 1):
        handle.notifications.publish(
            source="task_progress",
            status="update",
            metadata={"item": item, "current": index, "total": total},
            dedupe_key=f"task_progress:{handle.task.read()['task_id']}",
        )
    return total
```

These notifications are persisted when the agent inbox uses an
`AgentInboxStore`. `dedupe_key` keeps the newest progress update for the same
task visible to the model.

## Sending A Message To A Background Subagent

When the background task is itself an `Agent`, the dispatch response also
advertises `task_activity` and `task_message(task_id=..., message=...)`.

```python
message_result = agent.tool_library(
    [("call_7", "task_message", {"task_id": task_id, "message": "Continue with compatibility mode."})]
)
print(message_result.tool_calls[0].result)
```

If the subagent is still running, the message is delivered into its local
inbox and will be consumed on the next provider boundary. If it already interrupted
but has a checkpoint, msgFlux resumes it with the same `task_id`.

## Status Updates With The Handle

Grant `notifications.publish` when the tool should publish lightweight status
updates.

```python
@mf.tool_config(background=True, handle={"notifications": ["publish"]})
def process_items(
    items: list[str],
    handle: mf.Hidden,
) -> int:
    """Process items and publish task-scoped status updates."""
    handle.notifications.publish(
        status="prepare",
        hint="Background work has started.",
        metadata={"total": len(items)},
        dedupe_key="process-items-status",
    )
    for index, item in enumerate(items, 1):
        handle.notifications.publish(
            status="process",
            metadata={"item": item, "current": index, "total": len(items)},
            dedupe_key="process-items-status",
        )
    return len(items)
```

For background tools, `handle.notifications` is automatically bound to the
current `task_id`.

## Dynamic Tool Mutation With The Handle

The `handle` dictionary exposes only named operations without exposing the
parameter to the model.

The current handle supports:

- `handle.tools.register(tool)`
- `handle.tools.remove(tool_name)`
- `handle.tools.list()`

```python
import msgflux as mf


def multiply(x: int) -> int:
    """Multiply a number by two."""
    return x * 2


@mf.tool_config(handle={"tools": ["list", "register"]})
def enable_multiplier(handle: mf.Hidden) -> list[str]:
    """Register the multiply tool."""
    handle.tools.register(multiply)
    return handle.tools.list()


@mf.tool_config(handle={"tools": ["list", "remove"]})
def disable_tool(
    handle: mf.Hidden,
    name: str,
) -> list[str]:
    """Remove a tool by name."""
    handle.tools.remove(name)
    return handle.tools.list()
```

If a hidden-handle tool adds a new background tool, the task control functions
are registered automatically in the same library.

## Related Pages

- [Tools](index.md)
- [Tool Config](config.md#runtime-injection-options)
- [Task Runtime](../../../../anatomy/task-runtime.md)

## Example Scripts

- `examples/background_task_wait_demo.py`
- `examples/background_task_notifications_demo.py`
- `examples/background_task_status_updates_demo.py`
