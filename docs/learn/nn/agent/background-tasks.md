# Background Tasks

Background tasks let a tool start work now and return the final result later.

The current design is intentionally small:

- `background=True` dispatches the tool and returns a `task_id`
- `task_status(task_id)` returns task state and progress
- `task_wait(task_id)` blocks until the task completes, fails, or times out
- `task_output(task_id)` returns the final output
- `task_list()` lists tasks visible in the current `ToolLibrary`
- `task_stop(task_id)` requests a cooperative stop
- completed and failed tasks can also be delivered back to the agent as a
  passive notification
- `inject_notification=True` lets a tool publish agent-visible status updates
- `inject_library=True` lets a tool add or remove tools dynamically
- `task_activity(task_id)` and `task_message(task_id, message)` are only
  exposed when the library contains a background subagent

This page focuses on the current behavior, not future multi-agent planning.

## Mental Model

```text
tool call
  -> immediate dispatch response with task_id
  -> background execution continues
  -> task state lives in TaskStore
  -> result is consumed later through task tools or notifications
```

## Example 1: Basic Background Tool

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
# The `long_sum` tool is running in the background with task_id='...'
# Use that task_id with `task_status`, `task_stop`, `task_wait`, or `task_output`

tasks = agent.tool_library([("call_2", "task_list", {})])
task_id = tasks.tool_calls[0].result[0]["task_id"]

state = agent.tool_library([("call_3", "task_status", {"task_id": task_id})])
print(state.tool_calls[0].result)

result = agent.tool_library([("call_4", "task_output", {"task_id": task_id})])
print(result.tool_calls[0].result)
```

## Example 1B: Waiting For A Background Task

Sometimes the agent has nothing useful to do until the task finishes.

```python
wait_result = agent.tool_library(
    [("call_5", "task_wait", {"task_id": task_id, "timeout": 5.0})]
)
print(wait_result.tool_calls[0].result)
```

When the task completes, `task_wait` returns the same payload as
`task_output(task_id)`. If the task fails, it returns the failed payload. If
the timeout is reached first, it returns a timeout payload with the current
task status and progress.

## Example 1D: Stopping A Background Task

`task_stop(task_id)` requests a cooperative stop.

```python
stop_result = agent.tool_library([("call_6", "task_stop", {"task_id": task_id})])
print(stop_result.tool_calls[0].result)
```

If the task has not started yet, msgFlux may stop it immediately. If it is
already running, the stop is observed at the next cooperative checkpoint. For
background subagents, that means before the next provider call.

## Example 1C: Reading Subagent Activity

`task_activity(task_id)` returns a compact list of activity entries, but only
for background subagent tasks.

```python
activity = agent.tool_library([("call_6", "task_activity", {"task_id": task_id})])
print(activity.tool_calls[0].result)
```

For background subagents it can include compact tool call entries such as:

```python
[
    "Status: Task queued.",
    "Status: Task running.",
    "ToolCall: search_docs({'query': 'task runtime'})",
]
```

## Example 2: Reporting Progress

Use `inject_task=True` when the tool should update its own progress.

```python
import time
import msgflux as mf


@mf.tool_config(background=True, inject_task=True)
def process_items(items: list[str], task) -> int:
    """Process items and report progress."""
    task.set_running(stage="prepare", message="Preparing work")

    total = len(items)
    for index, item in enumerate(items, 1):
        time.sleep(0.2)
        task.update_progress(
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

## Example 3: Passive Notification Back Into The Agent

Completed and failed tasks are injected back into the next provider call as a
synthetic user message:

```text
<system_note>
<notifications>
<notification source="task" ref="abcd1234" status="completed">
tool=long_sum
hint=Use task_output(task_id='abcd1234') if you need the result.
</notification>
</notifications>
</system_note>
```

That means the model can recover task output without polling manually on every
turn, as long as the prompt tells it what to do with these notifications.

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

## Example 3B: Progress Notifications

The same injected `task` handle can publish lightweight agent-visible updates.

```python
@mf.tool_config(background=True, inject_task=True)
def process_items(items: list[str], task) -> int:
    """Process items and publish progress notifications."""
    total = len(items)
    for index, item in enumerate(items, 1):
        task.notify(
            source="task_progress",
            status="update",
            metadata={"item": item, "current": index, "total": total},
            dedupe_key=f"task_progress:{task.task_id}",
        )
    return total
```

These notifications are persisted when the agent inbox uses an
`AgentInboxStore`. `dedupe_key` keeps the newest progress update for the same
task visible to the model.

## Example 3D: Sending A Message To A Background Subagent

When the background task is itself an `Agent`, the dispatch response also
advertises `task_activity` and `task_message(task_id=..., message=...)`.

```python
message_result = agent.tool_library(
    [("call_7", "task_message", {"task_id": task_id, "message": "Continue with compatibility mode."})]
)
print(message_result.tool_calls[0].result)
```

If the subagent is still running, the message is delivered into its local
inbox and will be consumed on the next provider boundary. If it already stopped
but has a checkpoint, msgFlux resumes it with the same `task_id`.

## Example 3C: Status Updates With `inject_notification`

Use `inject_notification=True` when the tool should publish lightweight status
updates without depending on the full `task` handle.

```python
@mf.tool_config(background=True, inject_notification=True)
def process_items(items: list[str], notification) -> int:
    """Process items and publish task-scoped status updates."""
    notification.update(
        "prepare",
        hint="Background work has started.",
        metadata={"total": len(items)},
        dedupe_key="process-items-status",
    )
    for index, item in enumerate(items, 1):
        notification.update(
            "process",
            metadata={"item": item, "current": index, "total": len(items)},
            dedupe_key="process-items-status",
        )
    return len(items)
```

For background tools, the injected `notification` handle is automatically bound
to the current `task_id`, so the agent sees a normal notification envelope with
`ref="<task_id>"`.

## Example 4: Dynamic Tool Mutation With `inject_library`

`inject_library=True` exposes a small `tool_library` handle to the tool.

The current handle supports:

- `tool_library.add(tool)`
- `tool_library.remove(tool_name)`
- `tool_library.list_tools()`

```python
import msgflux as mf


def multiply(x: int) -> int:
    """Multiply a number by two."""
    return x * 2


@mf.tool_config(inject_library=True)
def enable_multiplier(tool_library) -> list[str]:
    """Register the multiply tool."""
    tool_library.add(multiply)
    return tool_library.list_tools()


@mf.tool_config(inject_library=True)
def disable_tool(tool_library, name: str) -> list[str]:
    """Remove a tool by name."""
    tool_library.remove(name)
    return tool_library.list_tools()
```

If the injected tool adds a new background tool, the task tools are registered
automatically in the same library.

## Current Limits

This is the current scope of the implementation:

- task state is in-memory
- notifications are injected from `_prepare_model_execution()`
- `inspect_model_execution_params()` peeks notifications without consuming them
- there is no checkpoint resume yet
- there is no mailbox or agent-to-agent messaging yet

## Related Pages

- [Tools](tools.md)
- [Task Runtime](../../../anatomy/task-runtime.md)

## Example Scripts

- `examples/background_task_wait_demo.py`
- `examples/background_task_notifications_demo.py`
- `examples/background_task_status_updates_demo.py`
