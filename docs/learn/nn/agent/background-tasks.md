# Background Tasks

Background tasks let a tool start work now and return the final result later.

The current design is intentionally small:

- `background=True` dispatches the tool and returns a `task_id`
- `allow_background=True` lets the model choose background execution per call
  through the reserved `run_in_background` tool argument
- `task_status(task_id)` returns task state and progress
- `task_wait(task_id)` blocks until the task completes, fails, or times out
- `task_output(task_id)` returns the final output
- `task_list()` lists tasks visible in the current `ToolLibrary`
- `task_interrupt(task_id)` requests a cooperative interrupt
- completed and failed tasks can also be delivered back to the agent as a
  passive notification
- `inject_notification=True` lets a tool publish agent-visible status updates
- `inject_handle=True` lets a tool add or remove tools dynamically
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

## Task Messaging And Subagent Continuation

When a tool is allowed to run in the background, msgFlux creates a task record
and returns a task id to the model. The task record lives in the task store and
tracks lifecycle state such as queued, running, paused, completed, failed, and
interrupted.

For normal background tools, the task id is mostly an operational handle:

- `task_status(task_id=...)` reads lifecycle and progress.
- `task_output(task_id=...)` reads the final result once available.
- `task_wait(task_id=...)` waits for completion.
- `task_interrupt(task_id=...)` requests interruption.

Some background tools also support messages after dispatch. The built-in
`AgentTool` does this for subagents. In that case, `task_message` is the way
for the root model to send another message to the same running or resumable
task:

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
# Use that task_id with `task_status`, `task_interrupt`, `task_wait`, or `task_output`

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

## Example 1C: Letting The Model Choose Background Execution

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

## Example 1D: Stopping A Background Task

`task_interrupt(task_id)` requests a cooperative interrupt.

```python
interrupt_result = agent.tool_library([("call_6", "task_interrupt", {"task_id": task_id})])
print(interrupt_result.tool_calls[0].result)
```

If the task has not started yet, msgFlux may interrupt it immediately. If it is
already running, the interrupt is observed at the next cooperative checkpoint. For
background subagents, that means before the next provider call.

## Example 1E: Reading Subagent Activity

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
synthetic system message:

```text
<system_note>
<notification>
source: task
ref: abcd1234
status: completed
tool: long_sum
hint: Use task_output(task_id='abcd1234') if you need the result.
</notification>
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
inbox and will be consumed on the next provider boundary. If it already interrupted
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
to the current `task_id`, so the agent sees a normal notification block with
`ref: <task_id>`.

## Example 4: Dynamic Tool Mutation With `inject_handle`

`inject_handle=True` exposes a small `handle` to the tool.

The current handle supports:

- `handle.add(tool)`
- `handle.remove(tool_name)`
- `handle.list_tools()`

```python
import msgflux as mf


def multiply(x: int) -> int:
    """Multiply a number by two."""
    return x * 2


@mf.tool_config(inject_handle=True)
def enable_multiplier(handle) -> list[str]:
    """Register the multiply tool."""
    handle.add(multiply)
    return handle.list_tools()


@mf.tool_config(inject_handle=True)
def disable_tool(handle, name: str) -> list[str]:
    """Remove a tool by name."""
    handle.remove(name)
    return handle.list_tools()
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
