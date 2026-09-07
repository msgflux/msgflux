# Execution Event Streaming

Use an execution event stream when an application needs the whole agent
lifecycle rather than only the final answer. Assistant content, model requests,
tool activity, failures, and run boundaries share one ordered stream.

## Usage

`stream_events()` is an async iterator. It runs the Agent and yields events as
they occur. Set `OPENAI_API_KEY` in the environment, then consume the events
inside an async function. This example defines a minimal Agent, prints streamed
text, and captures the final output and run outcome. The complete Agent
response is included in `message.end`; the following `run.end` closes the
execution without repeating that potentially large content:

```python
import asyncio

import msgflux as mf
import msgflux.nn as nn

agent = nn.Agent(
    name="incident_analyst",
    model=mf.Model.chat_completion(
        "openai/gpt-5.6-luna",
        api_mode="responses",
    ),
    config={"stream": True},
)

incident_log = """\
09:02 - Scanner A stopped sending inventory updates.
09:07 - Orders continued reserving stock from the last known snapshot.
09:18 - Scanner A was restarted and queued updates began arriving.
09:23 - Two orders had overlapping reservations for SKU-1842.
09:31 - New reservations for SKU-1842 were paused.
"""


async def main() -> None:
    final_output = None
    outcome = None

    task = (
        "Summarize the failure sequence, customer impact, and immediate next "
        f"action from this incident log:\n\n{incident_log}"
    )

    async for event in agent.stream_events(task):
        if event.type == "message.delta":
            print(event.data["delta"], end="", flush=True)
        elif event.type == "message.end":
            final_output = event.data["content"]
        elif event.type == "run.end":
            outcome = event.data["outcome"]

    print(f"\n\noutcome={outcome}")
    print(f"final_output={final_output}")


asyncio.run(main())
```

`stream_events()` and `config={"stream": True}` control different layers.
`stream_events()` exposes Agent lifecycle events; the config enables incremental
output from the model. Without `stream=True`, the lifecycle stream still emits
events such as `model.request`, `message.end`, and `run.end`, but it has no
model chunks from which to produce `message.delta`.

## Watching An Existing Thread

`stream_events()` starts and owns one execution. Use `watch(thread_id)` when a
UI, websocket, or worker needs to attach to a thread that may already be
running:

```python
import asyncio

import msgflux as mf
import msgflux.nn as nn

agent = nn.Agent(
    name="incident_analyst",
    model=mf.Model.chat_completion(
        "openai/gpt-5.6-luna",
        api_mode="responses",
    ),
    config={"stream": True},
)

incident_log = """\
09:02 - Scanner A stopped sending inventory updates.
09:07 - Orders continued reserving stock from the last known snapshot.
09:18 - Scanner A was restarted and queued updates began arriving.
09:23 - Two orders had overlapping reservations for SKU-1842.
09:31 - New reservations for SKU-1842 were paused.
"""


def render_snapshot(snapshot) -> None:
    """Render state that existed before the live subscription began."""
    print(f"watching thread={snapshot.thread_id}")

    if snapshot.streaming_message:
        print(snapshot.streaming_message, end="", flush=True)

    for tool in snapshot.running_tools:
        print(f"\nalready running tool={tool.tool_name}")


def update_ui(event) -> None:
    """Apply one live event to this example's terminal UI."""
    if event.type == "message.delta":
        print(event.data["delta"], end="", flush=True)
    elif event.type == "tool.start":
        print(f"\nrunning tool={event.data['tool_name']}")


async def main() -> None:
    thread_id = "warehouse-incident-42"
    task = (
        "Explain why the overlapping reservations require reconciliation and "
        f"recommend the immediate operational action from this log:\n\n{incident_log}"
    )
    run = asyncio.create_task(
        agent.acall(
            task,
            scope=mf.ExecutionScope(thread_id=thread_id),
        )
    )

    async with agent.watch(thread_id) as watcher:
        render_snapshot(watcher.snapshot)

        async for event in watcher:
            update_ui(event)
            if event.type in {"run.end", "run.error"}:
                break

    await run


asyncio.run(main())
```

Entering the async context atomically captures a `ThreadSnapshot` and subscribes
to later events. Events produced after the capture are buffered until iteration
starts, so there is no gap between reading the snapshot and receiving live
updates.

The snapshot contains:

| Field | Description |
| --- | --- |
| `messages` | Latest durable `ChatMessages` when the Agent has a checkpoint store; otherwise `None` |
| `active_runs` | All process-local runs currently active in the thread, including nested agents |
| `running_tools` | Tool calls that started but have not settled |
| `background_tasks` | Background tasks that were dispatched but have not settled |
| `active_run` | Convenience access to the most recently started active run |
| `streaming_message` | Convenience access to that run's accumulated assistant output |

Each active-run snapshot also carries accumulated clear-text reasoning and
reasoning summaries when the provider exposes them. This lets a reconnecting UI
render the partial state first and then apply only the new deltas.

Agent executions publish to the process-local event hub even when
`stream_events()` is not active. Without watchers, the runtime maintains only
the state needed for a live snapshot: it does not allocate subscriber queues or
retain a replayable event log. Once runs, tools, and background tasks settle,
that live projection is discarded. Durable conversation recovery continues to
come from the checkpoint store.

`watch()` observes but does not start, resume, or cancel a run. Leaving its
context only unsubscribes that watcher. It also does not replay events that
occurred before the snapshot; their current effects are represented by the
snapshot instead.

When the model itself streams, the event iterator owns and consumes that model
stream. Do not separately consume a `ModelStreamResponse`: the content is
already delivered through `message.delta`, reasoning through
`reasoning.delta`, provider summaries through `reasoning_summary.delta`, and
the complete output through `message.end`.

Reasoning events preserve their order relative to visible output. They are
also emitted for model calls that lead to a tool invocation, before
`tool.start`, rather than being hidden inside the Agent's tool loop:

```text
model.request
reasoning_summary.delta
model.response
tool.start
tool.end
model.request
reasoning_summary.delta
message.delta
model.response
```

Providers that expose clear-text reasoning produce `reasoning.delta`. OpenAI
Responses produces `reasoning_summary.delta`; its opaque reasoning state is
retained for same-provider replay but is not presented as chain-of-thought. A
non-streaming model response emits its available reasoning or summary as one
delta before `model.response`.

## Event Shape

Every `ExecutionEvent` includes:

| Field | Description |
| --- | --- |
| `type` | Stable lifecycle event name |
| `timestamp` | UTC ISO 8601 emission time |
| `data` | Event-specific payload |
| `run_id` | Current run identity |
| `source_path` | Nested `type:name` identities from the observed root to the emitter |

Consume events in iterator order. Live event ordinals are not persisted or
exposed; a future durable event log would own a separate storage sequence.
Timestamps are useful for diagnostics but should not be used to reorder events.

`run.start` carries the less frequently changing execution context in its
`data`: `thread_id`, `namespace`, `root_run_id`, and `parent_run_id` for a
child operation when one exists. Foreground subagents may participate in the
same logical run as their caller; their nested `source_path` distinguishes
them. All following events remain correlatable through `run_id` and
`source_path` without repeating context on every token delta.

## Core Events

| Event | Meaning |
| --- | --- |
| `run.start` | Execution was accepted |
| `turn.start` | An agent turn started |
| `model.request` | A request is being sent to a model |
| `model.response` | The model response settled; includes compact token usage and timing when available |
| `message.start` | Assistant output presentation started |
| `message.delta` | Assistant content chunk |
| `reasoning.delta` | Reasoning content chunk, when exposed by the provider |
| `reasoning_summary.delta` | Provider reasoning-summary chunk |
| `message.end` | Complete assistant output |
| `tool.start` | A validated tool call is starting |
| `tool.blocked` | A tool policy rejected the call before execution; includes the public arguments and reason |
| `tool.update` | Intermediate tool progress |
| `tool.end` | Tool execution completed or failed |
| `task.start` | A background task was dispatched |
| `task.update` | Background task status or progress changed |
| `task.end` | A background task completed, failed, paused, or was interrupted |
| `compaction.start` | Context compaction started after the threshold policy approved it |
| `compaction.end` | Compaction completed or failed; includes compact usage when available but never the context view |
| `turn.end` | The turn reached a terminal boundary; it has no output payload |
| `run.end` | The run completed and reports its outcome |
| `run.error` | Execution failed; the iterator raises after this event |

## Model Metrics

`model.response` is emitted once for every LM request. For a streamed model
response, it arrives after the model's content deltas have been drained, when
terminal usage and timing are available:

```python
import asyncio

import msgflux as mf
import msgflux.nn as nn

agent = nn.Agent(
    name="incident_analyst",
    model=mf.Model.chat_completion(
        "openai/gpt-5.6-luna",
        api_mode="responses",
    ),
    config={"stream": True},
)

incident_log = """\
09:02 - Scanner A stopped sending inventory updates.
09:07 - Orders continued reserving stock from the last known snapshot.
09:18 - Scanner A was restarted and queued updates began arriving.
09:23 - Two orders had overlapping reservations for SKU-1842.
09:31 - New reservations for SKU-1842 were paused.
"""


async def main() -> None:
    task = (
        "Summarize the failure sequence, customer impact, and immediate next "
        f"action from this incident log:\n\n{incident_log}"
    )

    async for event in agent.stream_events(task):
        if event.type == "model.response":
            usage = event.data.get("usage", {})
            timing = event.data.get("timing", {})

            print("input tokens:", usage.get("input_tokens"))
            print("output tokens:", usage.get("output_tokens"))
            print("cached input tokens:", usage.get("cached_input_tokens"))
            print("latency (ms):", timing.get("latency_ms"))
            print("TTFT (ms):", timing.get("ttft_ms"))


asyncio.run(main())
```

The payload is intentionally compact. `usage` contains `input_tokens`,
`output_tokens`, and `cached_input_tokens` when reported; provider-native raw
usage, derived totals, and cache percentages are omitted. `timing` contains
`source`, `latency_ms`, and streaming `ttft_ms` when available.

Metrics stay on their individual `model.response` events instead of being
summed into `run.end`. This keeps multi-step tool loops and nested agents
auditable: each model call retains its own `run_id` and `source_path`, and a
consumer can aggregate only the calls relevant to its view.

Closing an event iterator early requests cancellation through the execution's
abort signal. Async execution is also cancelled locally so an abandoned client
does not retain ownership of the stream.

## Complete-Output Transformations

For Agents, a `Hook(event="transform_output", ...)` receives an `OutputContext`
with `output`, runtime `vars`, and the execution `scope`. Replacing `ctx.output`
changes the value presented to the caller without changing the canonical
assistant message stored in history. If the model streams tokens, the event
iterator accumulates assistant content and sets
`message.start.data["buffered"]` to `True`. It then emits the transformed value
once in `message.end`; raw assistant deltas are not exposed.

Reasoning, tool, and progress events are independent and continue to arrive
while assistant content is buffered. Reasoning and summaries are never passed
through `transform_output`.

### Stream completion and terminal hooks

An Agent stream owns its terminal checkpoint. When the provider closes the
`ModelStreamResponse`, the runtime first invokes `before_run_end`, persists the
settled conversation, and then invokes `after_run_end`. The same ordering is
used by synchronous and asynchronous Agent calls. A consumer that closes
`stream_events()` also waits for the execution task to finish its cancellation
cleanup, so the checkpoint is not left in the temporary `streaming` state.

See [Canonical Responses vs. Presented Output](hooks.md#canonical-responses-vs-presented-output)
for a complete example that replaces an artifact reference in the presented
event output while preserving the original reference in `ChatMessages`.

## Event Isolation

Each direct `stream_events()` call owns an independent delivery channel.
Concurrent calls therefore do not receive each other's events through those
iterators. The process-local hub additionally routes events by `thread_id`, so a
thread watcher intentionally sees every run, foreground nested execution, and
background task belonging to that thread. Use `run_id` and `source_path` to
separate the root Agent, ToolLibrary, called tool, and subagent while retaining
delivery order.

A detached background task can outlive the root `stream_events()` iterator. A
thread watcher continues receiving its `task.update`, nested Agent, tool, and
terminal events after that root iterator closes.

The built-in hub is process-local. It covers concurrent Agents and background
workers in the same Python process, but it is not a distributed broker and does
not survive a process restart. After a restart, `watch()` restores the durable
conversation snapshot and observes only newly produced live events. Applications
running multiple worker processes need to route one thread to one process or
bridge events through their own broker.

## Live OpenAI Validation

The repository includes a paid integration script for the Responses protocol:

```bash
uv run python scripts/validate_openai_event_streaming.py
```

It loads `OPENAI_API_KEY` from `.env`, uses `store=False`, and validates basic
token streaming, complete-output transformation, a local tool loop, and a
foreground AgentTool delegation. The `reasoning` scenario also verifies that
OpenAI Responses summaries precede visible output. Use `--scenario basic`,
`--scenario reasoning`, `--scenario transform`, `--scenario tool`, or
`--scenario nested` to run one case.
