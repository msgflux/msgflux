# Execution Event Streaming

This page describes the internal event path behind `Module.stream_events()` and
`Agent.watch(thread_id)`. The public usage contract lives in the
[Agent event-streaming guide](../learn/nn/agent/event-streaming.md).

## Design Goals

The runtime needs to support two related but different operations:

- start an execution and consume all of its events with `stream_events()`;
- attach to an existing thread, reconstruct its current presentation state,
  and consume only future events with `watch()`.

The implementation preserves these invariants:

1. reasoning, reasoning summaries, and assistant output retain provider order;
2. nested agents and background tasks remain observable;
3. reconnecting cannot lose an event between snapshot capture and subscription;
4. events are not duplicated merely because both APIs are active;
5. live events are not persisted as a second conversation history;
6. no subscriber queue or completed projection is retained without need.

## Layered Flow

The event path has four layers:

```text
provider stream
    ↓
ModelStreamResponse ordered LM journal
    ↓
Module/Agent lifecycle adapter
    ↓
_EventSink
    ├── direct _AsyncEventChannel → stream_events()
    └── process-local EventHub    → watch(thread_id)
```

Each layer has a narrow responsibility:

- `ModelStreamResponse` preserves provider event order without knowing about
  agents, threads, or presentation.
- `Module` and `Agent` convert model events into execution lifecycle events.
- `_EventSink` adds execution identity and timestamps exactly once.
- `EventHub` reduces events into live thread state and fans them out to active
  watchers.

## Ordered Model Events

Providers do not expose reasoning uniformly. A streaming response may contain:

```text
reasoning.delta
reasoning_summary.delta
output.delta
```

The old channel-specific consumers remain available, but consuming those
queues concurrently cannot reconstruct cross-channel order. For that reason,
`BaseStreamResponse` also owns a response-local ordered journal of
`LMStreamEvent` objects.

Every producer call such as `add_reasoning()`, `add_reasoning_summary()`, or
`add()` appends one event to that journal. `consume_events()` first replays the
events already produced and then follows future events until the response
settles.

The journal is multicast rather than a single-consumer queue. This matters for
normal streamed Agent calls: the runtime can consume the response to publish
thread events while application code independently consumes or replays the same
`ModelStreamResponse`. The journal belongs to the response object and disappears
with it; it is not a global or durable event log.

## Lifecycle Adaptation

`Module._aconsume_event_response()` maps ordered LM events into public execution
events:

| LM event | Execution event |
| --- | --- |
| `output.delta` | `message.delta` |
| `reasoning.delta` | `reasoning.delta` |
| `reasoning_summary.delta` | `reasoning_summary.delta` |

For streamed responses, `model.response` is registered as a consumer finalizer.
It is emitted only after the stream settles, when response type, token usage,
latency, and TTFT are available. Non-streaming responses emit their available
reasoning and summary immediately before `model.response`.

Intermediate model responses that select a tool are drained before the tool is
executed. Consequently the observable order remains:

```text
model.request
reasoning_summary.delta
model.response
tool.start
tool.end
```

This prevents tool-loop reasoning from becoming an opaque internal detail.
A policy rejection instead produces one `tool.blocked` event with the public
arguments and reason. Because the implementation never starts, it does not
produce `tool.start` or `tool.end`.

## Event Identity

`_EventSink.emit()` creates the immutable `ExecutionEvent`. It adds:

- a UTC ISO 8601 timestamp;
- the current `run_id`;
- a nested `source_path` built from `event_source()` contexts;
- thread, namespace, parent-run, and root-run identity on `run.start`.

Context that remains stable for the run is intentionally carried by
`run.start`, rather than repeated on every token delta. A nested path can look
like:

```text
agent:root
agent:root → tool_library:root_tools → agent:reviewer
background:research → agent:researcher
```

There is no public `sequence` field. Consumers use iterator order. A durable
event log, if introduced later, would need its own storage-level sequence and
replay semantics instead of persisting a redundant in-memory ordinal.

## Direct Execution Streams

`stream_events()` creates an `_AsyncEventChannel` and runs the module under its
sink. Every emitted event is delivered to that channel and also published once
to the shared hub.

The iterator owns the execution it starts. If its consumer closes early, the
runtime requests cancellation through the execution's abort signal and cancels
the local async task. On success it finishes with `message.end`, `turn.end`, and
`run.end`; on failure it emits `run.error` before re-raising.

The local channel ends with its root execution. It therefore cannot by itself
follow a background task that outlives the root. That is the role of a thread
watcher.

## Publication Without `stream_events()`

An Agent run installs a hub-only sink even when it is called through `call()` or
`acall()`. This is required because a watcher may attach after the run starts.
Waiting until the first watcher appears would make earlier partial state
impossible to reconstruct.

A non-streaming result emits its terminal message and run boundaries before the
call returns. A `ModelStreamResponse` is finalized by a detached event consumer:

- async calls retain an `asyncio.Task` until it settles;
- sync calls run the consumer through the shared executor with a copied
  `contextvars` context.

Because the model journal is multicast, this detached consumer does not steal
chunks from application code.

## Process-Local Event Hub

`EventHub` is an internal process-wide service keyed by `thread_id`. Every
publication performs two actions under one reentrant lock:

1. reduce the event into the thread's live projection;
2. enqueue the same event for the thread's current watchers.

It does not retain completed events. When no run, tool, or background task is
active, the projection is removed as soon as the final watcher unsubscribes.
When no watcher exists, no subscriber queue is allocated.

The live projection contains only state required to open a UI mid-execution:

- active runs and their namespaces;
- accumulated assistant output;
- clear-text reasoning and reasoning summaries;
- running tool calls and their arguments;
- unsettled background tasks and their latest progress.

Streaming chunks are kept as lists and joined only when a snapshot is created.
This avoids quadratic string concatenation while still providing complete
partial text to reconnecting clients.

## Atomic Watch Subscription

The unsafe form of reconnect is:

```text
load snapshot ── event occurs ── subscribe
                         ↑ event is lost
```

`ThreadWatcher.__aenter__()` instead asks the hub to load durable messages,
capture the live projection, and register the watcher while holding the same
lock used for publication:

```text
hub lock
  ├── load latest ChatMessages
  ├── copy active runs, tools, and tasks
  └── register subscriber
release lock
```

Publishers that arrive during this operation wait for the lock and then enqueue
their events to the newly registered watcher. Events are buffered in its async
queue until iteration begins.

The returned `ThreadSnapshot` combines two kinds of state:

- durable conversation messages loaded from the Agent's checkpoint store;
- ephemeral process-local state reduced by the hub.

Leaving the context unsubscribes only that watcher. It does not cancel, pause,
or resume the observed execution, and the watcher does not terminate
automatically at `run.end` because another run or background task may continue
on the same thread.

## Nested Agents And Background Tasks

Nested modules inherit the active event sink through `contextvars`.
`event_source()` extends their `source_path`, and run identity separates
parallel or nested work without creating another public event stream.

Background work crosses an executor boundary, so `BackgroundTaskDispatcher`
reinstalls a hub sink and its execution context in the worker. Task state
transitions produce:

```text
task.start   # dispatched or requeued
task.update  # running or progress changed
task.end     # completed, failed, paused, or interrupted
```

An agent running as a background task publishes its nested model, reasoning,
message, and tool events through that sink. The task projection remains active
after the root Agent reaches `run.end`, allowing a later watcher to reconnect
and continue observing it.

## Durability Boundary

Execution events are presentation and observability data. They are deliberately
not written to `CheckpointStore`:

- `ChatMessages` remains the durable interaction timeline;
- task stores remain authoritative for durable background-task state;
- the hub contains only current in-process presentation state;
- reconnect after process restart begins from durable state and receives only
  newly produced events.

This avoids duplicating assistant messages in both conversation history and an
event log. It also keeps future replay design independent from live websocket
delivery.

The built-in hub does not coordinate multiple Python processes. Deployments
with several workers must route a thread to one process or bridge events through
an external broker. Such a broker can reuse `ExecutionEvent`, but it should not
be hidden inside `CheckpointStore`.

## Main Implementation Files

| File | Responsibility |
| --- | --- |
| `src/msgflux/_private/response.py` | Ordered, multicast LM event journal |
| `src/msgflux/nn/events.py` | Model-response event adaptation and compact metrics |
| `src/msgflux/nn/modules/module.py` | Lifecycle boundaries, direct streams, and detached consumers |
| `src/msgflux/runtime/events.py` | Event shape, sink, timestamps, and source identity |
| `src/msgflux/runtime/event_hub.py` | Live projections, atomic watch subscription, and fan-out |
| `src/msgflux/runtime/background.py` | Background execution context and task publication |
| `src/msgflux/tasks/handle.py` | Task progress and terminal transition events |
