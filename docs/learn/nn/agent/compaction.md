# Conversation Compaction

Conversation compaction keeps the model-visible context bounded without
rewriting the Agent's canonical history. Install a `CompactionExtension` with
a `CompactionPolicy` for automatic compaction, or call the manual API for an
existing paused or failed execution.

## Enable Automatic Compaction

Set `OPENAI_API_KEY` in the environment, then run this example. The deliberately
small `context_capacity` makes the second turn trigger compaction without
requiring a genuinely long conversation:

```python
import asyncio

import msgflux as mf
from msgflux import nn

mf.load_dotenv()

history = mf.ChatMessages(
    thread_id="warehouse-incident-42",
    namespace="incident_analyst",
)
history.begin_turn(turn_id="initial-report")
history.add_user("Scanner A stopped publishing inventory updates at 09:02.")
history.add_assistant(
    "Orders may reserve stale inventory until the scanner recovers."
)
history.end_turn()

agent = nn.Agent(
    name="incident_analyst",
    model=mf.Model.chat_completion(
        "openai/gpt-5.6-luna",
        api_mode="responses",
    ),
    system_prompt="Preserve incident facts, decisions, and unresolved actions.",
    extensions=[
        nn.CompactionExtension(
            nn.CompactionPolicy(
                context_capacity=64,
                trigger_ratio=0.8,
                reserved_output_tokens=0,
                safety_margin_tokens=0,
            )
        )
    ],
)


async def main() -> None:
    answer = await agent.acall(
        "Scanner A is online again. What should operations verify next?",
        messages=history,
        scope=mf.ExecutionScope(
            thread_id="warehouse-incident-42",
            run_id="follow-up",
        ),
    )

    operation = history.latest_compaction()
    print(answer)
    print(operation["reason"])
    print(operation["views"][0]["format"])


asyncio.run(main())
```

The extension counts the complete upcoming request, including the new user
message, but compacts only through the latest completed turn. The new task
therefore remains verbatim after the compacted view. With OpenAI Responses, the
view comes from the native compact endpoint and its format is `provider`.

The small capacity above is only for making the example deterministic. In an
application, omit `context_capacity` when the model profile already reports the
correct context window.

## Configure The Threshold

`CompactionPolicy` combines four values:

| Field | Default | Meaning |
| --- | ---: | --- |
| `trigger_ratio` | `0.8` | Compact when estimated input reaches this fraction of the context capacity. |
| `reserved_output_tokens` | `4096` | Keep this many tokens available for model output. |
| `safety_margin_tokens` | `1024` | Reserve additional space for estimation and protocol overhead. |
| `context_capacity` | `None` | Override the model's known context capacity. |

The effective trigger is the smaller of the ratio limit and the capacity left
after both reserves. If neither the model nor the policy provides a capacity,
automatic compaction remains disabled: unknown does not mean unlimited.

Models own token counting. OpenAI Responses uses the provider input-token count
endpoint. Codex subscription models use msgFlux's conservative heuristic, as do
models without an exact counter.
Compatible providers can opt into the same server-side behavior by attaching an
`OpenAIResponsesContextAdapter` to the corresponding API mode. Merely supporting
Responses generation does not enable native context operations.

## Portable And Provider Views

The default model contract supports two complete view formats:

- `messages` is a portable summary produced through an ordinary model call.
- `provider` contains opaque continuation state for one provider and API mode.

OpenAI Responses uses its native compact endpoint and preserves the complete
returned output as a provider view. The `openai-codex` subscription provider uses
the current Codex protocol: a `compaction_trigger` control appended to an SSE
Responses request. It reconstructs a complete view from the retained user and
developer messages plus the opaque output. The trigger is never saved in history.
The response must contain exactly one encrypted compaction item; a normal answer,
a malformed checkpoint or an HTTP failure does not become a successful compaction. Chat Completions and other models use the
portable summary fallback unless they implement their own native compactor.
A `ModelGateway` always requests a portable view because a later fallback may
select a different provider.

Provider views are intentionally not decoded into summaries. If the newest
compaction has no portable view and does not match the selected provider and
API mode, msgFlux raises an error instead of silently discarding context.

## History And Checkpoints

Compaction appends an operation to `ChatMessages`; it does not remove or mark
old messages inactive. The operation records:

- the completed-turn item through which history was compacted;
- one complete model-visible view;
- provider, model, API mode, token estimate, and compaction usage metadata.

From the second operation onward, `parent_compaction_id` points to the previous
compaction operation. This records projection ancestry without duplicating
messages or changing the active thread head.

For a request, msgFlux materializes the newest compatible view and appends only
timeline items after that operation's boundary. Older compactions remain in
canonical history for audit and forks, but their views are not stacked into the
prompt.

When the Agent has a `checkpoint_store`, it saves the new operation before the
model request. Replaying the same boundary does not compact it a second time.
See [Runtime](runtime.md#checkpointing) for checkpoint configuration.

## Compact A Saved Execution Manually

Manual compaction accepts an `ExecutionScope`, not a message list. It loads the
latest checkpoint for that thread and bypasses the automatic token threshold.
No `CompactionExtension` is required for an explicit call; installed
`before_compaction` hooks still run with `reason="manual"` and may veto it.

For an existing Agent with an atomic checkpoint store and a paused or failed run:

```python
from msgflux.runtime import ExecutionScope

scope = ExecutionScope(thread_id="warehouse-incident-42")
result = await agent.acompact(scope=scope)
print(result.status, result.reason, result.run_id)
```

The scope selects the saved conversation. An optional `run_id` must identify its
latest saved run; older runs cannot be compacted. Synchronous applications use
`agent.compact(scope=scope)` instead. The scope also carries the workspace,
account and authorization context used by the host.

Only the prefix through the latest completed turn is compacted. The current
unfinished turn stays verbatim. The operation preserves the run ID, paused or
failed status, pending approval batch, counters and extension state. It neither
resumes execution nor grants approval. A pending approval remains pending;
a tool that already started and has an uncertain outcome requires reconciliation
before compaction is allowed.

`CompactionResult` contains `thread_id`, `run_id`, `status`, `reason`, optional
`operation_id` and `usage`. Its status is `completed` after the checkpoint commit,
or `skipped` with one of these reasons:

| Reason | Meaning |
| --- | --- |
| `no_completed_turn` | There is no completed prefix to compact. |
| `already_compacted` | That boundary already has a compaction. |
| `policy_skipped` | A lifecycle hook declined the operation. |

Active calls, background work and competing maintenance on the same Agent thread
raise `CompactionBusyError` or `CompactionRecoveryRequiredError`. Uncertain tool
effects and unresolved command receipts require reconciliation. The embedded
host must also ensure that no other process still owns the execution.

The checkpoint store must support atomic commits. The operation commits against
the revision it loaded: a competing writer causes `CheckpointConflictError`
instead of overwriting its changes. Provider failures and cancellation before
commit leave the saved checkpoint intact. Cancelling after a successful commit
does not undo that commit. Close an Agent only after its maintenance call has
finished or been cancelled and joined.

See [AgentService HTTP](service-http.md#compact-a-paused-or-failed-run) for the
service endpoint and bound session API.

## Observe Compaction

`stream_events()` and `watch()` expose two compact events:

- `compaction.start` reports the threshold decision and source boundary.
- `compaction.end` reports completion or failure, the selected format, and
  usage when available.

The events do not contain the summary or opaque provider state. See
[Execution Event Streaming](event-streaming.md#core-events) for the shared
event contract.

## Customize The Decision

`CompactionExtension` explicitly enables the `context_compaction` capability
and implements the `before_compaction` lifecycle hook. Additional hooks may
modify its `BeforeCompaction` decision sequentially, for example to skip
compaction for a particular tenant.

A standalone hook does not enable automatic compaction. A replacement
extension must return `nn.CONTEXT_COMPACTION_CAPABILITY` from its
`capabilities()` method and contribute a `before_compaction` hook. The Model
still owns token counting and creation of the compacted view; hooks control only
the decision and threshold metadata.

See [Hooks](hooks.md) for lifecycle-hook registration and replacement rules.
