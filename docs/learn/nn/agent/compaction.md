# Conversation Compaction

Conversation compaction keeps the model-visible context bounded without
rewriting the Agent's canonical history. Enable it by installing a
`CompactionExtension` with a `CompactionPolicy`.

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
endpoint; models without an exact counter use msgFlux's conservative heuristic.
Compatible providers can opt into the same server-side behavior by attaching an
`OpenAIResponsesContextAdapter` to the corresponding API mode. Merely supporting
Responses generation does not enable native context operations.

## Portable And Provider Views

The default model contract supports two complete view formats:

- `messages` is a portable summary produced through an ordinary model call.
- `provider` contains opaque continuation state for one provider and API mode.

OpenAI Responses uses its native compact endpoint and preserves the complete
returned output as a provider view. Chat Completions and other models use the
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

A standalone hook does not enable compacting by accident. A replacement
extension must return `nn.CONTEXT_COMPACTION_CAPABILITY` from its
`capabilities()` method and contribute a `before_compaction` hook. The Model
still owns token counting and creation of the compacted view; hooks control only
the decision and threshold metadata.

See [Hooks](hooks.md) for lifecycle-hook registration and replacement rules.
