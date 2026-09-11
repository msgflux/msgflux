# Compaction And Context Views

This page describes the implemented durability and projection contract for
conversation compaction.

## Ownership

Compaction crosses three layers, with one responsibility in each:

- `Agent` coordinates the safe history boundary, lifecycle hook, checkpoint,
  and execution events.
- `CompactionExtension` applies the default threshold policy through
  `before_compaction` and explicitly enables the `context_compaction`
  capability; applications may replace that policy.
- `Model` owns token counting and creates the complete compacted context view.
  Provider endpoints and wire formats remain below this boundary.

The checkpoint store only persists the resulting timeline. It does not count
tokens, call a model, or mutate earlier history.

## Request Boundary

The Agent checks compaction after it has prepared the upcoming model request.
The estimate therefore includes the current user message, rendered system
prompt, tool catalog, task context, and drained notifications.

Only a completed turn can end a compacted prefix. At the start of a new turn,
the latest completed turn becomes the source boundary and the current user
message remains in the suffix. A conversation with no completed turn cannot be
compacted automatically.

Compaction is not repeated inside the same turn or tool loop. If the newest
operation already refers to the selected completed-turn boundary, the Agent
continues without another count or compaction call.

## Context Capacity And Counting

Every chat model exposes:

```text
context_capacity: int | None
```

The base contract first uses an explicit positive `context_length`, then the
model profile's context limit. `None` means unknown, not unlimited. A
`CompactionPolicy.context_capacity` override takes precedence; without either
source, the default extension skips automatic compaction.

The default policy computes:

```text
ratio_limit = int(context_capacity * trigger_ratio)
reserve_limit = context_capacity - reserved_output_tokens - safety_margin_tokens
trigger_tokens = max(1, min(ratio_limit, reserve_limit))
```

Compaction starts when the estimated input is greater than or equal to this
trigger. Token counting belongs to `Model`: OpenAI Responses uses its native
input-token count endpoint, while the base model contract provides a
provider-neutral heuristic. A `ModelGateway` counts with the selected model and
passes through the active `model_preference`.

Native context operations are declared per API mode through a context adapter.
`OpenAIResponsesContextAdapter` prepares and decodes the compatible
`/responses/input_tokens` and `/responses/compact` operations. Credentials,
base URL construction, retry, abort handling, and HTTP errors remain transport
responsibilities. Providers must opt in explicitly; sharing the Responses
generation protocol alone does not imply native compaction support.

## Append-Only Operation

Compaction appends one operation to `ChatMessages`. It never changes, removes,
or disables earlier items:

```python
{
    "type": "compaction",
    "item_id": "itm_01...",
    "parent_compaction_id": "itm_previous...",  # omitted for the first one
    "reason": "threshold",
    "compacted_through_item_id": "itm_00...",
    "views": [
        {
            "format": "messages",
            "items": [
                {
                    "role": "system",
                    "content": (
                        "<conversation_summary>\n"
                        "Portable continuation state\n"
                        "</conversation_summary>"
                    ),
                }
            ],
        }
    ],
    "metadata": {
        "provider": "openai",
        "model_id": "gpt-5.6-luna",
        "api_mode": "responses",
        "input_tokens_before": 118400,
        "estimate_source": "provider",
        "usage": {"input_tokens": 118400, "output_tokens": 920},
    },
    "timestamp": "...",
}
```

Each operation currently contains the single complete view returned by
`Model.compact_context()` or `Model.acompact_context()`. A view is not a patch
against an older summary.

`parent_compaction_id` links successive projections. It is an audit and future
ancestry reference only: materialization still selects the newest operation,
and no mutable Scope head is introduced by compaction v1.

`ChatMessages.add_compaction()` validates that the source item is a completed
turn event, that at least one view exists, and that provider views declare both
their provider and API mode.

## View Formats

A portable compactor returns a `messages` view. The base implementation asks
the model for a continuation summary and stores it as a system message wrapped
in `<conversation_summary>`.

A native compactor can return a provider-bound view:

```python
{
    "format": "provider",
    "provider": "openai",
    "api_mode": "responses",
    "items": [...],
}
```

Provider items are opaque. `ChatMessages` preserves their order and content
without trying to extract a portable summary.

OpenAI Responses uses the standalone compact endpoint and stores its complete
output this way. OpenAI Chat Completions uses the portable fallback. A
`ModelGateway` always disables native compaction because its normal fallback
behavior may select another provider for a later request.

## Materialization And Repeated Compaction

The newest compaction operation defines the context base. Projection is:

```text
newest compatible complete view
  + timeline items appended after compacted_through_item_id
```

Earlier operations remain available for audit and forks, but their views do
not stack into the prompt. This keeps model context bounded after repeated
compactions and removes the need for an `active` flag.

Materialization first looks for a provider view matching the active provider
and API mode, then for a portable `messages` view. If neither is available, it
raises an actionable error instead of dropping the compacted prefix. The
canonical append-only timeline remains unchanged by materialization.

## Hook And Event Boundaries

Before creating a view, the Agent first requires an installed extension with
the `context_compaction` capability, then sends `BeforeCompaction` through the
`before_compaction` lifecycle hooks. Its payload contains:

- canonical messages and runtime `scope`/`vars`;
- estimated input tokens and whether the estimate is provider-native or
  heuristic;
- context capacity and computed trigger;
- reason, source item boundary, and `compact`/`skip` action.

The built-in extension replaces the capacity, trigger, and action according to
its policy. If the final action is `skip`, the Agent emits no compaction events.

For an approved operation, the order is:

```text
compaction.start
Model.compact_context() or Model.acompact_context()
append compaction operation
checkpoint save, when configured
compaction.end(status="completed")
model.request
```

If counting or the policy hook fails, no operation has started. If view
creation, append, or checkpoint persistence fails after `compaction.start`, the
Agent emits `compaction.end` with `status="failed"` and re-raises the error.

Event payloads contain boundary, capacity, estimate, trigger, format, model id,
and compact usage when available. They deliberately omit summary text and
opaque provider items. Publication uses the ordinary execution event hub, so
both `stream_events()` and `watch()` observe the same operation.

## Durability

The Agent appends the operation before saving a running checkpoint. If the
checkpoint succeeds but event delivery is interrupted, replay sees that the
latest operation already covers the selected boundary and does not compact it
again.

Without a checkpoint store, the same operation remains in the caller's
in-memory `ChatMessages`. Checkpoint state contains the complete canonical
timeline, including old source items and all compaction operations; only the
model-facing projection is bounded.

## OpenAI Endpoint Constraint

The standalone Responses compact endpoint is stateless and compatible with a
locally managed, ZDR-oriented timeline. Its input must still fit the model
context window, so the threshold reserves output and safety space and must
trigger before overflow.

OpenAI parameters used for ordinary generation, such as `store`, reasoning
effort, and maximum output tokens, are not forwarded to the compact endpoint.
The returned output is continuation state for Responses, not a
provider-neutral summary.

## Design References

- [OpenAI standalone compact endpoint](https://developers.openai.com/api/docs/guides/compaction#standalone-compact-endpoint)
- [Pi harness v2 design](https://github.com/earendil-works/pi/blob/harness-v2/j4/packages/agent/docs/harness-v2.md)
