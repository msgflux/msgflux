# ToolFlowControl

`ToolFlowControl` is the extension point for custom tool loops in msgFlux.

It exists for the cases where the default `Agent` tool loop is not enough, but
you still want to reuse the rest of the stack: prompt assembly, provider
adapters, tool execution, retries, history handling, and response preparation.

In practice, `ToolFlowControl` is the alternative to modifying the original
agent loop directly.

## When To Use It

Use a flow control when the model must produce structured state that drives a
multi-step tool loop.

Common examples:

- ReAct-style `thought -> action -> observation` loops
- planners that alternate between planning and execution
- evaluators that decide whether another tool round is needed
- custom tool policies that cannot be expressed with the default tool call path

Do not use a flow control when the default tool calling behavior already
matches the problem. A custom flow control adds a new internal contract and
should exist only when the loop itself is part of the feature.

## What A Flow Control Owns

A `ToolFlowControl` does not execute tools by itself. It owns the logic that
interprets model output as loop state.

Its responsibilities are:

- `extract_flow_result`: read the model output and decide whether the loop is
  complete or whether tool calls must be executed
- `inject_results`: place tool outputs back into the structured state
- `build_history`: convert the current step into the next round of assistant
  history

The async methods mirror the same contract. Most flow controls can reuse the
default async implementations.

## Lifecycle

The loop looks like this:

```text
user/task
  -> Agent prepares prompt and provider params
  -> model returns structured flow state
  -> flow control extracts tool calls
  -> ToolLibrary executes tools
  -> flow control injects observations
  -> flow control appends step history
  -> model runs again or final answer is returned
```

The important boundary is that `ToolFlowControl` works on structured state,
not on raw provider payloads.

## Provider Hooks

Most flow controls can use their logical generation schema directly. For those
cases, the base class defaults are enough:

- `build_provider_response_format(...) -> None`
- `normalize_provider_response(raw_response, ...) -> raw_response`

These hooks only matter when the provider should see a different schema than
the runtime loop shape.

That situation appears when:

- a provider requires a stricter schema than the runtime structure
- the flow wants a model-friendly transport shape, but runtime code wants a
  normalized shape
- part of the schema depends on runtime context, such as the active tool list

ReAct is the current example of this pattern.

## Why This Matters

Without `ToolFlowControl`, custom reasoning loops tend to leak into the core
agent loop. That creates two problems:

- new reasoning styles become hard to add without editing stable infrastructure
- provider-specific schema concerns end up mixed with generic tool execution

`ToolFlowControl` keeps the extension localized. The default tool loop stays
intact, while custom loops get a place to define their own structure and state
transitions.

## Related Pages

- [Logical vs Provider Schema](logical-vs-provider-schema.md)
- [ReAct Provider Schemas](react-provider-schemas.md)
