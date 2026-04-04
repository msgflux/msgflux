# ReAct Provider Schemas

ReAct is the current example of a flow control whose runtime shape should not
be exposed directly to the provider.

Its logical shape is simple:

```python
class Action(Struct):
    name: str
    arguments: dict[str, Any] | None
```

That is the right shape for runtime code. It is not the right shape for strict
structured output providers.

## Why ReAct Needs A Provider Schema

The type of `arguments` is not fixed ahead of time.

It depends on the selected tool. That means a static field like
`arguments: dict[str, Any]` is too broad for providers that require an explicit
schema.

ReAct solves this by building a provider-specific response format from the
active tool list.

## Provider-Facing Shape

Instead of exposing a generic `arguments` field, ReAct flattens the selected
tool parameters into each action variant.

Example:

```json
{
  "thought": "Store the fields in one call.",
  "actions": [
    {
      "name": "store_fields",
      "fields": {
        "entries": [
          {"key": "city", "value": "Austin"},
          {"key": "country", "value": "USA"}
        ]
      }
    }
  ],
  "final_answer": null
}
```

This gives the provider a concrete schema:

- the action `name` is fixed per variant
- tool parameters are explicit
- parameter dictionaries can still use the usual lowering rules

## Runtime Normalization

After the provider response is decoded, msgFlux normalizes it back to the
logical ReAct shape:

```python
{
    "thought": "Store the fields in one call.",
    "actions": [
        {
            "name": "store_fields",
            "arguments": {
                "fields": {"city": "Austin", "country": "USA"}
            },
        }
    ],
    "final_answer": None,
}
```

That is the structure consumed by the flow control and by tool execution.

## Why This Design Won

This approach keeps three properties at the same time:

- tool arguments stay readable at runtime
- providers receive a strict, explicit schema
- tool parameters remain typed by the real tool annotations

It is better than:

- keeping `arguments` as a generic pair list forever
- sending JSON as strings
- trying to model every possible JSON shape with a single broad union

## Related Pages

- [ToolFlowControl](tool-flow-control.md)
- [Logical vs Provider Schema](logical-vs-provider-schema.md)
- [Dict Lowering and Restoration](dict-lowering-and-restoration.md)
