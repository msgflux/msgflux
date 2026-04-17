# Dict Lowering and Restoration

Strict structured output providers often accept closed objects more reliably
than arbitrary maps.

That becomes a problem for types such as:

- `dict[str, str]`
- `list[dict[str, int]]`
- nested structures that contain `dict[K, V]`

msgFlux solves this with a transport-only lowering step.

## Lowered Shape

A logical `dict[K, V]` is encoded as an object with `entries`:

```json
{
  "entries": [
    {"key": "city", "value": "Austin"},
    {"key": "country", "value": "USA"}
  ]
}
```

This keeps the provider-facing schema closed and explicit, which is friendlier
to strict structured output systems.

## Where Lowering Happens

Lowering belongs to the provider-facing layer, not to the public runtime API.

That means:

- the user still works with `dict[K, V]`
- the provider receives the lowered `entries` form
- the response is restored back to a normal `dict`

The same strategy is also used when tool parameters contain dictionaries.

## Restoration

Restoration is schema-aware. msgFlux does not blindly turn every `entries`
object into a dictionary.

The restore step uses the original type information to rebuild values such as:

- `dict[str, T]`
- `dict[int, T]`
- nested dictionaries
- dictionaries inside lists, tuples, and structured objects

The restore logic is centralized so the same rules can be reused in provider
adapters and in tool parameter reconstruction.

## Why This Is Better Than Changing The Runtime Type

Changing the runtime type would leak provider limitations into application code.

For example, replacing a logical `dict[str, str]` with a permanent
`entries: list[{key, value}]` shape would make internal code less readable just
to satisfy one provider.

Lowering keeps the runtime contract intact while still allowing strict
structured outputs.

## Related Pages

- [Logical vs Provider Schema](logical-vs-provider-schema.md)
- [ReAct Provider Schemas](react-provider-schemas.md)
