# Logical vs Provider Schema

msgFlux separates two different schema layers:

- the logical schema used by the runtime
- the provider-facing schema used to request structured output

These are often the same. They should not be assumed to be the same.

## Logical Schema

The logical schema is the shape that msgFlux wants to work with after the model
response is consumed and normalized.

Examples:

- a signature turned into a `msgspec.Struct`
- a generation schema such as `ChainOfThought`
- a flow control shape such as `ReAct(actions=[...])`

This is the schema that downstream code should reason about.

## Provider Schema

The provider schema is the shape exposed to a specific model provider for
structured output.

Its job is different:

- satisfy provider restrictions
- make the output easy for the model to produce
- encode runtime values in a provider-compatible way

For some providers, the logical schema can be sent directly. For others, a
transport shape is required first.

## Why The Split Exists

This split exists because provider constraints are not the same as runtime
needs.

Examples:

- OpenAI structured outputs reject arbitrary `dict[str, T]` objects, so msgFlux
  lowers them to an `entries` representation
- ReAct wants to consume `Action(arguments: dict[str, Any])` at runtime, but
  OpenAI needs a concrete schema built from the active tool list

If msgFlux forced the logical schema to always match provider constraints, the
runtime model would become harder to read and less reusable.

## The Contract

When a provider-specific transport schema is needed, the flow is:

```text
logical schema
  -> provider response format
  -> model output
  -> provider normalization
  -> logical schema again
```

In code, the two relevant hooks are:

- `build_provider_response_format(...)`
- `normalize_provider_response(...)`

For plain structured outputs, the provider may also lower and restore specific
types without involving a flow control.

## Design Rule

The runtime should stay optimized for readability and correctness.

The provider schema should stay optimized for compatibility and model
generation.

If a schema starts looking unnatural in runtime code because of provider
restrictions, it is usually a sign that a provider-facing transport schema is
missing.

## Related Pages

- [ToolFlowControl](tool-flow-control.md)
- [Dict Lowering and Restoration](dict-lowering-and-restoration.md)
- [ReAct Provider Schemas](react-provider-schemas.md)
