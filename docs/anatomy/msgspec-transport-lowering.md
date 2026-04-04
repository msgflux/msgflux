# msgspec Transport Lowering

`src/msgflux/utils/msgspec.py` is where msgFlux turns the idea of
"logical schema versus provider schema" into concrete code.

This file is not only a bag of helpers. It contains the transport-typing layer
for structured outputs:

- recursive lowering of logical `msgspec` schemas into provider-compatible ones
- schema-aware restoration back into logical runtime values
- local validation of what the OpenAI lowering layer is willing to support

If `openai.py` is the adapter boundary, `msgspec.py` is the type-translation
engine behind it.

## Why This File Exists

The runtime wants readable types:

- `dict[str, str]`
- `list[dict[str, int]]`
- `Optional[dict[str, T]]`
- `ReAct(arguments: dict[str, Any] | None)` after normalization

Strict structured-output providers do not always accept those shapes directly.

So msgFlux needs an intermediate layer that can:

- preserve the logical schema
- compile a stricter transport schema
- restore the provider payload afterward

That work lives here.

## The Three Core Helpers

The transport layer is centered around three helpers:

- `lower_msgspec_struct_for_openai(...)`
- `restore_transport_value(...)`
- `restore_openai_structured_output(...)`

They each solve a different part of the contract.

## 1. `lower_msgspec_struct_for_openai(...)`

This function compiles a logical `msgspec.Struct` into an OpenAI-compatible
transport struct.

Its job is recursive:

- walk every field in the struct
- inspect the field's type hint
- lower unsupported shapes when a transport representation exists
- reject shapes that should fail early

The main lowered case is:

```text
dict[K, V]
  -> {entries: list[{key: K, value: V}]}
```

This is done without mutating the original logical struct. Instead, the helper
creates runtime-generated transport structs with `msgspec.defstruct(...)`.

### What It Preserves

The compiler preserves the logical schema as the source of truth.

That means:

- downstream code still reasons about the original struct
- provider adapters use the compiled transport struct only for request/response
- restoration later targets the original logical type again

### What It Rejects

The compiler also acts as a preflight validator for the OpenAI structured
output subset supported by msgFlux.

Examples of rejected shapes:

- `Any`
- `dict` without explicit `K` and `V`
- `list` without explicit item type
- unions other than `Optional[T]`
- `set` and `frozenset`
- mapping abstractions such as `Mapping[K, V]`

This is a design choice. The provider could reject some of these later, but the
library chooses to fail early with a local error that points to the exact field
path.

## 2. `restore_transport_value(...)`

This function performs the reverse operation.

It takes:

- a decoded transport value
- the original logical type hint

and restores the value recursively.

This function is intentionally broader than an OpenAI-only helper. It is used
in two important places:

- provider output normalization
- tool argument restoration before local execution

That reuse matters because it keeps the same restoration rules across provider
and tool boundaries.

### Strict And Non-Strict Modes

`restore_transport_value(...)` has two modes:

- `strict=True`
- `strict=False`

`strict=True` is used for provider structured outputs. In that mode, the
function expects the provider transport shape exactly and raises when the shape
does not match.

`strict=False` is used when restoring tool kwargs. That path is more permissive
because values may already be in logical form instead of transport form.

This distinction is subtle but important.

It allows the same restore engine to support both:

- "OpenAI must have returned the transport wrapper"
- "tool params may already be plain Python dicts"

## 3. `restore_openai_structured_output(...)`

This helper is the OpenAI-specific wrapper around
`restore_transport_value(...)`.

It simply fixes the OpenAI assumptions:

- `dict_factory=dotdict`
- `strict=True`

That makes it the right final step when decoding a provider response that used
transport lowering.

## Dict Lowering Example

Logical schema:

```python
class Output(Struct):
    metadata: dict[str, int]
```

Transport schema:

```text
Output
  -> OutputMetadataMap
     -> entries: list[OutputMetadataEntry]
```

Transport payload:

```json
{
  "metadata": {
    "entries": [
      {"key": "count", "value": 2},
      {"key": "retries", "value": 1}
    ]
  }
}
```

Restored logical payload:

```python
{"metadata": {"count": 2, "retries": 1}}
```

## Typed Restoration, Not Blind Conversion

The restore layer is not a generic "if it has `entries`, turn it into dict"
rule.

It uses the target type hint to decide how to rebuild values such as:

- primitive scalars
- enums
- nested structs
- tuples
- optional values
- dictionary keys with non-string types

This is why cases such as `dict[int, str]` can work correctly after
structured-output decoding.

## Relationship To Tool Execution

This file is not only about provider outputs.

The same restoration logic is reused by `LocalTool` before calling Python
implementations. That means a provider can return transport-lowered values,
ReAct can normalize them into logical action arguments, and local tools can
still receive correctly typed parameters.

Conceptually:

```text
logical schema
  -> lower for provider
  -> provider payload
  -> restore to logical response
  -> extract tool args
  -> restore tool kwargs
  -> local tool implementation
```

That shared restoration path is one of the main reasons the transport work
stays coherent across the stack.

## ASCII Diagram

This is the type-translation flow:

```text
logical msgspec.Struct
  -> lower_msgspec_struct_for_openai(...)
  -> transport msgspec.Struct
  -> response_format in provider
  -> provider returns JSON payload
  -> decode transport struct
  -> struct_to_dict(...)
  -> restore_openai_structured_output(...)
  -> logical runtime payload
```

And for tool params:

```text
tool annotation
  -> transport-shaped argument
  -> restore_transport_value(..., strict=False)
  -> logical Python kwargs
  -> LocalTool implementation
```

## Why This Shape Matters

Without this file, provider adapters would either:

- leak transport-specific shapes into runtime code, or
- duplicate custom lowering/restoration logic in multiple places

This module centralizes the type system boundary:

- compile transport structs once
- restore values once
- validate supported cases once

That is what makes the higher-level provider and flow-control code readable.

## Related Pages

- [OpenAI Chat Completion](openai-chat-completion.md)
- [Logical vs Provider Schema](logical-vs-provider-schema.md)
- [Dict Lowering and Restoration](dict-lowering-and-restoration.md)
- [ToolLibrary](tool-library.md)
