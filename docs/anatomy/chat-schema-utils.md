# Chat Schema Utils

`src/msgflux/utils/chat.py` looks broad at first glance because it contains both
message helpers and schema helpers.

From an anatomy perspective, the most important role of this module is:

- building provider-facing `response_format` payloads
- generating JSON schema for tools
- centralizing ChatML-shaped helper blocks used across providers and agents

This file is not the type-lowering engine. That job belongs to
[msgspec Transport Lowering](msgspec-transport-lowering.md).

Instead, `chat.py` is the schema-envelope layer.

## Why This File Exists

There are two related but different jobs in the structured-output stack:

- transform Python/msgspec types into transport-compatible shapes
- wrap those shapes into the exact JSON schema envelopes expected by providers

`msgspec.py` handles the first job.

`chat.py` handles the second.

That separation matters because the code that says:

```text
dict[K, V] -> entries wrapper
```

is not the same code that says:

```text
schema -> {"type": "json_schema", "json_schema": {...}}
```

## The Main Responsibilities

The schema-related responsibilities in this module are:

- `response_format_from_msgspec_struct(...)`
- `response_format_from_json_schema(...)`
- `hint_to_schema(...)`
- `generate_json_schema(...)`
- `generate_tool_json_schema(...)`

Those functions feed different parts of the runtime.

## 1. `response_format_from_msgspec_struct(...)`

This helper turns a `msgspec.Struct` into the OpenAI `response_format` object.

It does more than a plain `msgspec.json.schema(...)` call.

The function:

- generates the schema from the struct
- dereferences `$ref` definitions into a single inlined schema
- forces `additionalProperties: false` on object nodes
- forces all object properties into `required`
- removes the root title
- wraps the result in OpenAI's `json_schema` envelope

This is the point where a `msgspec.Struct` becomes a provider-ready structured
output contract.

### Why Inlining Matters

Inlining the schema definitions keeps the final `response_format` self-contained.

That makes the provider-facing payload easier to reason about and avoids
leaving schema references unresolved when the request is built.

### Why `additionalProperties: false` Matters

This is one of the reasons `chat.py` must stay aligned with the lowering layer.

By default, this helper closes object schemas aggressively. That is correct for
strict OpenAI structured outputs, but it means open-ended maps cannot be passed
through directly as plain JSON object schemas.

That is why `dict[K, V]` has to be lowered earlier into explicit `entries`
objects.

## 2. `response_format_from_json_schema(...)`

This helper is deliberately simple.

It wraps an already-built JSON schema into the provider envelope:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "...",
    "schema": { ... },
    "strict": true
  }
}
```

This is the path used when the provider schema is assembled dynamically without
going through a plain `msgspec.Struct`.

The current example is ReAct, which builds a provider-specific JSON schema from
the active tool list.

So the division is:

- `response_format_from_msgspec_struct(...)` for struct-driven schemas
- `response_format_from_json_schema(...)` for already-assembled JSON schemas

## 3. `hint_to_schema(...)`

This helper converts Python type hints into tool-parameter JSON schema
fragments.

It is used for tool schemas, not for the main structured output path.

That distinction is important:

- provider response schemas for `generation_schema` come from `msgspec`
- tool parameter schemas come from Python annotations and docstrings

It supports cases such as:

- primitives
- `Literal[...]`
- `List[T]`
- `Union[...]` and `Optional[T]`
- `Dict[K, V]`

Just like the structured-output path, `Dict[K, V]` is lowered to an
`entries`-based shape for compatibility with strict providers.

## 4. `generate_json_schema(...)`

This helper assembles a function-style schema from a tool class or callable.

It combines:

- the tool name
- the cleaned docstring description
- parsed `Args:` descriptions
- the annotation-derived parameter schema
- required vs optional field decisions

This is the schema-time bridge between a Python tool implementation and the
runtime/provider layers that need a JSON schema description of that tool.

## 5. `generate_tool_json_schema(...)`

This is the final wrapper used by `ToolLibrary`.

It converts the function-style schema into the provider-facing tool format:

```json
{
  "type": "function",
  "function": {
    "name": "...",
    "description": "...",
    "parameters": { ... },
    "strict": true
  }
}
```

This is what native tool-calling providers expect.

## Where `ToolLibrary` Uses It

The flow looks like this:

```text
Python tool annotations + docstring
  -> hint_to_schema(...)
  -> generate_json_schema(...)
  -> generate_tool_json_schema(...)
  -> ToolLibrary.get_tool_json_schemas()
  -> Agent / provider
```

That is why `chat.py` sits on the path from local tool declaration to provider
tool exposure.

## ChatML Helpers

The other half of the file is the `ChatBlock` / `ChatML` helpers.

These are intentionally thin utilities that normalize message construction for:

- user messages
- assistant messages
- system messages
- tool-call blocks
- multimodal parts such as image, audio, and file items

They matter architecturally because they keep providers and modules from
manually rebuilding ChatML dictionaries everywhere.

This is especially visible in places like:

- system prompt injection in providers
- tool call history construction
- multimodal user input handling

## Relationship To Other Layers

This file should be read together with:

- [OpenAI Chat Completion](openai-chat-completion.md)
- [msgspec Transport Lowering](msgspec-transport-lowering.md)
- [ToolLibrary](tool-library.md)

The division of labor is:

- `msgspec.py` lowers and restores types
- `chat.py` builds provider-ready schema envelopes and tool schemas
- provider adapters send those envelopes to real APIs

## ASCII Diagram

This is the schema path that runs through `chat.py`:

```text
msgspec.Struct
  -> response_format_from_msgspec_struct(...)
  -> OpenAI json_schema envelope
  -> provider request
```

And for tools:

```text
tool annotations + docstring
  -> hint_to_schema(...)
  -> generate_json_schema(...)
  -> generate_tool_json_schema(...)
  -> ToolLibrary.get_tool_json_schemas()
  -> Agent / provider
```

## Why This Shape Matters

Without `chat.py`, provider adapters would need to know too much about:

- how to inline and close JSON schema objects
- how to map tool annotations to function schemas
- how to build message blocks consistently

This module keeps those concerns centralized and reusable.

It is not the deepest type layer, but it is the layer that turns internal
schemas into the exact payloads that the rest of the stack can hand off to a
provider.

## Related Pages

- [OpenAI Chat Completion](openai-chat-completion.md)
- [msgspec Transport Lowering](msgspec-transport-lowering.md)
- [ToolLibrary](tool-library.md)
