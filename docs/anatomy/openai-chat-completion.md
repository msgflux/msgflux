# OpenAI Chat Completion

`src/msgflux/models/providers/openai.py` contains several OpenAI-backed model
classes, but the architectural center of the file is `OpenAIChatCompletion`.

This provider is where msgFlux translates the generic `Agent` contract into the
OpenAI chat completions API.

That translation is not a thin pass-through. The provider is responsible for:

- building OpenAI request parameters
- adapting msgFlux output contracts to OpenAI constraints
- decoding provider responses into `ModelResponse`
- restoring provider transport payloads back to logical runtime shapes
- enforcing option combinations that should fail early

## Why This Provider Matters

This module is one of the main boundaries between msgFlux runtime semantics and
provider semantics.

Upstream code thinks in terms such as:

- `generation_schema`
- `typed_parser`
- `ToolFlowControl`
- `prefilling`
- `tool_definitions`

OpenAI expects:

- `messages`
- `response_format`
- `tools`
- `tool_choice`
- provider-specific payload shapes

`OpenAIChatCompletion` is the adapter between those two worlds.

## Main Flow

The non-streaming path looks like this:

```text
Agent
  -> OpenAIChatCompletion.__call__(...)
  -> _validate_chat_completion_options(...)
  -> _build_generation_params(...)
  -> _generate(...)
     -> _prepare_generate_kwargs(...)
     -> _execute_model(...)
     -> _process_model_output(...)
  -> ModelResponse
```

The async path mirrors the same structure through `acall(...)` and
`_agenerate(...)`.

## Two Preparation Stages

There are two preparation steps, and they solve different problems.

### 1. `_build_generation_params(...)`

This method builds the OpenAI request envelope:

- normalizes `messages`
- injects `system_prompt`
- keeps `prefilling`
- expands `tool_definitions` into native `tools` and `tool_choice` when present

This is still a provider-neutral view of the request shape.

### 2. `_prepare_generate_kwargs(...)`

This is where schema logic becomes provider-specific.

It decides how msgFlux output contracts should be exposed to OpenAI.

That includes:

- `typed_parser`
- canonical `generation_schema`
- flow-control metadata carried through `ToolDefinitions`
- the OpenAI `response_format`
- transport normalization metadata

This is the method where logical schema and provider schema stop being the same
thing.

## Logical Schema vs Provider Schema

The provider follows the split documented in
[Logical vs Provider Schema](logical-vs-provider-schema.md).

At this layer:

- `generation_schema` is the canonical msgFlux runtime schema
- `transport_generation_schema` is the OpenAI-facing schema metadata

The transport metadata contains two pieces:

- `decoder_schema`
- `normalize`

Conceptually:

```text
generation_schema
  -> maybe lower for OpenAI
  -> response_format
  -> OpenAI returns payload
  -> decode with decoder_schema
  -> normalize transport payload
  -> validate against generation_schema
```

This keeps provider constraints from leaking into the runtime contract.

## Structured Output Branches

`_prepare_generate_kwargs(...)` has three main branches for non-streaming
structured output.

### 1. Typed Parser

If `typed_parser` is set, the provider does not build an OpenAI structured
output schema from `generation_schema`.

Instead:

- raw text is returned by the model
- the parser decodes that text
- optionally, msgFlux validates the parsed output against
  `generation_schema`

This branch is parser-oriented rather than provider-schema-oriented.

### 2. Plain `generation_schema`

If `generation_schema` is present and is not a `ToolFlowControl`, the provider
derives an OpenAI `response_format` from it.

If necessary, the schema is lowered first. This is where cases such as
`dict[K, V]` become provider-compatible transport shapes.

### 3. `ToolFlowControl`

If `generation_schema` is a `ToolFlowControl`, the provider asks the flow
control whether it wants to override the provider-facing schema.

That happens through:

- `build_provider_response_format(...)`
- `normalize_provider_response(...)`

This is how ReAct can send a provider-specific action schema while still
consuming a normalized runtime shape afterward.

## Response Decoding

Once OpenAI returns a completion, `_process_completion_model_output(...)`
converts it into a `ModelResponse`.

There are four major result shapes:

- native tool call payloads
- text completions
- structured outputs
- audio outputs

The structured path is the important one for the current architecture.

It does the following:

```text
OpenAI content string
  -> decode transport payload
  -> convert struct to dict when needed
  -> apply transport normalizer
  -> validate against canonical generation_schema
  -> return dotdict payload
```

That last validation step matters. It means the provider does not trust the
transport schema alone. The final runtime object is still checked against the
logical schema expected by msgFlux.

## Tool Calls And Structured Outputs

This provider handles two distinct tool-related modes:

### Native OpenAI Tool Calls

If OpenAI returns `tool_calls`, the provider builds a `ToolCallAggregator` and
returns a `ModelResponse` with `response_type="tool_call"`.

At that point the provider stops. The loop continues in `Agent`.

### Structured Tool Loops

If the agent is using a `ToolFlowControl` such as ReAct, the provider does not
rely on native `tool_calls`.

Instead it:

- receives a structured response payload
- decodes it
- normalizes it back to the logical flow shape
- returns that shape to `Agent`

So the provider supports both tool systems, but they are separate branches.

## Early Validation

`_validate_chat_completion_options(...)` exists to reject incompatible
combinations before an OpenAI request is made.

Today that includes:

- `prefilling` + `generation_schema`
- `stream=True` + `typed_parser`

The first rule is especially important because `prefilling` appends an
assistant message into the prompt, while `generation_schema` expects the model
to produce a structured payload from the start.

Conceptually:

```text
prefilling
  -> "continue from this assistant text"

generation_schema
  -> "start a strict structured output payload"
```

Those are conflicting instructions in this provider path, so the provider fails
fast instead of sending an ambiguous request.

## Streaming

The streaming path is intentionally simpler than the structured-output path.

In streaming mode, the provider:

- creates a `ModelStreamResponse`
- consumes chunks from OpenAI
- aggregates text, reasoning, and native tool call deltas
- sets response metadata as the stream completes

It does not combine `stream=True` with typed-parser decoding, and structured
schema-heavy normalization is not the primary path there.

## ASCII Diagram

This is the provider's main decision tree:

```text
Agent params
  -> __call__ / acall
  -> validate options
  -> build generation params
  -> prepare generate kwargs
       |
       +--> typed_parser branch
       |
       +--> generation_schema branch
       |      |
       |      +--> lower schema for OpenAI if needed
       |      +--> build response_format
       |
       +--> ToolFlowControl branch
              |
              +--> build_provider_response_format(...)
              +--> keep normalize_provider_response(...)
  -> execute OpenAI request
  -> process completion output
       |
       +--> tool_calls -> ToolCallAggregator
       |
       +--> text -> plain ModelResponse
       |
       +--> structured
              |
              +--> decode transport payload
              +--> normalize transport payload
              +--> validate against generation_schema
  -> ModelResponse
```

## Relationship To The Rest Of The System

This provider should be read together with:

- [Agent](agent.md)
- [ToolFlowControl](tool-flow-control.md)
- [Logical vs Provider Schema](logical-vs-provider-schema.md)
- [ReAct Provider Schemas](react-provider-schemas.md)

The design line is:

- `Agent` assembles the runtime contract
- `OpenAIChatCompletion` adapts that contract to OpenAI
- the provider returns normalized output back to the runtime contract

## Why This Shape Matters

Without this adapter layer, provider restrictions would leak into signatures,
generation schemas, flow controls, and tool execution.

This module keeps those concerns localized:

- OpenAI-specific request formatting stays here
- OpenAI-specific transport schemas stay here
- final runtime validation still points back to msgFlux contracts

That balance is the main reason the newer transport-schema work is sustainable.
