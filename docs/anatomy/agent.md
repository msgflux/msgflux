# Agent

`Agent` is the orchestration center of msgFlux.

Most of the library's higher-level behavior passes through it:

- input preparation
- prompt assembly
- provider execution
- structured output handling
- native tool calling
- custom tool loops via `ToolFlowControl`
- response formatting and delivery

If you only draw one module to understand the runtime, draw `Agent` first.

## Mental Model

`Agent` does not own every detail. It coordinates specialized subsystems.

```text
input/message
  -> Agent
     -> templates and task preparation
     -> ToolLibrary schemas
     -> provider call
     -> response processing
        -> native tool loop, or
        -> ToolFlowControl loop, or
        -> final response formatting
```

The important part is the branching after the provider call. `Agent` can follow
three main paths:

- plain response
- native provider tool calling
- custom flow control loop

## Main Flow

The normal synchronous path looks like this:

```text
message / kwargs
  -> Agent.forward(...)
  -> _prepare_inputs(...)
  -> _execute_model(...)
     -> _prepare_model_execution(...)
     -> generator(...)
     -> provider model call
  -> _process_model_response(...)
     -> maybe native tool loop
     -> maybe ToolFlowControl loop
     -> _prepare_response(...)
  -> response
```

The async path mirrors the same structure through `aforward`, `_aexecute_model`,
and `_aprocess_model_response`.

## Generation Schema And Signature

One of the easiest ways to misunderstand `Agent` is to treat
`generation_schema`, `typed_parser`, and `signature` as unrelated features.

They are different entry points into the same contract-building phase.

### Direct `generation_schema`

When `generation_schema` is passed directly, it defines the structured output
shape that the provider should produce and that the runtime should consume.

Typical cases:

- a plain `msgspec.Struct` output
- a reasoning schema such as `ChainOfThought`
- a custom flow control schema such as `ReAct`

### `typed_parser`

`typed_parser` is not the same thing as `generation_schema`.

It influences how the expected output is described and parsed, but it does not
replace the role of `generation_schema` as the runtime output contract.

In practice:

- `generation_schema` defines the shape of structured output
- `typed_parser` defines a parser-oriented output strategy

That is why these features appear together in the preparation path, but they
should not be mentally merged into one mechanism.

### `signature`

A signature is a higher-level configuration source.

When `signature` is provided, `Agent` compiles several internal pieces from it:

- input annotations
- task template
- instructions
- expected output
- output struct used as generation schema

So, from the perspective of `Agent` anatomy, a signature is not just a prompt
shortcut. It is a contract compiler.

### Signature + Generation Schema

The most important special case is combining a signature with an explicit
`generation_schema`.

In that path, `Agent` does not discard one in favor of the other. Instead, it
fuses them:

- the signature still defines the typed output fields
- the explicit `generation_schema` remains the outer structure
- the signature output becomes the type of `final_answer`

This is what allows combinations such as:

- signature + `ChainOfThought`
- signature + `ReAct`
- signature + other reasoning schemas with `final_answer`

Conceptually, the merge looks like this:

```text
signature outputs
  -> StructFactory.from_signature(...)
  -> Output struct

generation_schema with final_answer
  + Output struct as final_answer type
  -> fused generation schema
```

That fused schema is what later enters provider preparation and response
normalization.

## What Happens Before The Model Call

`_prepare_model_execution(...)` assembles the provider-facing execution params.

That step is where `Agent` combines:

- current chat `messages`
- rendered `system_prompt`
- `prefilling`
- `stream`
- `typed_parser`
- `generation_schema`
- tool schemas from `ToolLibrary`

When no custom flow control is involved, tool schemas can be passed directly to
the provider as native tool definitions.

When the generation schema is a `ToolFlowControl`, the behavior changes:

- tool schemas are rendered into the system prompt using the flow's template
- native provider tool calling is disabled for that request
- flow-specific tool schema metadata is passed to the provider adapter

This is how ReAct can keep a custom loop without changing the default agent
loop.

## Contract Compilation View

A useful way to read the module is to separate two stages:

```text
configuration stage
  -> signature
  -> generation_schema
  -> typed_parser
  -> templates / instructions / annotations

execution stage
  -> messages
  -> system prompt
  -> provider params
  -> model output
  -> loops / normalization / final response
```

`Agent` sits across both stages.

That is why its code can look broader than a plain runtime orchestrator:
it is both the place where the contract is assembled and the place where that
contract is executed.

## Execution Branches

After the provider returns, `Agent` decides how to continue based on
`response_type` and `generation_schema`.

### 1. Plain Response

If the model returns plain text or a normal structured object, `Agent` just
extracts the raw data and formats the final response.

```text
provider response
  -> _extract_raw_response(...)
  -> _prepare_response(...)
  -> response_mode handling
  -> final output
```

### 2. Native Provider Tool Calls

If the provider returns a native tool call payload, `Agent` enters the built-in
tool loop.

```text
provider response_type == "tool_call"
  -> _process_tool_call_response(...)
  -> raw_response.get_calls()
  -> _process_tool_call(...)
  -> ToolLibrary(...)
  -> raw_response.insert_results(...)
  -> messages.extend(tool_messages)
  -> _execute_model(...) again
```

This loop continues until the provider stops returning tool calls.

### 3. ToolFlowControl

If `generation_schema` is a subclass of `ToolFlowControl`, `Agent` delegates
the loop semantics to the flow control.

```text
provider structured response
  -> _process_tool_flow_control_response(...)
  -> flow_control.extract_flow_result(...)
     -> complete? yes -> finish
     -> complete? no  -> execute tools
  -> _process_tool_call(...)
  -> ToolLibrary(...)
  -> flow_control.inject_results(...)
  -> flow_control.build_history(...)
  -> _execute_model(...) again
```

This keeps the loop generic:

- `Agent` owns orchestration
- `ToolFlowControl` owns loop semantics
- `ToolLibrary` owns tool execution

## Agent And ToolLibrary

`Agent` does not execute tools directly. It always delegates tool execution to
`ToolLibrary`.

The dependency is narrow and intentional:

```text
Agent
  -> ToolLibrary.get_tool_json_schemas()
  -> ToolLibrary.get_tool_annotations()
  -> ToolLibrary(...) / ToolLibrary.acall(...)
```

That means `Agent` relies on `ToolLibrary` for two separate concerns:

- schema-time concerns:
  tool JSON schemas and parameter annotations
- runtime concerns:
  executing tool calls and collecting results

This is why `ToolLibrary` matters even when the active provider never uses
native tool calling directly.

## ASCII Diagram

This is the overall runtime picture:

```text
                     +--------------------+
                     |   user / message   |
                     +---------+----------+
                               |
                               v
                     +--------------------+
                     |       Agent        |
                     |  _prepare_inputs   |
                     +---------+----------+
                               |
                               v
                     +--------------------+
                     | _prepare_model_    |
                     |    execution       |
                     +---------+----------+
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
      +--------------------+      +--------------------+
      | native tool schema |      | ToolFlowControl    |
      | or plain schema    |      | prompt/schema path |
      +----------+---------+      +----------+---------+
                 |                           |
                 +-------------+-------------+
                               |
                               v
                     +--------------------+
                     |  provider/generator |
                     +---------+----------+
                               |
                               v
                     +--------------------+
                     | _process_model_    |
                     |    response        |
                     +----+---------+-----+
                          |         |
            +-------------+         +-------------------+
            |                                         |
            v                                         v
 +-----------------------+               +--------------------------+
 | native tool loop      |               | ToolFlowControl loop     |
 | get_calls()           |               | extract_flow_result()    |
 | ToolLibrary()         |               | ToolLibrary()            |
 | insert_results()      |               | inject_results()         |
 | rerun model           |               | build_history()          |
 +-----------+-----------+               | rerun model              |
             |                           +------------+-------------+
             |                                        |
             +-------------------+--------------------+
                                 |
                                 v
                       +--------------------+
                       | _prepare_response  |
                       +---------+----------+
                                 |
                                 v
                       +--------------------+
                       |    final output    |
                       +--------------------+
```

## Why This Shape Matters

This module boundary keeps msgFlux extensible without collapsing everything
into provider code or into custom agents.

The design goal is:

- `Agent` should stay the stable coordinator
- providers should adapt transport concerns
- `ToolFlowControl` should define custom loop semantics
- `ToolLibrary` should stay the execution boundary for tools

Once you read the system through that lens, the newer transport-schema work
becomes much easier to place.

## Related Pages

- [ToolFlowControl](tool-flow-control.md)
- [Logical vs Provider Schema](logical-vs-provider-schema.md)
- [ReAct Provider Schemas](react-provider-schemas.md)
