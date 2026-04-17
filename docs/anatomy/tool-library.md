# ToolLibrary

`ToolLibrary` is the execution boundary for tools in msgFlux.

It sits between orchestration and implementation:

- `Agent` decides that a tool must be called
- `ToolLibrary` resolves and executes that call
- each `Tool` instance performs the actual local or remote work

This separation is important because schema concerns and execution concerns
meet here.

## What It Owns

`ToolLibrary` owns four responsibilities:

- registering local and remote tools
- exposing tool schemas to other modules
- executing prepared tool calls
- collecting results into a uniform `ToolResponses` object

It does not decide when a tool should be called. That remains the job of the
provider response path or of a `ToolFlowControl`.

## Two Phases

`ToolLibrary` participates in two different phases of the runtime.

### 1. Schema-Time

Before the model runs, other modules ask `ToolLibrary` for metadata:

- `get_tool_json_schemas()`
- `get_tool_annotations()`

Those methods are used for different reasons.

`get_tool_json_schemas()` supports:

- native provider tool calling
- prompt rendering for custom tool loops
- dynamic provider schemas such as ReAct action variants

`get_tool_annotations()` supports:

- transport restoration after structured output decoding
- typed reconstruction of lowered values before local tool execution

### 2. Runtime

Once tool calls are produced, `ToolLibrary.forward(...)` or
`ToolLibrary.aforward(...)` executes them and returns a `ToolResponses`
container.

That runtime path applies tool configuration rules such as:

- `return_direct`
- `call_as_response`
- `spawn`
- `inject_vars`
- `inject_message`
- `inject_messages`
- `handoff`
- `disable_input`

This keeps the execution policy centralized instead of spreading it across
`Agent`, provider adapters, and tool implementations.

## The Execution Flow

The synchronous path looks like this:

```text
tool_callings
  -> ToolLibrary.forward(...)
  -> resolve tool by name
  -> apply tool config
  -> prepare call params
  -> execute tools with scatter_gather
  -> collect ToolCall results
  -> return ToolResponses
```

The async path mirrors the same structure through `aforward(...)` and
`ascatter_gather(...)`.

## Local And Remote Tools

The library can store both local and MCP-backed tools behind the same
interface.

```text
ToolLibrary
  -> LocalTool
  -> MCPTool
```

That means the caller does not need different orchestration logic for:

- a Python function
- an `nn.Module`-style tool
- a proxied MCP tool

The library normalizes all of them into a single execution surface.

## Typed Restoration

One of the most important newer responsibilities of this layer is restoring
transport-lowered arguments back to the logical tool parameter types.

For local tools, that work happens in `LocalTool._restore_transport_params(...)`
using the original annotations.

That means a model can return a provider-friendly transport shape such as:

```json
{
  "fields": {
    "entries": [
      {"key": "city", "value": "Austin"},
      {"key": "country", "value": "USA"}
    ]
  }
}
```

and the local tool still receives:

```python
{"fields": {"city": "Austin", "country": "USA"}}
```

This is the point where transport compatibility is converted back into runtime
types.

## Agent Relationship

The relationship between `Agent` and `ToolLibrary` is intentionally narrow.

```text
Agent
  -> ToolLibrary.get_tool_json_schemas()
  -> ToolLibrary.get_tool_annotations()
  -> ToolLibrary(...) / ToolLibrary.acall(...)
```

That gives `Agent` just enough surface area to:

- expose tools to providers
- build custom flow-control schemas
- execute tool calls

without turning `Agent` itself into a tool registry or argument decoder.

## ASCII Diagram

This is the role of `ToolLibrary` in the larger flow:

```text
                    +----------------------+
                    |        Agent         |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
   +------------------------+        +------------------------+
   | get_tool_json_schemas  |        | get_tool_annotations   |
   +-----------+------------+        +------------+-----------+
               |                                  |
               +----------------+-----------------+
                                |
                                v
                     +----------------------+
                     |     ToolLibrary      |
                     | register / resolve   |
                     | config / execute     |
                     +----------+-----------+
                                |
          +---------------------+----------------------+
          |                                            |
          v                                            v
 +----------------------+                    +----------------------+
 |      LocalTool       |                    |       MCPTool        |
 | restore params       |                    | proxy remote tool    |
 | call Python impl     |                    | call MCP client      |
 +----------+-----------+                    +----------+-----------+
            |                                           |
            +-------------------+-----------------------+
                                |
                                v
                     +----------------------+
                     |    ToolResponses     |
                     |  ToolCall entries    |
                     +----------------------+
```

## Why This Shape Matters

If tool execution lived directly in `Agent`, every new transport rule,
injection rule, or remote-tool integration would make the orchestrator more
fragile.

If schema restoration lived only in provider code, local tool execution would
become provider-specific.

`ToolLibrary` keeps that boundary clean:

- `Agent` stays focused on orchestration
- providers stay focused on transport and decoding
- tool implementations stay focused on business logic

## Related Pages

- [Agent](agent.md)
- [ToolFlowControl](tool-flow-control.md)
- [Dict Lowering and Restoration](dict-lowering-and-restoration.md)
