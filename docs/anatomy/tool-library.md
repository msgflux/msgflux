# ToolLibrary

`ToolLibrary` is the execution boundary for tools in msgFlux.

It sits between orchestration and implementation:

- `Agent` decides that a tool must be called
- `ToolLibrary` resolves and executes that call
- each `Tool` instance performs the actual local or remote work

This separation is important because schema concerns and execution concerns
meet here.

## What It Owns

`ToolLibrary` owns five responsibilities:

- registering local and remote tools
- routing tools into buckets when a bucket captures their kind
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

## On-Demand Tools

`ToolLibrary` can also keep tools registered without exposing them immediately
to the model.

The contract is intentionally small:

- `@tool_config(on_demand=True)` keeps the tool out of
  `get_tool_json_schemas()` and `get_tool_annotations()`
- the tool is stored in `ToolLibrary.on_demand_tools` and can still be found by
  `tool_search`
- if at least one on-demand tool exists, `ToolLibrary` injects `tool_search`
- `tool_search` can search both local and MCP-backed on-demand tools
- keyword searches return matching tool names, and `description=True` includes
  tool metadata
- `select:tool_name` promotes matching tools by calling `ToolLibrary.add(...)`
  again without the `on_demand` flag

This is useful when a session can register a large number of tools but should
keep the active tool context small.

A hidden `ToolLibraryHandle` is the natural companion feature here: a tool can
add a new on-demand tool at runtime, and `ToolLibrary` will expose `tool_search`
automatically if needed.

`tool_search` itself is a builtin tool, not a method implemented inside
`ToolLibrary`. Runtime-aware tools inherit `ToolLibraryOperator`, receive a
`ToolLibraryHandle`, and use that handle to search, describe, select, or add
tools. This keeps `ToolLibrary` focused on registration, removal, schema
exposure, and execution, while operational behavior lives in builtin/runtime
components.

## Tool Buckets

Some tools are not independent public tools. They are better represented as
members of another tool.

`ToolBucket` is the base type for that pattern:

```python
class ToolBucket:
    tool_kind = "bucket"
    capture_kind: str

    def add(self, tool: ToolMetadata) -> None:
        ...
```

The registration rule is:

- if a tool has `on_demand=True`, it goes to `on_demand_tools`
- otherwise, if a bucket exists for its `tool_kind`, the bucket captures it
- otherwise, the tool is registered normally

For agents, `nn.Agent` exposes `tool_kind="agent"`. `AgentTool` is a bucket
with `capture_kind="agent"`, so adding an agent to a library that already has
`AgentTool` updates the single public `agent(name, message)` tool instead of
exposing the agent as a separate tool.

```python
library = ToolLibrary(
    name="team",
    tools=[
        AgentTool(),
        reviewer_agent,
        planner_agent,
    ],
)
```

The model only sees `agent(...)`. The bucket description and usage guidance are
refreshed on the wrapping `LocalTool`, so provider schemas and prompt guidance
reflect the captured agents.

`AgentTool` still receives runtime context as hidden runtime kwargs. The public
agent parameters stay as `agent(name, message)`, while `ToolLibrary` passes the
current `messages`, `vars`, and execution `scope` into the bucket. The bucket
then forwards those runtime values only when the selected subagent's own
`tool_config` requests them.

On-demand tools use the same path. An on-demand agent first lives in
`on_demand_tools`; when `tool_search` receives `select:agent_name`,
`ToolLibrary.add(...)` runs again and the agent is captured by `AgentTool`.

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
