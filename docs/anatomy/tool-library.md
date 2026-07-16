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
- routing tools into buckets when a bucket capture matches their configuration
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
container. Programmatic single-tool calls use `execute(...)` and
`aexecute(...)`. All four entry points share the same preparation and execution
pipeline.

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
  -> execute prepared calls with scatter_gather
  -> collect ToolCall results
  -> return ToolResponses
```

The async path mirrors the same structure through `aforward(...)` and
`ascatter_gather(...)`. `execute(...)` and `aexecute(...)` run one prepared call
and return its result directly. They still apply transport restoration, hidden
parameter removal, runtime injection, retry, telemetry, spawn, and background
dispatch rules.

The resolved state passed through this loop is represented by
`PreparedToolExecution` in `msgflux.tools.dataclasses`. Model calls to schemas
that are absent or currently hidden raise the internal `ToolNotAvailableError`
from `msgflux.tools.exceptions`; the loop converts that condition into the
normal `ToolCall.error` response.

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
- on-demand tools are captured by the builtin `search_tools` bucket
- if at least one on-demand tool exists, `ToolLibrary` registers `search_tools`
- `search_tools` can search both local and MCP-backed on-demand tools
- text and `/regex/` queries return compact matching tool summaries
- an exact-name query promotes the existing wrapper in place

This is useful when a session can register a large number of tools but should
keep the active tool context small.

An explicitly configured `ToolHandle` is the natural companion feature here: a
tool with `tools.register` access can add a new on-demand tool at runtime, and
`ToolLibrary` will expose `search_tools` automatically if needed.

`search_tools` is both a builtin operator and a `ToolBucket` with
`capture={"on_demand": True}`. It owns the searchable metadata and promotes a
selected tool through its scoped handle. `ToolBucketManager` changes the
ownership edge and exposure flags without unregistering the wrapper. Search
behavior remains in the builtin tool.

Background task control tools follow the same pattern through `ToolBackground`.
The `task_status`, `task_wait`, `task_output`, `task_interrupt`,
`task_activity`, and `task_message` tools are callable objects with
reserved background tool kinds: the common controls use `"background"`,
activity uses `"background_activity"`, and messaging uses
`"background_message"`. `ToolLibrary` asks `ToolBackground` to reconcile the
surface from currently registered tools. `ToolBackground` derives the common
controls from background execution and the optional controls from the union of
declared `background_capabilities`.

## Tool Buckets

Some tools are not independent public tools. They are better represented as
members of another tool.

`ToolBucket` is the base type for that pattern:

```python
class ToolBucket:
    tool_kind = "bucket"
    capture: Mapping[str, Any]

    def add(self, tool: ToolMetadata) -> None:
        ...

    def refresh(self) -> None:
        ...
```

`capture` matches entries in `tool_config`. For example,
`{"tool_kind": "agent", "on_demand": False}` captures regular agents, while
`{"on_demand": True}` captures every on-demand tool. Every entry must match.
`capture["tool_kind"]` can name one kind or several kinds separated by `|`,
such as `"catalog|orders"`.

Capture also provides structural selectors. `source` accepts `tool`, `bucket`,
or `any` and defaults to `tool`. `name` matches an exact canonical name.
`capabilities={"all": [...]}` requires every semantic label, while
`capabilities={"any": [...]}` requires at least one. `match.any` provides OR
alternatives; predicates in each alternative and at the top level use AND.

`policy`, `source`, and `match` are reserved control entries rather than
ordinary `tool_config` predicates. `name` and `capabilities` are evaluated from
normalized metadata. Other entries retain exact `tool_config` matching.

`ToolLibrary` normalizes `on_demand=False` for every registered tool, including
plain callables, `Tool` instances, and manually constructed `ToolMetadata`.
Opting into on-demand registration therefore always requires an explicit
`on_demand=True`.

### Capture Policies

The reserved `capture["policy"]` entry does not participate in predicate
matching. It lists optional restrictions that the bucket applies after a tool
matches. `handle` is currently the only supported policy:

```python
class ReadOnlyBucket(ToolBucket):
    capture = {
        "tool_kind": "operation",
        "on_demand": False,
        "policy": {
            "handle": {
                "tools": ["list", "get"],
            }
        },
    }
```

For a handle policy, every domain and action requested by the captured tool
must be a subset of the policy. Unknown policy names and undeclared handle
actions fail registration. Without `capture["policy"]`, the bucket trusts the
captured tool and does not restrict its declared handle access. Built-in
buckets intentionally use this trusted default.

Two bucket captures that accept the same source cannot overlap. This makes
routing deterministic without a priority system. Disjoint canonical-name or
configuration selectors can coexist, as can a leaf bucket using `source=tool`
and a composition bucket using `source=bucket`. A kind bucket that coexists
with on-demand tools should include
`"on_demand": False`, leaving `{"on_demand": True}` to `search_tools`. The
base bucket rejects captured tools that configure `background` or
`allow_background`; configure those flags on the bucket itself instead. A
bucket candidate may retain those flags when another bucket composes it.

Before registration, `ToolBucket.add(...)` can stage constructor-provided
tools. Once the bucket belongs to a library, `bucket.tools` becomes a read-only,
live metadata view derived from the library. The base refresh hook does nothing;
a subclass can rebuild presentation data such as its description and usage
guidance from that view.

The registration rule is:

- if a registered bucket matches the tool configuration, the bucket captures it
- otherwise, the tool is registered normally

When a bucket is registered, it also captures matching tools that are already
registered. Therefore, the order of `tools` in `ToolLibrary(...)` does not
change capture behavior.

Late bucket registration is transactional from the library's perspective. The
library registers the bucket once, assigns matching candidates to it, and sets
their `exposed` configuration to `False`. If any candidate is rejected, it
removes the ownership edges, restores the previous exposure values, and
unregisters the new bucket. A duplicate derived mode or another bucket-specific
validation error therefore leaves the previous library surface intact.

One undo journal spans the complete recursive registration. Registering a
staged bucket, its staged child bucket, and that child's tools is one
transaction rather than several nested commits. Rollback runs the recorded
ownership and registration operations in reverse order, restoring staged
metadata at every level and preventing descendants from retaining missing
owners.

After each successful capture or release, `ToolBucketManager` synchronizes the
wrapping tool's description, public annotations, and usage guidance from the
bucket. This lets a bucket keep a fixed public schema or intentionally derive
one from its contents without replacing the wrapping `Tool` instance.

### Nested Ownership

Bucket composition is represented as an exclusive ownership tree over one flat
registry. Every executable wrapper, including captured tools and nested
buckets, lives exactly once in `ToolLibrary.library`. `tool_owners[child] =
bucket` stores the structural edge; there is no second captured-tool registry
and no redundant `captured` flag.

`tool_configs[name]["exposed"]` controls the model-facing projection and
defaults to `True` when omitted. Capture changes it to `False`, so schema,
annotation, display-name, and usage-guidance builders skip that node without
removing its executable wrapper. `ToolBucket.tools` is a non-owning view that
materializes fresh `ToolMetadata` for the bucket's direct children.

On-demand activation changes this internal projection and its ownership edge
in place. It does not unregister or reconstruct the wrapper. This is required
for deferred buckets because their captured descendants must remain attached
while the bucket moves from `search_tools` to the public projection or another
matching bucket.

Canonical names are unique across the complete tree, not only among public
roots. This lets execution and removal resolve one node without relying on an
ambiguous path.

Buckets do not execute `ToolMetadata.impl` directly. A bucket that declares a
hidden `tools` parameter receives a `ToolBucketHandle`. That handle is an
execution facade over `ToolLibrary`: `tools(...)` and `tools.acall(...)` enter
the same pipeline as normal calls, while `tools.list()` returns captured
descendants. The handle is bound to the current bucket, so a call is rejected
when its target is not a direct or transitive descendant.

This creates two distinct projections of the same ownership graph:

- the model schema projection contains only public roots
- the bucket execution projection contains only that bucket's descendants

Nested buckets can therefore reuse agents, task controls, and future
interpreter capabilities without moving implementations between objects or
bypassing runtime injection.

`ToolBucketGraph` is the read-only structural view over the flat registry,
configuration map, and ownership edges. It owns traversal, node and owner
lookup, nested-first matching, capture candidates, global-name validation,
descendant authorization, overlap validation, and cycle detection.

`ToolBucketManager` receives references to those same canonical maps. It owns
capture, release, on-demand promotion, live child metadata, bucket binding, and
presentation propagation. It does not store a second collection of tools.
`ToolBucket` remains responsible for evaluating one selector and enforcing its
capture policy.

`ToolLibrary` owns canonical wrapper registration, background reconciliation,
the transaction boundary, and execution. `ToolRegistrationTransaction` holds
the undo journal used by recursive registration. Keeping graph queries and
bucket mutations in separate focused objects prevents ownership logic from
spreading through the execution loop.

The library traverses this tree for four operations:

1. **Routing:** nested buckets are checked before their parents. A newly added
   agent can still reach an `AgentTool` owned by another bucket.
2. **Ownership:** bucket names identify structural nodes, and each node has at
   most one parent. A candidate matching multiple parents is rejected.
3. **Removal:** captured nodes can be located below a public root. A bucket
   cannot be removed while it owns children.
4. **Background reconciliation:** background sources and generated task
   controls remain discoverable when `AgentTool` or `TaskTool` is nested.

Bucket registration is bottom-up. A new bucket first captures matching visible
candidates and refreshes its presentation. Only then may an existing
composition bucket capture it. The parent therefore receives a fully assembled
child, regardless of constructor order.

Presentation updates move upward. When an inner bucket captures or releases a
child, the manager synchronizes its wrapper, calls the parent's `refresh()` over
its live metadata view, and repeats to the public root. An outer interpreter
catalog can therefore reflect a late-added Reviewer through its nested
`AgentTool` without exposing Reviewer directly.

Cycles are rejected before ownership changes. If bucket `a` selects bucket `b`
by name and `b` selects `a`, registration of the second edge fails and the
existing tree remains intact. Composition currently uses consuming semantics
only; one node cannot be referenced by multiple bucket parents.

For agents, `nn.Agent` is normalized to `tool_config["tool_kind"]="agent"`.
`AgentTool` is a bucket with
`capture={"tool_kind": "agent", "on_demand": False}`, so adding an agent to
a library that already has `AgentTool` updates the single public
`agent(name, message)` tool instead of exposing the agent as a separate tool.

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

The public agent parameters stay as `agent(name, message)`. `AgentTool` receives
a hidden bucket proxy, and that proxy carries the current runtime context into
the selected agent's normal library execution path. Values such as `messages`
and `vars` are injected only when the selected subagent's own `tool_config`
requests them.

On-demand tools use the same path. An on-demand agent is first captured by
`search_tools`; when it receives an exact-name query, `ToolLibrary.add(...)` runs
again with `on_demand=False`, and `AgentTool` captures it.

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
