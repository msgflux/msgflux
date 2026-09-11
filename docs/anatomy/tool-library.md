# ToolLibrary

`ToolLibrary` is the execution boundary for tools in msgFlux.

It sits between orchestration and implementation:

- `Agent` decides that a tool must be called
- `ToolLibrary` resolves and executes that call
- each `Tool` instance performs the actual local or remote work

This separation is important because schema concerns and execution concerns
meet here.

## Internal Layout

The implementation is split by ownership instead of keeping registration,
provider contracts, dispatch, and execution in one module:

| Module | Responsibility |
| --- | --- |
| `msgflux.tools.catalog` | Execution-free catalog entries, views, choices, references, and native bindings |
| `msgflux.tools.specs` | Provider-neutral dispatch, context-binding, and loading declarations |
| `nn.modules.tool.definitions` | Compiled executable definitions and the declaration compiler |
| `nn.modules.tool.registry` | Stable definition ownership and catalog projection |
| `nn.modules.tool.execution` | Plans and policy payloads passed through the runtime |
| `nn.modules.tool.extensions` | Dispatch strategies, context providers, policies, and their registry |
| `nn.modules.tool.implementations` | Local Python and MCP execution adapters |
| `nn.modules.tool.execution_runtime` | Preparation, hooks, dispatch, telemetry, and outcome normalization |
| `nn.modules.tool.library` | Public facade, registration, buckets, catalog state, and lifecycle ownership |

`msgflux.nn.modules.tool` preserves the public imports for `Tool`, `LocalTool`,
`MCPTool`, and `ToolLibrary`. The former `nn.modules.tool_runtime` path remains a
compatibility facade, but Models import catalog contracts directly from
`msgflux.tools.catalog`. This keeps provider imports independent from the neural
module and executor runtime.

## What It Owns

`ToolLibrary` owns these responsibilities:

- registering local and remote tools
- routing tools into buckets when a bucket capture matches their configuration
- compiling each tool once into a stable logical definition
- projecting provider-neutral catalog entries from those definitions
- executing prepared tool calls
- normalizing execution feedback into `ToolOutcome` values

It does not decide when a tool should be called. That remains the job of the
provider response path or of a `ToolFlowControl`.

## Registry And Executor Ownership

`ToolRegistry` owns stable `ToolDefinition` values and produces `ToolRef` and
`ToolCatalogView` projections. It deliberately does not register executor
modules. Executor ownership belongs to the `ToolLibrary` facade so one module
has exactly one place in the module tree and `state_dict`.

The current facade keeps directly exposed executors in its public `ModuleDict`.
Bucket-captured definitions are indexed by the same registry, while their
executors move to a separate library-owned `ModuleDict`. Buckets retain only
`ToolRef` values. Consequently, public, deferred, and bucket-captured tools all
resolve through one definition registry without making hidden tools appear in
the model-facing module surface or giving a bucket direct executor access.

Deep copies preserve the identity between each copied definition and its copied
executor because both are owned by the `ToolLibrary` facade, not independently
by the registry.

## Two Phases

`ToolLibrary` participates in two different phases of the runtime.

### 1. Definition And Catalog Time

When a tool is added, `ToolLibrary` first compiles its implementation, schema,
dispatch policy, feedback policy, context bindings, loading policy, and
normalized declaration into one immutable `ToolDefinition`. Registration,
bucket routing, background capability discovery, catalog projection, and
runtime execution all read that definition.

Callables and `Tool` adapters pass through a short-lived `ToolDeclaration`
during normalization. It exists only long enough to build the executor and
compile the definition; the library does not retain a mutable registration
metadata object. `ToolLibrary.add(...)` can also receive an already compiled
`ToolDefinition`, which lets extensions contribute the same canonical contract
without reopening `tool_config` interpretation.

`tool_config` remains a declaration frontend on the incoming callable. The
library no longer keeps a parallel `tool_configs` registry after compilation.
The definition's `declaration` field contains the registration-time snapshot
needed by arbitrary bucket capture rules and compatibility lifecycle hooks;
typed behavior uses the definition's `dispatch`, `feedback`, `context`,
`loading`, and `kind` fields. Mutating a callable's or executor's `tool_config`
after registration does not change library behavior. Remove and add the tool
again to compile a new definition.

Before the model runs, other modules ask `ToolLibrary` for a catalog view. The
current compatibility methods remain available:

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

### Canonical Catalog Projection

`ToolCatalogView` is the canonical, immutable catalog snapshot. The registry
projects it from stable definitions plus thread-local loaded names. Every entry
retains the provider-neutral schema, annotations, strict mode, execution
namespace, native bindings, tool kind, display metadata, and stable `ToolRef`.
The view therefore contains enough information for a Model adapter without
accessing an executor or reading `tool_config` again.

The library explicitly selects which registered definitions belong to its
model-facing surface. A normal executable bucket appears as one public tool;
its captured children remain resolvable in the registry but do not leak into
the catalog. The deferred-search bucket intentionally exposes its captured
children because they are the searchable definitions.

Search is identified by the entry's `catalog_role="search"`, not by a reserved
tool name. `entries` remains stable for the lifetime of the snapshot, while
`visible_entries()` derives the portable callable surface. It exposes loaded
or directly selected deferred tools and includes the portable search entry only
while unresolved deferred tools remain.

`Agent` and `Model` use `ToolCatalogView` directly. `ToolCatalog` remains a
temporary compatibility input for callers that pass pre-built function schemas
straight to a Model. Registration, loading, filtering, choice validation, and
new integrations belong to the canonical view.

### 2. Runtime

The Model decodes provider calls into `ToolIntent` values. The Agent passes
those intents to `ToolLibrary.execute_intents(...)` or
`ToolLibrary.aexecute_intents(...)`, which execute the canonical runtime path
and return ordered `ToolOutcome` values.

`forward(...)`, `aforward(...)`, and `ToolResponses` adapt that canonical path
for custom `ToolFlowControl` implementations that still use tuple calls.

Feedback is declarative at this boundary. `ToolLibrary` attaches the compiled
`FeedbackSpec` to each outcome, but does not decide whether that feedback ends
the Agent run. Agent extensions own that orchestration decision.

That runtime path applies tool configuration rules such as:

- `return_direct`
- `call_as_response`
- `detached`
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
ToolIntent
  -> ToolLibrary.execute_intents(...)
  -> resolve tool by name
  -> read compiled ToolDefinition
  -> run before_tool runtime policies
  -> prepare call params
  -> run before_tool lifecycle hooks
  -> build ToolExecutionPlan
  -> run before_dispatch lifecycle hooks
  -> run before_dispatch runtime policies
     -> blocked: emit tool.blocked and return a tool error
  -> execute tools with scatter_gather
  -> run after_tool lifecycle hooks
  -> run after_tool runtime policies
  -> return one ToolOutcome per ToolIntent
```

The async path mirrors the same structure through `aexecute_intents(...)` and
`ascatter_gather(...)`. `ToolExecutionPlan` freezes the selected tool, visible
arguments, runtime arguments, dispatch mode, and return policy before either
path dispatches it.

`execute_intents(...)` and `aexecute_intents(...)` are the primary execution
boundary. The older `forward(...)` and `aforward(...)` APIs convert tuple calls
to `ToolIntent`, execute that same path, and render the outcomes as
`ToolResponses`. The compatibility object is therefore no longer part of the
Agent or canonical runtime path.

Both pre-execution hook chains are sequential per call. The first `block`
short-circuits the remaining handlers for that event invocation, including
hooks inherited from the Agent when a library extension already denied the
call. It does not stop preparation of sibling calls. `tool.blocked` is the
terminal event for the denied call; `tool.start`, `after_tool`, and `tool.end`
belong only to implementations that actually started.

Runtime policies use the owned extension registry shared with dispatchers.
They operate on `ToolIntent`, `ToolExecutionPlan`, and `ToolOutcome` rather than
the compatibility hook events. Policies are sequential and monotonic: the first
blocked outcome ends that phase for one intent. A failure before execution
fails closed, while a failure in `after_tool` preserves the outcome already
produced. Lifecycle hooks remain inside this canonical policy envelope during
the compatibility period.

Runtime argument injection uses the same registry. `runtime_inputs` compiles to
immutable `ContextBinding` values on `ToolDefinition`; the model sees neither
the binding nor its target parameter. `ToolContextProvider` resolves each
source against a per-intent `ToolRuntimeContext`, allowing custom runtime data
without another conditional in the library core. Handles are scoped to the
current tool call before they enter that context.

`before_dispatch` receives the public arguments and
`ToolDefinition.declaration` alongside the selected `dispatch_mode`. Runtime
injections remain private. Extensions may
reduce `background` or `detached` to `foreground`, but cannot promote attached
execution into a detached mode. When a mode changes, the library rebuilds the
reserved call arguments before dispatch.

## Extensions And Core Invariants

`ToolLibraryExtension` owns optional packages of tools, hooks, setup, and
cleanup. It can also validate a canonical definition before registration and
observe successful add, remove, and clear boundaries. Those callbacks run
sequentially; the core no longer needs capability-specific branches to maintain
extension state. `ToolDispatch` owns one open dispatch name in the runtime
registry. Library lifecycle hooks run before hooks inherited from an owning
Agent. The extension mechanism powers deferred tool search, background task
controls, MCP server integration, and application-defined dispatch modes.

Foreground, background, and detached execution are default `ToolDispatch`
extensions rather than branches selected by the core. A tool may select another
registered mode through `tool_config(dispatch="name")`. Batch execution remains
concurrent because every selected dispatcher contributes an async operation to
the same gather boundary.

The core continues to own deterministic registration and bucket routing,
transport restoration, runtime injection, abort handling, telemetry, and plan
dispatch. Extensions can participate at stable lifecycle boundaries but cannot
bypass these invariants.

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

## Deferred Tools

`ToolLibrary` can also keep tools registered without exposing them immediately
to the model.

The contract is intentionally small:

- `@tool_config(defer_loading=True)` keeps the tool out of the initial portable
  callable surface
- deferred tools are captured by the builtin `tool_search` bucket
- if at least one deferred tool exists, `ToolLibrary` registers `tool_search`
- `tool_search` can search both local and MCP-backed deferred tools
- keyword searches return matching tool names, and `description=True` includes
  tool metadata
- `select=["tool_name"]` records matching names in the current
  `ChatMessages.metadata`

This is useful when a session can register a large number of tools but should
keep the active tool context small.

An explicitly injected `ToolLibraryHandle` is the natural companion feature
here: a tool can add a new deferred tool at runtime, and `ToolLibrary` will
expose `tool_search` automatically if needed.

`ToolSearchExtension` installs `tool_search`, which is both a builtin operator
and a `ToolBucket` with
`capture={"defer_loading": True}`. It owns stable references to the searchable
tools and receives their presentation as execution-free entries, while the
thread's `ChatMessages` owns which names have been loaded. The shared library is
never mutated by a selection, so concurrent threads can expose different tool
subsets safely.

Background task control tools follow the same pattern through
`BackgroundTasksExtension`, backed by `ToolBackground`.
The `task_status`, `task_wait`, `task_output`, `task_interrupt`,
`task_activity`, and `task_message` tools are callable objects with
reserved background tool kinds: the common controls use `"background"`,
activity uses `"background_activity"`, and messaging uses
`"background_message"`. `BackgroundTasksExtension` reconciles that surface from
registration callbacks and owns the disabled-control state. `ToolBackground`
derives the common controls from background execution and the optional controls
from the union of declared `background_capabilities`.

## Tool Buckets

Some tools are not independent public tools. They are better represented as
members of another tool.

`ToolBucket` is the base type for that pattern:

```python
class ToolBucket:
    tool_kind = "bucket"
    capture: Mapping[str, Any]

    def add(self, ref: ToolRef) -> None:
        ...

    def refresh(self, entries: tuple[ToolBucketEntry, ...] = ()) -> None:
        ...
```

`capture` matches entries in `tool_config`. For example,
`{"tool_kind": "agent", "defer_loading": False}` captures regular agents, while
`{"defer_loading": True}` captures every deferred tool. Every entry must match.
`capture["tool_kind"]` can name one kind or several kinds separated by `|`,
such as `"catalog|orders"`.

Two bucket captures cannot overlap. This makes routing deterministic without a
priority system. A kind bucket that coexists with deferred tools should include
`"defer_loading": False`, leaving `{"defer_loading": True}` to `tool_search`. The
base executable bucket rejects captured tools that configure
`background`, `allow_background`, `detached`, `call_as_response`,
`return_direct`, or `handoff`; these model-loop policies belong to the
public bucket. `ToolSearchTool` overrides this validation because it catalogs
deferred tools rather than proxy-executing them behind its own call.

`ToolLibrary` routes matching tools to `ToolBucket.add(...)`. The base method
keeps only stable `ToolRef` values in `bucket.tools` and rejects duplicate
names. Executors remain registered on the library, while canonical definitions
remain in `ToolRegistry`. After membership changes, the library projects each
definition to an immutable, execution-free `ToolBucketEntry` and calls
`bucket.refresh(entries)`. A subclass can rebuild derived presentation data
such as its description and usage guidance without reaching into a child
implementation.

Buckets cannot obtain a child executor from either `bucket.tools` or a
`ToolBucketEntry`. An injected
`ToolBucketHandle` accepts `handle(tool_name, **arguments)` and
`handle.acall(...)`, validates membership, resolves the captured `LocalTool` or
`MCPTool`, and creates a canonical `ToolIntent`. The intent re-enters the same
policy, hook, argument-preparation, dispatch, abort, event, and telemetry path
used by a top-level tool call. The handle unwraps a successful `ToolOutcome`
back to the plain value expected by bucket implementations; a failed outcome is
raised at the proxy boundary and becomes the public bucket call's error. The
bucket never executes a child implementation directly. The handle also exposes
`list_entries()` and `get_entry(name)` when a runtime decision needs captured
presentation metadata.

The registration rule is:

- if a registered bucket matches the tool configuration, the bucket captures it
- otherwise, the tool is registered normally

When a bucket is registered, it also captures matching tools that are already
registered. Therefore, the order of `tools` in `ToolLibrary(...)` does not
change capture behavior.

For agents, `nn.Agent` is normalized to `tool_config["tool_kind"]="agent"`.
`AgentTool` is a bucket with
`capture={"tool_kind": "agent", "defer_loading": False}`, so adding an agent to
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

The public parameters stay as `agent(name, message)`. `AgentTool` receives a
scoped bucket handle carrying the parent call context. When it selects a child,
`ToolLibrary` reapplies that child's configuration: for example,
`inject_messages=True` gives the child agent a copy of the parent's history,
while an agent without that option receives no inherited history. The scoped
handle itself keeps the original context references; copying happens only while
preparing a selected agent call. Agent execution namespaces are normalized from
the agent's canonical module name, so a public tool alias does not change its
checkpoint identity.

Deferred agents are captured by `tool_search` and represented as logical tools
in `ToolCatalog`. Once loaded for a thread, the provider can call the agent by
its own tool name without promoting it into the shared `AgentTool` bucket.

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
