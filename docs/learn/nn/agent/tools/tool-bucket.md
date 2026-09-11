# Tool Bucket

`ToolBucket` groups several logical tools behind one callable tool. The model
sees only the bucket schema. At runtime, a scoped handle redirects the chosen
call through `ToolLibrary`, preserving local/MCP wrappers, retries, runtime
injection, errors, and telemetry.

This is the common routing primitive behind two built-ins:

| Built-in | Captures | Public surface |
|----------|----------|----------------|
| [`AgentTool`](agent-tool.md) | Agents with `tool_kind="agent"` | `agent(name, message)` |
| [`ToolSearchTool`](tool-search.md) | Tools with `defer_loading=True` | `tool_search(...)` |

A bucket is useful when a large set of implementations share one stable entry
point. It reduces schema count without discarding the identity, configuration,
or implementation of the captured tools.

## Define And Register A Bucket

A bucket defines a public name and schema, plus a `capture` mapping. In this
example, both catalog and order tools become targets of one
`operations(name, query)` tool:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools import ToolBucket


@mf.tool_config(tool_kind="catalog")
def find_product(query: str) -> str:
    """Find products by name or SKU."""
    return f"Product match: {query}"


@mf.tool_config(tool_kind="orders")
def find_order(query: str) -> str:
    """Find orders by customer or order ID."""
    return f"Order match: {query}"


class OperationsTool(ToolBucket):
    """Dispatch a query to one operations tool."""

    name = "operations"
    capture = {
        "tool_kind": "catalog|orders",
        "defer_loading": False,
    }
    tool_config = {"runtime_inputs": ["handle"]}
    annotations = {
        "name": str,
        "query": str,
        "handle": mf.Hidden,
        "return": str,
    }

    def __call__(self, name: str, query: str, *, handle) -> str:
        return handle(name, query=query)

    async def acall(self, name: str, query: str, *, handle) -> str:
        return await handle.acall(name, query=query)


library = nn.ToolLibrary(
    name="store",
    tools=[find_product, OperationsTool(), find_order],
)

print(library.get_tool_names())
# ["operations"]
```

Registration order is deliberately irrelevant. `ToolLibrary` routes a new
matching tool into an existing bucket, and a newly registered bucket captures
matching tools already in the library. Captured tools are removed from the
top-level callable surface. The bucket retains stable `ToolRef` values rather
than implementations or mutable tool metadata. The library supplies immutable
`ToolBucketEntry` projections when the bucket needs names, descriptions, or
guidance for its presentation. Execute a captured tool only through
`handle(...)` or `handle.acall(...)`. The handle carries the original `messages` and `vars`
objects. The selected child's own injection policy decides whether messages are
shared or copied; agent children receive an isolated history, while ordinary
tools retain reference semantics. Add and remove tools through the library or
its handle so capture rules and presentation refresh remain correct.

The handle is a proxy over the complete library pipeline, not a shortcut to the
captured Python callable. The selected call is represented as a normal tool
intent, then passes through runtime policies, lifecycle hooks, runtime input
resolution, the configured dispatcher, abort handling, and tool events. A
policy that blocks the selected child therefore prevents its implementation
from running, just as it would for a directly exposed tool. Successful calls
still return the child's plain result from `handle(...)` or `handle.acall(...)`;
failures raise at the proxy boundary so the public bucket reports the error to
the model.

## Capture Rules

`capture` matches normalized `tool_config` fields by equality. The special
`tool_kind` value can contain several alternatives separated by `|`:

```python
capture = {"tool_kind": "catalog|orders", "defer_loading": False}
```

All fields must match. In the example, a catalog tool with
`defer_loading=True` does not enter `OperationsTool`; it is captured by tool
search instead. This gives the catalog one clear owner.

The library validates ownership before mutating its state:

- two buckets whose capture rules can match the same tool are rejected;
- duplicate captured names are rejected;
- a bucket cannot be removed while it still owns tools;
- empty or duplicate `tool_kind` alternatives are rejected;
- an executable bucket rejects captured tools configured with model-loop
  behavior: `background`, `allow_background`, `detached`, `call_as_response`,
  `return_direct`, or `handoff`.

The last rule keeps ownership explicit. The public bucket controls whether its
model-visible call runs in the background, returns directly, or changes the
tool loop. A hidden child retains its own wrapper behavior—argument restoration,
runtime injection, retry, errors, and telemetry—but cannot independently change
the parent loop. Configure model-loop options on the bucket itself.

A specialized catalog bucket may override `validate_capture`. The built-in
`ToolSearchTool` does this because it catalogs deferred tool references and later
exposes the selected tool as a normal public call; it does not proxy-execute
that child behind `tool_search(...)`.

## Refreshing The Public Tool

The library calls `refresh(entries)` after bucket membership changes. Override
that hook when the public description or usage guidance depends on the current
contents:

```python
from msgflux.tools import ToolBucketEntry


class NamedOperationsTool(OperationsTool):
    def refresh(self, entries: tuple[ToolBucketEntry, ...] = ()) -> None:
        names = ", ".join(sorted(entry.name for entry in entries)) or "none"
        self.description = f"Dispatch an operations query. Available: {names}."
```

After the hook runs, `ToolLibrary` copies the updated description and
`usage_guidance` to the model-facing bucket tool. `AgentTool` uses this to keep
its compact agent list current. An entry intentionally contains no executor;
inside a bucket call, use `handle.get_entry(name)` for presentation metadata
and `handle(name, ...)` or `await handle.acall(name, ...)` for execution.

## Membership And Concurrency

Bucket membership is library state. Static registration and
`handle.add(tool)` therefore affect every concurrent call that shares the same
`ToolLibrary`. Use separate libraries when different tenants truly have
different allowed implementations.

Deferred activation is the exception: tool search stores active names in each
thread's `ChatMessages`, so concurrent threads may expose different deferred
subsets without mutating shared bucket membership.

## ToolBucket Versus ToolBackground

`ToolBucket` presents several captured implementations through one dispatcher.
`ToolBackground` instead manages independently callable task-control tools
according to the capabilities declared by background sources. Use a bucket to
stabilize one public schema; use background capabilities when each task
operation must remain directly callable.

## Related Pages

- [Tool Config: tool_kind](config.md#tool_kind)
- [AgentTool](agent-tool.md)
- [Tool Search](tool-search.md)
- [Background Tasks](background-tasks.md)
