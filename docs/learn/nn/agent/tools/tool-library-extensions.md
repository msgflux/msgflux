# ToolLibrary Extensions

`ToolLibraryExtension` packages tools, lifecycle hooks, setup, and cleanup under
one removable owner. Use it for a capability that belongs to tool registration
or execution rather than to the surrounding Agent.

## Add Tools And Hooks

This extension contributes one tool and adjusts its validated arguments before
execution:

```python
from dataclasses import replace

from msgflux.nn import ToolLibrary, ToolLibraryExtension
from msgflux.nn.hooks import Hook


class MathTools(ToolLibraryExtension):
    def __init__(self):
        super().__init__("math")

    def tools(self):
        def double(value: int) -> int:
            """Double a value."""
            return value * 2

        return (double,)

    def hooks(self):
        def clamp(event):
            value = max(0, event.arguments["value"])
            return replace(event, arguments={"value": value})

        return (Hook(event="before_tool", handler=clamp),)


library = ToolLibrary("calculator", [], extensions=[MathTools()])
response = library([("call_1", "double", {"value": -3})])

assert response.tool_calls[0].result == 0
```

The contributed tool still passes through normal schema generation, runtime
injection, abort handling, telemetry, and `ToolExecutionPlan` dispatch. An
extension does not bypass the library core.

Library hooks run before hooks inherited from an owning Agent. `before_tool`
sees only model-visible arguments; injected handles, messages, and vars remain
protected runtime arguments. `after_tool` receives the result or error before
it becomes a `ToolCall` result.

Multiple extensions may contribute hooks for the same event. They run
sequentially in registration order. A `block` from `before_tool` or
`before_dispatch` stops the remaining handlers only for that call. The first
reason is returned to the model and emitted as `tool.blocked`; another tool call
in the same response is processed normally.

Use `before_dispatch` when a policy depends on the normalized dispatch mode:

```python
class ForegroundOnly(ToolLibraryExtension):
    def __init__(self):
        super().__init__("foreground_only")

    def hooks(self):
        def keep_attached(event):
            if event.dispatch_mode in {"background", "detached"}:
                return replace(event, dispatch_mode="foreground")
            return event

        return (Hook(event="before_dispatch", handler=keep_attached),)
```

This boundary exposes public arguments and the normalized configuration
snapshot compiled at registration, not injected runtime values or a mutable
executor attribute. It may reduce background/detached execution to foreground
or block the call; it cannot promote a foreground call into detached execution.

## Runtime Policies

`ToolPolicy` is the typed extension boundary for rules that must apply to every
canonical tool intent, including calls redirected through buckets and handles.
Policies are asynchronous, so they may consult a remote authorization or audit
service without blocking the event loop. The current execution abort signal
cancels an in-flight policy or dispatcher await.

This complete example blocks production deployments before the implementation
can run:

```python
import msgflux.nn as nn
from msgflux.tools import ToolIntent, ToolOutcome


class ProductionGuard(nn.ToolPolicy):
    def __init__(self):
        super().__init__("production_guard")

    async def before_tool(self, payload):
        if (
            payload.intent.name == "deploy"
            and payload.intent.arguments.get("environment") == "production"
        ):
            return ToolOutcome.failed(
                payload.intent,
                status="blocked",
                code="production_requires_approval",
                message="Production deployment requires approval.",
            )
        return payload


def deploy(environment: str) -> str:
    """Deploy the current release to one environment."""
    return f"deployed:{environment}"


library = nn.ToolLibrary(
    name="deployments",
    tools=[deploy],
    extensions=[ProductionGuard()],
)

outcome = library.execute_intents(
    [
        ToolIntent(
            id="call_1",
            name="deploy",
            arguments={"environment": "production"},
        )
    ]
)[0]

assert outcome.status == "blocked"
assert outcome.error.code == "production_requires_approval"
```

Policies run sequentially in registration order. The phases are:

1. `before_tool` receives the provider-neutral intent before runtime arguments
   are injected. It may replace only the intent arguments or return a blocked
   `ToolOutcome`.
2. `before_dispatch` receives the prepared `ToolExecutionPlan`. It may replace
   only its dispatch spec or block the call.
3. `after_tool` receives the outcome produced by the dispatcher and may replace
   its result fields.

The first blocked outcome stops later policies for that phase and affects only
that tool call. An exception in `before_tool` or `before_dispatch` fails closed.
An exception in `after_tool` preserves the outcome already produced.

Existing lifecycle hooks remain supported. Their order is: policy
`before_tool`, legacy `before_tool`, legacy `before_dispatch`, policy
`before_dispatch`, tool execution, legacy `after_tool`, then policy
`after_tool`.

## Custom Runtime Inputs

`ToolContextProvider` registers one or more sources that tools select through
`runtime_inputs`. This keeps provider lookup, network access, and runtime state
out of the ToolLibrary core.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools import ToolIntent


class TenantProvider(nn.ToolContextProvider):
    def __init__(self):
        super().__init__("tenant_runtime", sources=["tenant"])

    async def resolve(self, request):
        return request.context.require("vars")["tenant"]


@mf.tool_config(
    runtime_inputs=[
        nn.ContextBinding(source="tenant", parameter="tenant_id"),
    ]
)
def identify_order(order_id: str, tenant_id: str) -> str:
    """Return a tenant-qualified order identifier."""
    return f"{tenant_id}:{order_id}"


library = nn.ToolLibrary(
    name="orders",
    tools=[identify_order],
    extensions=[TenantProvider()],
)

outcome = library.execute_intents(
    [
        ToolIntent(
            id="call_1",
            name="identify_order",
            arguments={"order_id": "order_42"},
        )
    ],
    vars={"tenant": "acme"},
)[0]

assert outcome.result == "acme:order_42"
```

Providers run asynchronously in registration order and receive the canonical
definition, intent, and tool-scoped runtime context. Missing providers fail
during argument preparation, before tool execution or activity recording.

## Custom Dispatch Modes

`ToolDispatch` adds an execution mode without changing `ToolLibrary`. Its
`dispatch_name` is selected with `@tool_config(dispatch="...")`. Dispatchers are
asynchronous even when the legacy synchronous ToolLibrary API starts the call.

This example routes a tool to an external queue instead of executing its Python
implementation:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools import ToolIntent, ToolOutcome


class QueueDispatch(nn.ToolDispatch):
    def __init__(self):
        super().__init__("dispatch_queue", dispatch_name="queue")

    async def dispatch(self, request):
        # Replace this with an async queue or HTTP client.
        job_id = f"job:{request.plan.intent.id}"
        return ToolOutcome.dispatched(
            request.plan.intent,
            result={"job_id": job_id},
        )


@mf.tool_config(dispatch="queue")
def generate_report(report_id: str) -> str:
    """Generate a report through an external worker."""
    raise RuntimeError("The queue dispatcher owns this execution path")


library = nn.ToolLibrary(
    name="reports",
    tools=[generate_report],
    extensions=[QueueDispatch()],
)

outcome = library.execute_intents(
    [
        ToolIntent(
            id="call_1",
            name="generate_report",
            arguments={"report_id": "weekly"},
        )
    ]
)[0]

assert outcome.status == "dispatched"
assert outcome.result == {"job_id": "job:call_1"}
```

A dispatcher that still wants local execution calls
`await request.execute()` and may wrap or replace the resulting `ToolOutcome`.
The request also carries `ToolRuntimeContext`, including the current abort
signal, handle, messages, vars, task store, inbox, and activity recorder.

## Registration Lifecycle

An extension can validate compiled definitions and react to library membership
without reading a callable's mutable `tool_config`:

```python
from msgflux.nn import ToolLibrary, ToolLibraryExtension


events = []


def audit(event, name):
    events.append((event, name))


class SafeCatalog(ToolLibraryExtension):
    def __init__(self):
        super().__init__("safe_catalog")

    def validate_tool(self, library, definition):
        if definition.name == "delete_home":
            raise ValueError("delete_home is not allowed")

    def on_tool_added(self, library, definition):
        audit("added", definition.name)

    def on_tool_removed(self, library, definition):
        audit("removed", definition.name)

    def on_clear(self, library):
        audit("cleared", library.name)


def lookup(query: str) -> str:
    """Look up one catalog value."""
    return query


library = ToolLibrary("catalog", [], extensions=[SafeCatalog()])
library.add(lookup)
library.remove("lookup")
library.clear()

assert events == [
    ("added", "lookup"),
    ("removed", "lookup"),
    ("cleared", "catalog_tool_library"),
]
```

`validate_tool` runs sequentially in extension registration order before the
registry or executor containers change. The other callbacks run after the
corresponding mutation succeeds. They receive the canonical `ToolDefinition`,
so policy, loading, schema, and presentation values are the same snapshots used
by execution. These callbacks are synchronous registration boundaries; use
async execution policies and hooks for network work on the tool-call path.

## Register And Remove

Registration returns an ownership handle:

```python
extension = MathTools()
handle = library.register_extension(extension.name, extension)

handle.remove()
assert "double" not in library.get_tool_names()
```

Use `await handle.aremove()` when cleanup performs async I/O. Library extension
removal is immediate: tools and hooks can disappear from an execution already
in progress. Configure extensions before serving concurrent requests and treat
runtime removal as an explicit administrative operation.

## Built-In Extensions

`ToolLibrary` uses the same mechanism for optional capabilities:

- `ToolSearchExtension` installs the deferred-tool search bucket when the first
  `defer_loading=True` tool is registered.
- `BackgroundTasksExtension` validates background capabilities and reconciles
  the task-control tool surface through registration lifecycle callbacks. It
  also owns the set of controls explicitly disabled by the application.
- `MCPServersExtension` owns MCP connection, discovery, remote tools, and
  connection cleanup. The existing `mcp_servers=` argument constructs this
  extension for compatibility and convenience.

Tool buckets, argument restoration, runtime injection, abort checks, and
execution telemetry remain part of the library core because every extension
must preserve those invariants.
