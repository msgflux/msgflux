# Tool Config

The `@mf.tool_config` decorator adds special behaviors to tools.

`ToolLibrary` compiles this configuration when the tool is registered. Changing
the callable's `tool_config` afterward does not alter an already registered
tool; remove and add it again when an application intentionally needs a new
definition. This keeps concurrent executions on one stable schema and runtime
policy.

## required_permissions

Declare the capabilities a tool requires. The trusted application supplies live
grants through `ExecutionScope`; model arguments and runtime `vars` cannot grant
permissions.

```python
import msgflux as mf
from msgflux.nn import ToolLibrary
from msgflux.runtime import ExecutionScope, PermissionSet, execution_context


@mf.tool_config(required_permissions=["catalog.read"])
def lookup(sku: str) -> str:
    """Look up a catalog entry."""
    return f"Product {sku}"


library = ToolLibrary("catalog", [lookup])
with execution_context(scope=ExecutionScope(
    principal="user:42",
    permissions=PermissionSet(["catalog.read"]),
)):
    result = library.run("lookup", {"sku": "ABC"})
```

This example grants `catalog.read` only for the enclosed execution. Names match
exactly; wildcards and resource patterns are not supported. Nested scopes can
restrict grants but cannot widen them or switch the principal. See
[Live authority](../runtime.md#live-authority) for inheritance and serialization.

Requirements are frozen when registered and are not model-input properties.
The library checks them before argument injection and hooks, again before
dispatch, and at executor entry. Captured tools and background dispatch use the
same requirements; resuming a background agent checks the caller's current
grants before changing task state. Checkpoints do not restore authority.

Missing grants produce a blocked tool outcome with `tool_permission_denied`.
The live stream emits `tool.permission_denied` and `tool.blocked`, identifying
the tool call without copying arguments or grants into those events. This does
not redact other execution events. Direct `library.run`/`arun` calls raise
`RuntimeError`, like other blocked outcomes; direct local/MCP adapter calls
raise `PermissionError`.

!!! warning "Capability checks are not a sandbox"
    Tools without declared requirements remain executable for compatibility.
    Python implementations, policies and dispatchers are trusted host code;
    calling a decorated Python function directly does not enforce this boundary.
    Capability names do not restrict filesystem paths, network destinations or
    process access. Provider-hosted tools are outside the local dispatch gate.
    OS isolation, resource-scoped grants, user-approval suspension and durable
    audit replay are follow-up work, not guarantees of this API.

## description

Set `description` to replace the model-facing tool description without changing
the callable's docstring. When omitted, msgFlux snapshots the `description`
class attribute, function docstring, or `__call__` docstring when the tool is
registered.

```python
@mf.tool_config(description="Search the product catalog by SKU or product name.")
def search_products(query: str) -> str:
    """Internal implementation notes can remain here."""
    return query
```

Configure the description before adding the tool to `ToolLibrary`. Do not edit
the compiled definition directly; use `handle.remove(name)` and
`handle.add(configured_tool)` when replacing a dynamically registered tool.

## defer_loading

Set `defer_loading=True` for tools that should remain searchable without
occupying the model's initial callable surface. See [Tool Search](tool-search.md) for more details.

```python
import msgflux as mf


@mf.tool_config(defer_loading=True)
def search_archive(query: str) -> str:
    """Search the long-term archive."""
    return query
```

## feedback

Set `feedback` when an Agent extension should decide what happens after the
tool settles. The value is compiled into the tool's `FeedbackSpec` and copied
to its canonical `ToolOutcome`:

```python
import msgflux as mf


@mf.tool_config(feedback="approval")
def request_deployment(environment: str) -> str:
    """Prepare a deployment request for review."""
    return f"deployment:{environment}"
```

An extension can handle `approval` through the `resolve_tool_feedback`
lifecycle event. If no extension makes a return decision, the Agent sends
the outcome back to the model normally. See
[Tool Feedback Extensions](../extensions.md#tool-feedback-extensions).

Use either `feedback` or one of the compatibility aliases `return_direct`,
`handoff`, and `call_as_response`; combining them is rejected. Pass a
`FeedbackSpec` instead of a string when the mode requires static options.

## return_direct

When `return_direct=True`, the builtin `tool_feedback` Agent extension returns
the tool result as the final response instead of sending it back to the model.

Use cases:

- Reduce agent calls by designing tools that return user-ready outputs
- Agent as router - delegate to specialists and return their responses directly

???+ example

    === "Basic Usage"

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(return_direct=True)
        def get_report() -> str:
            """Return the report."""
            return "This is your detailed report..."

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-5.6-luna")
            tools = [get_report]

        agent = Assistant()
        response = agent("Give me the report")
        # Returns the tool result directly, no model formatting
        ```

    === "With Reasoning Models"

        Combine `return_direct` with reasoning models to optimize tool calls. The model reasons about which tool to use, but the result bypasses additional processing:

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(GROQ_API_KEY="...")

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-20b", reasoning_effort="low"
        )

        @mf.tool_config(return_direct=True)
        def get_report() -> str:
            """Return the report from user."""
            return "This is your detailed report..."

        class ReporterAgent(nn.Agent):
            model = model
            tools = [get_report]
            config = {"tool_choice": "required", "verbose": True}

        agent = ReporterAgent()
        response = agent("Give me the report")
        ```

    === "Report Generator"

        Combine with the `vars` runtime input for external processing:

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(GROQ_API_KEY="...")

        @mf.tool_config(return_direct=True, runtime_inputs=["vars"])
        def generate_formatted_report(**kwargs) -> str:
            """Generate a formatted sales report."""
            vars = kwargs.get("vars", {})
            date_range = vars.get("date_range", "Unknown")

            # Mock data - in production, query your database
            report = f"""
            Sales Report: {date_range}
            ─────────────────────────────
            Total Revenue: $124,500
            Total Orders: 847
            Average Order: $147.04
            Top Product: Widget Pro (234 units)
            ─────────────────────────────
            Generated automatically.
            """
            return report

        class Reporter(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-5.6-luna")
            tools = [generate_formatted_report]
            config = {"verbose": True}

        agent = Reporter()
        response = agent("Generate the Q3 report", vars={"date_range": "2024-Q3"})
        ```

## runtime_inputs

`runtime_inputs` binds values supplied by the ToolLibrary runtime to hidden tool
parameters. These parameters are removed from the schema shown to the model.

```python
import msgflux as mf


@mf.tool_config(runtime_inputs=["handle", "messages", "vars"])
def inspect_runtime(handle, messages, vars):
    """Inspect the current execution environment."""
    return {
        "tools": handle.list_tools(),
        "message_count": len(messages),
        "tenant": vars.get("tenant"),
    }
```

Built-in sources are `message`, `messages`, `vars`, and `handle`. Use
`ContextBinding` when the parameter name differs from the source or only one
value should be selected:

```python
import msgflux.nn as nn


@mf.tool_config(
    runtime_inputs=[
        nn.ContextBinding(
            source="vars",
            parameter="tenant_id",
            options={"key": "tenant_id"},
        )
    ]
)
def load_orders(tenant_id: str) -> str:
    """Load orders for the current tenant."""
    return tenant_id
```

Extensions may register additional sources with `ToolContextProvider`. See
[ToolLibrary Extensions](tool-library-extensions.md#custom-runtime-inputs).

### `vars`

The `vars` source gives tools access to the agent's variable dictionary.

Use cases:

- Pass external credentials (API keys, tokens)
- Share state between tools
- Extract information from tools without returning it to the model (e.g., store metadata, logs, or intermediate results in `vars` for later use)

???+ example

    === "External Credentials"

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(runtime_inputs=["vars"])
        def save_to_s3(**kwargs) -> str:
            """Save file to S3."""
            vars = kwargs.get("vars")
            token = vars["aws_token"]
            # Use token for S3 upload
            return "File saved successfully"

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-5.6-luna")
            tools = [save_to_s3]

        agent = Assistant()
        response = agent("Save my file", vars={"aws_token": "secret-123"})
        ```

    === "Named Parameters"

        Inject specific vars as named parameters:

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(
            runtime_inputs=[
                nn.ContextBinding(
                    source="vars",
                    parameter="api_key",
                    options={"key": "api_key"},
                ),
                nn.ContextBinding(
                    source="vars",
                    parameter="user_id",
                    options={"key": "user_id"},
                ),
            ]
        )
        def upload_file(**kwargs) -> str:
            """Upload user file."""
            api_key = kwargs["api_key"]
            user_id = kwargs["user_id"]
            return f"Uploaded for user {user_id}"

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-5.6-luna")
            tools = [upload_file]

        agent = Assistant()
        response = agent("Upload my file", vars={"api_key": "...", "user_id": "123"})
        ```

    === "Mutable State"

        Tools can modify vars for persistent state:

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(runtime_inputs=["vars"])
        def save_preference(name: str, value: str, **kwargs):
            """Save a user preference."""
            vars = kwargs.get("vars")
            vars[name] = value  # Modifies the vars dict
            return f"Saved {name} = {value}"

        @mf.tool_config(runtime_inputs=["vars"])
        def get_preference(name: str, **kwargs):
            """Get a user preference."""
            vars = kwargs.get("vars")
            return vars.get(name, "Not found")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-5.6-luna")
            tools = [save_preference, get_preference]

        agent = Assistant()

        user_vars = {}
        agent("Save my favorite color as blue", vars=user_vars)
        agent("What is my favorite color?", vars=user_vars)

        print(user_vars)  # {"favorite_color": "blue"}
        ```

## disable_input

With `disable_input=True`, the tool exposes no public input parameters to the
model. The tool is called as `tool_name()`, and any arguments supplied by the
model are ignored at runtime.

This is useful for:

- Specialist subagents that should be triggered without a task payload
- Tools that work only with injected context such as `message`, `messages`, or `vars`
- Internal routing tools where the coordinator should only decide whether to call

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn as nn

    @mf.tool_config(disable_input=True, runtime_inputs=["messages"])
    class Specialist(nn.Agent):
        """Specialist that works only from conversation context."""

        model = mf.Model.chat_completion("openai/gpt-5.6-luna")
        system_prompt = "You are a specialist. Use the conversation history."

    class Coordinator(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-5.6-luna")
        tools = [Specialist]
    ```

## `handle` Runtime Input

With `runtime_inputs=["handle"]`, the tool receives a tool-scoped
`ToolLibraryHandle` at runtime. The handle gives the implementation controlled
access to the library and to runtime resources associated with the current
call. It is not a model argument and is removed from the public tool schema.

The most common operations are:

| Area | Methods | Availability |
|------|---------|--------------|
| Tool library | `add`, `remove`, `list_tools`, `get_tool` | Any injected handle. |
| Current background task | `get_task`, `get_task_id`, `set_running`, `update_progress` | Only while the tool is executing as a background task. |
| Cooperative control | `raise_if_interrupted`, `raise_if_paused` | Only while the tool is executing as a background task. |
| Agent notifications | `notify`, `get_notification` | Any tool-scoped handle; delivery uses the active agent inbox. |
| Runtime stores | `get_task_store`, `get_agent_inbox` | Uses the resources inherited by the current execution. |

Calling a task-only method during normal inline execution raises a
`RuntimeError`. This catches tools that request the `handle` runtime input
but forgot to opt into `background=True` or model-selected
`allow_background=True`.

```python
import msgflux as mf


def lookup_customer(customer_id: str) -> str:
    """Look up customer details."""
    return customer_id


@mf.tool_config(runtime_inputs=["handle"])
def enable_lookup(handle: mf.Hidden) -> list[str]:
    """Register tools dynamically through the runtime handle."""
    handle.add(lookup_customer)
    return handle.list_tools()
```

`handle.add(...)` uses the same registration path as
`ToolLibrary.add(...)`. That means the new tool is normalized, validated,
routed into a matching [ToolBucket](tool-bucket.md), and reconciled with the
background task controls. The mutation affects the shared `ToolLibrary`; it is
not thread-local. By contrast, selecting an already registered deferred tool
through [Tool Search](tool-search.md) is stored in `ChatMessages` and is local
to that conversation.

For a background tool, the same handle can report durable task progress and
publish a lightweight notification without exposing runtime parameters to the
model:

```python
import msgflux as mf


@mf.tool_config(background=True, runtime_inputs=["handle"])
def reconcile_inventory(sku: str, handle: mf.Hidden) -> dict:
    """Reconcile one SKU and report task progress."""
    handle.set_running(stage="load", message=f"Loading {sku}")
    handle.raise_if_interrupted()
    handle.update_progress(
        stage="compare",
        message="Compared reservations with physical stock",
        current=1,
        total=1,
    )
    handle.notify(
        source="task_progress",
        status="reconciled",
        metadata={"sku": sku},
        dedupe_key=f"inventory:{handle.get_task_id()}",
    )
    return {"sku": sku, "status": "reconciled"}
```

The progress record is read with `task_status`; the notification is delivered
through the agent inbox; and the return value is retrieved with `task_output`
or `task_wait`. See [Background Tasks](background-tasks.md#reporting-progress)
for the complete lifecycle.

## `message` Runtime Input

The `message` runtime input receives the original message passed to the
agent. This is useful when the tool needs access to the full envelope object,
including `Message` fields that should not be part of the public tool schema.

Use cases:

- Agent-as-a-tool with declarative `Message` envelopes
- Tools that need `response_mode` side effects on the original message
- Access to `vars`, metadata, or other fields outside the public task schema

???+ example

    === "Hidden parameter"

        Use this when the function should explicitly name the injected
        `message` argument. `disable_input=True` hides all public parameters, so
        the model does not see or fill `message`.

        ```python
        @mf.tool_config(runtime_inputs=["message"], disable_input=True)
        def inspect_original_message(message) -> str:
            """Inspect the original message envelope."""
            if isinstance(message, mf.Message):
                return str(message.get("meta.trace_id"))
            return "No structured message available."
        ```

    === "From kwargs"

        Use this when the public signature should not include `message`.
        msgFlux injects it through `kwargs`.

        ```python
        @mf.tool_config(runtime_inputs=["message"])
        def inspect_original_message(**kwargs) -> str:
            """Inspect the original message envelope."""
            message = kwargs.get("message")
            if isinstance(message, mf.Message):
                return str(message.get("meta.trace_id"))
            return "No structured message available."
        ```

## `messages` Runtime Input

The `messages` runtime input receives the agent's internal state
(conversation history) as `messages` in kwargs. This is particularly useful for
**agent-as-a-tool** patterns where you want to pass the full conversation context
to a specialist agent.

Use cases:

- Agent-as-a-tool: Pass conversation history to specialist agents
- Safety/moderation checks on conversation
- Access multimodal context (e.g. images in conversation)
- Context-aware tool execution

???+ example

    === "Agent-as-Tool (Primary Use)"

        When an agent is used as a tool, the `messages` runtime input passes the conversation history so the specialist has full context:

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-5.6-luna")

        # With the messages runtime input, the specialist receives
        # the coordinator's conversation as messages
        @mf.tool_config(runtime_inputs=["messages"])
        class Specialist(nn.Agent):
            """Expert that needs conversation context."""

            model = model
            system_prompt = "You are a specialist."

        class Coordinator(nn.Agent):
            model = model
            system_prompt = "Route to specialists when needed."
            tools = [Specialist]
            config = {"verbose": True}

        coordinator = Coordinator()

        # When coordinator calls specialist, the full conversation
        # is passed via messages parameter
        response = coordinator("Help me with a complex problem")
        ```

    === "Safety Checker"

        Check conversation safety before responding:

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(runtime_inputs=["messages"])
        def check_safety(**kwargs) -> dict:
            """Check if the conversation is safe to continue."""
            messages = kwargs.get("messages", [])
            last_message = messages[-1]["content"] if messages else ""

            # Simple keyword-based safety check
            forbidden_keywords = ["hack", "exploit", "malware", "attack"]
            content_lower = last_message.lower()
            is_safe = not any(kw in content_lower for kw in forbidden_keywords)

            return {
                "safe": is_safe,
                "reason": None if is_safe else "Potentially harmful content detected"
            }

        class SafeAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-5.6-luna")
            system_prompt = "Always check safety before responding."
            tools = [check_safety]
            config = {"verbose": True}

        agent = SafeAgent()
        response = agent("Can you help me write a Python script?")
        ```

    === "Context-Aware Processing"

        Access images or other multimodal content from conversation:

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(runtime_inputs=["messages"])
        def analyze_shared_images(**kwargs) -> str:
            """Analyze all images shared in the conversation."""
            messages = kwargs.get("messages", [])

            images = []
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "image_url":
                            images.append(block["image_url"]["url"])

            if not images:
                return "No images found in conversation."

            return f"Found {len(images)} images to analyze."
        ```

## handoff

When `handoff=True`, the tool is configured for seamless agent-to-agent handoff:

- Sets `return_direct=True` and requests the `messages` runtime input
- Changes tool name to `transfer_to_{original_name}`
- Removes input parameters (equivalent to `disable_input=True`)

Unlike Agent-as-Tool, the Specialist's response bypasses the Coordinator entirely and goes directly to the user. The Coordinator only decides *who* handles the request.

```
              Input
                │
                ▼
  ┌────────────────────────────────┐
  │         Coordinator            │
  │                                │
  │   ┌──────────┐                 │
  │   │  Model   │──▶ "transfer_to │
  │   └──────────┘    Specialist()"│
  └──────────────┬─────────────────┘
                 │
                 │  + full conversation history
                 ▼
  ┌──────────────────────────────┐
  │         Specialist           │
  │          (Agent)             │
  │                              │
  │  receives full conversation  │
  │  context and takes ownership │
  └─────────────────┬────────────┘
                    │
                    ▼  (direct — Coordinator bypassed)
                  Output
```

???+ example

    ```python
    # pip install msgflux
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-5.6-luna")

    # Tool is now "transfer_to_TechnicalSupport" with no parameters
    @mf.tool_config(handoff=True)
    class TechnicalSupport(nn.Agent):
        """Specialist for technical issues, debugging, and troubleshooting."""
        model = model
        system_prompt = """You are a technical support specialist.
        Help users solve technical problems step by step."""
        config = {"verbose": True}

    class Coordinator(nn.Agent):
        """Routes user queries to the appropriate specialist."""
        model = model
        system_prompt = """You are a support coordinator.
        Transfer users to technical support for technical issues."""
        tools = [TechnicalSupport]
        config = {"verbose": True}

    coordinator = Coordinator()
    response = coordinator("My application crashes when I try to connect to the database")
    ```

## call_as_response

Return tool call parameters **without executing** the tool. Useful for extracting structured data.

Use cases:

- BI report parameter extraction
- API call preparation
- Form data collection

???+ example

    ```python
    # pip install msgflux
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-5.6-luna")

    @mf.tool_config(call_as_response=True)
    def generate_sales_report(
        start_date: str, end_date: str, metrics: list[str], group_by: str
    ) -> dict:
        """Generate a sales report within a given date range.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            metrics: List of metrics to include (e.g., ["revenue", "orders", "profit"]).
            group_by: Dimension to group data by (e.g., "region", "product", "sales_rep").

        Returns:
            A structured sales report as a dictionary.
        """
        return  # Never executed

    class BIAnalyst(nn.Agent):
        model = model
        system_prompt = """You're a BI analyst. When a user requests sales reports,
        you should simply complete the generate_sales_report tool call,
        extracting the requested metrics, dates, and groupings."""
        tools = [generate_sales_report]
        config = {"verbose": True}

    agent = BIAnalyst()
    response = agent(
        "I need a report of sales between July 1st and August 31st, 2025, "
        "showing revenue and profit, grouped by region."
    )
    # Returns the tool call parameters without executing the function
    ```

## dispatch

Select a dispatch mode registered by a `ToolDispatch` extension:

```python
@mf.tool_config(dispatch="queue")
def generate_report(report_id: str) -> str:
    """Generate a report through an external worker."""
    ...
```

The ToolLibrary fails the call if `queue` is not registered. `dispatch` cannot
be combined with `background`, `allow_background`, or `detached`. See
[ToolLibrary Extensions](tool-library-extensions.md#custom-dispatch-modes) for a
complete dispatcher.

## detached

Dispatch a tool without waiting for a result. The model receives confirmation
that the task started, but no eventual return value. The tool may be synchronous
or asynchronous.

Use cases:

- Detached operations (emails, notifications)
- Tasks that don't need to return a result to the model

???+ example

    ```python
    # pip install msgflux
    import asyncio
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    @mf.tool_config(detached=True)
    async def send_notification(user_id: str, message: str):
        """Send notification asynchronously. Will not generate a return."""
        # Simulate async operation (e.g., API call, email sending)
        await asyncio.sleep(2)
        print(f"Notification sent to {user_id}: {message}")

    class Notifier(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-5.6-luna")
        tools = [send_notification]
        config = {"verbose": True}

    agent = Notifier()

    # Agent returns immediately, notification is dispatched
    response = agent("Notify user123 that their order shipped")
    ```

## name_override

Assign a custom name to a tool:

```python
import httpx2

@mf.tool_config(name_override="search_repos")
def github_repository_search_v2_extended(query: str) -> str:
    """Search GitHub repositories."""
    url = "https://api.github.com/search/repositories"
    resp = httpx2.get(url, params={"q": query, "per_page": 3})
    repos = resp.json().get("items", [])
    return "\n".join(f"- {r['full_name']}" for r in repos)

# Tool is exposed as "search_repos" instead of the long function name
```

## background_capabilities

Use `background_capabilities` to declare optional controls supported by a
background-capable tool. It does not enable background execution by itself;
the tool must also use `background=True` or `allow_background=True`.

Every background-capable tool installs the common controls (`task_status`,
`task_list`, `task_wait`, `task_output`, and `task_interrupt`). Capabilities add
to that shared surface:

| Capability | Adds | Intended source |
|------------|------|-----------------|
| `activity` | `task_activity` | Any background implementation that records compact activity. |
| `message` | `task_message` | `Agent` and `AgentTool`, whose inbox and checkpoint runtime can continue a task. |

```python
@mf.tool_config(background=True, background_capabilities=["activity"])
def index_documents(path: str) -> str:
    """Index a document tree in the background."""
    return path
```

The library computes the union of capabilities declared by its registered
background sources and exposes only the required optional controls. Removing
the last source for a capability removes its control as well. `message` is
rejected for ordinary functions because they do not implement the agent
continuation contract. Background `Agent` and `AgentTool` sources default to
both `activity` and `message` when the option is omitted; ordinary tools default
to no optional capabilities.

See [Background Tasks: Background Capabilities](background-tasks.md#background-capabilities)
for execution semantics, task continuation, and the reason these controls are
managed by `ToolBackground` rather than captured by a bucket.

## tool_kind

Use `tool_kind` to label a tool for [ToolBucket](tool-bucket.md) routing. The label is stored
in the tool configuration and does not change the function name or its
model-facing schema.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools import ToolBucket

@mf.tool_config(tool_kind="catalog")
def find_product(query: str) -> str:
    """Find a product by name."""
    return query

class CommerceTool(ToolBucket):
    """Group commerce operations."""

    name = "commerce"
    capture = {"tool_kind": "catalog|orders", "defer_loading": False}
    annotations = {"return": str}

    def __call__(self) -> str:
        return "ready"

library = nn.ToolLibrary(
    name="store",
    tools=[find_product, CommerceTool()],
)

print(library.get_tool_names())
# ["commerce"]
```

`CommerceTool` captures the `catalog` tool regardless of their order during
library initialization. `capture` matches `tool_config` entries, and
`capture["tool_kind"]` accepts one kind or several kinds separated by `|`.
Overlapping captures are rejected. Configure `background` or
`allow_background` on the bucket, not on a tool it captures with the base
`ToolBucket` validation.

## display_name

Assign a human-readable name for UI surfaces and events while keeping the tool's
programmatic name stable:

```python
import msgflux as mf
import msgflux.nn as nn

@mf.tool_config(display_name="Repository Search")
def search_repos(query: str) -> str:
    """Search GitHub repositories."""
    return query


def edit(text: str) -> str:
    """Edit text in place."""
    return text


library = nn.ToolLibrary(name="assistant", tools=[search_repos, edit])
print(library.get_tool_display_names())
# {'search_repos': 'Repository Search', 'edit': 'edit'}
```

`ToolLibrary.get_tool_display_names()` returns a `tool_name -> display_name`
mapping. The model still calls `search_repos`, but clients such as a CLI can
show `Repository Search` to users. If `display_name` is not passed, msgFlux
falls back to the tool's registered name.

## usage_guidance

Add guidance that is rendered into the agent system prompt under
`<tool_usage_guidance>`. Use it for tool-specific "when/how to use" instructions
that should not live in the function description.

```python
import msgflux as mf
import msgflux.nn as nn

@mf.tool_config(
    display_name="Repository Search",
    usage_guidance=(
        "Use when the user asks for GitHub repositories. Prefer concise search "
        "queries with language, framework, or topic constraints."
    ),
)
def search_repos(query: str) -> str:
    """Search GitHub repositories."""
    return query


agent = nn.Agent(
    name="assistant",
    model=mf.Model.chat_completion("openai/gpt-5.6-luna"),
    tools=[search_repos],
)

print(agent.get_system_prompt())
```

The guidance is injected into the system prompt when you render it with
`agent.get_system_prompt()`.

## Builtin usage guidance

msgFlux also ships an opt-in guidance registry for builtin tools. This keeps
tool implementations neutral while letting applications such as a CLI attach
ready-to-use instructions when composing an agent.

Use `apply_tool_guidance()` to fill `usage_guidance` only when a tool does not
already define one:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools import apply_tool_guidance
from msgflux.tools.builtin import AgentTool, WebFetchTool, WebSearchTool

tools = apply_tool_guidance([AgentTool(), WebSearchTool(), WebFetchTool()])

agent = nn.Agent(
    name="assistant",
    model=mf.Model.chat_completion("openai/gpt-5.6-luna"),
    tools=tools,
)

print(agent.get_system_prompt())
```

Explicit guidance always wins:

```python
@mf.tool_config(usage_guidance="Use only for internal repository search.")
def web_search(query: str) -> str:
    """Search internal repositories."""
    ...

tools = apply_tool_guidance([web_search])
# Keeps "Use only for internal repository search."
```

## retry

Control retry behavior per tool. Accepts a [tenacity](https://tenacity.readthedocs.io/) decorator, `False` to disable, or `None` (default) to use env-based retry.

By default, all tools have automatic retry enabled using environment variables (`TOOL_STOP_AFTER_ATTEMPT`, `TOOL_STOP_AFTER_DELAY`). Use this parameter to customize or disable retry for specific tools.

???+ example

    === "Custom Retry"

        ```python
        from tenacity import retry, stop_after_attempt, wait_exponential

        @mf.tool_config(
            retry=retry(
                reraise=True,
                stop=stop_after_attempt(5),
                wait=wait_exponential(min=1, max=10),
            )
        )
        def call_external_api(query: str) -> str:
            """Call an unreliable external API."""
            import httpx2
            resp = httpx2.get("https://api.example.com/search", params={"q": query})
            resp.raise_for_status()
            return resp.json()["result"]
        ```

    === "Disable Retry"

        ```python
        @mf.tool_config(retry=False)
        def fast_lookup(key: str) -> str:
            """Fast local lookup that should not retry."""
            return cache[key]
        ```

    === "Default (env-based)"

        ```python
        # No retry parameter needed — uses env defaults
        @mf.tool_config(return_direct=True)
        def search(query: str) -> str:
            """Search with default retry behavior."""
            return do_search(query)
        ```
