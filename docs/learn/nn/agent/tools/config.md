# Tool Config

The `@mf.tool_config` decorator adds special behaviors to tools.

## return_direct

When `return_direct=True`, the tool result is returned directly as the final response instead of going back to the model.

Use cases:

- Reduce agent calls by designing tools that return user-ready outputs
- Agent as router - delegate to specialists and return their responses directly

???+ example

    === "Basic Usage"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(return_direct=True)
        def get_report() -> str:
            """Return the report."""
            return "This is your detailed report..."

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [get_report]

        agent = Assistant()
        response = agent("Give me the report")
        # Returns the tool result directly, no model formatting
        ```

    === "With Reasoning Models"

        Combine `return_direct` with reasoning models to optimize tool calls. The model reasons about which tool to use, but the result bypasses additional processing:

        ```python
        # pip install msgflux[openai]
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

        Combine with `inject_vars` for external processing:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(GROQ_API_KEY="...")

        @mf.tool_config(return_direct=True, inject_vars=True)
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
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [generate_formatted_report]
            config = {"verbose": True}

        agent = Reporter()
        response = agent("Generate the Q3 report", vars={"date_range": "2024-Q3"})
        ```

## inject_vars

With `inject_vars=True`, tools can access and modify the agent's variable dictionary.

Use cases:

- Pass external credentials (API keys, tokens)
- Share state between tools
- Extract information from tools without returning it to the model (e.g., store metadata, logs, or intermediate results in `vars` for later use)

???+ example

    === "External Credentials"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_vars=True)
        def save_to_s3(**kwargs) -> str:
            """Save file to S3."""
            vars = kwargs.get("vars")
            token = vars["aws_token"]
            # Use token for S3 upload
            return "File saved successfully"

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [save_to_s3]

        agent = Assistant()
        response = agent("Save my file", vars={"aws_token": "secret-123"})
        ```

    === "Named Parameters"

        Inject specific vars as named parameters:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_vars=["api_key", "user_id"])
        def upload_file(**kwargs) -> str:
            """Upload user file."""
            api_key = kwargs["api_key"]
            user_id = kwargs["user_id"]
            return f"Uploaded for user {user_id}"

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [upload_file]

        agent = Assistant()
        response = agent("Upload my file", vars={"api_key": "...", "user_id": "123"})
        ```

    === "Mutable State"

        Tools can modify vars for persistent state:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_vars=True)
        def save_preference(name: str, value: str, **kwargs):
            """Save a user preference."""
            vars = kwargs.get("vars")
            vars[name] = value  # Modifies the vars dict
            return f"Saved {name} = {value}"

        @mf.tool_config(inject_vars=True)
        def get_preference(name: str, **kwargs):
            """Get a user preference."""
            vars = kwargs.get("vars")
            return vars.get(name, "Not found")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
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

    @mf.tool_config(disable_input=True, inject_messages=True)
    class Specialist(nn.Agent):
        """Specialist that works only from conversation context."""

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        system_message = "You are a specialist. Use the conversation history."

    class Coordinator(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        tools = [Specialist]
    ```

## Runtime Injection Options

Tool config supports several `inject_*` flags for common agent state. Use the
`handle` access dictionary for runtime operations. `mf.Hidden` only hides a
parameter from the model-facing schema; it does not inject a value by itself.

| API | Runtime argument | Documentation |
|-----|------------------|---------------|
| `inject_vars=True` | `vars` | [inject_vars](#inject_vars) |
| `inject_message=True` | `message` | [inject_message](#inject_message) |
| `inject_messages=True` | `messages` | [inject_messages](#inject_messages) |
| `handle={...}` | `handle` | [handle](#handle) |
| `mf.Hidden` | Hidden schema parameter | [Tools: Hidden Tool Parameters](index.md#hidden-tool-parameters) |

## handle

`handle` is a dictionary from domain names to non-empty action lists. The tool
receives only those operations, and must declare `handle: mf.Hidden`. Missing or
`None` means no injection; `{}`, booleans, unknown names, and `"all"` are invalid.

| Domain | Actions |
|--------|---------|
| `notifications` | `publish` |
| `task` | `read`, `progress`, `activity`, `interrupt_check` |
| `tasks` | `list`, `read`, `wait`, `output`, `interrupt`, `activity`, `message` |
| `tools` | `list`, `get`, `register`, `remove` |
| `background` | `dispatch`, `resume` |

```python
import msgflux as mf


def lookup_customer(customer_id: str) -> str:
    """Look up customer details."""
    return customer_id


@mf.tool_config(handle={"tools": ["list", "register"]})
def enable_lookup(handle: mf.Hidden) -> list[str]:
    """Register tools dynamically through the runtime handle."""
    handle.tools.register(lookup_customer)
    return handle.tools.list()
```

## inject_message

With `inject_message=True`, the tool receives the original `message` passed to the
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
        @mf.tool_config(inject_message=True, disable_input=True)
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
        @mf.tool_config(inject_message=True)
        def inspect_original_message(**kwargs) -> str:
            """Inspect the original message envelope."""
            message = kwargs.get("message")
            if isinstance(message, mf.Message):
                return str(message.get("meta.trace_id"))
            return "No structured message available."
        ```

## inject_messages

With `inject_messages=True`, the tool receives the agent's internal state
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

        When an agent is used as a tool, `inject_messages` passes the conversation history so the specialist has full context:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # With inject_messages, the specialist receives
        # the coordinator's conversation as messages
        @mf.tool_config(inject_messages=True)
        class Specialist(nn.Agent):
            """Expert that needs conversation context."""

            model = model
            system_message = "You are a specialist."

        class Coordinator(nn.Agent):
            model = model
            system_message = "Route to specialists when needed."
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
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_messages=True)
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
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Always check safety before responding."
            tools = [check_safety]
            config = {"verbose": True}

        agent = SafeAgent()
        response = agent("Can you help me write a Python script?")
        ```

    === "Context-Aware Processing"

        Access images or other multimodal content from conversation:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        @mf.tool_config(inject_messages=True)
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

- Sets `return_direct=True` and `inject_messages=True`
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
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    # Tool is now "transfer_to_TechnicalSupport" with no parameters
    @mf.tool_config(handoff=True)
    class TechnicalSupport(nn.Agent):
        """Specialist for technical issues, debugging, and troubleshooting."""
        model = model
        system_message = "You are a technical support specialist."
        instructions = "Help users solve technical problems step by step."
        config = {"verbose": True}

    class Coordinator(nn.Agent):
        """Routes user queries to the appropriate specialist."""
        model = model
        system_message = "You are a support coordinator."
        instructions = "Transfer users to technical support for technical issues."
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
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

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
        system_message = """You're a BI analyst. When a user requests sales reports,
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

## spawn

Dispatch a tool without waiting for a result. The model receives a confirmation that the task was started, but no return value. Requires async tool.

Use cases:

- Spawned operations (emails, notifications)
- Tasks that don't need to return a result to the model

???+ example

    ```python
    # pip install msgflux[openai]
    import asyncio
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    @mf.tool_config(spawn=True)
    async def send_notification(user_id: str, message: str):
        """Send notification asynchronously. Will not generate a return."""
        # Simulate async operation (e.g., API call, email sending)
        await asyncio.sleep(2)
        print(f"Notification sent to {user_id}: {message}")

    class Notifier(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        tools = [send_notification]
        config = {"verbose": True}

    agent = Notifier()

    # Agent returns immediately, notification is dispatched
    response = agent("Notify user123 that their order shipped")
    ```

## name_override

Assign a custom name to a tool:

```python
import httpx

@mf.tool_config(name_override="search_repos")
def github_repository_search_v2_extended(query: str) -> str:
    """Search GitHub repositories."""
    url = "https://api.github.com/search/repositories"
    resp = httpx.get(url, params={"q": query, "per_page": 3})
    repos = resp.json().get("items", [])
    return "\n".join(f"- {r['full_name']}" for r in repos)

# Tool is exposed as "search_repos" instead of the long function name
```

## background_capabilities

Use `background_capabilities` to declare the optional task controls a
background-capable tool implements. Valid values are `activity` and `message`.
This option requires `background=True` or `allow_background=True`.

```python
@mf.tool_config(background=True, background_capabilities=["activity"])
def index_documents(path: str) -> str:
    """Index a document tree in the background."""
    return path
```

The declaration installs `task_activity` while the tool is registered. `message`
installs `task_message`; it is currently implemented by background agents
through their inbox and checkpoint runtime, so it is rejected for other source
kinds. `Agent` and `AgentTool` use both capabilities by default when this
option is omitted.

## tool_kind

Use `tool_kind` to label a tool for `ToolBucket` routing. The label is stored
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
    capture = {"tool_kind": "catalog|orders", "on_demand": False}
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

Buckets trust captured tools by default. To restrict the handle access a bucket
will accept, declare the optional `handle` policy inside `capture`:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools import ToolBucket

@mf.tool_config(
    tool_kind="operation",
    handle={"tools": ["list"]},
)
def list_registered_tools(handle: mf.Hidden) -> list[str]:
    """List registered tools."""
    return handle.tools.list()

class ReadOnlyOperations(ToolBucket):
    """Capture read-only tool operations."""

    name = "read_only_operations"
    capture = {
        "tool_kind": "operation",
        "on_demand": False,
        "policy": {
            "handle": {
                "tools": ["list", "get"],
            }
        },
    }
    annotations = {"return": str}

    def __call__(self) -> str:
        return ",".join(sorted(self.tools))

library = nn.ToolLibrary(
    name="read_only",
    tools=[ReadOnlyOperations(), list_registered_tools],
)

print(library.get_tool_names())
# ["read_only_operations"]
```

Every handle domain and action requested by a captured tool must be included in
the policy. A bucket without `capture["policy"]` applies no capture restriction;
built-in buckets use this trusted behavior by default. `handle` is currently the
only supported capture policy.

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
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
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

tools = [
    AgentTool(),
    *apply_tool_guidance([WebSearchTool(), WebFetchTool()]),
]

agent = nn.Agent(
    name="assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
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
            import httpx
            resp = httpx.get("https://api.example.com/search", params={"q": query})
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
