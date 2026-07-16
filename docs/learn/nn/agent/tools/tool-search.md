# Tool Search

Tool search keeps rarely used tools out of the model's initial tool list until
the model explicitly asks to load them.

Mark a tool as on-demand with `tool_config(on_demand=True)`:

```python
import msgflux as mf
import msgflux.nn as nn

@mf.tool_config(on_demand=True)
def query_finance_report(company: str) -> str:
    """Query archived finance reports for a company."""
    return f"Finance report for {company}"

agent = nn.Agent(
    name="analyst",
    model=mf.Model.chat_completion(
        "openai/gpt-5.6-luna",
        reasoning_effort="none",
    ),
    tools=[query_finance_report],
)
```

When at least one on-demand tool exists, msgFlux exposes a runtime tool named
`search_tools`. The on-demand tools are searchable but are not included in the
normal callable tool schemas until selected.

Internally, `search_tools` is a `ToolBucket` with
`capture={"source": "any", "on_demand": True}`. This keeps the search index
and deferred metadata together for both regular tools and bucket tools.

```python
schema_names = [
    schema["function"]["name"]
    for schema in agent.tool_library.get_tool_json_schemas()
]
print(schema_names)
# ["search_tools"]
```

## Search Before Loading

A normal search returns compact `name: guidance` or `name: description` lines
without loading them:

```python
result = agent.tool_library(
    [("call_1", "search_tools", {"query": "finance report"})]
).tool_calls[0].result

print(result)
# query_finance_report: Query archived finance reports for a company.
```

Use `/pattern/` for a case-insensitive regex search and append `:K` to limit the
number of matches:

```python
result = agent.tool_library(
    [
        (
            "call_2",
            "search_tools",
            {"query": "/finance|billing/:3"},
        )
    ]
).tool_calls[0].result

print(result)
```

Regex extensions, backreferences, and quantified groups are rejected. Patterns
are limited to 128 characters and search bounded metadata.

## Load Tools

Pass an exact returned name through the same `query` argument to activate it:

```python
result = agent.tool_library(
    [
        (
            "call_3",
            "search_tools",
            {"query": "query_finance_report"},
        )
    ]
).tool_calls[0].result

print(result)
# loaded=query_finance_report
```

After selection, the tool is promoted into the model-facing schemas:

```python
schema_names = [
    schema["function"]["name"]
    for schema in agent.tool_library.get_tool_json_schemas()
]
print(schema_names)
# ["query_finance_report"]
```

If other on-demand tools remain, `search_tools` stays available. If no on-demand
tools remain, msgFlux removes `search_tools` from the exposed runtime tools.

Promotion does not remove or register the tool again. The library atomically
releases the ownership edge from `search_tools`, changes its internal loading
state, and routes it to another matching bucket when applicable. The executable
wrapper and any descendants keep their identity throughout activation.

Bucket tools can also be loaded on demand. Their child capture should explicitly
select `on_demand=False`, keeping the bucket's children disjoint from the Tool
Search capture:

```python
from msgflux.tools import ToolBucket

@mf.tool_config(on_demand=True)
class DeferredWorkspace(ToolBucket):
    """Expose workspace operations only after loading."""

    name = "workspace"
    capture = {"tool_kind": "workspace", "on_demand": False}
    annotations = {"return": str}

    def __call__(self) -> str:
        return "ready"
```

## Buckets And AgentTool

On-demand agents work with [Agent Tool](agent-tool.md) as well. When an
on-demand agent is loaded by exact name, the existing `AgentTool` bucket
captures it as an available `agent(name, message)` target.
