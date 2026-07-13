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
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=[query_finance_report],
)
```

When at least one on-demand tool exists, msgFlux exposes a runtime tool named
`tool_search`. The on-demand tools are searchable but are not included in the
normal callable tool schemas until selected.

Internally, `tool_search` is a `ToolBucket` with
`capture={"on_demand": True}`. This keeps the search index and the deferred
tool metadata together; selecting a tool promotes it back through normal
library registration.

```python
schema_names = [
    schema["function"]["name"]
    for schema in agent.tool_library.get_tool_json_schemas()
]
print(schema_names)
# ["tool_search"]
```

## Search Before Loading

A normal search returns compact `name: guidance` or `name: description` lines
without loading them:

```python
result = agent.tool_library(
    [("call_1", "tool_search", {"query": "finance report"})]
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
            "tool_search",
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
            "tool_search",
            {"query": "query_finance_report"},
        )
    ]
).tool_calls[0].result

print(result)
# loaded=query_finance_report
```

After selection, the tool is promoted into the normal library:

```python
schema_names = [
    schema["function"]["name"]
    for schema in agent.tool_library.get_tool_json_schemas()
]
print(schema_names)
# ["query_finance_report"]
```

If other on-demand tools remain, `tool_search` stays available. If no on-demand
tools remain, msgFlux removes `tool_search` from the exposed runtime tools.

## Buckets And AgentTool

On-demand agents work with [Agent Tool](agent-tool.md) as well. When an
on-demand agent is loaded by exact name, `ToolLibrary.add(...)` promotes it and the existing
`AgentTool` bucket captures it as an available `agent(name, message)` target.
