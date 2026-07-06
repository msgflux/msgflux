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

```python
schema_names = [
    schema["function"]["name"]
    for schema in agent.tool_library.get_tool_json_schemas()
]
print(schema_names)
# ["tool_search"]
```

## Search Before Loading

A normal search returns matching tool names without loading them:

```python
result = agent.tool_library(
    [("call_1", "tool_search", {"query": "finance report"})]
).tool_calls[0].result

print(result["matches"])
# ["query_finance_report"]
print(result["loaded"])
# []
```

Set `description=True` when the model needs details before deciding whether to
activate a tool:

```python
result = agent.tool_library(
    [
        (
            "call_2",
            "tool_search",
            {"query": "finance report", "description": True},
        )
    ]
).tool_calls[0].result

print(result["descriptions"])
```

## Select Tools

Use `select:name` to activate exact on-demand tools:

```python
result = agent.tool_library(
    [("call_3", "tool_search", {"query": "select:query_finance_report"})]
).tool_calls[0].result

print(result["loaded"])
# ["query_finance_report"]
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
on-demand agent is selected, `ToolLibrary.add(...)` promotes it and the existing
`AgentTool` bucket captures it as an available `agent(name, message)` target.

