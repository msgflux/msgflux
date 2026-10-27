# Tool Search

When the first deferred tool is registered, `ToolLibrary` installs
`ToolSearchExtension`. The extension owns the `tool_search` bucket while loaded
tool names remain isolated per thread in `ChatMessages`.

Tool search lets a catalog grow without placing every tool schema in every
request. The initial context stays approximately stable as rarely used tools
are added; mark those tools with `defer_loading=True`:

```python
import msgflux as mf
import msgflux.nn as nn


@mf.tool_config(defer_loading=True)
def query_finance_report(company: str) -> str:
    """Query archived finance reports for a company."""
    return f"Finance report for {company}"


agent = nn.Agent(
    name="analyst",
    model=mf.Model.chat_completion("openai/gpt-5.6-luna"),
    tools=[query_finance_report],
)
```

The name follows the OpenAI and Anthropic convention: the tool remains in the
logical `ToolCatalog`, but its full callable definition is deferred until the
model searches for it.

## Hosted And Portable Modes

OpenAI GPT-5.6 models using Responses compile deferred tools to the hosted
protocol:

```python
[
    {"type": "tool_search"},
    {
        "type": "function",
        "name": "query_finance_report",
        "defer_loading": True,
        # description and parameters omitted here
    },
]
```

OpenAI performs the search inside the same Responses request. Its
`tool_search_call` and `tool_search_output` items are retained in
`ChatMessages`, followed by the normal function call that msgFlux executes.

Anthropic Messages uses its native server-side search tool instead:

```python
[
    {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    },
    {
        "name": "query_finance_report",
        "description": "Query archived finance reports for a company.",
        "input_schema": {"type": "object", "properties": {}},
        "defer_loading": True,
    },
]
```

Anthropic documents both a regex and a BM25 search variant. msgFlux uses the
BM25 variant because the generic `tool_search` contract is a natural-language
query rather than a regular expression.

The same `@mf.tool_config(defer_loading=True)` definition drives both hosted
protocols; no Anthropic-specific tool option is required. Anthropic receives
every candidate function definition on every request, including deferred ones,
because the server needs them to expand discovered tools. Deferred declarations
keep `defer_loading: true` after local loading; an explicitly selected tool is
exposed directly, and the search tool itself is never deferred. Function
`input_schema` objects are passed through unchanged and are
not regenerated when the deferral flag is added. Keeping the declarations stable
preserves prompt-cache prefixes and the validity of signed thinking on later turns.

Anthropic mode also keeps automatic tool guidance stable. The optional changing
tool-budget warning is omitted from its system prompt; `max_tool_turns` is still
enforced. Application changes to the system prompt, tools, or earlier messages can
invalidate signed thinking. Keep those prefixes append-only; see
[Anthropic's thinking guide](https://platform.claude.com/docs/en/build-with-claude/thinking).

Anthropic returns `server_tool_use` and `tool_search_tool_result` blocks with
nested `tool_reference` entries. msgFlux retains those blocks in
`ChatMessages` as `tool_search_call` and `tool_search_output` items, and stores
the ordered assistant blocks as opaque Anthropic provider state so a later
request replays the assistant content unchanged. The same blocks are preserved
when the response is streamed. The server search is never executed as a client
function, and no `tool_result` is sent for its `srvtoolu_...` identifier.

Anthropic model support is caller responsibility. Use `native_tools=False` when
the selected model does not document support for the tool-search variants:

```python
model = mf.Model.chat_completion(
    "anthropic/claude-haiku-5-5",
    native_tools=False,
)
```

msgFlux does not substitute a different or more expensive model when hosted
search is unavailable. With `native_tools=False`, the generic `native_tools`
switch keeps the portable path described below.

The OpenAI implementation checks both the active model and API mode through
`supports_native_tool_search()`. Unknown models, Chat Completions, and providers
without hosted search use the portable fallback: msgFlux exposes its local
`tool_search` function first, then exposes a selected tool on the next model
turn. This works with any msgFlux provider that supports ordinary function
calling, including local tools, agents, and MCP tools.

!!! warning "Portable loading changes the tool prefix"

    Loading a portable deferred tool changes the tool payload on the following
    request. The feature remains correct, but that boundary can reduce provider
    prompt-cache hits. Hosted Responses search keeps discovery within the native
    protocol and avoids this intermediate schema transition.

## Thread-Local Loading

Loaded tools belong to a conversation, not to the shared `ToolLibrary`.
`ChatMessages.metadata` stores the loaded names under the library's catalog ID:

```python
from msgflux.chat_messages import ChatMessages

first = ChatMessages(thread_id="analysis-a")
second = ChatMessages(thread_id="analysis-b")

agent.tool_library(
    [("call_1", "tool_search", {"select": ["query_finance_report"]})],
    messages=first,
)

print(first.get_loaded_tools(agent.tool_library.name))
# {"query_finance_report"}
print(second.get_loaded_tools(agent.tool_library.name))
# set()
```

Use `get_tool_catalog_view(...)` to inspect the immutable definition snapshot
for one thread without compiling a provider payload:

```python
first_view = agent.tool_library.get_tool_catalog_view(first)
second_view = agent.tool_library.get_tool_catalog_view(second)

print([entry.name for entry in first_view.visible_entries()])
# ["query_finance_report"]
print([entry.name for entry in second_view.visible_entries()])
# ["tool_search"]
```

Both views retain `query_finance_report` in `entries`; only `loaded` and the
derived visible projection differ. A view requires a `ChatMessages` instance
with a configured `thread_id`, making accidental process-global activation
explicitly invalid.

If the caller has a runtime scope but no `ChatMessages`, pass its identity
explicitly:

```python
view = agent.tool_library.get_tool_catalog_view(thread_id="analysis-a")
```

An explicit ID creates an isolated snapshot with no loaded deferred names; it
does not create or mutate conversation history.

Each entry is an execution-free description containing its stable reference,
input schema, annotations, native bindings, loading state, and display
metadata. The search tool is marked by a semantic catalog role instead of a
reserved name. Once no unresolved deferred tools remain, the portable search
entry disappears from `visible_entries()` without mutating the underlying
catalog.

The executable catalog is not mutated. This means simultaneous threads can
share one agent safely while exposing different tool subsets. The state is also
copied and checkpointed with `ChatMessages`.

## Search And Load Locally

A search returns matching names without loading them:

```python
result = agent.tool_library(
    [("call_2", "tool_search", {"query": "finance report"})],
    messages=first,
).tool_calls[0].result

print(result["matches"])
# ["query_finance_report"]
print(result["loaded"])
# []
```

Use `select` to load exact matches for that thread:

```python
result = agent.tool_library(
    [
        (
            "call_3",
            "tool_search",
            {"select": ["query_finance_report"]},
        )
    ],
    messages=first,
).tool_calls[0].result

print(result["loaded"])
# ["query_finance_report"]
```

`query="select:query_finance_report"` remains supported as a compact selection
syntax. Set `description=True` to include display names, descriptions, usage
guidance, and tool kinds in search results.

After loading, `ToolCatalog.portable_tools()` includes the selected tool for
`first`; it continues to include only the local `tool_search` fallback for
`second`. Local functions, modules, agents, and MCP tools use the same catalog
and thread-state contract.
