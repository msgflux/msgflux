# Tools

Tools are interfaces that allow models to perform actions or query information.

### What are Tools?
1.  **Function Calling** - A tool is exposed as a function with defined name, parameters, and types
    - Example: `web_search(query: str)`
    - The model decides whether to call it and provides arguments

2.  **Extending Capabilities** - Tools allow you to:
    - Search for real-time data (news, stocks, databases)
    - Execute Code
    - Manipulate systems (send emails, schedule events)
    - Integrate with external APIs

3.  **Agent-based Orchestration** - The LLM acts as an agent that decides:
    - When to use a tool
    - Which tool to use
    - How to interpret the tool's output

In msgFlux, a **Tool can be any callable** (function, class with `__call__`/`acall` e.g. nn.Agent).

!!! info

    While more tools enable more actions, too many tools can confuse the model about which one to use.

!!! tip

    A good practice is to inform the model in the system prompt when it should use that tool.

### How Tool Calls Work

When the model decides to use a tool, the Agent intercepts the response, executes the function, appends the result to the conversation, and calls the model again. This loop continues until the model produces a final text response.

```
                        Input
                          │
                          ▼
┌──────────────────────────────────────────────┐
│            messages + tool schemas           │
└─────────────────────────┬────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │     Model     │ ──▶ "call get_weather(city)"
                  └───────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  get_weather  │ ──▶ "Sunny, 24°C"
                  └───────────────┘
                          │  result appended to messages
                          ▼
                  ┌───────────────┐
                  │     Model     │ ──▶ "The weather in Paris is sunny..."
                  └───────┬───────┘
                          │
                     More calls?
                    /            \
                  Yes             No
                   │               │
             (next cycle)     [ Output ]
```

---

???+ example

    === "GitHub API"

        Query GitHub's public API for repository information:

        ```python
        # pip install msgflux[openai]
        import httpx
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        def get_github_repo(owner: str, repo: str) -> str:
            """Get information about a GitHub repository.

            Args:
                owner: Repository owner (username or organization).
                repo: Repository name.

            Returns:
                Repository details including stars, forks, and description.
            """
            url = f"https://api.github.com/repos/{owner}/{repo}"
            response = httpx.get(url, timeout=10)

            if response.status_code == 404:
                return f"Repository {owner}/{repo} not found."

            if response.status_code != 200:
                return f"Error fetching repository: {response.status_code}"

            data = response.json()
            return f"""
            Repository: {data['full_name']}
            Description: {data.get('description', 'No description')}
            Stars: {data['stargazers_count']:,}
            Forks: {data['forks_count']:,}
            Language: {data.get('language', 'Unknown')}
            Open Issues: {data['open_issues_count']}
            URL: {data['html_url']}
            """

        class GithubAssistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You help users explore GitHub repositories."
            tools = [get_github_repo]
            config = {"verbose": True}

        response = agent("Tell me about the pytorch repository")
        ```

    === "File Operations"

        Real file system operation:

        ```python
        # pip install msgflux[openai]
        import os
        from pathlib import Path
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        def list_files(directory: str, pattern: str = "*") -> str:
            """List files in a directory matching a pattern.

            Args:
                directory: Path to the directory.
                pattern: Glob pattern to filter files (default: all files).

            Returns:
                List of matching files with sizes.
            """
            path = Path(directory).expanduser()

            if not path.exists():
                return f"Directory not found: {directory}"

            if not path.is_dir():
                return f"Not a directory: {directory}"

            files = list(path.glob(pattern))[:20]  # Limit results

            if not files:
                return f"No files matching '{pattern}' in {directory}"

            result = []
            for f in files:
                size = f.stat().st_size if f.is_file() else 0
                size_str = f"{size:,} bytes" if f.is_file() else "directory"
                result.append(f"  {f.name} ({size_str})")

            return f"Files in {directory}:\n" + "\n".join(result)

        def read_file(filepath: str, max_lines: int = 50) -> str:
            """Read content from a text file.

            Args:
                filepath: Path to the file.
                max_lines: Maximum lines to read (default: 50).

            Returns:
                File content or error message.
            """
            path = Path(filepath).expanduser()

            if not path.exists():
                return f"File not found: {filepath}"

            if not path.is_file():
                return f"Not a file: {filepath}"

            try:
                lines = path.read_text().splitlines()[:max_lines]
                content = "\n".join(lines)
                if len(lines) == max_lines:
                    content += "\n...[truncated]"
                return content
            except Exception as e:
                return f"Error reading file: {e}"

        class FileAssistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You help users explore files on their system."
            tools = [list_files, read_file]
            config = {"verbose": True}

        response = agent("List Python files in the current directory")
        ```

---

### Builtin Tools

msgFlux provides built-in tools that work out of the box:

#### WebFetch

`WebFetch` fetches web pages and converts them to Markdown. It uses a parser endpoint (default: `https://markdown.new/`) or falls back to semantic HTML parsing.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import WebFetch

class WebReader(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You help users understand web content."
    tools = [WebFetch]
    config = {"verbose": True}

agent = WebReader()
result = agent("Summarize the main points from https://news.ycombinator.com")
```

#### WebSearch

`WebSearch` performs web searches backed by either a retriever or a model:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import WebSearch

# Retriever-backed search (using Exa, Brave, Tavily, etc.)
retriever_search = WebSearch("retriever/exa")

# Model-backed search
model_search = WebSearch("model/openai/gpt-4o-search-preview")

# Or use environment variables:
# export MSGFLUX_TOOL_WEB_SEARCH_ENGINE="retriever/wikipedia"
env_search = WebSearch()

class Researcher(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You help users find up-to-date information."
    tools = [retriever_search, model_search, env_search]
    config = {"verbose": True}

agent = Researcher()
result = agent("What is the latest Python version?")
```

Supported retriever engines: `wikipedia`, `searxng`, `serpapi`, `brave`, `tavily`, `linkup`, `exa`, `arxiv`.

Supported model engines: any OpenAI-compatible model.

#### Parameters

- **`init_params`**: Passed when initializing the retriever or model backend.

- **`call_params`**: Passed on each retriever call (retriever engines only).

- **`goal`**: Optional call-time instruction for model-backed search only.
  It steers the model backend before it answers.

```python
# init_params: configure the backend at initialization
search = WebSearch(
    "retriever/exa",
    init_params={"include_text": True, "max_characters": 2000},
)

# call_params: passed on each call (retriever engines only)
result = search("Python news", call_params={"top_k": 5})

# goal: passed at call time for model-backed search only
model_search = WebSearch("model/openai/gpt-4o-search-preview")
result = model_search("Python news", goal="Answer with concise bullet points.")
```

Alternatively, read from environment variables:
```bash
export MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS='{"include_text": true}'
export MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS='{"top_k": 5}'
```

#### Weather

`Weather` gets current, forecast, or historical weather data for a location:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import Weather

weather = Weather()

class WeatherAssistant(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You help users understand weather conditions."
    tools = [weather]
    config = {"verbose": True}

agent = WeatherAssistant()
result = agent("Is it raining in Fortaleza right now?")
```

The public tool call accepts:

- **`location`**: A simple city/place name such as `"Fortaleza"`. Coordinates like `"-3.71722,-38.54306"` are also accepted when the user provides them.

- **`when`**: `"now"`, a relative time like `"+6h"` or `"-3d"`, or an ISO datetime.

```python
weather = Weather(
    engine="open_meteo",
    max_future_days=7,
    max_past_days=90,
    forecast_hours_when_now=6,
)

current = weather("Fortaleza")
forecast = weather("Fortaleza", when="+6h")
historical = weather("-3.71722,-38.54306", when="-3d")
```

If `engine` is not passed, `Weather` reads `MSGFLUX_TOOL_WEATHER_ENGINE`. When
neither is set, it defaults to `open_meteo`.

The tool returns a structured `dotdict` with:

- `location`: resolved name, coordinates, and resolution source
- `when`: requested target time, kind (`now`, `future`, or `past`), and whether it was clamped
- `weather`: temperature, apparent temperature, humidity, condition, rain, cloud cover, and wind data
- `forecast`: next hourly items when available
- `source`: provider and endpoint metadata
- `units`: units reported by the weather provider

---

???+ example

    === "Web Fetch"

        Extract text content from web pages using `httpx`:

        ```python
        # pip install msgflux[openai] beautifulsoup4
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import WebFetch

        # mf.set_envs(OPENAI_API_KEY="...")

        class WebReader(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You help users understand web content."
            tools = [WebFetch]
            config = {"verbose": True}

        agent = WebReader()

        response = agent("Summarize the main points from https://news.ycombinator.com")
        ```

    === "Web Search"

        Use a built-in web search tool backed by either a retriever or a model:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import WebSearch

        # Option 1: retriever-backed web search
        wikipedia_search = WebSearch(
            "retriever/wikipedia",
            call_params={"top_k": 2},
        )

        # Option 2: model-backed web search
        openai_search = WebSearch(
            "model/openai/gpt-4o-search-preview",
            init_params={
                "web_search_options": {"search_context_size": "low"},
            },
        )

        # Or read the engine and params from the environment:
        # export MSGFLUX_TOOL_WEB_SEARCH_ENGINE="retriever/wikipedia"
        # export MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS='{"language": "pt"}'
        # export MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS='{"top_k": 2}'
        env_search = WebSearch()

        class Researcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You help users find up-to-date information."
            tools = [wikipedia_search, openai_search, env_search]
            config = {"verbose": True}

        agent = Researcher()

        result = agent("What is the latest Python version?")
        ```

        The tool returns a `dict` with:

        - `data`: the search result payload
        - `annotations`: citation metadata when available

        `init_params` is unpacked into the backend constructor
        (`Retriever.web_search(...)` or `Model.chat_completion(...)`).
        `call_params` is supported only for retriever engines and is unpacked
        whenever the retriever is called. If these values are not passed
        explicitly, `WebSearch` reads the JSON objects from
        `MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS` and
        `MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS`.
        Model-backed search also accepts `goal` at call time to steer
        the model before it answers.

    === "Weather"

        Get current, forecast, or historical weather data:

        ```python
        # pip install msgflux[openai] httpx
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import Weather

        # mf.set_envs(OPENAI_API_KEY="...")

        weather = Weather(
            engine="open_meteo",
            max_future_days=7,
            max_past_days=90,
            forecast_hours_when_now=6,
        )

        class WeatherAssistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You help users understand weather conditions."
            tools = [weather]
            config = {"verbose": True}

        agent = WeatherAssistant()

        response = agent("Will it rain in Fortaleza in the next few hours?")
        ```

        You can also call the tool directly:

        ```python
        current = weather("Fortaleza")
        forecast = weather("Fortaleza", when="+6h")
        historical = weather("-3.71722,-38.54306", when="-3d")
        ```

        The tool returns structured weather data instead of prose, so the agent
        can decide how to summarize current conditions, forecasts, and
        historical observations.

    === "Wikipedia Search"

        Use msgflux's built-in Wikipedia retriever as a tool:

        ```python
        # pip install msgflux[openai] wikipedia
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # Create Wikipedia search tool from built-in retriever
        wikipedia = mf.Retriever.web_search("wikipedia")

        class Researcher(nn.Module):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a research assistant with access to Wikipedia.",
            tools = [wikipedia]
            config = {"verbose": True}

        response = agent("Tell me about the history of the Python programming language")
        ```

### Registry

Use `mf.Registry` to collect tools with a decorator instead of building a list by hand.
This is especially useful when tools are spread across multiple files or when you want
to keep them decoupled from the agent definition.

```python
import msgflux as mf

tools = mf.Registry()

@tools
def add(a: float, b: float) -> str:
    """Sum two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The result of the addition.
    """
    return f"{a} + {b} = {a + b}"

@tools(name="multiply")
def mul(a: float, b: float) -> str:
    """Multiply two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The result of the multiplication.
    """
    return f"{a} * {b} = {a * b}"

print(tools.to_list())   # [<function add ...>, <function mul ...>]
print(tools.to_items())  # {"add": <function add ...>, "multiply": <function mul ...>}
```

The decorator captures the callable's name automatically (`.__name__` for functions,
`.name` attribute if available). Pass `name=` to override:

| Usage | Captured name |
|-------|---------------|
| `@tools` | `add` (from `__name__`) |
| `@tools("custom")` | `custom` |
| `@tools(name="custom")` | `custom` |

Pass the registry directly to an Agent:

```python
import msgflux.nn as nn

class Calculator(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = tools.to_list()
```

You can have multiple independent registries — each is an isolated instance:

```python
read_tools = mf.Registry()
write_tools = mf.Registry()

@read_tools
def get_data(key: str) -> str:
    """Retrieve data by key."""
    return f"value_for_{key}"

@write_tools
def save_data(key: str, value: str) -> str:
    """Persist a key-value pair."""
    return f"saved {key}={value}"

print(len(read_tools))   # 1
print(len(write_tools))  # 1
```

### Writing Good Tools

#### Tool Names

A well-defined tool is fundamental for the model to understand **when** and **how** to use it. The model reads the tool's name, description (docstring), and parameter definitions to decide if it should call the tool and with what arguments.

Poor tool definitions lead to:

- The model not calling the tool when it should
- Incorrect parameter values being passed
- Confusion when multiple tools have similar names

???+ example "Tool Names and Description"

    === "Good Tool Name"

        A simple, descriptive name helps the model quickly understand the tool's purpose. Combined with a clear docstring and well-documented parameters, the model can make accurate decisions about when to use the tool.

        **Best practices:**

        - Use short, action-oriented names (`search`, `send_email`)
        - Document the purpose in the docstring
        - Describe each parameter with type hints and descriptions

        ```python
        def web_search(query: str) -> str:
            """Search for content similar to query.

            Args:
                query: Term to search on the web.

            Returns:
                Results similar to query.
            """
            pass
        ```

    === "Bad Tool Name"

        Long, complex names with unnecessary prefixes confuse the model. Missing or poor descriptions make it impossible for the model to understand when to use the tool.

        **Common problems:**

        - Overly long names with implementation details (`superfast_brave_web_search`)
        - Redundant parameter names (`query_to_search` instead of `query`)
        - Missing docstrings or parameter descriptions
        - No type hints

        ```python
        def superfast_brave_web_search(query_to_search: str) -> str:
            pass  # No docstring, no parameter description
        ```


#### Tool Returns

The way a tool returns information affects how well the model interprets and uses the result.

???+ example "Return Value Best Practices"

    === "Basic Return"

        Returns the value, but model must infer context:

        ```python
        def add(a: float, b: float) -> float:
            """Sum two numbers."""
            return a + b  # Returns: 8
        ```

    === "Descriptive Return"

        Provides context that helps the model respond naturally:

        ```python
        def add(a: float, b: float) -> str:
            """Sum two numbers."""
            c = a + b
            return f"The sum of {a} plus {b} is {c}"
        ```

    === "Instructive Return"

        Guides the model on how to use the result:

        ```python
        def add(a: float, b: float) -> str:
            """Sum two numbers."""
            c = a + b
            return f"The calculation is complete. Tell the user: {a} + {b} = {c}"
        ```


### Tool Choice

Control how the model selects tools.

**Options:**

| Value | Behavior |
|-------|----------|
| `"auto"` | Model decides whether to use tools (default) |
| `"required"` | Model must call at least one tool |
| `"none"` | Model cannot use tools |
| `"tool_name"` | Model must call the specific tool |

???+ example

    === "auto (default)"

        Model decides when to use tools:

        ```python
        # pip install msgflux[openai] wikipedia
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        wikipedia = mf.Retriever.web_search("wikipedia")

        class Researcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [wikipedia]
            config = {"tool_choice": "auto", "verbose": True}

        agent = Researcher()

        # Model may or may not use the tool
        response = agent("What is the capital of France?")  # Probably won't use tool
        response = agent("Tell me about quantum entanglement")  # Will likely use tool
        ```

    === "required"

        Force the model to always use a tool:

        ```python
        # pip install msgflux[openai] wikipedia
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        wikipedia = mf.Retriever.web_search("wikipedia")

        class Researcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [wikipedia]
            config = {"tool_choice": "required", "verbose": True}

        agent = Researcher()

        # Model MUST call a tool before responding
        response = agent("What is photosynthesis?")
        ```

    === "Specific Tool"

        Force a specific tool to be called:

        ```python
        # pip install msgflux[openai] wikipedia
        import httpx
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        wikipedia = mf.Retriever.web_search("wikipedia")

        def search_github(query: str) -> str:
            """Search GitHub repositories."""
            resp = httpx.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "per_page": 5}
            )
            repos = resp.json().get("items", [])
            return "\n".join(f"- {r['full_name']}: {r['description']}" for r in repos)

        class SearchAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [search_github, wikipedia]
            # Always use GitHub
            config = {"tool_choice": "search_github", "verbose": True}

        agent = SearchAgent()
        response = agent("Find machine learning projects")
        ```

    === "none"

        Disable tool usage temporarily:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [wikipedia, search_github]
            # Tools disabled
            config = {"tool_choice": "none", "verbose": True}

        # Model will respond without using any tools
        response = agent("What do you know about Python?")
        ```

    === "Router Pattern"

        Use `required` for routing agents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        @mf.tool_config(return_direct=True)
        class PythonExpert(nn.Agent):
            """Expert in Python programming."""

            model = model
            system_message = "You are a Python expert."

        @mf.tool_config(return_direct=True)
        class RustExpert(nn.Agent):
            """Expert in Rust programming."""

            model = model
            system_message = "You are a Rust expert."

        class Router(nn.Agent):
            """Routes programming questions to the right expert."""
            model = model
            system_message = "Route questions to the appropriate expert."
            tools = [PythonExpert, RustExpert]
            config = {"tool_choice": "required", "verbose": True}

        router = Router()

        # Router MUST pick an expert
        response = router("How do I handle errors in Rust?")
        ```

!!! note "Interaction with `tool_filter`"

    `tool_choice` is resolved after runtime filtering.

    - If `tool_filter` removes a specific tool configured in `tool_choice`, the Agent falls back to `"auto"` for that request
    - If `tool_filter` removes all tools, tool usage is disabled for that request

### Structured Tool Parameters

Use `msgspec.Struct` when a tool parameter needs a complex object shape. This
gives the model a strict object schema and lets the tool receive typed Python
objects instead of loose dictionaries.

```python
from typing import Literal

import msgspec


class TodoItem(msgspec.Struct):
    content: str
    active_form: str
    status: Literal["pending", "in_progress", "completed"]


def write_todos(todos: list[TodoItem]) -> str:
    """Persist the current task checklist.

    Args:
        todos: Complete todo list for the current session.
    """
    first = todos[0]
    assert isinstance(first, TodoItem)
    return f"Saved {len(todos)} todos."
```

When the model calls `write_todos`, msgFlux restores each item in `todos` to a
`TodoItem` instance before executing the tool. This is preferable to
`list[dict[str, str]]` for complex inputs because the generated tool schema
spells out the allowed fields and status values.

This works with OpenAI-compatible strict tool schemas:

```python
agent = nn.Agent(
    name="developer_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=[write_todos],
)
```

Use this pattern for stable contracts. Avoid adding fields that only save a few
runtime conveniences if they cost prompt/tool-call tokens, such as an `id` field
that the runtime can derive internally.

### Runtime Tool Filtering

Use `tool_filter` when you need to change which tools are exposed on a **per-request** basis, without rebuilding the agent.

**Rules:**

- `tool_filter` must contain exactly one key: `"allow"` or `"block"`
- The value can be a single tool name or a list of tool names
- `{"block": "*"}` disables all tools for that request
- Runtime `tool_filter=...` overrides the value loaded from `message_fields`

???+ example

    === "Allow Only Specific Tools"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        def web_search(query: str) -> str:
            """Search the web."""
            return f"Search results for: {query}"

        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            return f"Computed: {expression}"

        def browser(url: str) -> str:
            """Open a web page."""
            return f"Fetched: {url}"

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [web_search, calculator, browser]
            config = {"verbose": True}

        agent = Assistant()

        response = agent(
            "Search for the latest Python release and calculate 2 + 2",
            tool_filter={"allow": ["web_search", "calculator"]},
        )
        ```

    === "Block Specific Tools"

        ```python
        response = agent(
            "Answer this without opening pages",
            tool_filter={"block": "browser"},
        )
        ```

    === "Disable All Tools Temporarily"

        ```python
        response = agent(
            "Answer only from model knowledge",
            tool_filter={"block": "*"},
        )
        ```

### Tool Filter from Message

You can also load `tool_filter` from the input `Message` using `message_fields`.

This is useful when another module or router decides which tools should be available before the Agent runs.

???+ example

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    def web_search(query: str) -> str:
        """Search the web."""
        return f"Search results for: {query}"

    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return f"Computed: {expression}"

    class ControlledAgent(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        tools = [web_search, calculator]
        message_fields = {
            "task": "content",
            "tool_filter": "control.tool_filter",
        }

    agent = ControlledAgent()

    msg = mf.Message(content="What is 25 * 17?")
    msg.set("control.tool_filter", {"allow": "calculator"})

    response = agent(msg)
    ```

!!! tip

    Runtime kwargs still take precedence:

    ```python
    response = agent(
        msg,
        tool_filter={"block": "*"},  # overrides msg.control.tool_filter
    )
    ```

### Limiting Tool Loops

Use `config["max_tool_turns"]` to cap how many **completed tool rounds** an Agent can execute in a single request.

When the limit is reached:

- The next attempted tool round is not executed
- The Agent makes one more model call with tools disabled
- The model gets a final chance to produce a plain answer

This is useful to avoid runaway tool loops while still allowing a graceful final response.

???+ example

    ```python
    # pip install msgflux[openai] wikipedia
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    wikipedia = mf.Retriever.web_search("wikipedia")

    class Researcher(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        system_message = "Use tools when needed, but finish with a concise answer."
        tools = [wikipedia]
        config = {
            "tool_choice": "auto",
            "max_tool_turns": 2,
            "verbose": True,
        }

    agent = Researcher()

    response = agent("Compare Python and Ruby and keep the answer short")
    ```

!!! note

    `max_tool_turns` is different from `tool_choice="none"`:

    - `tool_choice="none"` disables tools from the start
    - `max_tool_turns` allows some tool usage first, then forces a final no-tools round if the loop keeps going

### Async Tools

When your agent runs asynchronously with `acall()`, prefer writing async tools as well. This ensures non-blocking execution and better performance when tools perform I/O operations.

???+ note "Sync vs Async Tools"

    === "Async Tool (Recommended)"

        ```python
        import httpx

        async def fetch_data(url: str) -> str:
            """Fetch data from a URL asynchronously."""
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                return response.text
        ```

    === "Sync Tool"

        ```python
        import httpx

        def fetch_data(url: str) -> str:
            """Fetch data from a URL."""
            response = httpx.get(url, follow_redirects=True)
            return response.text
        ```

You can also implement a class-based async tool using the `acall` method:

???+ example

    ```python
    import httpx

    class WebFetcher:
        """Fetch content from web pages."""

        def __init__(self, timeout: int = 30):
            self.timeout = timeout

        async def acall(self, url: str) -> str:
            """Fetch content from URL asynchronously."""
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                return response.text
    ```

### Class-based Tools

Tools can be implemented as classes with `__call__` or `acall` methods. This is useful when you need to maintain state or configure the tool at initialization.

???+ example "Class-based Tool"

    === "Basic Class Tool"

        ```python
        from typing import Optional
        import httpx

        class GitHubSearch:
            """Search GitHub repositories."""

            def __init__(self, max_results: Optional[int] = 5):
                self.max_results = max_results

            def __call__(self, query: str) -> str:
                """Search for repositories matching query.

                Args:
                    query: Search term for repositories.
                """
                url = "https://api.github.com/search/repositories"
                params = {"q": query, "per_page": self.max_results}
                response = httpx.get(url, params=params, timeout=10)

                if response.status_code != 200:
                    return f"Error: {response.status_code}"

                data = response.json()
                results = []
                for repo in data.get("items", []):
                    results.append(f"- {repo['full_name']} ({repo['stargazers_count']}⭐)")

                return "\n".join(results) if results else "No repositories found."
        ```

    === "Override Tool Name"

        Use the `name` attribute to override the class name:

        ```python
        import httpx

        class GitHubRepoSearchV2:
            name = "search_repos"  # Exposed as "search_repos" instead of class name

            def __init__(self, max_results: int = 5):
                self.max_results = max_results

            def __call__(self, query: str) -> str:
                """Search GitHub for repositories."""
                url = "https://api.github.com/search/repositories"
                resp = httpx.get(url, params={"q": query, "per_page": self.max_results})
                repos = resp.json().get("items", [])
                return "\n".join(f"- {r['full_name']}" for r in repos) or "No results."
        ```

### Return Types

Tools can return any data type. Non-string returns are automatically serialized using `msgspec.json.encode` before being passed to the model.

???+ note "Tool Return Examples"

    === "String Return"

        ```python
        def add(a: float, b: float) -> str:
            """Sum two numbers."""
            return f"The sum of {a} plus {b} is {a + b}"
        ```

    === "Dict Return"

        ```python
        from typing import Dict

        def web_search(query: str) -> Dict[str, str]:
            """Search for content."""
            return {
                "title": "Result title",
                "snippet": "Result snippet",
                "url": "https://example.com"
            }
        ```

    === "List Return"

        ```python
        from typing import List

        def get_top_results(query: str) -> List[Dict]:
            """Get top search results."""
            return [
                {"title": "Result 1", "url": "..."},
                {"title": "Result 2", "url": "..."}
            ]
        ```

### Tool Config

The `@mf.tool_config` decorator adds special behaviors to tools.

#### return_direct

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

#### inject_vars

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

#### disable_input

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

#### inject_message

With `inject_message=True`, the tool receives the original `message` passed to the
agent. This is useful when the tool needs access to the full envelope object,
including `Message` fields that should not be part of the public tool schema.

Use cases:

- Agent-as-a-tool with declarative `Message` envelopes
- Tools that need `response_mode` side effects on the original message
- Access to `vars`, metadata, or other fields outside the public task schema

???+ example

    ```python
    @mf.tool_config(inject_message=True)
    def inspect_original_message(message, **kwargs) -> str:
        """Inspect the original message envelope."""
        if isinstance(message, mf.Message):
            return str(message.get("meta.trace_id"))
        return "No structured message available."
    ```

#### inject_messages

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

#### handoff

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

#### call_as_response

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

#### spawn

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

#### name_override

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

#### display_name

Assign a human-readable name for UI surfaces and events while keeping the tool's
programmatic name stable:

```python
@mf.tool_config(display_name="Repository Search")
def search_repos(query: str) -> str:
    """Search GitHub repositories."""
    ...
```

The model still calls `search_repos`, but clients such as a CLI can show
`Repository Search` to users.

#### usage_guidance

Add guidance that is rendered into the agent system prompt under
`<tool_usage_guidance>`. Use it for tool-specific "when/how to use" instructions
that should not live in the function description.

```python
@mf.tool_config(
    display_name="Repository Search",
    usage_guidance=(
        "Use when the user asks for GitHub repositories. Prefer concise search "
        "queries with language, framework, or topic constraints."
    ),
)
def search_repos(query: str) -> str:
    """Search GitHub repositories."""
    ...
```

#### Builtin usage guidance

msgFlux also ships an opt-in guidance registry for builtin tools. This keeps
tool implementations neutral while letting applications such as a CLI attach
ready-to-use instructions when composing an agent.

Use `apply_tool_guidance()` to fill `usage_guidance` only when a tool does not
already define one:

```python
import msgflux as mf
from msgflux.tools import apply_tool_guidance
from msgflux.tools.builtin import WebFetch, WebSearch

tools = apply_tool_guidance([WebSearch(), WebFetch()])

agent = mf.nn.Agent(
    name="assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    tools=tools,
)
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

#### retry

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

### Agent-as-Tool

Agents can be used as tools for other agents, enabling hierarchical task delegation, also known as **SubAgents**. Using AutoParams makes this pattern especially clean: the class name becomes the tool name, and the docstring becomes the tool description.

The Coordinator calls the Specialist as any other tool. The result returns to the Coordinator's model, which synthesizes it into the final response.

```
              Input
                │
                ▼
  ┌──────────────────────────────┐
  │         Coordinator          │
  │                              │
  │   ┌──────────┐               │
  │   │  Model   │──▶ "call      │
  │   └──────────┘    Specialist │
  └─────────────┬────────────────┘
                │  call(task)
                ▼
  ┌──────────────────────────────┐
  │         Specialist           │
  │        (SubAgent)            │
  │                              │
  │  processes task independently│
  │  may call its own tools      │
  └─────────────┬────────────────┘
                │  result
                ▼
  ┌──────────────────────────────┐
  │         Coordinator          │
  │                              │
  │   ┌──────────┐               │
  │   │  Model   │──▶ synthesized│
  │   └──────────┘    response   │
  └─────────────┬────────────────┘
                │
                ▼
                Output
```

???+ note "Agent-as-Tool Examples"

    === "Health Team"

        A coordinator agent delegates to specialist agents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Nutritionist(nn.Agent):
            """Specialist in nutrition, diet planning, and healthy eating habits.
            Consult for meal plans, dietary recommendations, and nutritional advice."""

            model = model
            system_message = "You are a certified nutritionist."
            instructions = """Create clear and practical meal plans tailored to the user's goals.
            Be objective, technical, and structured."""

        class FitnessTrainer(nn.Agent):
            """Specialist in fitness, exercise routines, and physical training.
            Consult for workout plans, training schedules, and exercise guidance."""

            model = model
            system_message = "You are a certified personal trainer."
            instructions = """Design workout routines based on the user's fitness level and goals.
            Focus on safety, progression, and sustainability."""

        class HealthCoordinator(nn.Agent):
            """Coordinates health specialists to provide comprehensive wellness advice."""

            model = model
            system_message = "You coordinate a team of health specialists."
            instructions = "Delegate user requests to the appropriate specialist."
            tools = [Nutritionist, FitnessTrainer]
            config = {"verbose": True}

        coordinator = HealthCoordinator()

        response = coordinator("I want to lose 10kg and build muscle")
        ```

    === "Research Team"

        Multiple research specialists with a coordinator:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class AcademicResearcher(nn.Agent):
            """Expert in academic research with peer-reviewed sources.
            Use for scholarly inquiries and scientific topics."""

            model = model
            system_message = "You are an academic researcher."
            expected_output = "Provide academic-level analysis with citations."

        class MarketResearcher(nn.Agent):
            """Expert in market research and competitive analysis.
            Use for business intelligence and market sizing."""

            model = model
            system_message = "You are a market research analyst."
            expected_output = "Provide actionable business insights."

        class TechnicalResearcher(nn.Agent):
            """Expert in technical documentation and APIs.
            Use for programming questions and library comparisons."""

            model = model
            system_message = "You are a technical researcher."
            expected_output = "Provide technical details with code examples."

        class ResearchCoordinator(nn.Agent):
            model = model
            system_message = "You coordinate research specialists."
            instructions = "Delegate to the appropriate researcher based on the query type."
            tools = [
                AcademicResearcher,
                MarketResearcher,
                TechnicalResearcher
            ]
            config = {"verbose": True}

        coordinator = ResearchCoordinator()

        response = coordinator("Compare FastAPI vs Flask for building REST APIs")
        ```

    === "Agent Router"

        Route requests directly to specialists using `return_direct`:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        @mf.tool_config(return_direct=True)
        class PythonExpert(nn.Agent):
            """Expert in Python performance optimization."""

            model = model
            system_message = "You specialize in Python performance."

        @mf.tool_config(return_direct=True)
        class JavaScriptExpert(nn.Agent):
            """Expert in JavaScript and Node.js."""

            model = model
            system_message = "You specialize in JavaScript."

        class Router(nn.Agent):
            model = model
            system_message = "Route programming questions to the right expert."
            tools = [PythonExpert, JavaScriptExpert]
            config = {"verbose": True}

        router = Router()

        # Response comes directly from the specialist
        response = router("How do I optimize a Python loop?")
        ```

    === "Handoff Pattern"

        Seamless conversation handoff between agents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # Enable handoff - transfers conversation history
        @mf.tool_config(handoff=True)
        class StartupSpecialist(nn.Agent):
            """Specialist in scaling digital startups.
            Use for growth strategies, metrics, and funding."""

            model = model
            system_message = "You are a startup scaling expert."

        class BusinessConsultant(nn.Agent):
            model = model
            system_message = """You are a business consultant.
            If the context is a startup, transfer to the specialist."""
            tools = [StartupSpecialist]
            config = {"verbose": True}

        consultant = BusinessConsultant()

        # Conversation is handed off to specialist
        response = consultant(
            "My SaaS has a CAC of $120 and LTV of $600. How do I scale?"
        )
        ```

### MCP

The **Model Context Protocol (MCP)** allows agents to connect to external tool servers. MCP servers expose tools that can be called by the agent, enabling integration with filesystems, databases, APIs, and other services.

Configure MCP servers using the `mcp_servers` attribute:

???+ example

    === "Stdio Transport"

        Connect to an MCP server via standard I/O:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class FileAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            mcp_servers = [{
                "name": "filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            }]
            config = {"verbose": True}

        agent = FileAgent()
        response = agent("List all files in the current directory")
        ```

    === "HTTP Transport"

        Connect to an MCP server via HTTP:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class APIAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            mcp_servers = [{
                "name": "api",
                "transport": "http",
                "base_url": "http://localhost:8000",
                "headers": {"Authorization": "Bearer token"}
            }]

        agent = APIAgent()
        ```

    === "With Tool Configuration"

        Apply `tool_config` options to MCP tools:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class ConfiguredAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            mcp_servers = [{
                "name": "filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "include_tools": ["read_file", "write_file"],
                "tool_config": {
                    "read_file": {"inject_vars": True}
                }
            }]

        agent = ConfiguredAgent()
        ```

    === "Build Your Own (FastMCP)"

        Build a Python MCP server with [FastMCP](https://github.com/jlowin/fastmcp) and
        connect it to an Agent — no Node.js required.

        **1. Create the server** (`my_server.py`):

        ```python
        # /// script
        # requires-python = ">=3.10"
        # dependencies = ["fastmcp"]
        # ///
        """MCP server — launch with: uv run my_server.py"""
        from fastmcp import FastMCP

        mcp = FastMCP("my-server")

        @mcp.tool()
        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        @mcp.tool()
        def get_weather(city: str) -> str:
            """Return the current weather for a city."""
            # replace with a real API call
            return f"It's sunny in {city}, 24°C"

        if __name__ == "__main__":
            mcp.run()
        ```

        The `# /// script` block is [uv inline script metadata](https://docs.astral.sh/uv/guides/scripts/).
        `uv run my_server.py` installs `fastmcp` automatically in an isolated environment —
        no `pip install` or `pyproject.toml` changes needed.

        **2. Connect via Agent** (`main.py`):

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class MyAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "You are a helpful assistant."
            mcp_servers = [{
                "name": "my",
                "transport": "stdio",
                "command": "uv",
                "args": ["run", "my_server.py"],
            }]

        agent = MyAgent()
        response = agent("What is 3 + 4? Also, what's the weather in São Paulo?")
        print(response)
        # The agent has access to my__add and my__get_weather.
        ```

        !!! tip "Tool namespacing"
            Tools are prefixed with the server `name`: `my__add`, `my__get_weather`.
            Use `include_tools` / `exclude_tools` to control which tools are exposed.

**Server Configuration Options:**

| Option | Description |
|--------|-------------|
| `name` | Namespace for tools from this server |
| `transport` | `"stdio"` or `"http"` |
| `command` | Command to start the server (stdio only) |
| `args` | Command arguments (stdio only) |
| `cwd` | Working directory (stdio only) |
| `env` | Environment variables (stdio only) |
| `base_url` | Server URL (http only) |
| `headers` | Additional HTTP headers (http only) |
| `auth` | Authentication provider — `BearerTokenAuth`, `APIKeyAuth`, etc. (http only) |
| `include_tools` | Allowlist of tools to expose |
| `exclude_tools` | Blocklist of tools to hide |
| `tool_config` | Per-tool configuration options such as `display_name`, `usage_guidance`, retry, and injection behavior |
