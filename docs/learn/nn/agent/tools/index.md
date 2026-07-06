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

### More Tool Topics

- [Tool Config](config.md): per-tool behavior such as `return_direct`, runtime injection, retries, display names, and usage guidance.
- [Builtin Tools](builtin.md): built-in web, weather, agent, skill, and runtime tools.
- [Tool Search](tool-search.md): keep rarely used tools on demand and activate them with `tool_search`.
- [Background Tasks](background-tasks.md): background dispatch, task tools, progress, notifications, and `task_message`.
- [AgentTool](agent-tool.md): route many agents through one `agent(name, message)` tool and tool bucket.
- [MCP](mcp.md): connecting external Model Context Protocol servers.


!!! info

    While more tools enable more actions, too many tools can confuse the model about which one to use.

!!! tip

    Use [`usage_guidance`](config.md#usage_guidance) when the model needs explicit guidance about when or how to use a tool.

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

msgFlux ships built-in tools for common runtime and integration workflows. See [Builtin Tools](builtin.md) for the full catalog.

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

### Agents as Tools

An `nn.Agent` can be registered directly as a tool. In this direct pattern,
each agent appears to the model as its own callable tool, using the agent name,
description, and task signature.

Use this when the coordinator should choose between a small number of specialist
agents directly. If you want the model to see only one public
`agent(name, message)` tool that routes to many agents, use
[AgentTool](agent-tool.md) instead.

???+ example "Direct Agent Tools"

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

## Next Topics

- [Tool Config](config.md)
- [Builtin Tools](builtin.md)
- [Tool Search](tool-search.md)
- [Background Tasks](background-tasks.md)
- [Agent Tool](agent-tool.md)
- [MCP](mcp.md)
