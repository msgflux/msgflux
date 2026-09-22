# Builtin Tools

msgFlux provides built-in tools that work out of the box:

## Workspace tools

`ReadFileTool`, `WriteTool`, `EditTool`, `DeleteTool` and `ApplyPatchTool`
operate through the live workspace binding, not directly on host paths.
`DeleteTool` removes one UTF-8 file and uses the same exact-content review
and approval mechanism as write/edit. It does not implement recursive `rm`.

```python
from msgflux.tools.builtin import DeleteTool, EditTool, ReadFileTool, WriteTool

tools = [ReadFileTool(cwd="/project"), WriteTool(cwd="/project"),
         EditTool(cwd="/project"), DeleteTool(cwd="/project")]
```

This creates tools whose relative paths start at virtual `/project`. Pass them
to an Agent with an execution environment and exact resource grants. Configure
`AgentApprovals` when changes require user confirmation; a standalone tool call
does not create a confirmation UI. See the
[runtime guide](../runtime.md#write-edit-and-delete-tools-with-agent-previews)
for configuration, review previews and checkpoint recovery.

### List, glob and grep

Install query dependencies with `uv add 'msgflux[workspace-tools]'` in your
application. These tools use the same workspace API on local and in-memory
backends, without invoking shell commands:

```python
from msgflux.tools.builtin import GlobTool, GrepTool, LsTool

tools = [
    LsTool(cwd="/project", max_entries=10_000),
    GlobTool(cwd="/project", max_results=1_000),
    GrepTool(cwd="/project", max_results=100, max_file_bytes=1_000_000),
]
```

This configures host-side limits without adding budget parameters to every model
call. The model sees `ls(path)`, `glob(pattern, path)` and `grep(pattern, path)`.
`path` defaults to `.` and identifies a directory; `cwd` is a virtual path, not
a host path and not a security boundary.

`ls` returns `{"path": ..., "entries": [{"name": ..., "kind": ...}]}` for one
directory. `glob` returns `{"matches": [{"path": ..., "kind": ...}],
"truncated": ...}`. Its patterns support `*`, `?`, character classes and `**`
for recursive path segments. `grep` searches UTF-8 text line by line, returning
matches with virtual `path`, one-based `line` and `text`, plus skipped-file
information and a truncation flag. Non-UTF-8, NUL-containing and oversized
files are skipped instead of being loaded without a bound.
Each matching line is previewed up to 4,096 characters, with its own `truncated`
flag; use `ReadFileTool` with `offset`/`limit` to inspect the surrounding text.
Grep's output budget counts encoded match and skipped-file records, excluding
the small enclosing JSON object. Reaching the result limit conservatively marks
the result as truncated even when it happens to equal the number of matches.

Glob and grep apply nested `.gitignore` rules **starting at the selected search
directory**. They do not inspect ancestor directories, global Git configuration
or `.git/info/exclude`. Search from the project root when project-wide ignore
rules must apply. The `.git` directory is always pruned. `ls` intentionally shows
all entries, including ignored names. Neither recursive tool follows symlinks
or other unsafe entry types.

Grant `filesystem.list` for each directory visited and `filesystem.read` for
existing `.gitignore` files and each file searched by grep. A list grant does
not imply a read grant. An inaccessible ignore file or unignored target raises
an error; it is not silently omitted as though the search were complete.

Traversal is bounded by host-configured depth, node count and elapsed-time
checks. Grep additionally bounds file bytes, match output and regex search time.
Result limits report truncation; traversal limits raise. Time checks are
cooperative and cannot interrupt a blocking backend operation. Large results
can still be handled by `ToolOutputOffloadExtension`; offload does not remove
the need to bound search work.

## WebFetchTool

`WebFetchTool` fetches web pages and converts them to Markdown. It uses a parser endpoint (default: `https://markdown.new/`) or falls back to semantic HTML parsing.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import WebFetchTool

class WebReader(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_prompt = "You help users understand web content."
    tools = [WebFetchTool]
    config = {"verbose": True}

agent = WebReader()
result = agent("Summarize the main points from https://news.ycombinator.com")
```

## WebSearchTool

`WebSearchTool` performs web searches backed by either a retriever or a model:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import WebSearchTool

# Retriever-backed search (using Exa, Brave, Tavily, etc.)
retriever_search = WebSearchTool("retriever/exa")

# Model-backed search
model_search = WebSearchTool("model/openai/gpt-4o-search-preview")

# Or use environment variables:
# export MSGFLUX_TOOL_WEB_SEARCH_ENGINE="retriever/wikipedia"
env_search = WebSearchTool()

class Researcher(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_prompt = "You help users find up-to-date information."
    tools = [retriever_search, model_search, env_search]
    config = {"verbose": True}

agent = Researcher()
result = agent("What is the latest Python version?")
```

Supported retriever engines: `wikipedia`, `searxng`, `serpapi`, `ceramic`, `brave`, `tavily`, `linkup`, `exa`, `arxiv`.

Supported model engines: any OpenAI-compatible model.

### WebSearch Parameters

- **`init_params`**: Passed when initializing the retriever or model backend.

- **`call_params`**: Passed on each retriever call (retriever engines only).

- **`goal`**: Optional call-time instruction for model-backed search only.
  It steers the model backend before it answers.

```python
# init_params: configure the backend at initialization
search = WebSearchTool(
    "retriever/exa",
    init_params={"include_text": True, "max_characters": 2000},
)

# call_params: passed on each call (retriever engines only)
result = search("Python news", call_params={"top_k": 5})

# goal: passed at call time for model-backed search only
model_search = WebSearchTool("model/openai/gpt-4o-search-preview")
result = model_search("Python news", goal="Answer with concise bullet points.")
```

Alternatively, read from environment variables:
```bash
export MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS='{"include_text": true}'
export MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS='{"top_k": 5}'
```

## WeatherTool

`WeatherTool` gets current, forecast, or historical weather data for a location:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import WeatherTool

weather = WeatherTool()

class WeatherAssistant(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_prompt = "You help users understand weather conditions."
    tools = [weather]
    config = {"verbose": True}

agent = WeatherAssistant()
result = agent("Is it raining in Fortaleza right now?")
```

The public tool call accepts:

- **`location`**: A simple city/place name such as `"Fortaleza"`. Coordinates like `"-3.71722,-38.54306"` are also accepted when the user provides them.

- **`when`**: `"now"`, a relative time like `"+6h"` or `"-3d"`, or an ISO datetime.

```python
weather = WeatherTool(
    engine="open_meteo",
    max_future_days=7,
    max_past_days=90,
    forecast_hours_when_now=6,
)

current = weather("Fortaleza")
forecast = weather("Fortaleza", when="+6h")
historical = weather("-3.71722,-38.54306", when="-3d")
```

If `engine` is not passed, `WeatherTool` reads `MSGFLUX_TOOL_WEATHER_ENGINE`. When
neither is set, it defaults to `open_meteo`.

The tool returns a structured `dotdict` with:

- `location`: resolved name, coordinates, and resolution source
- `when`: requested target time, kind (`now`, `future`, or `past`), and whether it was clamped
- `weather`: temperature, apparent temperature, humidity, condition, rain, cloud cover, and wind data
- `forecast`: next hourly items when available
- `source`: provider and endpoint metadata
- `units`: units reported by the weather provider

## Examples

???+ example "Builtin tool examples"

    === "Web Fetch"

        Extract text content from web pages using `httpx2`:

        ```python
        # pip install msgflux beautifulsoup4
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import WebFetchTool

        # mf.set_envs(OPENAI_API_KEY="...")

        class WebReader(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_prompt = "You help users understand web content."
            tools = [WebFetchTool]
            config = {"verbose": True}

        agent = WebReader()

        response = agent("Summarize the main points from https://news.ycombinator.com")
        ```

    === "Web Search"

        Use a built-in web search tool backed by either a retriever or a model:

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import WebSearchTool

        # Option 1: retriever-backed web search
        wikipedia_search = WebSearchTool(
            "retriever/wikipedia",
            call_params={"top_k": 2},
        )

        # Option 2: model-backed web search
        openai_search = WebSearchTool(
            "model/openai/gpt-4o-search-preview",
            init_params={
                "web_search_options": {"search_context_size": "low"},
            },
        )

        # Or read the engine and params from the environment:
        # export MSGFLUX_TOOL_WEB_SEARCH_ENGINE="retriever/wikipedia"
        # export MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS='{"language": "pt"}'
        # export MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS='{"top_k": 2}'
        env_search = WebSearchTool()

        class Researcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_prompt = "You help users find up-to-date information."
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
        explicitly, `WebSearchTool` reads the JSON objects from
        `MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS` and
        `MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS`.
        Model-backed search also accepts `goal` at call time to steer
        the model before it answers.

    === "WeatherTool"

        Get current, forecast, or historical weather data:

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import WeatherTool

        # mf.set_envs(OPENAI_API_KEY="...")

        weather = WeatherTool(
            engine="open_meteo",
            max_future_days=7,
            max_past_days=90,
            forecast_hours_when_now=6,
        )

        class WeatherAssistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_prompt = "You help users understand weather conditions."
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
        # pip install msgflux wikipedia
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # Create Wikipedia search tool from built-in retriever
        wikipedia = mf.Retriever.web_search("wikipedia")

        class Researcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_prompt = "You are a research assistant with access to Wikipedia."
            tools = [wikipedia]
            config = {"verbose": True}

        response = agent("Tell me about the history of the Python programming language")
        ```
