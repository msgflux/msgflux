# MCP

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
