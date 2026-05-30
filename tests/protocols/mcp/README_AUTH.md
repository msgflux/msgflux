# MCP HTTP Auth Validation

This directory includes a minimal authenticated HTTP MCP server used by tests and
manual validation.

Run the automated E2E auth test:

```bash
uv run pytest tests/protocols/mcp/test_http_auth_live.py
```

Run the server with Docker:

```bash
docker compose -f tests/protocols/mcp/docker-compose.auth.yml up --build
```

In another shell, validate the client path:

```bash
MCP_AUTH_URL=http://127.0.0.1:8765/mcp \
MCP_AUTH_TOKEN=test-token \
uv run python tests/protocols/mcp/check_auth_http_client.py
```

The manual client should print:

```text
tools=['whoami']
authenticated:docker
```
