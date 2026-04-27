# ⇄ Channels

Channels expose msgFlux modules through external interfaces. The first channel
is the HTTP server, which exposes registered `Agent` instances through an
OpenAI-compatible Chat Completions endpoint.

This lets any OpenAI SDK client call specialized msgFlux agents as if they were
regular chat models. The request `model` selects the agent name registered in
the server.

## Install

The HTTP server depends on FastAPI and Uvicorn:

```bash
pip install "msgflux[server,openai]"
```

When working inside this repository, use:

```bash
uv run --extra server --extra openai msgflux server examples/server_streaming_agent.py:registry --host 127.0.0.1
```

The default server address is:

```text
http://127.0.0.1:8010/v1
```

## Registry

Create a Python file that exports a `ChannelRegistry`. The server loads this
object from the CLI target.

```python
import msgflux as mf
from msgflux import nn

mf.load_dotenv()


class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You are a concise customer support specialist."


class BillingAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You are a concise billing support specialist."


registry = mf.ChannelRegistry()
registry.agent(SupportAgent(), name="support")
registry.agent(BillingAgent(), name="billing")
```

Run it:

```bash
uv run --extra server --extra openai msgflux server examples/server_streaming_agent.py:registry --host 127.0.0.1
```

Use `--port` to override the default `8010`:

```bash
uv run --extra server --extra openai msgflux server examples/server_streaming_agent.py:registry --host 127.0.0.1 --port 9000
```

## Chat Completions

The server currently exposes:

```text
POST /v1/chat/completions
```

The supported request fields are:

| Field | Description |
|---|---|
| `model` | Registered agent name, such as `support` or `billing`. |
| `messages` | OpenAI-compatible chat messages. |
| `stream` | Enables OpenAI-compatible SSE streaming when `true`. |
| `run_config` | msgFlux extension for runtime execution config. |

`run_config` currently supports:

| Key | Description |
|---|---|
| `vars` | Runtime variables passed to the Agent. |
| `model_preference` | Optional model preference forwarded to the Agent. |
| `tool_filter` | Optional tool filtering metadata forwarded to the Agent. |
| `kwargs` | Extra runtime kwargs forwarded to `Agent.acall`. |

The endpoint does not persist sessions. Each request should contain the
messages required for that turn.

## cURL

Non-streaming request:

```bash
curl -s http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "support",
    "messages": [
      {
        "role": "user",
        "content": "Meu pedido A1002 ainda nao chegou. O que aconteceu?"
      }
    ],
    "run_config": {
      "vars": {
        "tenant": "acme",
        "tier": "priority"
      }
    }
  }'
```

Streaming request:

```bash
curl -N http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "billing",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "A fatura INV-44 falhou. O que devo fazer agora?"
      }
    ],
    "run_config": {
      "vars": {
        "tenant": "acme",
        "tier": "priority"
      }
    }
  }'
```

The streaming response uses Server-Sent Events with OpenAI-compatible chunks and
ends with:

```text
data: [DONE]
```

## msgFlux Client

The `msgflux` model provider is a thin OpenAI-compatible client for a msgFlux
server. It uses `MSGFLUX_BASE_URL` when set, otherwise it defaults to
`http://127.0.0.1:8010/v1`.

```python
import asyncio
import msgflux as mf

mf.load_dotenv()


async def main():
    model = mf.Model.chat_completion(
        "msgflux/support",
        variables={
            "tenant": "acme",
            "tier": "priority",
        },
    )
    response = await model.acall(
        [{"role": "user", "content": "Meu pedido A1002 ainda nao chegou."}],
        stream=True,
    )

    async for chunk in response.consume():
        print(chunk, end="", flush=True)


asyncio.run(main())
```

For a different server port:

```bash
MSGFLUX_BASE_URL=http://127.0.0.1:9000/v1 uv run --extra openai python examples/server_streaming_client.py
```

## Pre-processing

Use `registry.pre()` to mutate the `AgentRun` before `Agent.acall`. Prefer
passing server-side data through `run.variables` instead of injecting extra
`system` messages into the client-provided `messages`.

```python
@registry.pre()
def add_server_context(_request, context, run):
    run.variables = {
        **run.variables,
        "tenant": run.variables.get("tenant", "default"),
        "tier": run.variables.get("tier", "standard"),
        "agent_name": context.agent_name,
        "request_id": context.request_id,
    }
    return run
```

Use `@registry.pre("support")` to target a single agent, or `@registry.pre()`
to run the processor for every registered agent.

## Examples

The repository includes a multi-agent streaming example:

```bash
uv run --extra server --extra openai msgflux server examples/server_streaming_agent.py:registry --host 127.0.0.1
```

In another terminal:

```bash
uv run --extra openai python examples/server_streaming_client.py
```

The example exposes two agents in the same server:

| Agent | Model path | Purpose |
|---|---|---|
| `support` | `msgflux/support` | Order support. |
| `billing` | `msgflux/billing` | Invoice and payment support. |
