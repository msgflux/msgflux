# ⇄ Channels

Channels expose msgFlux modules through external interfaces. The first channel
is the HTTP server, which exposes registered `Agent` instances through an
OpenAI-compatible Chat Completions endpoint.

This lets any OpenAI SDK client call specialized msgFlux agents as if they were
regular chat models. The request `model` selects the agent name registered in
the server.

Source repository:

```text
https://github.com/msgflux/msgflux
```

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

The request is converted into an `AgentRun` before the Agent is executed:

| HTTP input | Runtime field | Agent call |
|---|---|---|
| `messages` | `run.messages` | `Agent.acall(messages=...)` |
| `stream` | `run.stream` | `Agent.acall(stream=...)` |
| `run_config.vars` | `run.variables` | `Agent.acall(vars=...)` |
| `run_config.model_preference` | `run.model_preference` | `Agent.acall(model_preference=...)` |
| `run_config.tool_filter` | `run.tool_filter` | `Agent.acall(tool_filter=...)` |
| `run_config.kwargs` | `run.kwargs` | expanded into `Agent.acall(**kwargs)` |

Use `run_config.kwargs` for Agent runtime inputs that are not part of the
OpenAI Chat Completions shape, such as `task`, `task_context` and
`task_multimodal`.

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

Multimodal request:

```bash
curl -N http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vision",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image in one short paragraph."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://raw.githubusercontent.com/msgflux/msgflux/main/docs/assets/logo.png"
            }
          }
        ]
      }
    ],
    "run_config": {
      "vars": {
        "tenant": "acme"
      }
    }
  }'
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

Use `registry.pre()` to mutate the `AgentRun` before `Agent.acall`. A
pre-processor can:

- Rewrite `run.messages`.
- Add or replace `run.variables`.
- Override `run.stream`.
- Set `run.model_preference` or `run.tool_filter`.
- Add `run.kwargs` such as `task`, `task_context` or `task_multimodal`.

Prefer passing server-side data through `run.variables` instead of injecting
extra `system` messages into the client-provided `messages`.

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

Processors can mutate the `run` object in place and return it, return `None` to
keep the current run, or return a mapping with replacement fields:

```python
@registry.pre("billing")
def force_billing_vars(_request, _context, run):
    return {
        "messages": run.messages,
        "vars": {
            **run.variables,
            "department": "billing",
        },
        "kwargs": run.kwargs,
    }
```

Returning a mapping replaces the provided fields. Mutate `run` in place when
you want to merge with the current execution state.

### Convert messages to task

Agents can be called with `messages`, but many msgFlux Agent designs are more
natural with `task`, `task_context` and templates. A pre-processor can unwrap
the OpenAI messages and call the Agent through the task interface instead.

```python
def last_user_text(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


@registry.pre("support")
def messages_to_task(_request, _context, run):
    task = last_user_text(run.messages)

    # Clear messages so the Agent receives only the rendered task.
    run.messages = []
    run.kwargs = {
        **run.kwargs,
        "task": task,
        "task_context": {
            "tenant": run.variables.get("tenant", "default"),
            "tier": run.variables.get("tier", "standard"),
        },
    }
    return run
```

This produces a call equivalent to:

```python
await agent.acall(
    messages=[],
    vars=run.variables,
    task="...",
    task_context={...},
)
```

### Convert OpenAI image content to task_multimodal

OpenAI clients often send multimodal content inside `messages[].content`. Agents
expect multimodal inputs through `task_multimodal`, so the channel can translate
the request shape before execution.

```python
class VisionAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "Describe images clearly and concisely."


registry.agent(VisionAgent(), name="vision")


def split_openai_content(content):
    if isinstance(content, str):
        return content, []

    text_parts = []
    image_urls = []

    for block in content or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "image_url":
            image = block.get("image_url")
            if isinstance(image, dict):
                image_urls.append(image.get("url"))
            elif isinstance(image, str):
                image_urls.append(image)

    return "\n".join(text_parts).strip(), [url for url in image_urls if url]


@registry.pre("vision")
def openai_images_to_task_multimodal(_request, _context, run):
    last_user = next(
        (
            message
            for message in reversed(run.messages)
            if message.get("role") == "user"
        ),
        {},
    )
    task, image_urls = split_openai_content(last_user.get("content"))

    run.messages = []
    run.kwargs = {
        **run.kwargs,
        "task": task or "Describe the image.",
        "task_multimodal": {
            "image": image_urls,
        },
    }
    return run
```

`task_multimodal` accepts the same media keys supported by Agent:

| Key | Example value |
|---|---|
| `image` | URL, local path, or list of image sources. |
| `audio` | URL, local path, or list of audio sources. |
| `video` | URL, local path, or list of video sources. |
| `file` | URL, local path, or list of file sources. |

You can also bypass message conversion and pass Agent kwargs directly:

```json
{
  "model": "vision",
  "messages": [],
  "run_config": {
    "kwargs": {
      "task": "Describe this image.",
      "task_multimodal": {
        "image": "https://raw.githubusercontent.com/msgflux/msgflux/main/docs/assets/logo.png"
      }
    }
  }
}
```

## Post-processing

Use `registry.post()` to transform the Agent output before the HTTP response is
encoded. Post-processors receive `(output, context, run)`.

```python
@registry.post("support")
def add_support_footer(output, _context, _run):
    if not isinstance(output, str):
        return output
    return output + "\n\nIf this does not solve it, contact support."
```

Return `None` to keep the original output. Return a new value to replace it.
For non-streaming responses, a mapping can control the final assistant message:

```python
@registry.post("billing")
def normalize_billing_output(output, context, _run):
    return {
        "response": str(output),
        "reasoning_content": f"Handled by agent `{context.agent_name}`.",
    }
```

The adapter maps `response` or `answer` to `message.content`, and maps
`reasoning` or `reasoning_content` to `message.reasoning_content`.

For streaming responses, the Agent usually returns a `ModelStreamResponse`.
Avoid converting it into a string in post-processing unless you intentionally
want to buffer the stream and lose incremental token delivery.

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
