# Prompt Cache Warmup

Prompt warmup sends the agent's stable prompt prefix to the model provider before the first real task. Providers with prompt caching can prefill the system prompt and tool schemas, so the next request with the same prefix can start faster or report cached input tokens.

Use warmup when your agent has a large static system prompt, many tools, or loaded skills. It is not a replacement for response caching: no task message is sent, no checkpoint is written, and no conversation history is updated.

## Basic Usage

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn


def lookup_ticket(ticket_id: str) -> str:
    return f"{ticket_id}: open"


class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = """
    You are a support agent.
    Follow the escalation policy and use tools when needed.
    """
    tools = [lookup_ticket]


agent = SupportAgent()

# Sends only the rendered system prompt and tool schemas.
agent.warmup_system_prompt()

response = agent("Check ticket MSGFLUX-42")
print(response)
```

## Async Usage

```python
agent = SupportAgent()

await agent.awarmup_system_prompt()
response = await agent.acall("Check ticket MSGFLUX-42")
```

## Fire And Forget

Warmup can run in the background while the process continues initializing:

```python
agent = SupportAgent()

agent.warmup_system_prompt(background=True)

# Continue booting your CLI, server, or worker.
```

For async applications:

```python
await agent.awarmup_system_prompt(background=True)
```

Background warmup does not return the provider response. Errors are logged by the background task runner.

## What Is Sent

Warmup intentionally bypasses runtime state. It sends:

- The rendered `system_prompt`
- Tool schemas from the agent's `ToolLibrary`
- The provider's configured warmup token limit

It does not send:

- User task messages
- Chat history
- Checkpointer state
- Agent inbox notifications
- Typed parsers or generation schemas
- Response templates

This keeps warmup focused on the stable prefix that providers can cache.

## Runtime Vars

If the system prompt uses runtime variables, pass the same `vars` you expect to use for the real request:

```python
agent.warmup_system_prompt(vars={"tenant": "acme"})

response = agent(
    "Summarize the ticket",
    vars={"tenant": "acme"},
)
```

Cache hits depend on the provider seeing the same prompt prefix. If the rendered system prompt changes between warmup and the real call, the provider may miss the cache.

## Tool Filtering

Warmup accepts the same `tool_filter` shape used by agent execution:

```python
agent.warmup_system_prompt(tool_filter={"allow": ["lookup_ticket"]})
```

Use this when the real request will also restrict tools. A different tool schema list changes the prompt prefix and can reduce cache hits.

## Model Gateway

When the agent uses a `ModelGateway`, warmup routes through the gateway and accepts `model_preference`:

```python
agent.warmup_system_prompt(model_preference="fast")
```

The selected model must support `warmup_system_prompt`.

## Provider Token Limit

OpenAI-compatible chat providers default to one generated token for warmup. This keeps the request cheap while still forcing the provider to process the prompt.

Override it on the model if a provider needs a different minimum:

```python
model = mf.Model.chat_completion(
    "openai/gpt-4.1-mini",
    warmup_max_tokens=1,
)
```

OpenAI Chat Completions and Groq reject zero completion tokens, so `0` is not a portable warmup value.

## Reading Cache Usage

The provider response is returned in foreground mode:

```python
raw_response = agent.warmup_system_prompt()
print(raw_response.usage)
```

OpenAI reports prompt cache hits in usage metadata:

- Chat Completions: `usage.prompt_tokens_details.cached_tokens`
- Responses API: `usage.input_tokens_details.cached_tokens`

msgflux currently implements this warmup path for chat completions.

## Practical Guidance

- Keep the warmup prefix stable.
- Put large static instructions, tools, and skills before dynamic user content.
- Warmup is most useful above the provider's prompt-cache threshold.
- Do not expect warmup to persist conversation state.
- Use foreground warmup during tests so failures are visible.
- Use background warmup during app startup when failures should not block serving.

