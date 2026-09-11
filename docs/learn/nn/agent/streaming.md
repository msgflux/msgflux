# Streaming

Streaming lets the agent emit tokens as they are generated instead of waiting for the full response. This enables real-time display in terminals, chat UIs, and HTTP endpoints.

Enable streaming with `config={"stream": True}`. The agent returns a `ModelStreamResponse` object whose `consume()` method is an async generator that yields chunks.

## Basic Usage

???+ example

    === "Async"

        ```python
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"stream": True}

        agent = Assistant()

        response = await agent.acall("Tell me a story")

        async for chunk in response.consume():
            print(chunk, end="", flush=True)
        ```

    === "Sync"

        In sync mode, the stream runs in a background thread. `consume()` is still an async generator — use it inside an event loop or poll the response after the stream completes:

        ```python
        import time
        import msgflux as mf
        import msgflux.nn as nn

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"stream": True}

        agent = Assistant()

        response = agent("Tell me a story")

        # Wait for the stream to finish
        for _ in range(100):
            if response.metadata is not None:
                break
            time.sleep(0.1)

        # After completion, the full content is in response.data
        print(response.data)
        ```

    === "FastAPI"

        `consume()` is an async generator, so it plugs directly into `StreamingResponse`:

        ```python
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        import msgflux as mf
        import msgflux.nn as nn

        app = FastAPI()

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"stream": True}

        agent = Assistant()

        @app.get("/chat")
        async def chat(query: str):
            response = await agent.acall(query)
            return StreamingResponse(
                response.consume(),
                media_type="text/plain",
            )
        ```

## Response Object

When `stream=True`, the agent returns a `ModelStreamResponse` instead of a plain string. Key attributes:

| Attribute / Method | Description |
|---|---|
| `response.consume()` | Async generator that yields `str` chunks until the stream ends. |
| `response.consume_events()` | Async generator that preserves the provider order across output, reasoning, and reasoning-summary deltas. |
| `response.data` | Full accumulated content after the stream completes (or `None` while streaming). |
| `response.response_type` | `"text_generation"` or `"tool_call"` — set when the first content token arrives. |
| `response.metadata` | Usage and request timing — set when the stream finishes. `None` while streaming, useful as a completion signal. |
| `response.first_chunk_event` | Event that fires on the very first token (reasoning or content). |

## Streaming with Reasoning

When the model supports reasoning (`return_reasoning=True`), the compatibility
methods `consume()` and `consume_reasoning()` remain independent. Use
`consume_reasoning()` to read the chain of thought separately:

```python
model = mf.Model.chat_completion(
    "groq/openai/gpt-oss-120b",
    reasoning_effort="low",
    return_reasoning=True,
)

class Solver(nn.Agent):
    model = model
    system_prompt = "Solve the problem."
    config = {"stream": True}

agent = Solver()
response = await agent.acall("What is 15 * 7 + 3?")

print("Thinking:")
async for chunk in response.consume_reasoning():
    print(chunk, end="", flush=True)

print("\n\nAnswer:")
async for chunk in response.consume():
    print(chunk, end="", flush=True)
```

When relative ordering matters, consume the canonical event stream instead:

```python
async for event in response.consume_events():
    if event.type == "reasoning.delta":
        show_reasoning(event.data)
    elif event.type == "reasoning_summary.delta":
        show_summary(event.data)
    elif event.type == "output.delta":
        show_answer(event.data)
```

`consume_events()` omits absent channels. A model that exposes only a reasoning
summary starts with `reasoning_summary.delta`; it does not produce a placeholder
`reasoning.delta` containing `None`.

For the full dual-queue architecture, event system, and internal details, see [Reasoning — Streaming](reasoning.md#3-streaming).

## See Also

- [Reasoning](reasoning.md) — Ordered LM events and channel-specific consumers
- [Chat Completion — Streaming](../../models/chat_completion.md#6-streaming) — Model-level streaming reference
