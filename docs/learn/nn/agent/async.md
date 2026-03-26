# Async

Every Agent supports async execution via `acall`. This allows the agent to run without blocking the event loop, making it essential for concurrent execution, web frameworks, and pipelines that call multiple agents in parallel.

## Basic Usage

Replace the sync call `agent(...)` with `await agent.acall(...)`:

???+ example

    === "Sync"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Answer concisely."

        agent = Assistant()
        response = agent("What is the capital of Japan?")
        print(response)  # "Tokyo"
        ```

    === "Async"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Answer concisely."

        agent = Assistant()
        response = await agent.acall("What is the capital of Japan?")
        print(response)  # "Tokyo"
        ```

The return type is the same — `acall` is the async equivalent of `__call__`, both route through `forward` / `aforward` respectively.

## Concurrent Agents

The real power of async is running multiple agents concurrently. Use `asyncio.gather` to dispatch several calls at once:

```python
import asyncio
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.chat_completion("openai/gpt-4.1-mini")

class Summarizer(nn.Agent):
    model = model
    instructions = "Summarize the text in one sentence."

class Translator(nn.Agent):
    model = model
    instructions = "Translate the text to Portuguese."

summarizer = Summarizer()
translator = Translator()

text = "Quantum computing uses qubits that can exist in superposition..."

# Both run concurrently — total time ≈ max(summarizer, translator)
summary, translation = await asyncio.gather(
    summarizer.acall(text),
    translator.acall(text),
)

print(summary)
print(translation)
```

## Functional Concurrency

`msgflux.nn.functional` provides higher-level concurrency primitives that handle threading and error collection:

???+ example

    === "amap_gather — same agent, multiple inputs"

        Apply one agent to a list of inputs concurrently:

        ```python
        import msgflux.nn as nn
        import msgflux.nn.functional as F

        class Classifier(nn.Agent):
            model = model
            instructions = "Classify the sentiment as positive, negative, or neutral."

        agent = Classifier()

        reviews = [
            "This product is amazing!",
            "Terrible experience, never again.",
            "It's okay, nothing special.",
        ]

        results = await F.amap_gather(
            agent.acall,
            args_list=[(r,) for r in reviews],
        )

        for review, result in zip(reviews, results):
            print(f"{review[:30]}... → {result}")
        ```

    === "ascatter_gather — different agents, different inputs"

        Dispatch different agents with different inputs concurrently:

        ```python
        import msgflux.nn as nn
        import msgflux.nn.functional as F

        summarizer = Summarizer()
        translator = Translator()

        results = await F.ascatter_gather(
            [summarizer.acall, translator.acall],
            args_list=[
                ("Explain quantum computing in detail...",),
                ("The weather is beautiful today.",),
            ],
        )

        summary, translation = results
        ```

## Web Frameworks

`acall` integrates naturally with async web frameworks:

???+ example

    === "FastAPI"

        ```python
        from fastapi import FastAPI
        import msgflux as mf
        import msgflux.nn as nn

        app = FastAPI()

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Answer the user's question."

        agent = Assistant()

        @app.get("/ask")
        async def ask(query: str):
            response = await agent.acall(query)
            return {"response": response}
        ```

## See Also

- [Streaming](streaming.md) — Async streaming with `consume()` and `consume_reasoning()`
- [Functional API](../functional.md) — `amap_gather`, `ascatter_gather`, `aspawn`
