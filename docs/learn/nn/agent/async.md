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

The real power of async is running multiple agents concurrently. `msgflux.nn.functional` provides concurrency primitives that handle error collection and integrate with telemetry:

???+ example

    === "abcast_gather — same input, multiple agents"

        Broadcast the same input to several agents and gather the results. Total time is roughly the slowest agent, not the sum:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        import msgflux.nn.functional as F

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

        summary, translation = await F.abcast_gather(
            [summarizer, translator], text
        )

        print(summary)
        print(translation)
        ```

    === "amap_gather — same agent, multiple inputs"

        Apply one agent to a list of inputs concurrently:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        import msgflux.nn.functional as F

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

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
            agent,
            args_list=[(r,) for r in reviews],
        )

        for review, result in zip(reviews, results):
            print(f"{review[:30]}... → {result}")
        ```

    === "ascatter_gather — different agents, different inputs"

        Dispatch different agents with different inputs concurrently:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        import msgflux.nn.functional as F

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Summarizer(nn.Agent):
            model = model
            instructions = "Summarize the text in one sentence."

        class Translator(nn.Agent):
            model = model
            instructions = "Translate the text to Portuguese."

        summarizer = Summarizer()
        translator = Translator()

        results = await F.ascatter_gather(
            [summarizer, translator],
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
- [Functional API](../functional.md) — `abcast_gather`, `amap_gather`, `ascatter_gather`, `aspawn`
