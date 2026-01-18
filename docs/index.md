---
hide:
  - navigation
  - toc
---

<div class="hero fade-in-up" markdown>

# msgFlux { .gradient-text }

**An open-source framework for building multimodal AI applications** { .subtitle }

<p style="margin: 2rem 0;">
    <a href="quickstart/" class="md-button md-button--primary">
        :material-rocket-launch: Get Started
    </a>
    <a href="learn/models/model/" class="md-button">
        :material-book-open: Documentation
    </a>
</p>

```bash
pip install msgflux
```

</div>

---

## :material-shield-check: Core Principles

<div class="grid cards" markdown>

-   :material-shield-lock:{ .lg .middle } **Privacy First**

    ---

    msgFlux does not collect or transmit user data. All telemetry is fully controlled by the user and remains local, ensuring data sovereignty and compliance.

-   :material-puzzle:{ .lg .middle } **Designed for Simplicity**

    ---

    Core building blocks—**Model**, **DataBase**, **Parser**, and **Retriever**—provide a unified and intuitive interface to interact with diverse AI resources.

-   :material-lightning-bolt:{ .lg .middle } **Powered by Efficiency**

    ---

    Leverages high-performance libraries like **Msgspec**, **Uvloop**, **Jinja**, and **Ray** for fast, scalable, and concurrent applications.

-   :material-cog:{ .lg .middle } **Practical**

    ---

    Workflow API inspired by `torch.nn`, enabling seamless composition with native Python. Advanced **versioning and reproducibility** out of the box.

</div>

---

## :material-cube-outline: High-Level Modules

msgFlux introduces a set of high-level modules designed to streamline **multimodal inputs and outputs**. These modules encapsulate common AI pipeline tasks:

<div class="grid cards" markdown>

-   :material-robot:{ .lg } **Agent**

    ---

    Orchestrates multimodal data, instructions, context, tools, and generation schemas. The cognitive core of complex workflows.

-   :material-microphone:{ .lg } **Speaker**

    ---

    Converts text into natural-sounding speech, enabling voice-based interactions.

-   :material-text-to-speech:{ .lg } **Transcriber**

    ---

    Transforms spoken language into text, supporting speech-to-text pipelines.

-   :material-image-edit:{ .lg } **Designer**

    ---

    Generates visual content from prompts and images, combining textual and visual modalities.

-   :material-database-search:{ .lg } **Retriever**

    ---

    Searches and extracts relevant information based on queries, ideal for grounding models in external knowledge.

-   :material-brain:{ .lg } **Predictor**

    ---

    Wraps predictive models (e.g., scikit-learn) for smooth integration into larger workflows.

</div>

---

## :material-code-braces: Quick Example

=== "Chat Completion"

    ```python
    from msgflux.models import ChatCompletion

    model = ChatCompletion(provider="openai", model="gpt-4")

    response = model.call(
        messages=[{"role": "user", "content": "Hello!"}]
    )

    print(response.content)
    ```

=== "Text Embeddings"

    ```python
    from msgflux.models import TextEmbedder

    embedder = TextEmbedder(provider="openai")

    embeddings = embedder.call(
        texts=["Hello world", "msgFlux is awesome"]
    )

    print(embeddings.shape)
    ```

=== "Text-to-Speech"

    ```python
    from msgflux.models import TextToSpeech

    tts = TextToSpeech(provider="openai")

    audio = tts.call(
        text="Hello from msgFlux!",
        voice="alloy"
    )

    audio.save("output.mp3")
    ```

=== "Neural Network Module"

    ```python
    from msgflux.nn import Agent

    agent = Agent(
        model="gpt-4",
        instructions="You are a helpful assistant",
        tools=[search_tool, calculator_tool]
    )

    result = agent("What's the weather in Paris?")
    print(result)
    ```

---

## :material-speedometer: Why msgFlux?

<div class="feature-box" markdown>

### :material-layers-triple: Unified Interface

Work with **text**, **vision**, **speech**, and more through a single, consistent API. No need to learn different SDKs for each provider.

</div>

<div class="feature-box" markdown>

### :material-swap-horizontal: Provider Agnostic

Easily switch between **OpenAI**, **Anthropic**, **Google**, **Mistral**, and more without changing your code structure.

</div>

<div class="feature-box" markdown>

### :material-timer-sand: Production Ready

Built-in support for **async operations**, **retries**, **error handling**, and **observability**. Deploy with confidence.

</div>

---

## :material-rocket-launch-outline: Ready to Build?

<div style="text-align: center; margin: 3rem 0;" markdown>

[Get Started with msgFlux](quickstart/){ .md-button .md-button--primary }
[Explore Examples](learn/models/model/){ .md-button }
[View on GitHub :fontawesome-brands-github:](https://github.com/msgflux/msgflux){ .md-button }

</div>
