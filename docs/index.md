---
title: msgFlux
hide:
  - navigation
  - toc
---

# msgFlux { .homepage-title }

<div class="hero-split">
<div class="hero-left">
<img src="./assets/msgflux.png" alt="msgFlux" width="260" />
</div>
<div class="hero-right">
<p class="tagline"><em>Dynamic</em> AI Systems</p>
<div class="tabbed-set tabbed-alternate" data-tabs="0:2">
<input checked="checked" id="__tabbed_0_1" name="__tabbed_0" type="radio" />
<input id="__tabbed_0_2" name="__tabbed_0" type="radio" />
<div class="tabbed-labels">
<label for="__tabbed_0_1">pip</label>
<label for="__tabbed_0_2">uv</label>
</div>
<div class="tabbed-content">
<div class="tabbed-block">
<div class="highlight"><pre><code>pip install msgflux</code></pre></div>
</div>
<div class="tabbed-block">
<div class="highlight"><pre><code>uv add msgflux</code></pre></div>
</div>
</div>
</div>
</div>
</div>

Traditional software relies on predefined rules — but the real world is unpredictable. AI systems need to be **dynamic**: flexible enough to adapt, reason, and react to situations you never explicitly coded for. Language models make this possible.

msgFlux gives you a **PyTorch-like API** to build these systems. Compose agents, tools, and workflows into **programs** — where each module has a clear responsibility and the system as a whole can handle whatever comes its way.

Unlike frameworks that force you into a single paradigm, msgFlux supports **both declarative and prompting** approaches. Write structured modules that know where to read and write, compose them with `Inline` for dynamic workflows, or craft your own prompts. The *flux* is in the name: workflows that flow and adapt.

*tl;dr* Think of msgFlux as **PyTorch for AI systems** — modular, composable, and built for the real world. Get started on [GitHub](https://github.com/msgflux/msgflux) and [PyPI](https://pypi.org/project/msgflux/).


!!! info "Getting Started: Install msgFlux and set up your model"

    ```bash
    pip install msgflux[openai]
    ```

    === "OpenAI"

        Authenticate by setting the `OPENAI_API_KEY` env variable or using `set_envs`.

        ```python linenums="1"
        import msgflux as mf

        mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        ```

    === "Groq"

        Authenticate by setting the `GROQ_API_KEY` env variable.

        ```python linenums="1"
        import msgflux as mf

        mf.set_envs(GROQ_API_KEY="...")

        model = mf.Model.chat_completion("groq/llama-3.3-70b-versatile")
        ```

    === "Ollama"

        Install [Ollama](https://ollama.ai) and pull your model first:

        ```bash
        ollama pull llama3.2
        ```

        ```python linenums="1"
        import msgflux as mf

        model = mf.Model.chat_completion("ollama/llama3.2")
        ```

    === "Other providers"

        msgFlux supports 12+ providers. Any provider with an OpenAI-compatible API works:

        ```python linenums="1"
        import msgflux as mf

        # Together AI
        model = mf.Model.chat_completion("together/meta-llama/Llama-3.3-70B-Instruct-Turbo")

        # Cerebras
        model = mf.Model.chat_completion("cerebras/llama-3.3-70b")

        # OpenRouter
        model = mf.Model.chat_completion("openrouter/anthropic/claude-sonnet-4")

        # SambaNova
        model = mf.Model.chat_completion("sambanova/Meta-Llama-3.1-8B-Instruct")

        # vLLM (self-hosted)
        model = mf.Model.chat_completion("vllm/meta-llama/Llama-3.1-8B-Instruct")
        ```


---

## 1) **Three ways** to define AI behavior.

msgFlux supports three styles for defining what an agent does. You can mix and match them freely.

**Declarative** — Use `signature` to define inputs and outputs. msgFlux handles the prompt:

```python
class Extractor(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    signature = "text -> summary: str, topics: list[str]"
```

**Prompting** — Write your own system message and instructions for full control:

```python
class Writer(nn.Agent):
    """Expert technical writer."""

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You are an expert technical writer."
    instructions = "Write a clear, concise summary of the given topic."
    expected_output = "A 2-3 paragraph summary in markdown format."
```

**Message-driven** — Bind inputs and outputs to **Message fields** via `message_fields` and `response_mode`. This is the preferred approach inside pipelines, where modules share a single `Message` and each one reads from and writes to named fields:

```python
class SentimentAnalyzer(nn.Agent):
    """Analyzes the sentiment of a given text."""

    model         = mf.Model.chat_completion("openai/gpt-4.1-mini")
    signature     = "text -> sentiment: str, confidence: float, reasoning: str"
    message_fields = {"task_inputs": "review"}  # reads from msg.review
    response_mode  = "sentiment"                # writes result to msg.sentiment

analyzer = SentimentAnalyzer()

msg = mf.Message()
msg.review = "I loved the movie, but the ending was disappointing."
analyzer(msg)

print(msg.sentiment)            # dotdict — access fields like a dict or with dot notation
print(msg.sentiment.confidence) # 0.82
print(msg.sentiment["reasoning"])
```

The agent never exposes its internal field names to the caller — the caller only works with `msg.review` and `msg.sentiment`. This makes modules easy to compose and reorder.


!!! info "Build agents for any task"

    Try the examples below after configuring your model above. Each tab sets up an agent for a different task. When subclassing `nn.Agent`, the class name becomes the agent's `name` and the docstring becomes its `description`.

    === "Q&A"

        ```python linenums="1"
        agent = nn.Agent("Assistant", model)
        agent("What is the capital of France?")
        ```

        **Possible Output:**
        ```text
        'The capital of France is Paris.'
        ```

    === "Tools"

        ```python linenums="1"
        def get_weather(city: str) -> str:
            """Get current weather for a city."""
            return httpx.get(f"https://wttr.in/{city}?format=3").text

        class WeatherAssistant(nn.Agent):
            """Checks real-time weather using external APIs."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [get_weather]

        assistant = WeatherAssistant()
        assistant("What's the weather like in Tokyo?")
        ```

    === "Structured Output"

        ```python linenums="1"
        class SentimentAnalyzer(nn.Agent):
            """Analyzes the sentiment of a given text."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "text -> sentiment: str, confidence: float, reasoning: str"

        analyzer = SentimentAnalyzer()
        result = analyzer("I loved the movie, but the ending was disappointing.")
        ```

        **Possible Output:**
        ```text
        {'sentiment': 'mixed', 'confidence': 0.82, 'reasoning': 'Positive overall but negative about the ending.'}
        ```

    === "Extraction"

        ```python linenums="1"
        class PIXExtractor(nn.Agent):
            """Extracts PIX transfer data from user messages."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "text -> amount: float, key_type: Literal['email', 'phone', 'cpf'], key_id: str"

        extractor = PIXExtractor()
        result = extractor("Send $50 to john@example.com")
        ```

        **Possible Output:**
        ```text
        {'amount': 50.0, 'key_type': 'email', 'key_id': 'john@example.com'}
        ```

    === "Chain of Thought"

        ```python linenums="1"
        from msgflux.generation.reasoning import ChainOfThought

        class MathSolver(nn.Agent):
            """Solves math problems with step-by-step reasoning."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = ChainOfThought

        solver = MathSolver()
        result = solver("Two dice are tossed. What is the probability that the sum equals two?")
        ```

        **Possible Output:**
        ```text
        {'reasoning': 'Each die has 6 faces → 36 outcomes. Only (1,1) sums to 2 → P = 1/36.', 'final_answer': '1/36 ≈ 0.0278'}
        ```

    === "ReAct"

        ```python linenums="1"
        from msgflux.generation.reasoning import ReAct

        class ResearchAgent(nn.Agent):
            """Reasons step-by-step and uses tools to find answers."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = ReAct
            tools = [search_web, calculator]

        agent = ResearchAgent()
        agent("What is the mass of the Earth divided by the mass of the Moon?")
        ```

        The agent iterates: **think** → **act** (call tools) → **observe** → repeat until `final_answer`.


---

## 2) **Modules** — compose AI systems like PyTorch.

msgFlux's module system mirrors `torch.nn`. Every component inherits from `nn.Module`, supports `forward()` / `aforward()` for sync and async, automatic submodule registration via `__setattr__`, parameter management, and built-in telemetry. Compose multiple modules to create a **program** — a self-contained AI system where each piece has a clear responsibility.


!!! info "Compose modules into programs"

    === "Pipeline"

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class ResearchPipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.researcher = nn.Agent(
                    name="researcher",
                    model=model,
                    instructions="Research the given topic thoroughly.",
                )
                self.writer = nn.Agent(
                    name="writer",
                    model=model,
                    instructions="Write a clear summary based on the research.",
                )

            def forward(self, message):
                message = self.researcher(message)
                message = self.writer(message)
                return message

        pipeline = ResearchPipeline()

        msg = mf.Message()
        msg.input.researcher = "How do transformers work?"
        pipeline(msg)
        ```

    === "ModuleDict"

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class Router(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.agents = nn.ModuleDict({
                    "classifier": nn.Agent(
                        name="classifier",
                        model=model,
                        instructions="Classify the user's intent.",
                    ),
                    "responder": nn.Agent(
                        name="responder",
                        model=model,
                        instructions="Respond based on the classified intent.",
                    ),
                })

            def forward(self, message):
                message = self.agents["classifier"](message)
                message = self.agents["responder"](message)
                return message
        ```

    === "Multimodal"

        Combine different modalities in a single pipeline:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class MeetingAssistant(nn.Module):
            """Transcribes audio and generates structured meeting notes."""

            def __init__(self):
                super().__init__()
                self.transcriber = nn.Transcriber(
                    name="transcriber",
                    model=mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe"),
                )
                self.summarizer = nn.Agent(
                    name="summarizer",
                    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
                    instructions="Generate structured meeting notes from the transcript.",
                )

            def forward(self, message):
                message = self.transcriber(message)
                message = self.summarizer(message)
                return message
        ```


??? "Why a PyTorch-like API?"

    Millions of developers already know PyTorch's patterns: `nn.Module`, `forward()`, submodule registration, `state_dict()`. By adopting the same conventions, msgFlux lets you **transfer your existing mental model** to AI system design.

    If you've built neural networks with PyTorch, you already know how to build AI programs with msgFlux.


---

## 3) **Inline** — dynamic workflows that flow and adapt.

`Inline` is a lightweight DSL for declaring entire pipelines as a single expression. Sequential steps (`->`), parallel branches (`[a, b]`), conditionals (`{cond ? a, b}`), and loops (`@{cond}: a;`) — all in one readable string. Every module reads from and writes to a shared `dotdict` message. This is the *flux* — the dynamic flow that gives the library its name.


!!! info "Orchestrate agents with a single expression"

    ```python linenums="1"
    import msgflux as mf
    import msgflux.nn as nn

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")


    class Router(nn.Agent):
        """Classifies user intent."""

        model = model
        signature = "text -> intent: Literal['technical', 'general']"


    class TechnicalExpert(nn.Agent):
        """Answers technical questions with precision and depth."""

        model = model
        system_message = "You are a technical expert. Be precise and detailed."


    class GeneralAssistant(nn.Agent):
        """Answers general questions in a friendly, concise way."""

        model = model
        system_message = "You are a friendly assistant. Be concise."


    router, expert, assistant = Router(), TechnicalExpert(), GeneralAssistant()


    def classify(msg):
        msg.intent = router(msg.question)

    def expert_answer(msg):
        msg.answer = expert(msg.question)

    def general_answer(msg):
        msg.answer = assistant(msg.question)


    flux = mf.Inline(
        "classify -> {intent == 'technical' ? expert_answer, general_answer}",
        {
            "classify": classify,
            "expert_answer": expert_answer,
            "general_answer": general_answer,
        },
    )

    msg = mf.dotdict(question="How does backpropagation work?")
    flux(msg)
    print(msg.answer)
    ```

    The `Router` agent classifies the intent at runtime, and `Inline` **conditionally routes** to the right expert — the pipeline adapts to the input. No `if/else` in Python, just a declarative expression.


---

## 4) **Beyond Agents** — a complete AI toolkit.

<div class="grid cards">
<div>
<strong>Multimodal Modules</strong><br>
<code>nn.Speaker</code>, <code>nn.Transcriber</code>, <code>nn.MediaMaker</code> — text-to-speech, speech-to-text, and image generation as composable pipeline steps.
</div>
<div>
<strong>MCP Protocol</strong><br>
Full Model Context Protocol support with Stdio and HTTP transports, authentication, tool discovery, and resource management.
</div>
<div>
<strong>Telemetry</strong><br>
OpenTelemetry-based tracing for every module, tool call, and LLM request. Zero overhead when disabled.
</div>
<div>
<strong>12+ Providers</strong><br>
OpenAI, Groq, Cerebras, Ollama, Together, OpenRouter, SambaNova, vLLM, JinaAI, and more — all through a unified API.
</div>
<div>
<strong>Async + Parallel</strong><br>
Every module supports <code>forward()</code> and <code>aforward()</code>. Use <code>F.scatter_gather</code> and <code>F.map_gather</code> for parallel execution.
</div>
<div>
<strong>Production Ready</strong><br>
Built-in response caching, retry logic with exponential backoff, fast <code>msgspec</code> serialization, and structured logging.
</div>
</div>
