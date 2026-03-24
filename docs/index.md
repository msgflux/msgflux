---
title: msgFlux
hide:
  - navigation
  - toc
---

# msgFlux { .homepage-title }

<div class="hero-split">
<div class="hero-left">
<img src="./assets/msgflux.png" alt="msgFlux" width="400" />
</div>
<div class="hero-right">
<p class="tagline"><em>Dynamic</em> AI Systems</p>
<div class="tabbed-set tabbed-alternate" data-tabs="0:2">
<input checked="checked" id="__tabbed_0_1" name="__tabbed_0" type="radio" />
<input id="__tabbed_0_2" name="__tabbed_0" type="radio" />
<div class="tabbed-labels">
<label for="__tabbed_0_1">uv</label>
<label for="__tabbed_0_2">pip</label>
</div>
<div class="tabbed-content">
<div class="tabbed-block">
<div class="highlight"><pre><code>uv add msgflux</code></pre></div>
</div>
<div class="tabbed-block">
<div class="highlight"><pre><code>pip install msgflux</code></pre></div>
</div>
</div>
</div>
</div>
</div>

msgFlux is an open-source framework for building dynamic AI systems using **composable modules**. It provides a lightweight definition of how AI systems are structured and how their components interact. Architecture, data flow, and prompts are independent layers that compose freely, adapting to different ways of thinking and building.

## **AI Systems *not* ML Systems**

**ML systems** are systems *for* AI — training, evaluating, and deploying models. **AI systems** are systems *with* AI — software where pretrained models are components within a larger application. The model is not the product; it is a building block. You are not training a model — you are *programming with one*. This is the space msgFlux occupies.

## **Declarative and Imperative**

One of the core ideas in msgFlux is that **interaction style is a module-level decision**. Each module can operate in one of two complementary modes — and both have native access to **vars**: runtime variables rendered into Jinja2 templates and optionally injected into tools.

- **Imperative**: the module receives inputs and vars explicitly and returns outputs directly.

- **Declarative**: the module declares where it reads data — including *vars* — from a shared message object.

=== "Imperative"

    The agent receives input and vars directly — like calling any Python function:

    ```python linenums="1"
    import msgflux as mf
    import msgflux.nn as nn

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    class SupportAgent(nn.Agent):
        model = model
        system_message = "You are a helpful support agent."
        instructions = """
        You are assisting {{ user_name }}.
        {% if is_vip %} Prioritize this customer.{% endif %}
        """

    agent = SupportAgent()

    vars = {"user_name": "Alice", "is_vip": True}  # (1)!

    result = agent("My dashboard is not loading after the last update.", vars=vars)
    print(result)  # (2)!
    ```

    1. `vars` flow into Jinja2 templates at runtime — `{{ user_name }}` renders into the instructions and `{% if is_vip %}` conditionally adds a priority note.
    2. Output is **returned explicitly** — the caller receives the result immediately.

=== "Declarative"

    The agent reads input from `msg.issue`, pulls vars from `msg.variables`, and writes to `msg.solution`:

    ```python linenums="1" hl_lines="13 14"
    import msgflux as mf
    import msgflux.nn as nn

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    class SupportAgent(nn.Agent):
        model = model
        system_message = "You are a helpful support agent."
        instructions = """
        You are assisting {{ user_name }}.
        {% if is_vip %} Prioritize this customer.{% endif %}
        """
        message_fields = {"task_inputs": "issue", "vars": "variables"}  # (1)!
        response_mode  = "solution"  # (2)!

    agent = SupportAgent()

    variables = {"user_name": "Alice", "is_vip": True}  # (3)!

    msg = mf.Message()
    msg.issue     = "My dashboard is not loading after the last update."
    msg.variables = variables
    agent(msg)

    print(msg.solution)  # (4)!
    ```

    1. **Reads input** from `msg.issue` and **reads vars** from `msg.variables` — the agent knows where to find its data.
    2. **Writes to** `msg.solution` — the result is placed back on the shared message.
    3. Vars are extracted from the message and rendered into Jinja2 templates — `{{ user_name }}` and `{% if is_vip %}` resolve automatically.
    4. After execution, the result is available on the message — no return value needed.

In the imperative model, a module behaves like a regular Python callable. Inputs and vars are passed directly, execution is explicit, and outputs are immediately returned. This is ideal for simple pipelines, scripts, or cases where control flow is clear and localized.

In the declarative model, a module is configured with knowledge about the structure of the message it operates on. Instead of receiving arguments, it knows *which fields to read* — including vars — and *which fields to populate*. This enables complex workflows where data flows through multiple modules without manual wiring, making composition and orchestration significantly easier.

## **Prompting and Programming**

On top of this interaction model, msgFlux deliberately distinguishes between **programming** and **prompting**, treating them as complementary but separate responsibilities.

- **Prompting** is where you define behavior *expressively*. Instead of embedding logic into code, you describe intent, instructions, roles, and constraints directly in natural language. These prompts are written explicitly and intentionally, but remain scoped by the signatures and modules that contain them.

- **Programming** is where you define the system structurally. This includes defining modules, agents, and especially **[signatures](https://dspy.ai/learn/programming/signatures/)**: typed, explicit contracts that describe inputs and outputs. Signatures formalize the behavior of a component and allow reasoning, validation, and optimization at the code level.

=== "Programming"

    Define behavior through a **signature** — a typed contract that specifies inputs and outputs. msgFlux generates the prompt and parses the structured result:

    ```python linenums="1"
    import msgflux as mf
    import msgflux.nn as nn
    from typing import Literal

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    class ClassifySentiment(mf.Signature):
        """Classify the sentiment of a sentence."""  # (1)!

        sentence: str = mf.InputField(desc="Text to analyze")
        sentiment: Literal["positive", "negative", "neutral"] = mf.OutputField()
        confidence: float = mf.OutputField(desc="Score between 0 and 1")

    class Classifier(nn.Agent):
        model = model
        signature = ClassifySentiment

    classifier = Classifier()
    result = classifier("I loved the movie, but the ending was disappointing.")
    # {'sentiment': 'neutral', 'confidence': 0.75}
    ```

    1. The docstring of a `Signature` becomes the agent's **instructions** — it tells the agent *what to do*.

=== "Prompting"

    Define behavior through **natural language** — system message, instructions, and expected output. You control exactly what the model sees:

    ```python linenums="1"
    import msgflux as mf
    import msgflux.nn as nn

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    class Classifier(nn.Agent):
        """Expert sentiment analyst."""

        model = model
        system_message = "You are a sentiment analysis expert."
        instructions = (
            "Analyze the sentiment of the given text. "
            "Consider nuance — a review can be mostly positive with negative aspects."
        )
        expected_output = "A JSON with 'sentiment' (positive/negative/neutral) and 'confidence' (0-1)."

    classifier = Classifier()
    result = classifier("I loved the movie, but the ending was disappointing.")
    ```

In this model, prompts are not loose strings passed around arbitrarily. They are written artifacts that live inside well-defined modules, constrained by signatures and executed within a programmed architecture.

By combining imperative and declarative modules with a clear separation between programming (signatures and structure) and prompting (written intent), msgFlux bridges classic software engineering and modern LLM-based development. The result is a system that scales from simple experiments to complex, production-ready AI applications while remaining explicit, composable, and maintainable.

## Modules

{!init_chat_completion_model.md!}


---

### Agent

msgFlux supports multiple styles for defining what an agent does. You can write explicit prompts for full control, use **signatures** to declare typed inputs and outputs, bind to fields on a shared **message** for pipeline composition, or give agents access to **tools** and **vars** that flow through the system. You can even use one agent as a **tool** for another. These styles compose freely — pick the right one for each component.

!!! info "Build Agents"

    Try the examples below after configuring your model above. Each tab demonstrates a different style or capability.

    === "Context"

        Pass additional context alongside the task — the agent grounds its answer on the provided information:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class Support(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Help the customer based on their account information."
            config = {"verbose": True, "include_date": True}

        agent = Support()

        account_info = """
        Name: Alice Johnson
        Plan: Premium
        Last payment: 2026-03-10
        Storage used: 45GB / 100GB
        """

        agent("Can I upgrade my storage?", context_inputs=account_info)
        ```

    === "Signature"

        Use `signature` to define inputs and outputs — msgFlux generates the prompt and parses structured output:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class Extractor(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "text -> summary: str, topics: list[str], sentiment: str"
            config = {"verbose": True}

        extractor = Extractor()
        result = extractor("The new iPhone has an amazing camera but the battery life is disappointing.")
        ```

        **Possible Output:**
        ```text
        {'summary': '...', 'topics': ['iPhone', 'camera', 'battery'], 'sentiment': 'mixed'}
        ```


    === "ReAct"

        Agents that reason step-by-step and use tools to find answers. `WebFetch` is a built-in tool that fetches web pages as Markdown:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ReAct
        from msgflux.tools.builtin import WebFetch

        class ResearchAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = ReAct
            tools = [WebFetch]
            config = {"verbose": True}

        agent = ResearchAgent()
        agent("What is the mass of the Earth divided by the mass of the Moon?")
        ```

        The agent iterates: **think** → **act** (call tools) → **observe** → repeat until `final_answer`.

    === "Vars"

        `vars` inject runtime context into the agent's Jinja2 **templates** and into tools via `inject_vars`. The model never sees injected vars directly — they flow through the system behind the scenes.

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        @mf.tool_config(inject_vars=True)
        def get_balance(**kwargs) -> str:
            """Look up the customer's current balance."""
            customer_id = kwargs["vars"]["customer_id"]
            balances = {"C-1234": "$1,250.00", "C-5678": "$340.75"}
            return balances.get(customer_id, "Customer not found.")

        class BankAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "You are helping customer {{customer_name}}."
            tools = [get_balance]
            config = {"verbose": True}

        agent = BankAgent()
        agent("What's my balance?", vars={"customer_name": "Alice", "customer_id": "C-1234"})
        ```

        `customer_name` renders into the instructions template. `customer_id` is injected into `get_balance` via `kwargs["vars"]` — invisible to the model, but available to the tool.

    === "Agent-as-Tool"

        An agent can serve as a tool for another agent. Decorate with `@tool_config` and pass the **class** to `tools`:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        @mf.tool_config(return_direct=True)  # (2)!
        class SentimentClassifier(nn.Agent):
            """Classify the sentiment of a given text."""  # (1)!

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "sentence: str -> sentiment: str, confidence: float"
            config = {"verbose": True}

        class Orchestrator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [SentimentClassifier]
            config = {"verbose": True}

        orchestrator = Orchestrator()
        orchestrator("Classify: 'This product is terrible'")
        ```

        1. When an agent is used as a tool, the docstring becomes its **description** — this is what the parent agent sees when deciding which tool to call.
        2. `return_direct=True` means the Orchestrator returns the list of tool calls and their results directly, instead of passing them back to the model for a final response.

    === "Message-driven"

        Bind inputs and outputs to fields on a shared `Message` — the preferred approach inside pipelines:

        ```python linenums="1"
        import msgspec
        import msgflux as mf
        import msgflux.nn as nn

        class Sentiment(msgspec.Struct):
            reasoning: str
            sentiment: str
            confidence: float

        class SentimentAnalyzer(nn.Agent):
            model            = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = Sentiment
            message_fields   = {"task_inputs": "review"}
            response_mode    = "sentiment"
            config           = {"verbose": True}

        analyzer = SentimentAnalyzer()

        msg = mf.Message()
        msg.review = "I loved the movie, but the ending was disappointing."
        analyzer(msg)

        print(msg.sentiment)
        print(msg.sentiment.confidence)
        ```

        The agent reads from `msg.review`, extracts structured data into a `Sentiment` schema, and writes to `msg.sentiment`. This makes modules easy to compose and reorder.

    === "Vision + CoT"

        Pass an image and let the agent reason step-by-step about what it sees:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ChainOfThought

        class VisionAnalyzer(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = ChainOfThought
            config = {"verbose": True}

        analyzer = VisionAnalyzer()
        result = analyzer(
            "What is happening in this image?",
            task_multimodal_inputs={"image": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"},
        )
        ```

        **Possible Output:**
        ```text
        {'reasoning': 'The image shows a close-up of an ant on a light surface...', 'final_answer': 'A macro photograph of a Camponotus ant.'}
        ```


---

### **Other Modules**

Beyond `nn.Agent`, msgFlux provides specialized modules for different modalities — all sharing the same `nn.Module` API:

!!! info "Built-in modules"

    All modules support `message_fields` and `response_mode` — configure once, then just pass the message through:

    === "Transcriber"

        Speech-to-text transcription:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class MeetingTranscriber(nn.Transcriber):
            model          = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")
            message_fields = {"task_multimodal_inputs": "audio_path"}
            response_mode  = "transcript"
            response_format = "text"
            config         = {"language": "en"}

        transcriber = MeetingTranscriber()

        msg = mf.Message()
        msg.audio_path = "meeting.mp3"
        transcriber(msg)

        print(msg.transcript)
        ```

    === "Speaker"

        Text-to-speech synthesis:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class Narrator(nn.Speaker):
            model          = mf.Model.text_to_speech("openai/tts-1")
            message_fields = {"task_inputs": "text"}
            response_mode  = "audio"
            response_format = "mp3"
            prompt         = "Speak in a calm, professional tone."

        narrator = Narrator()

        msg = mf.Message()
        msg.text = "Welcome to msgFlux."
        narrator(msg)

        print(msg.audio)  # bytes
        ```

    === "Embedder"

        Text embeddings for semantic search and similarity:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class TextEmbedder(nn.Embedder):
            model          = mf.Model.text_embedding("openai/text-embedding-3-small")
            message_fields = {"task_inputs": "texts"}
            response_mode  = "vectors"

        embedder = TextEmbedder()

        msg = mf.Message()
        msg.texts = ["How do transformers work?", "Attention is all you need."]
        embedder(msg)

        print(len(msg.vectors))  # 2
        ```

    === "MediaMaker"

        Image and video generation:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class ImageGenerator(nn.MediaMaker):
            model          = mf.Model.text_to_image("openai/gpt-image-1")
            message_fields = {"task_inputs": "prompt"}
            response_mode  = "image"

        generator = ImageGenerator()

        msg = mf.Message()
        msg.prompt = "A sunset over the ocean, watercolor style."
        generator(msg)

        print(msg.image)
        ```

---

### **Compose Modules into Programs**

A composition of modules is a **program** — each module handles one responsibility, and they work together naturally.


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
                    config={"verbose": True},
                )
                self.writer = nn.Agent(
                    name="writer",
                    model=model,
                    instructions="Write a clear summary based on the research.",
                    config={"verbose": True},
                )

            def forward(self, topic):
                research = self.researcher(topic)
                summary = self.writer(research)
                return summary

        pipeline = ResearchPipeline()
        pipeline("How do transformers work?")
        ```

    === "Router"

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Router(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Agent(
                    name="classifier",
                    model=model,
                    signature="text -> intent: Literal['billing', 'technical', 'general']",
                )
                self.agents = nn.ModuleDict({
                    "billing": nn.Agent(name="billing", model=model, instructions="Handle billing queries."),
                    "technical": nn.Agent(name="technical", model=model, instructions="Handle technical support."),
                    "general": nn.Agent(name="general", model=model, instructions="Handle general queries."),
                })

            def forward(self, message):
                result = self.classifier(message)
                return self.agents[result["intent"]](message)

        router = Router()
        router("I need to update my payment method")
        ```

    === "Multimodal"

        Combine Transcriber, Agent, and Speaker in a single pipeline — audio in, audio out:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class MeetingAssistant(nn.Module):
            """Transcribes audio, generates meeting notes, and narrates the summary."""

            def __init__(self):
                super().__init__()
                self.transcriber = nn.Transcriber(
                    name="transcriber",
                    model=mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe"),
                    config={"language": "en"},
                )
                self.summarizer = nn.Agent(
                    name="summarizer",
                    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
                    instructions="Generate a concise meeting summary with action items.",
                    config={"verbose": True},
                )
                self.narrator = nn.Speaker(
                    name="narrator",
                    model=mf.Model.text_to_speech("openai/tts-1"),
                    response_format="mp3",
                )

            def forward(self, audio_path):
                transcript = self.transcriber(audio_path)
                summary = self.summarizer(transcript)
                audio = self.narrator(summary)
                return audio

        assistant = MeetingAssistant()
        audio_summary = assistant("meeting.mp3")
        ```

??? info "Why a PyTorch-like API?"

    Millions of developers already know PyTorch's patterns: `nn.Module`, `forward()`, submodule registration, `state_dict()`. By adopting the same conventions, msgFlux lets you **transfer your existing mental model** to AI system design.

    If you've built neural networks with PyTorch, you already know how to build AI programs with msgFlux.


---

## **Inline** — dynamic workflows that flow and adapt.

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
