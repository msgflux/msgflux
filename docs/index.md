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

msgFlux is an open-source framework for building dynamic AI systems with **composable modules**. It treats prompts, signatures, tools, and message flow as explicit program structure instead of ad-hoc glue. Architecture, data flow, and prompting remain separate layers, so systems can evolve by changing contracts, modules, or routes without forcing everything to change together.

## **AI Systems *not* ML Systems**

**ML systems** are systems *for* AI - they train, evaluate, and deploy models. **AI systems** are systems *with* AI - software where pretrained models operate as components inside a broader application. In that setting, the model is not the whole product; it is one building block among many. You are not optimizing weights - you are designing behavior, interfaces, and flow around a model. This is the space msgFlux occupies.

## **Declarative and Imperative**

One of the core ideas in msgFlux is that **interaction style is a module-level decision**. A module should be able to behave like a regular callable or like a message-bound operator, depending on the role it plays in the system. msgFlux therefore supports two complementary modes, and both have native access to **vars**: runtime variables rendered into Jinja2 templates and optionally injected into tools.

- **Imperative**: the module receives inputs and vars explicitly and returns outputs directly.

- **Declarative**: the module declares where it reads data from a shared message object.

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
        message_fields = {"task": "issue", "vars": "variables"}  # (1)!
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

In the imperative model, a module behaves like a regular Python callable. Inputs and vars are passed directly, execution is explicit, and outputs are immediately returned. This is ideal when the caller owns control flow and the composition should stay obvious at the call site - for example in scripts, local pipelines, or tightly scoped orchestration code.

In the declarative model, a module is configured with knowledge about the structure of the message it operates on. Instead of receiving arguments, it knows *which fields to read* and *which fields to populate*. This is especially useful once multiple agents, tools, or modalities are operating over shared state, because composition becomes a matter of declaring contracts instead of hand-wiring every edge in the flow.

## **Prompting and Programming**

On top of this interaction model, msgFlux deliberately distinguishes between **programming** and **prompting**, treating them as complementary but separate responsibilities. This view comes primarily from a PyTorch-style way of building systems: modules, composition, explicit interfaces, and controlled data flow. msgFlux also adopts **signatures** as a useful abstraction for LM programming, because typed contracts are a strong way to describe what a component should consume and produce.

- **Prompting** is where you define behavior *expressively*. Instead of embedding all behavior into code, you describe intent, instructions, roles, and constraints directly in natural language. These prompts are written explicitly and intentionally, but remain scoped by the signatures and modules that contain them.

- **Programming** is where you define the system structurally. This includes defining modules, agents, routing, and especially **[signatures](https://dspy.ai/learn/programming/signatures/)**: typed, explicit contracts that describe inputs and outputs. In msgFlux, signatures are one tool within a larger module system. They formalize the behavior of a component and make it possible to reason about, validate, and optimize it at the code level.

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
    result = classifier(sentence="I loved the movie, but the ending was disappointing.")
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

In this model, prompts are not loose strings passed around arbitrarily. They are written artifacts that live inside well-defined modules, constrained by signatures, and executed within a programmed architecture.

Declarative signatures can also make systems more resilient to model updates, because the contract stays stable even when the underlying model changes. The center of gravity shifts from micromanaging *how* a prompt is phrased to specifying *what* the component must produce.

By combining imperative and declarative modules with a clear separation between programming (signatures, structure, and flow) and prompting (written intent), msgFlux brings software architecture discipline to LM-based development. The result is a system that scales from simple experiments to complex AI applications while remaining explicit, composable, and maintainable.

## Modules

{!init_chat_completion_model.md!}


---

### Agent

Agents in msgFlux are flexible — prompt them directly, use **signatures** for typed I/O, bind to a shared **message**, inject **tools** and **vars**, or nest one agent inside another as a tool. Mix and match as needed.

!!! info "Build Agents"

    === "Context"

        Pass additional task context alongside the task — the agent grounds its answer on the provided information:

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

        agent("Can I upgrade my storage?", task_context=account_info)
        ```

    === "Multimodal"

        Pass PDFs directly to the agent — from a URL or a local file. The agent reads and reasons over the document content:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class AnalyzerAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")
            config = {"verbose": True}

        agent = AnalyzerAgent()

        # From URL
        response = agent(
            "Summarize the key contributions of this paper.",
            task_multimodal_inputs={"file": "https://arxiv.org/pdf/2106.09685.pdf"}
        )

        # From local file
        response = agent(
            "Summarize the key contributions of this paper.",
            task_multimodal_inputs={"file": "./lora.pdf"}
        )
        ```

        **Possible Output:**
        ```text
        The paper proposes LoRA (Low-Rank Adaptation), an efficient fine-tuning method
        that injects trainable low-rank matrices into frozen pre-trained weights.
        It reduces trainable parameters by up to 10,000x with no inference latency overhead.
        ```

    === "Signature"

        Use `signature` to define inputs and outputs — msgFlux generates the prompt and parses structured output:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class Extractor(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "text -> summary, topics: list[str], sentiment"
            config = {"verbose": True}

        extractor = Extractor()
        result = extractor(text="The new iPhone has an amazing camera but the battery life is disappointing.")
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
        agent("Fetch https://en.wikipedia.org/wiki/Earth and summarize the key facts about Earth's mass and composition.")
        ```

        The agent iterates: **think** → **act** (call tools) → **observe** → repeat until `final_answer`.

    === "Vars"

        `vars` inject runtime context into the agent's Jinja2 **templates** and into tools via `inject_vars`. The model never sees injected vars directly, they flow through the system behind the scenes.

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
        vars = {"customer_name": "Alice", "customer_id": "C-1234"}
        agent("What's my balance?", vars=vars)
        ```

        `customer_name` renders into the instructions template. `customer_id` is injected into `get_balance` via `kwargs["vars"]` — invisible to the model, but available to the tool.

    === "Agent-as-Tool"

        An agent can serve as a tool for another agent. Pass the **class** to `tools`. Use `@tool_config` when you need extra behavior, like routing the tool result directly to the caller:

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

    === "Structured Output"

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
            message_fields   = {"task": "review"}
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
            task_multimodal={"image": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"},
        )
        ```

        **Possible Output:**
        ```text
        {'reasoning': 'The image shows a close-up of an ant on a light surface...', 'final_answer': 'A macro photograph of a Camponotus ant.'}
        ```


---

### **Other Modules**

Beyond `nn.Agent`, msgFlux provides specialized modules for different modalities:

!!! info "Built-in modules"

    All modules support `message_fields` and `response_mode` — configure once, then just pass the message through:

    === "Transcriber"

        Speech-to-text transcription:

        ```python linenums="1"
        import msgflux as mf
        import msgflux.nn as nn

        class MeetingTranscriber(nn.Transcriber):
            model           = mf.Model.speech_to_text("openai/whisper-1")
            message_fields  = {"task_multimodal": {"audio": "audio_path"}}
            response_mode   = "transcript"
            response_format = "text"
            config          = {"language": "en"}

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
            model           = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
            message_fields  = {"task": "text"}
            response_mode   = "audio"
            response_format = "pcm"
            prompt          = "Speak in a calm, professional tone."

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
            message_fields = {"task": "texts"}
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
            message_fields = {"task": "prompt"}
            response_mode  = "image"
            config         = {"background": "transparent"}

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

        class Researcher(nn.Agent):
            model = model
            instructions = "Research the given topic thoroughly."
            config = {"verbose": True}

        class Writer(nn.Agent):
            model = model
            instructions = "Write a clear summary based on the research."
            config = {"verbose": True}

        class ResearchPipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.researcher = Researcher()
                self.writer = Writer()

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

        class Classifier(nn.Agent):
            model = model
            signature = "text -> intent: Literal['billing', 'technical', 'general']"

        class BillingAgent(nn.Agent):
            model = model
            instructions = "Handle billing queries."

        class TechnicalAgent(nn.Agent):
            model = model
            instructions = "Handle technical support."

        class GeneralAgent(nn.Agent):
            model = model
            instructions = "Handle general queries."

        class Router(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = Classifier()
                self.agents = nn.ModuleDict({
                    "billing": BillingAgent(),
                    "technical": TechnicalAgent(),
                    "general": GeneralAgent(),
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

        class Transcriber(nn.Transcriber):
            model = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")
            config = {"language": "en"}

        class Summarizer(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Generate a concise meeting summary with action items."
            config = {"verbose": True}

        class Narrator(nn.Speaker):
            model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
            response_format = "pcm"

        class MeetingAssistant(nn.Module):
            """Transcribes audio, generates meeting notes, and narrates the summary."""

            def __init__(self):
                super().__init__()
                self.transcriber = Transcriber()
                self.summarizer = Summarizer()
                self.narrator = Narrator()

            def forward(self, audio_path):
                transcript = self.transcriber(audio_path)
                summary = self.summarizer(transcript)
                audio = self.narrator(summary)
                return audio

        assistant = MeetingAssistant()
        audio_summary = assistant("./meeting.mp3")
        ```

??? info "Why a PyTorch-like API?"

    Millions of developers already know PyTorch's patterns: `nn.Module`, `forward()`, submodule registration, `state_dict()`. By adopting the same conventions, msgFlux lets you **transfer your existing mental model** to AI system design.

    If you've built neural networks with PyTorch, you already know how to build AI programs with msgFlux.


---

## **Inline**

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

## Acknowledgements

msgFlux is built around a select set of exceptional libraries that make the whole thing possible:

- **[msgspec](https://jcristharif.com/msgspec/)** — ultra-fast serialization and validation that underpins all data contracts in msgFlux
- **[Jinja2](https://jinja.palletsprojects.com/)** — the templating engine powering prompt composition, vars injection, and pipeline expressions
- **[Tenacity](https://tenacity.readthedocs.io/)** — reliable retry logic with exponential backoff for resilient model calls
- **[OpenTelemetry](https://opentelemetry.io/)** — the observability standard behind msgFlux's built-in tracing and telemetry

We are grateful to the authors and maintainers of these projects.
