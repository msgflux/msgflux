# Task and Context

The agent receives input through **task** (what to do) and **task_context**
(background information). When both are provided, they're combined using
XML-like tags in the final prompt.

## Imperative vs Declarative

There are two ways to pass inputs to an agent:

| Mode | How it works |
|------|--------------|
| **Imperative** (kwargs) | Pass parameters directly via function arguments |
| **Declarative** (message_fields) | Agent extracts inputs from a `mf.Message` object |

The **declarative approach** with `message_fields` shines when designing complex systems: instead of manually wiring inputs and outputs between agents, you configure each agent once and let them consume/produce from a shared `Message`. This lets you **focus on system design** rather than plumbing variables between components. See [Declarative Mode with Message](#declarative-mode-with-message) for full details.

## Input Parameters

| Parameter | Description | Init | Runtime |
|-----------|-------------|:----:|:-------:|
| `task` | Main task input (string or dict for templates) | | ✅ |
| `task_context` | Dynamic task context passed at call time | | ✅ |
| `context_cache` | Fixed context stored in the agent | ✅ | |
| `task_multimodal` | Multimodal inputs (image, audio, file) | | ✅ |
| `messages` | Conversation history (ChatML format) | | ✅ |
| `vars` | Variables for Agent, Templates and Tools | | ✅ |

## How Task and Context are Combined

When you pass `task_context`, the context is injected **inside the task** using XML-like tags:

```xml
<context>
Company: FinData Analytics
Industry: FinTech
Product: AI-powered risk analysis
</context>

<task>
Create a pitch for this client
</task>
```

This structure helps the model clearly distinguish between background
information (task context) and what it needs to do (task).

???+ example

    === "Task Context Inputs"

        Pass task as first argument and task context via `task_context`:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class SalesAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"verbose": True}

        agent = SalesAgent()

        task_context = """
        Company: FinData Analytics
        Industry: FinTech
        Product: AI-powered risk analysis
        """

        params = agent.inspect_model_execution_params(
            "Create a pitch for this client", task_context=task_context
        )
        print(params)
        ```

    === "Context Cache"

        Use `context_cache` for context that doesn't change between calls:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class CompanyAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            context_cache = """
            Company values:
            - Customer first
            - Innovation
            - Integrity
            """
            config = {"verbose": True}

        agent = CompanyAgent()
        task_context = "Customer tier: enterprise\nIssue: invoice delay"
        params = agent.inspect_model_execution_params(
            "Write a response to a customer complaint", task_context=task_context
        )
        print(params)
        ```

    === "Same Agent, Two Modes"

        The same agent logic can be used imperatively or declaratively:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # Define agent with message_fields for declarative mode
        class Scraper(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [scrape_website]
            templates = {"task": "Summarize the news on this site: {}"}
            message_fields = {"task": "content"} # Where to read
            response_mode = "summary"                   # Where to write

        scraper = Scraper()

        # Imperative: Pass directly, get directly
        response = scraper("https://example.com/news")
        print(response)  # Direct string response

        # Declarative: Read from Message, write to Message (returns None)
        msg = mf.Message(content="https://example.com/news")
        scraper(msg)
        print(msg.summary)  # Response stored in message
        ```

        **When to choose declarative:**

        - Building pipelines where agents pass data to each other
        - Production systems where you want agents to be self-documenting
        - When you want to **design the system once** and let data flow through it

## Templates

Templates use **Jinja2** syntax to format inputs and outputs. There are three template types:

| Template | Purpose | Data Source |
|----------|---------|-------------|
| `task` | Format the task/question sent to the model | `task` dict + [vars](vars.md) |
| `task_context` | Format background context | `task_context` dict + [vars](vars.md) |
| `response` | Format the model's output before returning | Model output fields + [vars](vars.md) |

!!! tip "Response Template + Generation Schema"
    The `response` template is especially powerful when combined with `generation_schema`. The model outputs structured data, and you use Jinja to transform it into a human-readable format. This separates **what the model extracts** from **how you present it**.

???+ note "Template Examples"

    === "Task Template"

        Use `templates={"task": ...}` to format the task input:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            templates = {
                "task": """
                {% if user_name %}
                My name is {{ user_name }}.
                {% endif %}
                {{ user_input }}
                """
            }
            config = {"verbose": True}

        agent = Assistant()

        response = agent(
            task={"user_input": "Who was Nikola Tesla?"},
            vars={"user_name": "Bruce Wayne"}
        )
        ```

    === "Task Context Template"

        Use `templates={"task_context": ...}` to format structured task context:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class SalesAgent(nn.Module):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            templates = {
                "task_context": """
                The client is **{{ client_name }}** in the **{{ industry }}** sector.

                Challenges:
                {%- for pain in pain_points %}
                - {{ pain }}
                {%- endfor %}
                """
            }
            config = {"verbose": True}

        agent = SalesAgent()

        response = agent(
            "Create a pitch",
            task_context={
                "client_name": "EcoSupply Ltd.",
                "industry": "Sustainable packaging",
                "pain_points": ["High costs", "Certification needs"]
            }
        )
        ```

    === "Response Template (String)"

        For plain text outputs, use `{}` as placeholder for the model response:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            templates = {
                "response": """
                {% if user_name %}
                Hi {{ user_name }},
                {% endif %}
                {}
                """
            }
            config = {"verbose": True}

        agent = Assistant()

        response = agent(
            "Who was Nikola Tesla?",
            vars={"user_name": "Bruce Wayne"}
        )
        # Output: "Hi Bruce Wayne,\n\nNikola Tesla was a Serbian-American inventor..."
        ```

    === "Response Template (Structured)"

        When using `generation_schema`, access output fields directly in the template:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgspec import Struct
        from typing import Optional

        # mf.set_envs(OPENAI_API_KEY="...")

        class SafetyCheck(Struct):
            safe: bool
            answer: Optional[str]

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Only respond if the question is safe."
            generation_schema = SafetyCheck
            templates={
                "response": """
                {% if safe %}
                Hi! {{ answer }}
                {% else %}
                Sorry, I can't answer that question.
                {% endif %}
                """
            }
            message_fields = {"task": "content"}
            response_mode = "assistant.output"
            config = {"verbose": True}

        agent = Assistant()

        msg = mf.Message(content="Who was Nikola Tesla?")
        agent(msg)
        print(msg.get("assistant.output"))
        # Model outputs: {"safe": true, "answer": "Nikola Tesla was..."}
        # Template formats: "Hi! Nikola Tesla was..."
        ```

    === "Response + Extraction"

        Extract structured data and format a personalized response:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgspec import Struct

        # mf.set_envs(OPENAI_API_KEY="...")

        class ClientInfo(Struct):
            client_name: str
            company_name: str
            industry: str
            pain_points: list[str]

        class Extractor(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are an information extractor."
            instructions = "Extract information from the customer's message."
            generation_schema = ClientInfo
            templates = {
                "response": """
                Dear {{ client_name }},

                I understand that {{ company_name }} operates in {{ industry }}.

                Your main challenges:
                {%- for pain in pain_points %}
                - {{ pain }}
                {%- endfor %}

                Our solution addresses these exact pain points.

                Best regards,
                {{ seller }}.
                """
        }


        task = """
        Hello, my name is John and I work at EcoSupply Ltd.,
        a sustainable packaging company. We face high logistics
        costs and need ecological certifications.
        """

        agent = Assistant()

        response = agent(task, vars={"seller": "Hal Jordan"})
        # Model extracts: {"client_name": "John", "company_name": "EcoSupply Ltd.", ...}
        # Template produces personalized letter
        ```

    === "Response + Reasoning"

        Extract only `final_answer` from reasoning schemas like ReAct:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ReAct

        # mf.set_envs(OPENAI_API_KEY="...")

        def web_search(query: str) -> str:
            """Search the web for information."""
            # Simplified example - in production, use a real search API
            return f"Search results for '{query}': ..."

        class Researcher(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            tools = [web_search]
            generation_schema = ReAct
            templates = {
                "task": "Research this topic: {}",
                "response": "{{ final_answer }}"  # Only return the final answer
            }
            config = {"verbose": True}

        agent = Researcher()

        # ReAct outputs: {"current_step": {...}, "final_answer": "..."}
        # Template extracts just the final_answer
        response = agent("What is the population of Tokyo?")
        ```

    === "Signature + Response"

        Combine signature with response template for clean tool outputs:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from typing import Literal

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Classify(mf.Signature):
            """Classify sentiment of a given sentence."""
            sentence: str = mf.InputField()
            sentiment: Literal["positive", "negative", "neutral"] = mf.OutputField()
            confidence: float = mf.OutputField()

        @mf.tool_config(return_direct=True)
        class SentimentClassifier(nn.Agent):
            """Specialist in sentiment analysis."""
            model = model
            signature = Classify
            templates = {
                "response":
                """The sentence was classified as {{ sentiment }},
                with a confidence of {{ (confidence * 100) | round(2) }}%."""
            }
            config = {"verbose": True}

        class Assistant(nn.Agent):
            model = model
            tools = [SentimentClassifier]
            config = {"verbose": True}

        assistant = Assistant()
        response = assistant("Classify: 'This book was amazing!'")
        # Output: "The sentence was classified as positive, with a confidence of 92.5%."
        ```

!!! info "Task Template Without task"
    When you configure a `task` template but don't pass `task`, the rendered template itself becomes the task. This is useful for scenarios where the prompt is fixed and only some component changes (like images or [vars](vars.md)).

## Multimodal Inputs

Pass images, audio, or files via `task_multimodal`. Requires a multimodal model (e.g., `gpt-4.1`, `gpt-4.1-mini`).

| Media | Single | Multiple |
|-------|:------:|:--------:|
| Image | ✅ | ✅ |
| Audio | ✅ | ❌ |
| Video | ✅ | ❌ |
| File  | ✅ | ❌ |

!!! warning "Model Compatibility"
    Not all models support all multimodal inputs. Before using video, audio, or file inputs, verify that your chosen model supports that media type. Check the model provider's documentation for supported input types.


???+ note "Multimodal Examples"

    === "Image"

        Single image or multiple images for comparison:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class VisionAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        agent = VisionAgent()

        # Single image (URL)
        response = agent(
            "Describe this image",
            task_multimodal={
                "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
        )

        # Multiple images for comparison
        response = agent(
            "Compare these two images",
            task_multimodal={
                "image": [
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
                ]
            }
        )

        # With task template (fixed prompt, variable image)
        # When task is not passed, the rendered template becomes the task
        class DescribeAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            templates = {"task": "Describe this image in {{ language }}."}

        describe_agent = DescribeAgent()

        # Inspect what the model would receive
        params = describe_agent.inspect_model_execution_params(
            task_multimodal={
                "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
            vars={"language": "Portuguese"}
        )
        print(params["messages"])
        # The task will be: "Describe this image in Portuguese."

        # Execute the agent
        response = describe_agent(
            task_multimodal={
                "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
            vars={"language": "Portuguese"}
        )
        ```

    === "Image (Message Fields)"

        Use `message_fields` for declarative mapping from Message objects:

        !!! info "Graceful Handling of Missing Fields"
            If an image cannot be retrieved from the declarative mapping (e.g., the field doesn't exist or is empty), the task will still be assembled normally — the missing image simply won't be included. This allows flexible pipelines where multimodal inputs are optional.

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # Single image mapping
        class VisionAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            message_fields = {
                "task": "user.query",
                "task_multimodal": {"image": "user.image_url"}
            }

        agent = VisionAgent()

        # Create message with structured data
        msg = mf.Message()
        msg.set("user.query", "What objects are in this image?")
        msg.set("user.image_url", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")

        response = agent(msg)

        # Multiple images mapping
        class ComparisonAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            message_fields = {
                "task": "user.query",
                "task_multimodal": {"image": "user.images"}
            }

        comparison_agent = ComparisonAgent()

        msg = mf.Message()
        msg.set("user.query", "Compare these two images and describe the differences.")
        msg.set("user.images", [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
        ])

        response = comparison_agent(msg)
        ```

    === "Audio"

        Transcribe or analyze audio files:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class AgentTranscriber(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-audio-preview")

        agent = AgentTranscriber()

        response = agent(
            "Transcribe this audio and identify the speaker's emotion",
            task_multimodal={"audio": "/path/to/recording.wav"}
        )
        ```

    === "File (PDF)"

        Analyze PDF documents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class AnalyzerAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")

        agent = AnalyzerAgent()

        # From URL
        response = agent(
            "Summarize the key findings of this paper",
            task_multimodal={"file": "https://arxiv.org/pdf/1706.03762.pdf"}
        )

        # From local file
        response = agent(
            "Extract the main conclusions",
            task_multimodal={"file": "./report.pdf"}
        )
        ```


### Customizing Multimodal Blocks

Use `image_block_kwargs` and `video_block_kwargs` in `config` to pass extra parameters directly to the underlying multimodal block. This is useful, for example, to control the image **detail level** supported by OpenAI models:

| Value | Behavior |
|-------|----------|
| `"auto"` | Model decides (default) |
| `"low"` | Fast, low-resolution analysis (512×512 px) |
| `"high"` | High-fidelity analysis, higher token cost |
| `"original"` | Preserves the original resolution — recommended for spatially-sensitive tasks (e.g., click-accuracy with gpt-4.1) |

???+ example "Image Detail Level"

    === "image_block_kwargs"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class VisionAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")
            config = {
                "image_block_kwargs": {"detail": "original"}
            }

        agent = VisionAgent()

        response = agent(
            "Identify which UI element the cursor is closest to",
            task_multimodal={
                "image": "https://example.com/screenshot.png"
            }
        )
        ```

    === "video_block_kwargs"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class VideoAgent(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")
            config = {
                "video_block_kwargs": {"fps": 1}
            }

        agent = VideoAgent()

        response = agent(
            "Describe what happens in this video",
            task_multimodal={
                "video": "./recording.mp4"
            }
        )
        ```

## Messages (Chat History)

Pass a list of messages in ChatML format to provide conversation history. The `messages` parameter has explicit opt-in semantics:

| Value | Behavior |
|-------|----------|
| Not passed (default) | Ephemeral — no side effects on external state |
| `[]` (empty list) | Accumulator — user input and tool calls are appended in-place |
| `[...]` (existing list) | Continue — extends the existing history |

The final assistant response is **never added automatically** — append it manually with `mf.ChatBlock.assist(response)` after each turn.

???+ note "Chat History Examples"

    === "Accumulator Pattern"

        Pass `messages=[]` once and let the agent accumulate history in-place.
        Only append the assistant reply manually after each turn:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Advisor(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a helpful camera advisor."

        agent = Advisor()
        history = []

        # Turn 1
        response = agent("I'm looking for a compact camera under $500.", messages=history)
        print(f"Assistant: {response}")
        history.append(mf.ChatBlock.assist(response))
        # history: [user, assistant]

        # Turn 2 — user input is added automatically, just append the reply
        response = agent("Which one has better low-light performance?", messages=history)
        print(f"Assistant: {response}")
        history.append(mf.ChatBlock.assist(response))

        # Turn 3
        response = agent("What about battery life?", messages=history)
        print(f"Assistant: {response}")
        ```

    === "return_messages"

        Use `config={"return_messages": True}` when you need the full internal
        message list returned alongside the response (e.g., for logging or inspection):

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Assistant(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            config = {"return_messages": True}

        agent = Assistant()

        # First turn — no history yet
        result = agent("Hi, my name is Peter Parker, and I'm a photographer.")
        print(result.response)

        # result.messages has user input (+ tool calls if any), but not the reply
        # Append it manually before the next turn
        messages = result.messages + [mf.ChatBlock.assist(result.response)]

        result = agent(
            "Can you recommend some cameras for my freelance job?",
            messages=messages
        )
        print(result.response)
        # The agent remembers you're Peter Parker and a photographer
        ```

    === "ChatBot Pattern"

        Complete chatbot loop with streaming. `messages=None` on the first turn
        means ephemeral — no list is needed until history starts accumulating:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class ChatBot(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            system_message = "You are a helpful assistant."
            config = {"stream": True, "return_messages": True}

        agent = ChatBot()
        messages = None  # ephemeral on first turn

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break

            result = agent(user_input, messages=messages)

            full_response = ""
            print("Assistant: ", end="", flush=True)
            for chunk in result.consume():
                print(chunk, end="", flush=True)
                full_response += chunk
            print()

            # Build history for the next turn (include the assistant reply)
            messages = result.messages + [mf.ChatBlock.assist(full_response)]
        ```

## Declarative Mode with Message

In declarative mode, you configure the agent once with `message_fields` and `response_mode`, then pass a `mf.Message` object. The agent reads its inputs from the Message and writes its output back into it — no manual wiring needed.

```python
class MyAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    message_fields = {
        "task": "user.query",                 # Read task from msg.user.query
        "task_context": "context.background"  # Read context from msg.context.background
    }
    response_mode = "agent.output"              # Write response to msg.agent.output
```

### Path Formats

| Path type | Syntax | Behavior |
|-----------|--------|----------|
| Simple field | `"query"` | Reads the top-level `query` field |
| Nested field | `"user.question"` | Reads `msg.user.question` via dot notation |
| **OR inputs** | `("path1", "path2", ...)` | Returns the **first non-None value** found |

### OR Inputs — Flexible Path Resolution

When a path is defined as a **tuple**, the agent tries each path in order and uses the first one that returns a non-None value. This is the **OR inputs** pattern.

It is especially powerful in dynamic pipelines where the same agent may receive input from different upstream steps — for example, when an `inline` function may or may not have produced a refined output:

```python
class QAAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    message_fields = {
        "task": ("refined.question", "user.question", "user.raw_input")
    }
    response_mode = "qa.answer"
```

If `refined.question` is present in the message, it is used. Otherwise the agent falls back to `user.question`, and then to `user.raw_input`.

OR inputs also work inside dict-valued fields (e.g. `task_multimodal`):

```python
class VisionAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    message_fields = {
        "task": ("user.query", "user.raw_input"),        # OR on task
        "task_multimodal": {
            "image": ("user.photo_url", "user.image")           # OR on image
        }
    }
    response_mode = "vision.result"
```

???+ example "OR Inputs in a Pipeline"

    A `Refiner` agent is an optional step that rewrites the user's question. The `Answerer` uses OR inputs so it works correctly whether or not the refiner ran:

    ```python
    # pip install msgflux[openai]
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    class Refiner(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        instructions = "Rewrite the question to be clearer and more specific."
        message_fields = {"task": "user.question"}
        response_mode = "refined.question"

    class Answerer(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        message_fields = {
            # Use refined question if available, fall back to original
            "task": ("refined.question", "user.question")
        }
        response_mode = "answer"

    refiner = Refiner()
    answerer = Answerer()

    # With refinement
    msg_a = mf.Message()
    msg_a.set("user.question", "how do i make it go faster?")
    refiner(msg_a)   # writes to msg_a.refined.question
    answerer(msg_a)  # uses refined.question
    print(msg_a.answer)

    # Without refinement — answerer falls back to user.question
    msg_b = mf.Message()
    msg_b.set("user.question", "What is the capital of France?")
    answerer(msg_b)  # uses user.question directly
    print(msg_b.answer)
    ```
