# Signatures

A **Signature** is a declarative specification of input/output behavior for an Agent. Instead of hand-crafting prompts, you define the semantic roles of inputs and outputs, and msgFlux handles the prompt engineering for you.

This [DSPy-inspired](https://dspy.ai/learn/programming/signatures/) feature automatically generates:

- **System prompt** with task description (from docstring)
- **Task template** with input placeholders
- **Generation schema** for structured output
- **Annotations** for agent-as-tool integration

### Why Use Signatures?

| Without Signature | With Signature |
|-------------------|----------------|
| Manual prompt engineering | Declarative task specification |
| String manipulation for inputs | Type-safe input/output fields |
| Ad-hoc output parsing | Automatic structured responses |
| Manual tool schema definition | Auto-generated annotations |

Signatures let you focus on **what** the task should accomplish, not **how** to prompt the model.

### Inline Signatures

For simple tasks, use the shorthand string notation with arrow syntax:

```python
"input_field: type -> output_field: type"
```

The default type is `str` when unspecified.

???+ example "Inline Signature Examples"

    === "Translation"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Translator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "english -> portuguese"

        agent = Translator()
        
        print(agent.task_template)

        response = agent(english="hello world")
        print(response.portuguese)  # "olá mundo"        
        ```

    === "Sentiment Classification"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # With types: specify field types explicitly
        class Extractor(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "text -> sentiment: Literal['positive', 'negative', 'neutral'], confidence: float"

        agent = Extractor()

        print(agent.task_template)
        
        result = agent(text="I love this product!")
        print(result.sentiment)    # "positive"
        print(result.confidence)   # 0.95
        ```

    === "Calculator"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        # Multiple inputs
        class Calculator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "expression: str, precision: int -> result: float"

        agent = Calculator()

        print(agent.task_template)

        result = agent(expression="sqrt(2)", precision=4)
        print(result.result)  # 1.4142     
        ```

### Class-Based Signatures

For complex tasks, class-based signatures provide full control with typed fields, descriptions, and docstrings. The class docstring becomes the instruction for the model.

```python
class TaskName(mf.Signature):
    """Task description that guides the model."""

    input_field: type = mf.InputField(desc="Field description")
    output_field: type = mf.OutputField(desc="Field description")
```

The docstrings of a class-based Signature become the Agent instructions.

???+ example "Class Signature Examples"

    === "Basic Classification"

        ```python
        # pip install msgflux[openai]
        from typing import Literal
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Classify(mf.Signature):
            """Classify the sentiment of a given sentence."""

            sentence: str = mf.InputField()
            sentiment: Literal["positive", "negative", "neutral"] = mf.OutputField()
            confidence: float = mf.OutputField(desc="Confidence score between 0 and 1")

        class Classifier(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Classify

        agent = Classifier()

        print(agent.task_template)

        response = agent(sentence="This book was super fun to read!")
        print(response.sentiment)    # "positive"
        print(response.confidence)   # 0.92     
        ```

    === "Complex Extraction"

        ```python
        # pip install msgflux[openai]
        from typing import List, Optional
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class ExtractEntities(mf.Signature):
            """Extract named entities from text with their types and context."""

            text: str = mf.InputField(desc="Text to analyze")
            entities: List[dict] = mf.OutputField(
                desc="List of {name, type, context} objects"
            )
            summary: str = mf.OutputField(desc="Brief summary of the text")

        class EntityExtractor(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = ExtractEntities

        agent = EntityExtractor()

        print(agent.task_template)

        result = agent(text="Apple CEO Tim Cook announced new products in Cupertino.")
        print(result)
        # result.entities = [
        #     {"name": "Apple", "type": "ORG", "context": "technology company"},
        #     {"name": "Tim Cook", "type": "PERSON", "context": "CEO of Apple"},
        #     {"name": "Cupertino", "type": "LOC", "context": "city in California"}
        # ] 
        ```

    === "With Detailed Instructions"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Translate(mf.Signature):
            """Translate text accurately while preserving meaning, tone, and cultural nuances.

            Guidelines:
            - Maintain the original tone (formal/informal)
            - Preserve idiomatic expressions when possible
            - Adapt cultural references appropriately
            """

            text: str = mf.InputField(desc="Text to translate")
            source_language: str = mf.InputField(desc="Source language code (e.g., 'en', 'pt')")
            target_language: str = mf.InputField(desc="Target language code")
            translation: str = mf.OutputField(desc="Translated text")
            notes: str = mf.OutputField(desc="Translation notes about cultural adaptations")

        class Translator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Translate

        agent = Translator()

        print(agent.task_template)

        result = agent(
            text="It's raining cats and dogs!",
            source_language="en",
            target_language="pt"
        )
        print(result.translation)  # "Está chovendo canivetes!"
        print(result.notes)        # "Adapted English idiom to Brazilian Portuguese equivalent"
        ```

### Field Types

Signatures support various field types for different use cases:

| Type | Description | Example |
|------|-------------|---------|
| `str` | Text (default) | `name: str` |
| `int`, `float` | Numbers | `count: int`, `score: float` |
| `bool` | Boolean | `is_valid: bool` |
| `Literal[...]` | Constrained choices | `sentiment: Literal["pos", "neg"]` |
| `dict[str, T]` | Dictionaries | `metadata: dict[str, float]` |
| `List[T]` | Lists | `tags: List[str]` |
| `Image` | Image input | `photo: Image` |
| `Audio` | Audio input | `recording: Audio` |
| `Video` | Video input | `clip: Video` |
| `File` | File input | `document: File` |

### InputField and OutputField

Both `InputField` and `OutputField` accept a `desc` (or `description`) parameter to provide additional context:

```python
import msgflux as mf

class Review(mf.Signature):
    """Analyze a product review."""

    review_text: str = mf.InputField(desc="The customer review text")
    product_name: str = mf.InputField(desc="Name of the product being reviewed")

    rating: int = mf.OutputField(desc="Rating from 1 to 5 stars")
    pros: List[str] = mf.OutputField(desc="List of positive aspects mentioned")
    cons: List[str] = mf.OutputField(desc="List of negative aspects mentioned")
```

!!! tip "Field Descriptions"
    Use `desc` to clarify ambiguous fields, specify constraints (e.g., "between 0 and 1"), or provide examples. This helps the model understand exactly what you expect.

### Multimodal Signatures

Use `Image`, `Audio`, `Video`, or `File` for multimodal inputs:

???+ example "Multimodal Signature"

    === "Class-based"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class ImageClassifier(mf.Signature):
            """Classify the content of an image and describe what you see."""

            photo: mf.Image = mf.InputField(desc="Image to analyze")
            label: str = mf.OutputField(desc="Main subject of the image")
            description: str = mf.OutputField(desc="Detailed description")
            confidence: float = mf.OutputField(desc="Confidence score 0-1")

        class Classifier(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = ImageClassifier

        agent = Classifier()

        # Task template automatically includes image placeholder
        print(agent.task_template)

        response = agent(task_multimodal={
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        })
        print(response.label)        # "Nature boardwalk"
        print(response.description)  # "A wooden boardwalk path..."
        ```

    === "Str-based"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Classifier(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1")
            instructions = "Classify the content of an image and describe what you see."
            signature = "photo: Image -> label, description, confidence: float"

        agent = Classifier()

        # Task template automatically includes image placeholder
        print(agent.task_template)

        response = agent(task_multimodal={
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        })
        print(response.label)        # "Nature boardwalk"
        print(response.description)  # "A wooden boardwalk path..."
        ```


### Passing Inputs

When using signatures, you can pass inputs in multiple ways:

???+ note "Input Methods"

    === "As Kwargs"

        Pass inputs as keyword arguments (recommended):

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Translator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "english -> portuguese"

        agent = Translator()
        response = agent(english="hello world")
        ```

    === "As Dict"

        Pass all inputs as a dictionary:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Translator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = "english -> portuguese"

        agent = Translator()
        response = agent({"english": "hello world"})
        ```

    === "With Task Context"

        Combine with `task_context`:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Summarize(mf.Signature):
            """Summarize text in a specific style."""
            text: str = mf.InputField()
            style: str = mf.InputField(desc="e.g., 'formal', 'casual', 'technical'")
            summary: str = mf.OutputField()

        class Summarizer(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Summarize

        agent = Summarizer()
        response = agent(
            text="Long article...",
            style="casual",
            task_context="Focus on the key takeaways"
        )
        ```

### Combining Signatures with Other Components

Signatures can be combined with other Agent components. Here's how they interact:

#### What You Can Combine

| Component | Behavior with Signature |
|-----------|------------------------|
| `system_message` | **Additive** - Included in the system prompt alongside signature-generated content |
| `instructions` | **Override** - If provided, takes precedence over the signature's docstring |
| `examples` | **Additive** - Combined with any examples defined in the signature |
| `system_extra_message` | **Additive** - Appended to the system prompt |
| `generation_schema` | **Fused** - Merged with signature outputs (e.g., ChainOfThought + Signature) |

#### What the Signature Controls

| Component | Behavior |
|-----------|----------|
| `task` template | **Generated** - Created from input fields, overwrites any existing task template |
| `expected_output` | **Generated** - Created from output fields |
| `annotations` | **Generated** - Created from input fields for tool integration |

???+ example "Combining Signature with System Components"

    === "With system_message"

        Add context that applies to all requests:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Translate(mf.Signature):
            """Translate text accurately."""
            text: str = mf.InputField()
            target_lang: str = mf.InputField()
            translation: str = mf.OutputField()

        class Translator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Translate
            system_message = "You are a professional translator specialized in technical documents."

        agent = Translator()
        print(agent.get_system_prompt())
        ```

    === "With instructions (Override)"

        Override the signature's docstring:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Summarize(mf.Signature):
            """Summarize the given text."""  # This will be ignored
            text: str = mf.InputField()
            summary: str = mf.OutputField()

        class Summarizer(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Summarize
            instructions = "Create a bullet-point summary with exactly 3 key points."

        agent = Summarizer()
        print(agent.get_system_prompt())
        ```

    === "With examples"

        Combine examples from multiple sources:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        class Classify(mf.Signature):
            """Classify sentiment."""
            text: str = mf.InputField()
            sentiment: str = mf.OutputField()

        class Classifier(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Classify
            # These examples are *combined* with any examples in the signature
            examples = [
                mf.Example(
                    inputs={"text": "I love it!"},
                    outputs={"sentiment": "positive"}
                ),
                mf.Example(
                    inputs={"text": "Terrible product."},
                    outputs={"sentiment": "negative"}
                ),
            ]

        print(agent.get_system_prompt())
        ```

    === "With generation_schema"

        Fuse reasoning strategies with typed outputs:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ChainOfThought

        # mf.set_envs(OPENAI_API_KEY="...")

        class Calculate(mf.Signature):
            """Solve the math problem."""
            problem: str = mf.InputField()
            answer: float = mf.OutputField()

        class Calculator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            signature = Calculate
            generation_schema = ChainOfThought

        agent = Calculator()
        response = agent(problem="What is 15% of 80?")
        print(response)
        ```

### Signatures as Tools

When an agent has a signature, its annotations are automatically configured based on the input fields. This makes it **ready to be used as a tool** with properly typed parameters.

???+ example "Signature-Based Agent as Tool"

    ```python
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    # Without signature: default annotation is "task: str"
    class BasicAgent(nn.Agent):
        model = model

    print(BasicAgent().annotations)  # {"task": str}

    # With signature: annotations match the input fields
    class AnalyzeSentiment(mf.Signature):
        """Analyze the sentiment of text."""
        text: str = mf.InputField(desc="Text to analyze")
        sentiment: str = mf.OutputField()
        score: float = mf.OutputField(desc="Score from -1 to 1")

    class SentimentAnalyzer(nn.Agent):
        model = model
        signature = AnalyzeSentiment

    print(SentimentAnalyzer().annotations)  # {"text": str}

    # Use as a tool - the coordinator sees: analyze_sentiment(text: str)
    class Coordinator(nn.Agent):
        model = model
        tools = [SentimentAnalyzer]
        system_message = "You help analyze customer feedback."
        config = {"verbose": True}

    coordinator = Coordinator()
    response = coordinator("What's the sentiment of 'I absolutely love this product!'")
    print(response)
    ```

!!! tip "Best Practices"
    - **Start simple**: Begin with inline signatures and evolve to class-based as needed
    - **Be semantic**: Choose clear, meaningful field names (e.g., `question` not `q`)
    - **Use descriptions**: Add `desc` for ambiguous fields or specific constraints
    - **Docstrings matter**: The class docstring becomes the model's instruction
    - **Trust the system**: Avoid over-engineering prompts in descriptions
