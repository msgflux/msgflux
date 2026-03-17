# Query Router with Signatures

Inspired by the [DSPy Deep Research tutorial](https://www.cmpnd.ai/blog/learn-dspy-deep-research.html), this guide shows how to apply the same declarative style in msgFlux — defining *what* each module does via `Signature`, and composing a router that dispatches queries to different specialists.

## What You'll Build

```
User Query
    │
    ▼
QueryClassifier ── Signature: query → topic, complexity, keywords
    │
    ├── "technical"  ──► TechnicalExpert  (answer + code example + references)
    ├── "business"   ──► BusinessAnalyst  (analysis + key points + recommendation)
    ├── "creative"   ──► CreativeAdvisor  (ideas + suggestions + inspiration)
    └── "general"    ──► GeneralAssistant (answer + follow-up questions)
                │
                ▼
         Final response on msg
```

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — String Signature (compact form)

The fastest way to declare an input/output contract is as a string directly on the `Agent`. The model will populate each output field with the correct type.

```python
import msgflux as mf
import msgflux.nn as nn
from typing import Literal

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class QuickClassifier(nn.Agent):
    """Classifies the user's query."""
    model = model
    signature = "query -> topic: str, complexity: Literal['simple', 'complex']"


classifier = QuickClassifier()
result = classifier("How does backpropagation work in neural networks?")
print(result)
# {'topic': 'technical', 'complexity': 'complex'}
```

This is the msgFlux equivalent of `dspy.Predict("query -> topic, complexity")` — declarative, no manual prompt engineering.

---

## Step 2 — Class-based Signature (rich form)

For more control, use the `Signature` base class with `InputField` and `OutputField`. Each field can carry a description that guides the model:

```python
from msgflux import Signature, InputField, OutputField


class QueryClassification(Signature):
    """Classify the query to route it to the best specialist."""

    query: str = InputField(desc="The user's question")

    topic: Literal["technical", "business", "creative", "general"] = OutputField(
        desc="The primary domain of the query"
    )
    complexity: Literal["simple", "complex"] = OutputField(
        desc="'simple' for direct answers, 'complex' for deep research"
    )
    keywords: list[str] = OutputField(
        desc="Key terms that identify the domain"
    )
```

Pass the class to `Agent` via the `signature` attribute:

```python
class QueryClassifier(nn.Agent):
    """Classifies and routes queries."""
    model = model
    signature = QueryClassification
    config = {"verbose": True}


classifier = QueryClassifier()
result = classifier("What pricing strategy should I use for a B2B SaaS product?")
print(result)
# {
#   'topic': 'business',
#   'complexity': 'complex',
#   'keywords': ['SaaS', 'B2B', 'pricing', 'strategy']
# }
```

!!! tip
    `config = {"verbose": True}` prints each model call, reasoning steps, and raw
    responses to the console — useful for understanding what the pipeline is doing at
    every stage.

---

## Step 3 — Specialists with Descriptive Signatures

Each specialist declares its own contract via `Signature`. Field descriptions steer the model toward well-structured, purpose-built outputs:

```python
from typing import Optional


# ── Technical Expert ─────────────────────────────────────────────────────────

class TechnicalAnswer(Signature):
    """Answer technical questions clearly with practical examples."""

    query: str = InputField(desc="The technical question")

    answer: str = OutputField(desc="Clear and accurate explanation")
    code_example: Optional[str] = OutputField(
        desc="Illustrative code snippet, if applicable"
    )
    references: list[str] = OutputField(
        desc="Relevant documentation or resources"
    )


class TechnicalExpert(nn.Agent):
    model = model
    signature = TechnicalAnswer
    config = {"verbose": True}


# ── Business Analyst ─────────────────────────────────────────────────────────

class BusinessAnswer(Signature):
    """Analyze business questions with strategic perspective."""

    query: str = InputField(desc="The business question")

    analysis: str = OutputField(desc="Situation analysis")
    key_points: list[str] = OutputField(desc="Key points to consider")
    recommendation: str = OutputField(desc="Practical, actionable recommendation")


class BusinessAnalyst(nn.Agent):
    model = model
    signature = BusinessAnswer
    config = {"verbose": True}


# ── Creative Advisor ─────────────────────────────────────────────────────────

class CreativeAnswer(Signature):
    """Generate creative and inspiring ideas."""

    query: str = InputField(desc="The creative challenge or question")

    ideas: list[str] = OutputField(desc="3 to 5 original ideas")
    suggestions: list[str] = OutputField(desc="Tips to develop the ideas further")
    inspiration: str = OutputField(desc="An inspiring phrase or concept")


class CreativeAdvisor(nn.Agent):
    model = model
    signature = CreativeAnswer
    config = {"verbose": True}


# ── General Assistant ─────────────────────────────────────────────────────────

class GeneralAnswer(Signature):
    """Answer general questions thoroughly."""

    query: str = InputField(desc="The user's question")

    answer: str = OutputField(desc="Direct and informative response")
    follow_up_questions: list[str] = OutputField(
        desc="Questions the user might want to explore next"
    )


class GeneralAssistant(nn.Agent):
    model = model
    signature = GeneralAnswer
    config = {"verbose": True}
```

---

## Step 4 — Composing the Router

`QueryRouter` orchestrates the flow: it classifies the query and dispatches to the matching specialist via a `ModuleDict`:

```python
from msgflux import Message


class QueryRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = QueryClassifier()
        self.experts = nn.ModuleDict({
            "technical": TechnicalExpert(),
            "business":  BusinessAnalyst(),
            "creative":  CreativeAdvisor(),
            "general":   GeneralAssistant(),
        })

    def forward(self, msg):
        # 1. Classify the query
        self.classifier(msg)
        topic = msg.topic

        # 2. Dispatch to the right specialist
        expert = self.experts.get(topic, self.experts["general"])
        expert(msg)

        return msg
```

Usage:

```python
router = QueryRouter()

msg = Message(query="How do I use backpropagation with PyTorch?")
router(msg)

print(f"Topic:      {msg.topic}")
print(f"Complexity: {msg.complexity}")
print(f"Answer:     {msg.answer}")
if msg.get("code_example"):
    print(f"Code:\n{msg.code_example}")
```

---

## Step 5 — Verbose in Action

With `config = {"verbose": True}` on each agent, the console shows the pipeline in real time:

```
[query_classifier][call_model]
[query_classifier][response] {'topic': 'technical', 'complexity': 'complex', ...}

[technical_expert][call_model]
[technical_expert][response] {'answer': '...', 'code_example': '...', ...}
```

To silence it in production, remove the flag or set it to `False`:

```python
class TechnicalExpert(nn.Agent):
    model = model
    signature = TechnicalAnswer
    config = {"verbose": False}
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from typing import Literal, Optional

# ── Model ─────────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


# ── Signatures ────────────────────────────────────────────────────────────────

class QueryClassification(Signature):
    """Classify the query to route it to the best specialist."""

    query: str = InputField(desc="The user's question")

    topic: Literal["technical", "business", "creative", "general"] = OutputField(
        desc="The primary domain of the query"
    )
    complexity: Literal["simple", "complex"] = OutputField(
        desc="'simple' for direct answers, 'complex' for deep research"
    )
    keywords: list[str] = OutputField(desc="Key terms that identify the domain")


class TechnicalAnswer(Signature):
    """Answer technical questions clearly with practical examples."""

    query: str = InputField(desc="The technical question")
    answer: str = OutputField(desc="Clear and accurate explanation")
    code_example: Optional[str] = OutputField(
        desc="Illustrative code snippet, if applicable"
    )
    references: list[str] = OutputField(desc="Relevant documentation or resources")


class BusinessAnswer(Signature):
    """Analyze business questions with strategic perspective."""

    query: str = InputField(desc="The business question")
    analysis: str = OutputField(desc="Situation analysis")
    key_points: list[str] = OutputField(desc="Key points to consider")
    recommendation: str = OutputField(desc="Practical, actionable recommendation")


class CreativeAnswer(Signature):
    """Generate creative and inspiring ideas."""

    query: str = InputField(desc="The creative challenge or question")
    ideas: list[str] = OutputField(desc="3 to 5 original ideas")
    suggestions: list[str] = OutputField(desc="Tips to develop the ideas further")
    inspiration: str = OutputField(desc="An inspiring phrase or concept")


class GeneralAnswer(Signature):
    """Answer general questions thoroughly."""

    query: str = InputField(desc="The user's question")
    answer: str = OutputField(desc="Direct and informative response")
    follow_up_questions: list[str] = OutputField(
        desc="Questions the user might want to explore next"
    )


# ── Agents ────────────────────────────────────────────────────────────────────

class QueryClassifier(nn.Agent):
    model = model
    signature = QueryClassification
    config = {"verbose": True}


class TechnicalExpert(nn.Agent):
    model = model
    signature = TechnicalAnswer
    config = {"verbose": True}


class BusinessAnalyst(nn.Agent):
    model = model
    signature = BusinessAnswer
    config = {"verbose": True}


class CreativeAdvisor(nn.Agent):
    model = model
    signature = CreativeAnswer
    config = {"verbose": True}


class GeneralAssistant(nn.Agent):
    model = model
    signature = GeneralAnswer
    config = {"verbose": True}


# ── Router ────────────────────────────────────────────────────────────────────

class QueryRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = QueryClassifier()
        self.experts = nn.ModuleDict({
            "technical": TechnicalExpert(),
            "business":  BusinessAnalyst(),
            "creative":  CreativeAdvisor(),
            "general":   GeneralAssistant(),
        })

    def forward(self, msg):
        self.classifier(msg)
        expert = self.experts.get(msg.topic, self.experts["general"])
        expert(msg)
        return msg


# ── Run ───────────────────────────────────────────────────────────────────────

router = QueryRouter()

queries = [
    "How do I implement an LRU cache in Python?",
    "What pricing model should I use for a freemium product?",
    "Give me ideas for a mental wellness app.",
    "What is the Kyoto Protocol?",
]

for query in queries:
    msg = Message(query=query)
    router(msg)

    print(f"\n{'─' * 60}")
    print(f"Query:    {msg.query}")
    print(f"Topic:    {msg.topic} ({msg.complexity})")
    print(f"Keywords: {msg.keywords}")

    for field in ("answer", "analysis", "ideas"):
        if msg.get(field):
            print(f"\n{field.capitalize()}:\n{msg[field]}")
            break
```

---

## Async Version

To process multiple queries in parallel:

```python
import asyncio
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message

# ... (definitions above) ...


class AsyncQueryRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = QueryClassifier()
        self.experts = nn.ModuleDict({
            "technical": TechnicalExpert(),
            "business":  BusinessAnalyst(),
            "creative":  CreativeAdvisor(),
            "general":   GeneralAssistant(),
        })

    async def aforward(self, msg):
        await self.classifier.acall(msg)
        expert = self.experts.get(msg.topic, self.experts["general"])
        await expert.acall(msg)
        return msg


async def main():
    router = AsyncQueryRouter()

    queries = [
        "How does Python's garbage collector work?",
        "How do I scale a startup from 10 to 100 employees?",
        "Ideas to gamify a language-learning app.",
    ]

    messages = [Message(query=q) for q in queries]

    results = await F.ascatter_gather(
        [router.acall] * len(messages),
        args_list=[(msg,) for msg in messages],
    )

    for msg in results:
        print(f"\n[{msg.topic}] {msg.query}")


asyncio.run(main())
```

---

## Next Steps

- **Add tools**: Combine `Signature` with `tools = [search_web, ...]` to build specialists that fetch live information (ReAct pattern).
- **Chain specialists**: Use `Inline` for multi-stage pipelines, e.g. `classifier -> expert -> summarizer`.
- **Inspect signatures**: `TechnicalAnswer.get_str_signature()` returns the signature string — useful for logging and auditing.
