# Query Router with Signatures

This guide shows how to build a query router in msgFlux using `Signature` to define typed input/output contracts per specialist, so each agent knows exactly what it receives and what it must return.

---

## The Problem

Here is the system most teams start with.

```
User Query
    │
    ▼
┌──────────────────────────────────────────┐
│           GeneralAssistant               │
│                                          │
│  what kind of question?  ←──→  answer   │
└──────────────────────────────────────────┘
         │ improvises the output shape
         ▼
  {"response": "..."} or {"answer": "..."} or just a string
```

- The model receives every query regardless of domain — technical, business, creative, general.
- Output shape is inconsistent. The model decides whether to return `answer`, `analysis`, `ideas`, or something else entirely.
- You cannot enforce structure per domain without bloating a single prompt with conditional instructions.
- You cannot add domain-specific guidance without it leaking into unrelated queries.
- There is no contract. You parse whatever the model decides to return.

You are pattern-matching on a guess.

---

## The Plan

We will build a `QueryRouter` that separates *classification* from *response*.

A typed `Signature` drives the classifier: it receives the user query and emits `topic`, `complexity`, and `keywords` as structured fields — no free-form parsing. Each topic maps to a specialist agent with its own `Signature` that constrains both the reasoning and the response shape. `TechnicalExpert` always returns `answer`, `code_example`, and `references`. `BusinessAnalyst` always returns `analysis`, `key_points`, and `recommendation`. The output is typed and predictable by construction.

---

## Architecture

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

The routing layer is:

- **Typed** — every field is declared; no free-form output parsing
- **Composable** — each specialist is an independent `Agent` you can test in isolation
- **Extensible** — add a new topic by adding one `Signature`, one `Agent`, and one entry in `ModuleDict`

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 — Classifying the Query

Before dispatching to a specialist, the router needs to know what kind of question arrived. The classifier reads the raw query and produces two decisions: which domain it belongs to (`topic`) and how much depth the answer requires (`complexity`).

```python
import msgflux as mf
import msgflux.nn as nn

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

---

## Step 2 — Expanding the Classification Contract

The compact string form is enough to get a decision, but the router also needs `keywords` — the terms that identify the domain and can be passed downstream to specialists. `QueryClassification` makes those three outputs explicit and adds per-field descriptions that sharpen the model's judgment for each one:

```python
class QueryClassification(mf.Signature):
    """Classify the query to route it to the best specialist."""

    query: str = mf.InputField(desc="The user's question")

    topic: Literal["technical", "business", "creative", "general"] = mf.OutputField(
        desc="The primary domain of the query"
    )
    complexity: Literal["simple", "complex"] = mf.OutputField(
        desc="'simple' for direct answers, 'complex' for deep research"
    )
    keywords: list[str] = mf.OutputField(
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

## Step 3 — Defining the Specialists

Each topic maps to a dedicated agent with its own output contract. Because the `Signature` is declared per specialist, the response shape is fixed — `TechnicalExpert` always returns `answer`, `code_example`, and `references`; `BusinessAnalyst` always returns `analysis`, `key_points`, and `recommendation`. There is no ambiguity about what to expect from each one.

**Technical** — answers a technical question with a code example and pointers to documentation:

```python
from typing import Optional

class TechnicalAnswer(mf.Signature):
    """Answer technical questions clearly with practical examples."""

    query: str = mf.InputField(desc="The technical question")

    answer: str = mf.OutputField(desc="Clear and accurate explanation")
    code_example: Optional[str] = mf.OutputField(desc="Illustrative code snippet, if applicable")
    references: list[str] = mf.OutputField(desc="Relevant documentation or resources")


class TechnicalExpert(nn.Agent):
    model = model
    signature = TechnicalAnswer
    config = {"verbose": True}
```

**Business** — breaks down a strategic question into situation analysis, key considerations, and a concrete recommendation:

```python
class BusinessAnswer(mf.Signature):
    """Analyze business questions with strategic perspective."""

    query: str = mf.InputField(desc="The business question")

    analysis: str = mf.OutputField(desc="Situation analysis")
    key_points: list[str] = mf.OutputField(desc="Key points to consider")
    recommendation: str = mf.OutputField(desc="Practical, actionable recommendation")


class BusinessAnalyst(nn.Agent):
    model = model
    signature = BusinessAnswer
    config = {"verbose": True}
```

**Creative** — generates original ideas and development tips for open-ended challenges:

```python
class CreativeAnswer(mf.Signature):
    """Generate creative and inspiring ideas."""

    query: str = mf.InputField(desc="The creative challenge or question")

    ideas: list[str] = mf.OutputField(desc="3 to 5 original ideas")
    suggestions: list[str] = mf.OutputField(desc="Tips to develop the ideas further")
    inspiration: str = mf.OutputField(desc="An inspiring phrase or concept")


class CreativeAdvisor(nn.Agent):
    model = model
    signature = CreativeAnswer
    config = {"verbose": True}
```

**General** — handles everything else with a direct answer and follow-up suggestions:

```python
class GeneralAnswer(mf.Signature):
    """Answer general questions thoroughly."""

    query: str = mf.InputField(desc="The user's question")

    answer: str = mf.OutputField(desc="Direct and informative response")
    follow_up_questions: list[str] = mf.OutputField(desc="Questions the user might want to explore next")


class GeneralAssistant(nn.Agent):
    model = model
    signature = GeneralAnswer
    config = {"verbose": True}
```

---

## Step 4 — Composing the Router

`QueryRouter` orchestrates the flow: it classifies the query and dispatches to the matching specialist via a `ModuleDict`:

```python
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

    async def aforward(self, msg):
        await self.classifier.acall(msg)
        expert = self.experts.get(msg.topic, self.experts["general"])
        await expert.acall(msg)
        return msg
```

???+ example

    ```python
    router = QueryRouter()

    msg = mf.Message(query="How do I use backpropagation with PyTorch?")
    router(msg)

    print(f"Topic:      {msg.topic}")
    print(f"Complexity: {msg.complexity}")
    print(f"Answer:     {msg.answer}")
    if msg.get("code_example"):
        print(f"Code:\n{msg.code_example}")
    ```

---

## Complete Script

```python
import msgflux as mf
import msgflux.nn as nn
from typing import Literal, Optional

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class QueryClassification(mf.Signature):
    """Classify the query to route it to the best specialist."""

    query: str = mf.InputField(desc="The user's question")

    topic: Literal["technical", "business", "creative", "general"] = mf.OutputField(
        desc="The primary domain of the query"
    )
    complexity: Literal["simple", "complex"] = mf.OutputField(
        desc="'simple' for direct answers, 'complex' for deep research"
    )
    keywords: list[str] = mf.OutputField(desc="Key terms that identify the domain")


class TechnicalAnswer(mf.Signature):
    """Answer technical questions clearly with practical examples."""

    query: str = mf.InputField(desc="The technical question")
    answer: str = mf.OutputField(desc="Clear and accurate explanation")
    code_example: Optional[str] = mf.OutputField(desc="Illustrative code snippet, if applicable")
    references: list[str] = mf.OutputField(desc="Relevant documentation or resources")


class BusinessAnswer(mf.Signature):
    """Analyze business questions with strategic perspective."""

    query: str = mf.InputField(desc="The business question")
    analysis: str = mf.OutputField(desc="Situation analysis")
    key_points: list[str] = mf.OutputField(desc="Key points to consider")
    recommendation: str = mf.OutputField(desc="Practical, actionable recommendation")


class CreativeAnswer(mf.Signature):
    """Generate creative and inspiring ideas."""

    query: str = mf.InputField(desc="The creative challenge or question")
    ideas: list[str] = mf.OutputField(desc="3 to 5 original ideas")
    suggestions: list[str] = mf.OutputField(desc="Tips to develop the ideas further")
    inspiration: str = mf.OutputField(desc="An inspiring phrase or concept")


class GeneralAnswer(mf.Signature):
    """Answer general questions thoroughly."""

    query: str = mf.InputField(desc="The user's question")
    answer: str = mf.OutputField(desc="Direct and informative response")
    follow_up_questions: list[str] = mf.OutputField(desc="Questions the user might want to explore next")


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

    async def aforward(self, msg):
        await self.classifier.acall(msg)
        expert = self.experts.get(msg.topic, self.experts["general"])
        await expert.acall(msg)
        return msg


router = QueryRouter()

queries = [
    "How do I implement an LRU cache in Python?",
    "What pricing model should I use for a freemium product?",
    "Give me ideas for a mental wellness app.",
    "What is the Kyoto Protocol?",
]

for query in queries:
    msg = mf.Message(query=query)
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

## Further Reading

- [Signatures](../learn/nn/agent/signatures.md) — declarative input/output contracts for agents
- [Async](../learn/nn/agent/async.md) — running agents and modules asynchronously
- [Debug](../learn/nn/agent/debug.md) — inspecting model calls and verbose output
