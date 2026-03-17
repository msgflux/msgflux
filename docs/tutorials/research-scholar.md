# Research Scholar Agent

Build a multi-stage research pipeline that decomposes a broad question into subtopics, gathers findings for each in parallel, then synthesizes a cited report — trading raw model autonomy for structured, auditable steps.

> **Inspired by**: [DSPy Deep Research tutorial](https://www.cmpnd.ai/blog/learn-dspy-deep-research.html) — the msgFlux take on the *Decomposed Workflow* pattern.

## What You'll Build

```
Research request
       │
       ▼
  Clarifier ──── Signature: query → clarified_question, scope
       │
       ▼
  Planner ─────── Signature: clarified_question → subtopics: list[str]
       │
       ▼
  ┌────┴──────────────────────────────────┐
  │    Gatherer × N  (parallel)           │
  │  Signature: subtopic → findings: str  │
  └────┬──────────────────────────────────┘
       │  list[findings]
       ▼
  Synthesizer ─── Signature: subtopics, gathered → report: str
       │
       ▼
  Annotator ───── Signature: report, gathered → annotated_report, citations
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

## Step 1 — Search Tool

A single plain-Python function that the Gatherer agent will call. Replace the mock with a real search backend (Tavily, Brave, Exa, Wikipedia retriever, etc.).

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message, Signature, InputField, OutputField
from typing import List


def search_web(query: str) -> str:
    """Search the web for information on a topic. Returns a summary of findings."""
    # Replace with: tavily_client.search(query) or similar
    mock_results = {
        "quantum computing basics":
            "Quantum computers use qubits that exploit superposition and entanglement. "
            "Key players: IBM, Google, IonQ. IBM's Eagle processor has 127 qubits.",
        "quantum computing applications":
            "Applications: drug discovery (molecular simulation), cryptography (Shor's algorithm), "
            "optimization (QAOA), finance (portfolio optimization).",
        "quantum computing challenges":
            "Main barriers: decoherence, error rates (~1%), need for near-absolute-zero cooling, "
            "limited qubit connectivity. Error correction requires ~1000 physical qubits per logical qubit.",
        "quantum computing timeline":
            "Experts predict fault-tolerant quantum computing by 2030-2035. "
            "Current NISQ era: noisy intermediate-scale quantum devices.",
    }
    for key, result in mock_results.items():
        if any(word in query.lower() for word in key.split()):
            return result
    return f"No results found for: {query!r}"
```

---

## Step 2 — Signatures

Each stage has a typed contract. The docstring becomes the model's instruction; the fields constrain its output.

```python
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class ClarifySignature(Signature):
    """Restate the research request as a precise question and define its scope."""

    query: str = InputField(desc="The original research request")

    clarified_question: str = OutputField(
        desc="A single, precise question that fully captures the intent"
    )
    scope: str = OutputField(
        desc="What the research should and should not cover (1-2 sentences)"
    )


class PlanSignature(Signature):
    """Break the research question into 3-5 focused subtopics to investigate."""

    clarified_question: str = InputField(desc="The precise research question")
    scope: str = InputField(desc="Boundaries of the research")

    subtopics: List[str] = OutputField(
        desc="3 to 5 subtopics, each narrow enough for a single focused search"
    )


class SynthesizeSignature(Signature):
    """Synthesize findings from multiple sources into a coherent research report."""

    clarified_question: str = InputField(desc="The research question being answered")
    subtopics: List[str] = InputField(desc="The subtopics that were researched")
    gathered: List[str] = InputField(desc="Findings for each subtopic, in order")

    report: str = OutputField(
        desc="A well-structured report answering the research question (3-5 paragraphs)"
    )


class AnnotateSignature(Signature):
    """Add inline citations to the report, linking claims to specific source findings."""

    report: str = InputField(desc="The synthesized research report")
    gathered: List[str] = InputField(desc="Source findings used to build the report")

    annotated_report: str = OutputField(
        desc="The report with [Source N] citations inline"
    )
    citations: List[str] = OutputField(
        desc="Numbered list of sources cited, e.g. '[1] Quantum computing basics: ...'"
    )
```

---

## Step 3 — Agents

```python
class Clarifier(nn.Agent):
    model = model
    signature = ClarifySignature
    config = {"verbose": True}


class Planner(nn.Agent):
    model = model
    signature = PlanSignature
    config = {"verbose": True}


class Gatherer(nn.Agent):
    """Researches a single subtopic using web search."""
    model = model
    tools = [search_web]
    signature = "subtopic -> findings: str"
    config = {"verbose": True}


class Synthesizer(nn.Agent):
    model = model
    signature = SynthesizeSignature


class Annotator(nn.Agent):
    model = model
    signature = AnnotateSignature
```

---

## Step 4 — Composing the Pipeline

`F.map_gather` runs the Gatherer on every subtopic in parallel, returning results in order:

```python
class ResearchScholar(nn.Module):
    def __init__(self):
        super().__init__()
        self.clarifier   = Clarifier()
        self.planner     = Planner()
        self.gatherer    = Gatherer()
        self.synthesizer = Synthesizer()
        self.annotator   = Annotator()

    def forward(self, msg):
        # 1. Clarify the request
        self.clarifier(msg)

        # 2. Plan subtopics
        self.planner(msg)

        # 3. Gather findings for each subtopic in parallel
        msg.gathered = F.map_gather(
            self.gatherer,
            [(sub,) for sub in msg.subtopics],
        )

        # 4. Synthesize into a report
        self.synthesizer(msg)

        # 5. Add citations
        self.annotator(msg)

        return msg


scholar = ResearchScholar()

msg = Message(query="How does quantum computing work and when will it be practical?")
scholar(msg)

print("Subtopics:", msg.subtopics)
print("\nAnnotated Report:\n", msg.annotated_report)
print("\nCitations:")
for c in msg.citations:
    print(" ", c)
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message, Signature, InputField, OutputField
from typing import List


# ── Tool ──────────────────────────────────────────────────────────────────────

def search_web(query: str) -> str:
    """Search the web for information on a topic."""
    mock_results = {
        "quantum computing basics":
            "Quantum computers use qubits exploiting superposition and entanglement. "
            "IBM's Eagle processor has 127 qubits.",
        "quantum computing applications":
            "Applications: drug discovery, cryptography (Shor's algorithm), "
            "optimization (QAOA), finance.",
        "quantum computing challenges":
            "Decoherence, ~1% error rates, near-absolute-zero cooling, "
            "error correction needs ~1000 physical qubits per logical qubit.",
        "quantum computing timeline":
            "Fault-tolerant QC expected 2030-2035. Current NISQ era.",
    }
    for key, result in mock_results.items():
        if any(word in query.lower() for word in key.split()):
            return result
    return f"No results found for: {query!r}"


# ── Signatures ────────────────────────────────────────────────────────────────

class ClarifySignature(Signature):
    """Restate the research request as a precise question and define its scope."""

    query: str = InputField(desc="The original research request")
    clarified_question: str = OutputField(
        desc="A single, precise question that fully captures the intent"
    )
    scope: str = OutputField(
        desc="What the research should and should not cover"
    )


class PlanSignature(Signature):
    """Break the research question into 3-5 focused subtopics to investigate."""

    clarified_question: str = InputField(desc="The precise research question")
    scope: str = InputField(desc="Boundaries of the research")
    subtopics: List[str] = OutputField(
        desc="3 to 5 subtopics, each narrow enough for a single focused search"
    )


class SynthesizeSignature(Signature):
    """Synthesize findings from multiple sources into a coherent research report."""

    clarified_question: str = InputField(desc="The research question")
    subtopics: List[str] = InputField(desc="Subtopics that were researched")
    gathered: List[str] = InputField(desc="Findings for each subtopic, in order")
    report: str = OutputField(
        desc="Well-structured report answering the question (3-5 paragraphs)"
    )


class AnnotateSignature(Signature):
    """Add inline citations to the report linking claims to source findings."""

    report: str = InputField(desc="The synthesized report")
    gathered: List[str] = InputField(desc="Source findings")
    annotated_report: str = OutputField(desc="Report with [Source N] citations inline")
    citations: List[str] = OutputField(
        desc="Numbered source list, e.g. '[1] Quantum basics: ...'"
    )


# ── Agents ────────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class Clarifier(nn.Agent):
    model = model
    signature = ClarifySignature
    config = {"verbose": True}


class Planner(nn.Agent):
    model = model
    signature = PlanSignature
    config = {"verbose": True}


class Gatherer(nn.Agent):
    """Researches a single subtopic using web search."""
    model = model
    tools = [search_web]
    signature = "subtopic -> findings: str"
    config = {"verbose": True}


class Synthesizer(nn.Agent):
    model = model
    signature = SynthesizeSignature


class Annotator(nn.Agent):
    model = model
    signature = AnnotateSignature


# ── Pipeline ─────────────────────────────────────────────────────────────────

class ResearchScholar(nn.Module):
    def __init__(self):
        super().__init__()
        self.clarifier   = Clarifier()
        self.planner     = Planner()
        self.gatherer    = Gatherer()
        self.synthesizer = Synthesizer()
        self.annotator   = Annotator()

    def forward(self, msg):
        self.clarifier(msg)
        self.planner(msg)
        msg.gathered = F.map_gather(
            self.gatherer,
            [(sub,) for sub in msg.subtopics],
        )
        self.synthesizer(msg)
        self.annotator(msg)
        return msg


# ── Run ───────────────────────────────────────────────────────────────────────

scholar = ResearchScholar()

msg = Message(query="How does quantum computing work and when will it be practical?")
scholar(msg)

print("Question:", msg.clarified_question)
print("Subtopics:", msg.subtopics)
print("\nReport:\n", msg.annotated_report)
print("\nCitations:")
for c in msg.citations:
    print(" ", c)
```

---

## Async Version

Replace `map_gather` with `amap_gather` to parallelize I/O-bound search calls:

```python
import asyncio


class ResearchScholar(nn.Module):
    def __init__(self):
        super().__init__()
        self.clarifier   = Clarifier()
        self.planner     = Planner()
        self.gatherer    = Gatherer()
        self.synthesizer = Synthesizer()
        self.annotator   = Annotator()

    async def aforward(self, msg):
        await self.clarifier.acall(msg)
        await self.planner.acall(msg)

        # All subtopics gathered concurrently
        msg.gathered = await F.amap_gather(
            self.gatherer.acall,
            [(sub,) for sub in msg.subtopics],
        )

        await self.synthesizer.acall(msg)
        await self.annotator.acall(msg)
        return msg


async def main():
    scholar = ResearchScholar()
    msg = Message(query="What are the main risks and opportunities in generative AI for healthcare?")
    await scholar.acall(msg)
    print(msg.annotated_report)


asyncio.run(main())
```

---

## Why Decompose?

| Monolithic agent | ResearchScholar |
|---|---|
| One prompt, all tools, all context | Each stage has one job and one signature |
| Hard to debug which step went wrong | `verbose=True` on each agent shows the exact call |
| Re-running = full research again | Failed synthesis? Re-run only `Synthesizer` |
| Parallel gathering impossible | `map_gather` runs N subtopics simultaneously |

The `Signature` docstring on each stage acts like a typed interface: clear enough to read, strict enough to test.
