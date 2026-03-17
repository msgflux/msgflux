# Intent Router: Resolving Agent Tool Sprawl with Signatures

Build an orchestrated system where a planner decomposes queries into typed intents and routes each sub-task to a specialized agent — keeping every agent small, observable, and focused.

> **Inspired by**: [Solving Agent Tool Sprawl with DSPy](https://viksit.substack.com/p/solving-agent-tool-sprawl-with-dspy)

---

## The Problem

When you give a single agent access to many tools, things break in subtle ways:

```python
# ❌ Naive approach — one agent, every tool
class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [search_docs, get_doc_by_id, get_incident_metrics, list_open_tickets, ...]
```

The agent must simultaneously decide *what* to do and *how* to do it. Under load it picks the wrong tool, skips steps, or combines calls in the wrong order. When it fails, you can only fix the prompt — there is no structure to improve.

**The fix**: separate *planning* (what intents does this query need?) from *execution* (which agent handles each intent?). A typed `Signature` makes the planner's contract explicit and debuggable.

---

## What You'll Build

```
User query
    │
    ▼
QueryPlanner (Signature)
    │
    └─ plan: [{subquery, intent}, ...]
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
 search  lookup  analyze
 Agent   Agent   Agent
    │       │       │
    └───────┴───────┘
            │
    context threaded between steps
            │
            ▼
    Final answer assembled
```

Each sub-agent has access **only** to the tools relevant to its intent. Results from earlier steps flow into later ones through a shared `context` field.

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1: Define Tools

Each tool is a plain Python function. Keeping them small and single-purpose makes routing decisions easier for the planner.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField


# ── Tools ───────────────────────────────────────────────────────────────────

def search_docs(query: str) -> str:
    """Search the knowledge base by keyword. Returns matching article titles and IDs."""
    # Replace with your real search backend (Elasticsearch, BM25, etc.)
    catalog = {
        "deployment": "deploy-101 · Deployment Guide, deploy-docker · Docker Setup, deploy-k8s · Helm Charts",
        "authentication": "auth-001 · Auth Overview, auth-jwt · JWT Configuration",
        "performance": "perf-tips · Performance Guide, perf-db · Database Tuning",
    }
    for keyword, results in catalog.items():
        if keyword in query.lower():
            return results
    return f"No articles found for: {query!r}"


def get_doc_by_id(doc_id: str) -> str:
    """Retrieve the full content of a knowledge base article by its ID."""
    docs = {
        "deploy-101": "## Deployment Guide\nPush to `main` triggers CI. After green, run `make deploy`.",
        "auth-001": "## Auth Overview\nJWT tokens, 24 h expiry, refreshed automatically by the SDK.",
        "perf-db": "## Database Tuning\nAdd indexes on `user_id` and `created_at`. Use connection pooling.",
    }
    return docs.get(doc_id, f"Document {doc_id!r} not found.")


def get_incident_metrics(severity: str = "all", last_days: int = 7) -> str:
    """Return aggregated incident metrics for the given severity and time window."""
    data = {
        "all":      f"Last {last_days}d — 12 incidents · MTTR 4.2 h · 3 critical · 9 medium",
        "critical": f"Last {last_days}d — 3 critical incidents · MTTR 2.1 h",
        "medium":   f"Last {last_days}d — 9 medium incidents · MTTR 5.8 h",
    }
    return data.get(severity, data["all"])
```

---

## Step 2: Specialized Agents

Each agent gets only the tools it needs. `config = {"verbose": True}` prints every tool call and its result — invaluable for debugging routing decisions.

```python
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class SearchAgent(nn.Agent):
    """Finds relevant articles using keyword search."""
    model = model
    tools = [search_docs]
    signature = "query, context -> results: str"
    config = {"verbose": True}


class LookupAgent(nn.Agent):
    """Fetches the full content of a specific document by ID."""
    model = model
    tools = [get_doc_by_id]
    signature = "query, context -> details: str"
    config = {"verbose": True}


class AnalyzeAgent(nn.Agent):
    """Computes incident metrics and surfaces trends."""
    model = model
    tools = [get_incident_metrics]
    signature = "query, context -> analysis: str"
    config = {"verbose": True}
```

---

## Step 3: Query Planner with a Signature

The planner is the heart of the system. A `Signature` makes its contract explicit: here are the inputs, here are the typed outputs, here is the docstring that becomes its instruction.

```python
from typing import List


class QueryPlanner(Signature):
    """Decompose the user question into an ordered list of sub-tasks.

    Each step must be assigned one of the available intents.
    Steps may depend on previous ones — include earlier results in the subquery
    so the next agent has all the context it needs.
    """

    question: str = InputField(desc="The full user question")
    available_intents: str = InputField(
        desc="Comma-separated intents the system can handle, with one-line descriptions"
    )

    plan: List[dict] = OutputField(
        desc=(
            "Ordered list of steps. Each step is a dict with keys: "
            "'subquery' (str) and 'intent' (one of the available intents)."
        )
    )
```

Wire it to a `LM` module — a lightweight wrapper that calls the model with the signature's prompt:

```python
class Planner(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = nn.LM(model=model, signature=QueryPlanner)

    def forward(self, msg):
        msg.plan = self.lm(
            question=msg.question,
            available_intents=(
                "search: find articles by keyword, "
                "lookup: retrieve a specific document by ID, "
                "analyze: compute incident metrics and trends"
            ),
        )["plan"]
        return msg
```

---

## Step 4: Orchestrator Module

The orchestrator runs the plan step by step, threading the accumulated context into each agent call so later steps can build on earlier results.

```python
class IntentRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.planner = Planner()
        self.agents = nn.ModuleDict({
            "search":  SearchAgent(),
            "lookup":  LookupAgent(),
            "analyze": AnalyzeAgent(),
        })

    def forward(self, msg):
        # 1. Decompose query into a typed plan
        self.planner(msg)

        # 2. Execute each step, threading context forward
        context_parts = []

        for i, step in enumerate(msg.plan):
            intent  = step["intent"]
            subquery = step["subquery"]
            agent   = self.agents.get(intent)

            if agent is None:
                print(f"[step {i}] Unknown intent {intent!r}, skipping.")
                continue

            context = "\n".join(context_parts) or "No prior context."
            result  = agent(query=subquery, context=context)

            # Accumulate context for the next step
            step_summary = f"Step {i} ({intent}): {result}"
            context_parts.append(step_summary)
            print(step_summary)

        msg.context = "\n".join(context_parts)
        return msg
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from typing import List


# ── Tools ────────────────────────────────────────────────────────────────────

def search_docs(query: str) -> str:
    """Search the knowledge base by keyword."""
    catalog = {
        "deployment": "deploy-101 · Deployment Guide, deploy-docker · Docker Setup",
        "authentication": "auth-001 · Auth Overview, auth-jwt · JWT Configuration",
        "performance": "perf-tips · Performance Guide, perf-db · Database Tuning",
    }
    for keyword, results in catalog.items():
        if keyword in query.lower():
            return results
    return f"No articles found for: {query!r}"


def get_doc_by_id(doc_id: str) -> str:
    """Retrieve a knowledge base article by ID."""
    docs = {
        "deploy-101": "## Deployment Guide\nPush to `main` triggers CI. Run `make deploy` after green.",
        "auth-001":   "## Auth Overview\nJWT tokens with 24 h expiry, auto-refreshed by the SDK.",
        "perf-db":    "## Database Tuning\nIndex `user_id` and `created_at`. Use connection pooling.",
    }
    return docs.get(doc_id, f"Document {doc_id!r} not found.")


def get_incident_metrics(severity: str = "all", last_days: int = 7) -> str:
    """Return aggregated incident metrics."""
    data = {
        "all":      f"Last {last_days}d — 12 incidents · MTTR 4.2 h · 3 critical",
        "critical": f"Last {last_days}d — 3 critical incidents · MTTR 2.1 h",
        "medium":   f"Last {last_days}d — 9 medium incidents · MTTR 5.8 h",
    }
    return data.get(severity, data["all"])


# ── Signature ─────────────────────────────────────────────────────────────────

class QueryPlanner(Signature):
    """Decompose the user question into an ordered list of sub-tasks.

    Each step must be assigned one of the available intents.
    Include earlier results in later subqueries so context flows forward.
    """

    question: str = InputField(desc="The full user question")
    available_intents: str = InputField(
        desc="Comma-separated intents with one-line descriptions"
    )

    plan: List[dict] = OutputField(
        desc=(
            "Ordered list of steps. Each step: "
            "'subquery' (str) and 'intent' (search | lookup | analyze)."
        )
    )


# ── Modules ───────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class SearchAgent(nn.Agent):
    """Finds relevant articles using keyword search."""
    model = model
    tools = [search_docs]
    signature = "query, context -> results: str"
    config = {"verbose": True}


class LookupAgent(nn.Agent):
    """Fetches full document content by ID."""
    model = model
    tools = [get_doc_by_id]
    signature = "query, context -> details: str"
    config = {"verbose": True}


class AnalyzeAgent(nn.Agent):
    """Computes incident metrics and surfaces trends."""
    model = model
    tools = [get_incident_metrics]
    signature = "query, context -> analysis: str"
    config = {"verbose": True}


class Planner(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = nn.LM(model=model, signature=QueryPlanner)

    def forward(self, msg):
        msg.plan = self.lm(
            question=msg.question,
            available_intents=(
                "search: find articles by keyword, "
                "lookup: retrieve a specific document by ID, "
                "analyze: compute incident metrics and trends"
            ),
        )["plan"]
        return msg


class IntentRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.planner = Planner()
        self.agents  = nn.ModuleDict({
            "search":  SearchAgent(),
            "lookup":  LookupAgent(),
            "analyze": AnalyzeAgent(),
        })

    def forward(self, msg):
        self.planner(msg)

        context_parts = []
        for i, step in enumerate(msg.plan):
            agent = self.agents.get(step["intent"])
            if agent is None:
                continue

            context = "\n".join(context_parts) or "No prior context."
            result  = agent(query=step["subquery"], context=context)

            step_summary = f"Step {i} ({step['intent']}): {result}"
            context_parts.append(step_summary)
            print(step_summary)

        msg.context = "\n".join(context_parts)
        return msg


# ── Run ───────────────────────────────────────────────────────────────────────

router = IntentRouter()

msg = Message()
msg.question = "What is our deployment process and how many critical incidents happened this week?"

router(msg)

print("\n--- Plan ---")
for step in msg.plan:
    print(f"  [{step['intent']}] {step['subquery']}")

print("\n--- Final Context ---")
print(msg.context)
```

**Sample output** (plan generated by the model, tool calls logged by `verbose`):

```
[search] calling search_docs(query='deployment process')
Step 0 (search): deploy-101 · Deployment Guide, deploy-docker · Docker Setup

[lookup] calling get_doc_by_id(doc_id='deploy-101')
Step 1 (lookup): ## Deployment Guide\nPush to `main` triggers CI...

[analyze] calling get_incident_metrics(severity='critical', last_days=7)
Step 2 (analyze): Last 7d — 3 critical incidents · MTTR 2.1 h

--- Plan ---
  [search]  Find articles about deployment process
  [lookup]  Retrieve content of deploy-101
  [analyze] Get critical incident count for the last 7 days

--- Final Context ---
Step 0 (search): deploy-101 · Deployment Guide, deploy-docker · Docker Setup
Step 1 (lookup): ## Deployment Guide\nPush to `main` triggers CI...
Step 2 (analyze): Last 7d — 3 critical incidents · MTTR 2.1 h
```

---

## Async Version

Replace `forward` with `aforward` and use `acall` to run the full pipeline without blocking:

```python
import asyncio


class IntentRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.planner = Planner()
        self.agents  = nn.ModuleDict({
            "search":  SearchAgent(),
            "lookup":  LookupAgent(),
            "analyze": AnalyzeAgent(),
        })

    async def aforward(self, msg):
        await self.planner.acall(msg)

        context_parts = []
        for i, step in enumerate(msg.plan):
            agent = self.agents.get(step["intent"])
            if agent is None:
                continue

            context = "\n".join(context_parts) or "No prior context."
            result  = await agent.acall(query=step["subquery"], context=context)

            context_parts.append(f"Step {i} ({step['intent']}): {result}")

        msg.context = "\n".join(context_parts)
        return msg


async def main():
    router = IntentRouter()
    msg = Message()
    msg.question = "Walk me through authentication and show any performance issues this week."
    await router.acall(msg)
    print(msg.context)


asyncio.run(main())
```

---

## Why This Works

| Naive agent | Intent Router |
|---|---|
| All tools in one prompt | Each agent has ≤ 2 tools |
| Fails silently on wrong tool choice | Planner contract is typed and logged |
| No way to improve routing | Swap `QueryPlanner` logic without touching agents |
| Debugging = prompt rewriting | `verbose=True` shows every tool call |

The `Signature` class is the key: its docstring becomes the planner's instruction, its `InputField`/`OutputField` types constrain the output, and future improvements to the planner do not require touching any agent.
