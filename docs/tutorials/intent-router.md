# Solving Agent Tool Sprawl with Intent Routing

Stop letting LLMs improvise across your tools. Use typed signatures and intent-based orchestration for routing that is observable, debuggable, and structured.

> **Inspired by**: [Solving Agent Tool Sprawl with DSPy](https://viksit.substack.com/p/solving-agent-tool-sprawl-with-dspy)

---

## The Problem

Here is the architecture most teams build first.

```
User query
    │
    ▼
┌──────────────────────────────────────────┐
│               SupportAgent               │
│                                          │
│  what to do?  ←──────→  how to do it?   │
└────────────────────┬─────────────────────┘
                     │ picks one (maybe wrong)
         ┌───────────┼───────────┬───────────┐
         ▼           ▼           ▼           ▼
    search_docs  get_doc_by_id  metrics  list_tickets  ...
```

You expose all your tools to the agent. The model sees everything and decides which tools to call. But you have no control layer. When it picks wrong, you add examples to the prompt — then more examples, then ALL CAPS WARNINGS. The model keeps improvising because the fundamental issue is not the prompt. You have delegated architectural decisions to a general-purpose model that does not know your system.

- The model cannot learn from mistakes.
- You cannot see why it chose Tool X over Tool Y.
- You cannot A/B test routing strategies.

You are debugging by vibes.

```python
# Naive approach — one agent, every tool
class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [search_docs, get_doc_by_id, get_incident_metrics, list_open_tickets, ...]
```

**The fix**: add an orchestration layer that decides which tools to call, in what order, and how to combine results. Separate *planning* (what intents does this query need?) from *execution* (which agent handles each intent?). A typed `Signature` makes the planner's contract explicit and debuggable.

---

## What You'll Build

```
User query
    │
    ▼
QueryPlanner (Signature + ChainOfThought)
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
    Final context assembled
```

Each sub-agent has access **only** to the tools relevant to its intent. Results from earlier steps flow into later ones through a shared `context` field. The orchestration layer is:

- **Observable** — every routing decision is a typed plan you can log and inspect
- **Programmable** — routing logic lives in code, not buried in a prompt
- **Debuggable** — `verbose=True` shows every tool call and its result

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1: Define Tools

Each tool is a plain Python function organized by intent. Keeping them small and single-purpose makes routing decisions easier for the planner.

```python
import msgflux as mf
import msgflux.nn as nn


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

Each agent gets only the tools it needs. `config = {"verbose": True}` prints every tool call and its result, making every routing decision visible.

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

The planner is the heart of the system. A `Signature` makes its contract explicit: here are the inputs, here are the typed outputs, here is the docstring that becomes its instruction. `ChainOfThought` adds a reasoning step before the model commits to a plan.

```python
from msgflux.generation.reasoning import ChainOfThought
from typing import Dict, List


class QueryPlanner(mf.Signature):
    """Decompose the user question into an ordered list of sub-tasks.

    Each step must be assigned one of the available intents.
    Steps may depend on previous ones — include earlier results in the subquery
    so the next agent has all the context it needs.
    """

    question: str = mf.InputField(desc="The full user question")
    available_intents: str = mf.InputField(
        desc="Comma-separated intents the system can handle, with one-line descriptions"
    )

    plan: List[Dict[str, str]] = mf.OutputField(
        desc=(
            "Ordered list of steps. Each step is a dict with keys: "
            "'subquery' (str) and 'intent' (one of the available intents)."
        )
    )


class PlannerAgent(nn.Agent):
    model = model
    signature = QueryPlanner
    generation_schema = ChainOfThought
    config = {"verbose": True}


class Planner(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent = PlannerAgent()

    def forward(self, msg):
        msg.plan = self.agent(
            question=msg.question,
            available_intents=(
                "search: find articles by keyword, "
                "lookup: retrieve a specific document by ID, "
                "analyze: compute incident metrics and trends"
            ),
        )["final_answer"]["plan"]
        return msg

    async def aforward(self, msg):
        msg.plan = (await self.agent.acall(
            question=msg.question,
            available_intents=(
                "search: find articles by keyword, "
                "lookup: retrieve a specific document by ID, "
                "analyze: compute incident metrics and trends"
            ),
        ))["final_answer"]["plan"]
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
        self.planner(msg)

        context_parts = []
        for i, step in enumerate(msg.plan):
            intent   = step["intent"]
            subquery = step["subquery"]
            agent    = self.agents.get(intent)

            if agent is None:
                print(f"[step {i}] Unknown intent {intent!r}, skipping.")
                continue

            context = "\n".join(context_parts) or "No prior context."
            result  = agent(query=subquery, context=context)

            step_summary = f"Step {i} ({intent}): {result}"
            context_parts.append(step_summary)
            print(step_summary)

        msg.context = "\n".join(context_parts)
        return msg

    async def aforward(self, msg):
        await self.planner.acall(msg)

        context_parts = []
        for i, step in enumerate(msg.plan):
            agent = self.agents.get(step["intent"])
            if agent is None:
                continue

            context = "\n".join(context_parts) or "No prior context."
            result  = await agent.acall(query=step["subquery"], context=context)

            step_summary = f"Step {i} ({step['intent']}): {result}"
            context_parts.append(step_summary)
            print(step_summary)

        msg.context = "\n".join(context_parts)
        return msg
```

---

## Complete Example

```python
import asyncio

import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought
from typing import Dict, List

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


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


class QueryPlanner(mf.Signature):
    """Decompose the user question into an ordered list of sub-tasks.

    Each step must be assigned one of the available intents.
    Include earlier results in later subqueries so context flows forward.
    """

    question: str = mf.InputField(desc="The full user question")
    available_intents: str = mf.InputField(
        desc="Comma-separated intents with one-line descriptions"
    )

    plan: List[Dict[str, str]] = mf.OutputField(
        desc=(
            "Ordered list of steps. Each step: "
            "'subquery' (str) and 'intent' (search | lookup | analyze)."
        )
    )


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


class PlannerAgent(nn.Agent):
    model = model
    signature = QueryPlanner
    generation_schema = ChainOfThought
    config = {"verbose": True}


class Planner(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent = PlannerAgent()

    def forward(self, msg):
        msg.plan = self.agent(
            question=msg.question,
            available_intents=(
                "search: find articles by keyword, "
                "lookup: retrieve a specific document by ID, "
                "analyze: compute incident metrics and trends"
            ),
        )["final_answer"]["plan"]
        return msg

    async def aforward(self, msg):
        msg.plan = (await self.agent.acall(
            question=msg.question,
            available_intents=(
                "search: find articles by keyword, "
                "lookup: retrieve a specific document by ID, "
                "analyze: compute incident metrics and trends"
            ),
        ))["final_answer"]["plan"]
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

    async def aforward(self, msg):
        await self.planner.acall(msg)

        context_parts = []
        for i, step in enumerate(msg.plan):
            agent = self.agents.get(step["intent"])
            if agent is None:
                continue

            context = "\n".join(context_parts) or "No prior context."
            result  = await agent.acall(query=step["subquery"], context=context)

            step_summary = f"Step {i} ({step['intent']}): {result}"
            context_parts.append(step_summary)
            print(step_summary)

        msg.context = "\n".join(context_parts)
        return msg


router = IntentRouter()

msg = mf.Message()
msg.question = "What is our deployment process and how many critical incidents happened this week?"

router(msg)

print("\n--- Plan ---")
for step in msg.plan:
    print(f"  [{step['intent']}] {step['subquery']}")

print("\n--- Final Context ---")
print(msg.context)


async def main():
    router = IntentRouter()
    msg = mf.Message()
    msg.question = "Walk me through authentication and show any performance issues this week."
    await router.acall(msg)
    print(msg.context)


asyncio.run(main())
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

## Why This Works

| Without intent routing | With intent routing |
|---|---|
| All tools in one prompt | Each agent has at most 2 tools |
| Model picks wrong tool, you rewrite the prompt | Planner contract is typed and logged |
| No visibility into routing decisions | Every plan is structured data you can inspect |
| Cannot improve routing without changing the prompt | Swap `QueryPlanner` logic without touching agents |

Routing is code, not hope. The `Signature` docstring becomes the planner's instruction, its `InputField`/`OutputField` types constrain the output, and `ChainOfThought` gives the model a reasoning step before it commits to a plan. When routing breaks, you have structure to improve — not just a prompt to rewrite.

---

## Further Reading

- [Generation Schemas](../learn/nn/agent/generation-schemas.md) — structuring model output with `msgspec.Struct`
- [Signatures](../learn/nn/agent/signatures.md) — declarative input/output contracts for agents
