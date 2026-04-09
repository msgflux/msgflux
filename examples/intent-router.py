# /// script
# dependencies = []
# ///

import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought
from typing import Dict, List

mf.load_dotenv()
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

    Each step in the plan MUST be a dict with exactly two keys:
    - "subquery": the question or instruction for that agent
    - "intent": one of the available intents (exact string match required)

    Steps may depend on previous ones — include earlier results in the subquery
    so the next agent has full context.

    Constraint: 'lookup' requires a document ID that can only come from a prior
    'search' step. Never emit 'lookup' as the first step.
    """

    question: str = mf.InputField(desc="The full user question")
    available_intents: str = mf.InputField(
        desc="Comma-separated intents the system can handle, with one-line descriptions"
    )

    plan: List[Dict[str, str]] = mf.OutputField(
        desc=(
            "Ordered list of steps. Every step must contain both keys: "
            "'subquery' (str) and 'intent' (exact value from available_intents). "
            "Example: [{\"subquery\": \"find auth docs\", \"intent\": \"search\"}, ...]"
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
                "search: find articles by keyword — returns article titles and IDs, "
                "lookup: retrieve a specific document by ID (requires an ID from a prior search step), "
                "analyze: compute incident metrics and trends"
            ),
        )["final_answer"]["plan"]  # final_answer is a dict here because a Signature is set; "plan" is its OutputField
        return msg

    async def aforward(self, msg):
        msg.plan = (await self.agent.acall(
            question=msg.question,
            available_intents=(
                "search: find articles by keyword — returns article titles and IDs, "
                "lookup: retrieve a specific document by ID (requires an ID from a prior search step), "
                "analyze: compute incident metrics and trends"
            ),
        ))["final_answer"]["plan"]  # same as above: final_answer["plan"] → Signature OutputField
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
            agent = self.agents.get(step.get("intent", ""))
            if agent is None:
                continue

            context = "\n".join(context_parts) or "No prior context."
            result  = next(iter(agent(query=step["subquery"], context=context).values()))

            step_summary = f"Step {i} ({step['intent']}): {result}"
            context_parts.append(step_summary)
            print(step_summary)

        msg.context = "\n".join(context_parts)
        return msg

    async def aforward(self, msg):
        await self.planner.acall(msg)

        context_parts = []
        for i, step in enumerate(msg.plan):
            agent = self.agents.get(step.get("intent", ""))
            if agent is None:
                continue

            context = "\n".join(context_parts) or "No prior context."
            result  = next(iter((await agent.acall(query=step["subquery"], context=context)).values()))

            step_summary = f"Step {i} ({step['intent']}): {result}"
            context_parts.append(step_summary)
            print(step_summary)

        msg.context = "\n".join(context_parts)
        return msg
