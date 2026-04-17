# Lead Scoring

<span class="tag tag-purple">Intermediate</span><span class="tag tag-gray">Signature</span>

Sales teams receive more inbound leads than they can manually evaluate. Prioritizing them means assessing each lead across multiple dimensions: demographic fit, engagement signals, budget alignment, and purchase timing.

## The Problem

The most common starting point looks like this.

```
Lead data (company, role, activity, budget, signals)
                 │
                 ▼
│              ScorerAgent                 │
│                                          │
│  assess everything  ←──→  one call       │
                 │ single score + rationale
                 ▼
         sales team acts on it
```

- A single prompt must evaluate demographic fit, engagement level, budget, and timing simultaneously — each requiring different reasoning.
- The rationale is a blended explanation. You cannot tell which dimension drove the score.
- Adding a new scoring dimension means reworking the entire prompt.
- Tuning the weight of each dimension requires iterative prompt engineering with no clear feedback loop.

You are guessing which leads matter.

---

## The Plan

We will build a scorer that decomposes evaluation into four independent dimensions — demographic fit, engagement signals, budget alignment, and purchase timing — runs them in parallel, and aggregates the results into a final weighted score.

Each dimension is evaluated in isolation by its own dedicated scorer, without interference from the others. All four scorers run at the same time, so total latency is the slowest dimension, not their sum. This also means each scorer can be tuned, replaced, or inspected independently.

An aggregator combines the four scores into a weighted result (0–100), assigns a tier (A–D), and recommends a next action for the sales team.

This tutorial uses the **imperative API**: modules are called directly with typed arguments and return plain dicts, which keeps the data flow explicit and easy to test in isolation.

---

## Architecture

```
lead_data: str
       │
       ▼           ▼           ▼           ▼       │
DemographicScorer  EngagementScorer  BudgetScorer  TimingScorer
  (parallel via bcast_gather)
       │           │           │           │
                         │  results: list[dict]
                         ▼
                   Aggregator(demographic_score=..., engagement_score=..., ...)
                         │
                         ▼
                   dict: final_score, tier, rationale, next_action
```

The key design choice is that each scorer is a separate `Agent` with its own `Signature`. The `Aggregator` receives only the four dimension scores — not the raw lead data. This separation means you can tune individual scorer prompts, replace one scorer, or adjust aggregation weights without touching the rest of the pipeline.

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 — Scorer Signatures

Each scorer returns a `score` (0.0–1.0) and a `rationale` explaining the assessment. The `strengths`, `gaps`, `hot_signals`, and `urgency_signals` fields capture dimension-specific evidence for downstream use.

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Signature, InputField, OutputField
from typing import Literal, List


model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class DemographicScore(Signature):
    """Score the lead's company and role fit against the ideal customer profile."""

    lead_data: str = InputField(
        desc="Lead information: company size, industry, role, location"
    )

    score: float = OutputField(
        desc="Fit score from 0.0 (poor fit) to 1.0 (perfect fit)"
    )
    rationale: str = OutputField(
        desc="One sentence explaining the score"
    )
    strengths: List[str] = OutputField(
        desc="ICP attributes the lead matches"
    )
    gaps: List[str] = OutputField(
        desc="ICP attributes the lead does not match"
    )


class EngagementScore(Signature):
    """Score the lead's engagement level based on recorded activity signals."""

    lead_data: str = InputField(
        desc="Engagement history: page views, content downloads, email opens, demo requests"
    )

    score: float = OutputField(
        desc="Engagement score from 0.0 (cold) to 1.0 (highly engaged)"
    )
    rationale: str = OutputField(desc="One sentence explaining the score")
    hot_signals: List[str] = OutputField(
        desc="Strong buying signals detected"
    )


class BudgetScore(Signature):
    """Score the lead's likely budget against the product's price range."""

    lead_data: str = InputField(
        desc="Budget indicators: company revenue, funding stage, spending mentions"
    )

    score: float = OutputField(
        desc="Budget fit score from 0.0 to 1.0"
    )
    rationale: str = OutputField(desc="One sentence explaining the score")
    estimated_budget_range: str = OutputField(
        desc="Estimated annual software budget based on available signals"
    )


class TimingScore(Signature):
    """Score the lead's purchase timing readiness."""

    lead_data: str = InputField(
        desc="Timing signals: contract renewal dates, recent triggers, urgency mentions"
    )

    score: float = OutputField(
        desc="Timing score from 0.0 (not ready) to 1.0 (ready to buy now)"
    )
    rationale: str = OutputField(desc="One sentence explaining the score")
    urgency_signals: List[str] = OutputField(
        desc="Events or signals that suggest near-term purchase intent"
    )
```

---

## Step 2 — Scorer Agents

Each agent is backed by the same model but bound to a different signature — so each has a focused, isolated view of the lead data. Because the signatures are separate, prompts do not bleed into each other and each scorer can be tuned independently.

```python
class DemographicScorer(nn.Agent):
    """Scores lead fit based on company profile and role."""
    model = model
    signature = DemographicScore
    config = {"verbose": True}


class EngagementScorer(nn.Agent):
    """Scores lead activity and engagement signals."""
    model = model
    signature = EngagementScore
    config = {"verbose": True}


class BudgetScorer(nn.Agent):
    """Scores budget fit based on company financials."""
    model = model
    signature = BudgetScore
    config = {"verbose": True}


class TimingScorer(nn.Agent):
    """Scores purchase timing readiness."""
    model = model
    signature = TimingScore
    config = {"verbose": True}
```

---

## Step 3 — Aggregation Signature

The aggregator receives the four dimension scores as named kwargs and returns a final weighted score on a 0–100 scale. Weights are encoded in the `final_score` field description so the model applies them consistently — no separate instruction needed.

```python
class AggregateScore(Signature):
    """Aggregate dimension scores into a final lead quality rating."""

    demographic_score: float = InputField(desc="ICP fit score (0-1)")
    engagement_score:  float = InputField(desc="Engagement level score (0-1)")
    budget_score:      float = InputField(desc="Budget fit score (0-1)")
    timing_score:      float = InputField(desc="Purchase timing score (0-1)")

    final_score: float = OutputField(
        desc="Weighted final score (0-100). Weights: engagement 35%, demographic 30%, timing 20%, budget 15%"
    )
    tier: Literal["A", "B", "C", "D"] = OutputField(
        desc="Lead tier: A=80+, B=60-79, C=40-59, D=<40"
    )
    rationale: str = OutputField(
        desc="2-3 sentence explanation of the overall score"
    )
    next_action: str = OutputField(
        desc="Recommended immediate next step for the sales team"
    )
    priority_rank: int = OutputField(
        desc="Priority rank relative to other leads scored in this batch (1=highest)"
    )


class Aggregator(nn.Agent):
    model = model
    signature = AggregateScore
```

---

## Step 4 — LeadScorer Module

`F.bcast_gather` broadcasts `{"lead_data": lead_data}` to all four scorers in parallel. The dict wrapping is required because each `Signature` generates a Jinja2 task template (`<lead_data>{{ lead_data }}</lead_data>`) from its `InputField` — passing a bare string would raise a `ValueError` at runtime.

Once the four scores are collected, `Aggregator` is called with named kwargs directly — `**scores` unpacks the `demographic_score`, `engagement_score`, `budget_score`, and `timing_score` keys. The agent's Signature maps each kwarg to its corresponding `InputField` template variable. Both the scorer results and the final output are returned as a plain dict.

```python
class LeadScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorers = [
            DemographicScorer(),
            EngagementScorer(),
            BudgetScorer(),
            TimingScorer(),
        ]
        self.aggregator = Aggregator()

    def forward(self, lead_data: str) -> dict:
        results = F.bcast_gather(self.scorers, {"lead_data": lead_data})

        scores = {
            "demographic_score": results[0]["score"],
            "engagement_score":  results[1]["score"],
            "budget_score":      results[2]["score"],
            "timing_score":      results[3]["score"],
        }
        details = {
            "demographic": results[0],
            "engagement":  results[1],
            "budget":      results[2],
            "timing":      results[3],
        }

        final = self.aggregator(**scores)
        return {**final, **scores, "score_details": details}

    async def aforward(self, lead_data: str) -> dict:
        results = await F.abcast_gather(
            self.scorers, {"lead_data": lead_data}
        )

        scores = {
            "demographic_score": results[0]["score"],
            "engagement_score":  results[1]["score"],
            "budget_score":      results[2]["score"],
            "timing_score":      results[3]["score"],
        }
        details = {
            "demographic": results[0],
            "engagement":  results[1],
            "budget":      results[2],
            "timing":      results[3],
        }

        final = await self.aggregator.acall(**scores)
        return {**final, **scores, "score_details": details}
```

---

## Examples

???+ example

    === "Single lead"

        ```python
        scorer = LeadScorer()

        result = scorer(
            "Company: PayStream, 200 employees, Series B fintech, San Francisco. "
            "Role: VP Engineering. "
            "Activity: Visited pricing page 4x this week, downloaded security whitepaper, "
            "attended live demo, replied to SDR email. "
            "Budget signals: $40M Series B, currently paying $8k/mo on Datadog. "
            "Timing: Current contract with Segment renews in 60 days."
        )

        print(f"Score: {result['final_score']:.1f}/100  |  Tier: {result['tier']}")
        print(f"Next action: {result['next_action']}")
        print(f"Rationale: {result['rationale']}")
        ```

    === "Ranked batch (sync)"

        Score a list of leads and sort by final score:

        ```python
        leads = [
            {
                "name": "Alice Chen — VP Engineering at FinTech Series B ($40M raised)",
                "data": (
                    "Company: PayStream, 200 employees, Series B fintech, San Francisco. "
                    "Role: VP Engineering. "
                    "Activity: Visited pricing page 4x this week, downloaded security whitepaper, "
                    "attended live demo, replied to SDR email. "
                    "Budget signals: $40M Series B, currently paying $8k/mo on Datadog. "
                    "Timing: Current contract with Segment renews in 60 days."
                ),
            },
            {
                "name": "Bob Martinez — Marketing Manager at SMB retail",
                "data": (
                    "Company: LocalShop, 12 employees, bootstrapped retail, Texas. "
                    "Role: Marketing Manager. "
                    "Activity: One blog post view last month, no other engagement. "
                    "Budget signals: Revenue ~$2M/year, no tech stack mentioned. "
                    "Timing: No renewal signals, exploring options casually."
                ),
            },
            {
                "name": "Carol Davis — CTO at Health Tech startup",
                "data": (
                    "Company: MedAnalytics, 80 employees, Seed-funded health tech, Boston. "
                    "Role: CTO. "
                    "Activity: Requested a trial account, asked detailed API questions in chat, "
                    "watched 3 product demos. "
                    "Budget signals: $5M seed, HIPAA compliance is a hard requirement. "
                    "Timing: Launching new product in Q2, needs infrastructure now."
                ),
            },
        ]

        scorer = LeadScorer()
        scored_leads = [(lead["name"], scorer(lead["data"])) for lead in leads]
        scored_leads.sort(key=lambda x: x[1]["final_score"], reverse=True)

        print("\n" + "=" * 60)
        print("LEAD SCORING RESULTS")
        print("=" * 60)

        for rank, (name, result) in enumerate(scored_leads, 1):
            print(f"\n#{rank} — {name}")
            print(f"   Score: {result['final_score']:.1f}/100  |  Tier: {result['tier']}")
            print(f"   Demographic: {result['demographic_score']:.2f}  "
                  f"Engagement: {result['engagement_score']:.2f}  "
                  f"Budget: {result['budget_score']:.2f}  "
                  f"Timing: {result['timing_score']:.2f}")
            print(f"   Next action: {result['next_action']}")
            print(f"   Rationale: {result['rationale']}")
        ```

    === "Concurrent batch (async)"

        Score all leads simultaneously — each lead's four scorers run in parallel, and multiple leads are also processed concurrently:

        ```python
        import asyncio


        async def main():
            scorer = LeadScorer()

            results = await F.amap_gather(
                scorer,
                args_list=[(lead["data"],) for lead in leads],
            )

            for lead, result in zip(leads, results):
                print(f"{lead['name']}: {result['final_score']:.1f} (Tier {result['tier']})")


        asyncio.run(main())
        ```

---

## Extending

### Adding a scoring dimension

Add a new `Signature`, wrap it in an `Agent`, and append it to `self.scorers`. Then add a corresponding `InputField` to `AggregateScore` and update the weight distribution in the `final_score` description:

```python
class TechFitScore(Signature):
    """Score technology stack alignment with the product's integration requirements."""
    lead_data: str = InputField(desc="Technology stack, integrations, API usage")
    score: float = OutputField(desc="Tech fit score 0.0-1.0")
    rationale: str = OutputField(desc="One sentence explanation")
    compatible_tools: List[str] = OutputField(desc="Tools that integrate with the product")


class TechFitScorer(nn.Agent):
    model = model
    signature = TechFitScore
    config = {"verbose": True}
```

Then add `tech_fit_score: float = InputField(...)` to `AggregateScore` and pass it to the aggregator in `forward`:

```python
scores = {
    ...,
    "tech_fit_score": results[4]["score"],
}
```

### Routing by tier after scoring

Filter and route the result immediately after the `LeadScorer` returns:

```python
result = scorer(lead_data)
if result["tier"] == "A":
    schedule_outreach(result)
```

### Inspecting dimension details

Each `results[i]` dict from `bcast_gather` carries the full scorer output — not just the score:

```python
result = scorer(lead_data)
demographic = result["score_details"]["demographic"]
print(demographic["strengths"])   # ['Series B', 'decision-maker role', ...]
print(demographic["gaps"])        # ['outside target geography', ...]

engagement = result["score_details"]["engagement"]
print(engagement["hot_signals"])  # ['attended live demo', 'replied to SDR email']
```

---

## Complete Script

??? example "Expand full script"
    ```python
    # /// script
    # dependencies = []
    # ///

    import msgflux as mf
    import msgflux.nn as nn
    import msgflux.nn.functional as F
    from msgflux import Signature, InputField, OutputField
    from typing import Literal, List

    mf.load_dotenv()


    class DemographicScore(Signature):
        """Score the lead's company and role fit against the ideal customer profile."""
        lead_data: str = InputField(desc="Company size, industry, role, location")
        score: float = OutputField(desc="ICP fit score 0.0-1.0")
        rationale: str = OutputField(desc="One sentence explanation")
        strengths: List[str] = OutputField(desc="ICP attributes matched")
        gaps: List[str] = OutputField(desc="ICP attributes not matched")


    class EngagementScore(Signature):
        """Score the lead's engagement level based on activity signals."""
        lead_data: str = InputField(desc="Page views, downloads, email opens, demo requests")
        score: float = OutputField(desc="Engagement score 0.0-1.0")
        rationale: str = OutputField(desc="One sentence explanation")
        hot_signals: List[str] = OutputField(desc="Strong buying signals")


    class BudgetScore(Signature):
        """Score the lead's likely budget against the product's price range."""
        lead_data: str = InputField(desc="Company revenue, funding stage, spending signals")
        score: float = OutputField(desc="Budget fit score 0.0-1.0")
        rationale: str = OutputField(desc="One sentence explanation")
        estimated_budget_range: str = OutputField(desc="Estimated annual software budget")


    class TimingScore(Signature):
        """Score the lead's purchase timing readiness."""
        lead_data: str = InputField(desc="Renewal dates, recent triggers, urgency signals")
        score: float = OutputField(desc="Timing score 0.0-1.0")
        rationale: str = OutputField(desc="One sentence explanation")
        urgency_signals: List[str] = OutputField(desc="Near-term purchase intent signals")


    class AggregateScore(Signature):
        """Aggregate dimension scores into a final lead quality rating."""
        demographic_score: float = InputField(desc="ICP fit score (0-1)")
        engagement_score:  float = InputField(desc="Engagement score (0-1)")
        budget_score:      float = InputField(desc="Budget fit score (0-1)")
        timing_score:      float = InputField(desc="Timing score (0-1)")
        final_score:   float = OutputField(
            desc="Weighted score 0-100. Weights: engagement 35%, demographic 30%, timing 20%, budget 15%"
        )
        tier:          Literal["A", "B", "C", "D"] = OutputField(
            desc="A=80+, B=60-79, C=40-59, D=<40"
        )
        rationale:     str = OutputField(desc="2-3 sentence overall explanation")
        next_action:   str = OutputField(desc="Recommended immediate next step")
        priority_rank: int = OutputField(desc="Rank in this batch (1=highest priority)")



    model = mf.Model.chat_completion("openai/gpt-4.1-mini")


    class DemographicScorer(nn.Agent):
        model = model
        signature = DemographicScore


    class EngagementScorer(nn.Agent):
        model = model
        signature = EngagementScore


    class BudgetScorer(nn.Agent):
        model = model
        signature = BudgetScore


    class TimingScorer(nn.Agent):
        model = model
        signature = TimingScore


    class Aggregator(nn.Agent):
        model = model
        signature = AggregateScore



    class LeadScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scorers = [
                DemographicScorer(),
                EngagementScorer(),
                BudgetScorer(),
                TimingScorer(),
            ]
            self.aggregator = Aggregator()

        def forward(self, lead_data: str) -> dict:
            results = F.bcast_gather(self.scorers, {"lead_data": lead_data})

            scores = {
                "demographic_score": results[0]["score"],
                "engagement_score":  results[1]["score"],
                "budget_score":      results[2]["score"],
                "timing_score":      results[3]["score"],
            }
            details = {
                "demographic": results[0],
                "engagement":  results[1],
                "budget":      results[2],
                "timing":      results[3],
            }

            final = self.aggregator(**scores)
            return {**final, **scores, "score_details": details}

        async def aforward(self, lead_data: str) -> dict:
            results = await F.abcast_gather(
                [s.acall for s in self.scorers], {"lead_data": lead_data}
            )

            scores = {
                "demographic_score": results[0]["score"],
                "engagement_score":  results[1]["score"],
                "budget_score":      results[2]["score"],
                "timing_score":      results[3]["score"],
            }
            details = {
                "demographic": results[0],
                "engagement":  results[1],
                "budget":      results[2],
                "timing":      results[3],
            }

            final = await self.aggregator.acall(**scores)
            return {**final, **scores, "score_details": details}



    leads = [
        {
            "name": "Alice Chen — VP Engineering at FinTech Series B ($40M raised)",
            "data": (
                "Company: PayStream, 200 employees, Series B fintech, San Francisco. "
                "Role: VP Engineering. "
                "Activity: Visited pricing page 4x this week, downloaded security whitepaper, "
                "attended live demo, replied to SDR email. "
                "Budget signals: $40M Series B, currently paying $8k/mo on Datadog. "
                "Timing: Current contract with Segment renews in 60 days."
            ),
        },
        {
            "name": "Bob Martinez — Marketing Manager at SMB retail",
            "data": (
                "Company: LocalShop, 12 employees, bootstrapped retail, Texas. "
                "Role: Marketing Manager. "
                "Activity: One blog post view last month, no other engagement. "
                "Budget signals: Revenue ~$2M/year, no tech stack mentioned. "
                "Timing: No renewal signals, exploring options casually."
            ),
        },
        {
            "name": "Carol Davis — CTO at Health Tech startup",
            "data": (
                "Company: MedAnalytics, 80 employees, Seed-funded health tech, Boston. "
                "Role: CTO. "
                "Activity: Requested a trial account, asked detailed API questions in chat, "
                "watched 3 product demos. "
                "Budget signals: $5M seed, HIPAA compliance is a hard requirement. "
                "Timing: Launching new product in Q2, needs infrastructure now."
            ),
        },
    ]

    scorer = LeadScorer()
    scored_leads = []

    for lead in leads:
        result = scorer(lead["data"])
        scored_leads.append((lead["name"], result))

    scored_leads.sort(key=lambda x: x[1]["final_score"], reverse=True)

    print("\n" + "=" * 60)
    print("LEAD SCORING RESULTS")
    print("=" * 60)

    for rank, (name, result) in enumerate(scored_leads, 1):
        print(f"\n#{rank} — {name}")
        print(f"   Score: {result['final_score']:.1f}/100  |  Tier: {result['tier']}")
        print(f"   Demographic: {result['demographic_score']:.2f}  "
              f"Engagement: {result['engagement_score']:.2f}  "
              f"Budget: {result['budget_score']:.2f}  "
              f"Timing: {result['timing_score']:.2f}")
        print(f"   Next action: {result['next_action']}")
        print(f"   Rationale: {result['rationale']}")
    ```

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — signatures, message fields, and structured output
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Functional API](../learn/nn/functional.md) — `bcast_gather`, `abcast_gather`, and parallel execution
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) — `ChainOfThought` and structured output
