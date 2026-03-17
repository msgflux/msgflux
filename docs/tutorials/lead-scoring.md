# Lead Scoring

Score inbound leads across multiple dimensions simultaneously — demographic fit, engagement signals, budget alignment, and purchase timing — then aggregate into a weighted final score with a ranked shortlist.

## What You'll Build

```
Lead data (company, role, activity, budget, signals)
       │
       ├───────────────────────────────────────────┐
       ▼           ▼           ▼           ▼       │
DemographicScorer  EngagementScorer  BudgetScorer  TimingScorer
  (parallel via bcast_gather)
       │           │           │           │
       └───────────┴─────┬─────┴───────────┘
                         │  scores: list[float]
                         ▼
                   Aggregator ── weighted average + Signature:
                                   scores → final_score, tier,
                                             rationale, next_action
```

Each scorer runs independently and simultaneously — total latency equals the slowest scorer, not the sum.

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Scorer Signatures

Each scorer returns a `score` (0.0–1.0) and a `rationale` explaining the assessment:

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message, Signature, InputField, OutputField
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

## Step 4 — Lead Scorer Module

`F.bcast_gather` broadcasts the same `lead_data` string to all four scorers in parallel:

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

    def forward(self, msg):
        # All four scorers run in parallel on the same lead_data
        results = F.bcast_gather(self.scorers, msg.lead_data)

        msg.demographic_score = results[0]["score"]
        msg.engagement_score  = results[1]["score"]
        msg.budget_score      = results[2]["score"]
        msg.timing_score      = results[3]["score"]

        # Store dimension details
        msg.score_details = {
            "demographic": results[0],
            "engagement":  results[1],
            "budget":      results[2],
            "timing":      results[3],
        }

        # Aggregate
        self.aggregator(msg)
        return msg
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message, Signature, InputField, OutputField
from typing import Literal, List


# ── Signatures ────────────────────────────────────────────────────────────────

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


# ── Agents ────────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class DemographicScorer(nn.Agent):
    model = model
    signature = DemographicScore
    config = {"verbose": True}


class EngagementScorer(nn.Agent):
    model = model
    signature = EngagementScore
    config = {"verbose": True}


class BudgetScorer(nn.Agent):
    model = model
    signature = BudgetScore
    config = {"verbose": True}


class TimingScorer(nn.Agent):
    model = model
    signature = TimingScore
    config = {"verbose": True}


class Aggregator(nn.Agent):
    model = model
    signature = AggregateScore


# ── Pipeline ──────────────────────────────────────────────────────────────────

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

    def forward(self, msg):
        results = F.bcast_gather(self.scorers, msg.lead_data)

        msg.demographic_score = results[0]["score"]
        msg.engagement_score  = results[1]["score"]
        msg.budget_score      = results[2]["score"]
        msg.timing_score      = results[3]["score"]

        msg.score_details = {
            "demographic": results[0],
            "engagement":  results[1],
            "budget":      results[2],
            "timing":      results[3],
        }

        self.aggregator(msg)
        return msg


# ── Leads ─────────────────────────────────────────────────────────────────────

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
    msg = Message(lead_data=lead["data"])
    scorer(msg)
    scored_leads.append((lead["name"], msg))

# Sort by final score
scored_leads.sort(key=lambda x: x[1].final_score, reverse=True)

print("\n" + "=" * 60)
print("LEAD SCORING RESULTS")
print("=" * 60)

for rank, (name, msg) in enumerate(scored_leads, 1):
    print(f"\n#{rank} — {name}")
    print(f"   Score: {msg.final_score:.1f}/100  |  Tier: {msg.tier}")
    print(f"   Demographic: {msg.demographic_score:.2f}  "
          f"Engagement: {msg.engagement_score:.2f}  "
          f"Budget: {msg.budget_score:.2f}  "
          f"Timing: {msg.timing_score:.2f}")
    print(f"   Next action: {msg.next_action}")
    print(f"   Rationale: {msg.rationale}")
```

---

## Async Batch Scoring

Score a large batch of leads concurrently — each lead's four scorers run in parallel, and multiple leads are processed simultaneously:

```python
import asyncio


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

    async def aforward(self, msg):
        results = await F.ascatter_gather(
            [s.acall for s in self.scorers],
            args_list=[(msg.lead_data,)] * len(self.scorers),
        )

        msg.demographic_score = results[0]["score"]
        msg.engagement_score  = results[1]["score"]
        msg.budget_score      = results[2]["score"]
        msg.timing_score      = results[3]["score"]

        await self.aggregator.acall(msg)
        return msg


async def main():
    scorer = LeadScorer()
    messages = [Message(lead_data=lead["data"]) for lead in leads]

    # All leads scored concurrently
    results = await F.ascatter_gather(
        [scorer.acall] * len(messages),
        args_list=[(msg,) for msg in messages],
    )

    for lead, msg in zip(leads, results):
        print(f"{lead['name']}: {msg.final_score:.1f} (Tier {msg.tier})")

asyncio.run(main())
```
