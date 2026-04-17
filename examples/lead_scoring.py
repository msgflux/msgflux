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
