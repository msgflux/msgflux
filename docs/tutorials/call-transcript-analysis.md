# Call Transcript Analysis

<span class="tag tag-purple">Intermediate</span>

Customer service teams review call quality manually — a supervisor listens to recordings, fills out a scorecard, and writes notes. At 5–10 minutes per call, a team handling hundreds of calls a day can review only a fraction of them.

## The Problem

The standard approach to quality assurance on call center transcripts looks like this.

```
Call transcript
       │
       ▼
┌──────────────────────────────────────────┐
│             SupervisorAgent              │
│                                          │
│   read transcript  ←──→  score it        │
└──────────────────────────────────────────┘
       │ one number, one comment
       ▼
   quality report
```

- A single pass without structure produces inconsistent results. Two reviewers score the same call differently.
- Phase-level detail is lost. A call where the customer started angry but ended satisfied looks the same as one that was neutral throughout.
- There is no evidence trail. The score exists but not why it was given — disputed calls can't be audited.
- Scale is the bottleneck. Volume grows faster than review capacity.

You are sampling, not monitoring.

---

## The Plan

We will build an analyzer that produces a structured breakdown of a call in a single pass: per-phase sentiment with a reason for each label, a sentiment arc across the conversation, resolution quality, and a predicted CSAT score.

The key design choice is asking the model to reason step by step before filling any field. Classifying sentiment across three temporal phases is not a lookup — the model must locate where each phase begins, identify the language that carries sentiment, and weigh the trajectory before committing to labels. Without a shared reasoning step, each output field is filled independently and the results can contradict each other: the closing phase labeled satisfied while the resolution is flagged as absent. With step-by-step reasoning, a single interpretation is built first, all fields follow from it, and that reasoning is returned alongside the results as an audit trail you can inspect for every call.

This tutorial uses the **imperative API**: the analyzer is called directly with a transcript string and returns a plain dict.

---

## Architecture

```
transcript: str
       │
       ▼
  CallAnalyzer (nn.Module)
       │
       ▼
  _Analyzer (nn.Agent)
    ├── signature = CallAnalysisSignature
    └── generation_schema = ChainOfThought
              │
              ▼
         raw output
    ├── reasoning     ← "Let's think step by step..."
    └── final_answer
          ├── opening_sentiment / opening_reason
          ├── middle_sentiment  / middle_reason
          ├── closing_sentiment / closing_reason
          ├── sentiment_trajectory / trajectory_summary
          ├── was_resolved / resolution_quality / resolution_reason
          └── csat_prediction
              │
              ▼ (unwrapped by CallAnalyzer.forward)
         dict: all output fields + "reasoning"
```

The `reasoning` field records *how* the model interpreted the conversation before committing to each label — invaluable for auditing disputed cases. The `CallAnalyzer.forward` method unwraps the CoT envelope with `result.get("final_answer", result)` so callers receive a flat dict with all fields, including `reasoning`, at the top level.

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 — Signature

The `Signature` encodes the full analytical contract: one input field and every structured field the model must produce. Separate `reason` fields for each sentiment and for the resolution verdict force the model to provide evidence, not just labels.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Signature, InputField, OutputField
from msgflux.generation.reasoning import ChainOfThought
from typing import Literal

mf.load_dotenv()

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class CallAnalysisSignature(Signature):
    """
    Analyze a customer service call transcript across three conversational
    phases and evaluate how well the issue was resolved.
    """

    transcript: str = InputField(
        desc=(
            "Full conversation transcript with speaker labels. "
            "Example format:\n"
            "[Customer]: Hello, my order hasn't arrived...\n"
            "[Agent]: I'm sorry to hear that, let me check..."
        )
    )

    # ── Phase sentiments ──────────────────────────────────────────────────────

    opening_sentiment: Literal["positive", "neutral", "frustrated", "angry"] = OutputField(
        desc="Customer sentiment in the opening phase (roughly the first third of the conversation)"
    )
    opening_reason: str = OutputField(
        desc="Specific words, tone, or cues from the opening that justify this sentiment"
    )

    middle_sentiment: Literal["positive", "neutral", "frustrated", "angry"] = OutputField(
        desc="Customer sentiment in the middle phase (roughly the central third)"
    )
    middle_reason: str = OutputField(
        desc="Specific words, tone, or cues from the middle that justify this sentiment"
    )

    closing_sentiment: Literal["positive", "neutral", "satisfied", "frustrated", "angry"] = OutputField(
        desc="Customer sentiment in the closing phase (roughly the final third)"
    )
    closing_reason: str = OutputField(
        desc="Specific words, tone, or cues from the closing that justify this sentiment"
    )

    # ── Trajectory ────────────────────────────────────────────────────────────

    sentiment_trajectory: Literal[
        "improved", "stable_positive", "stable_neutral", "stable_negative", "worsened", "volatile"
    ] = OutputField(
        desc="Overall arc of the customer's emotional state from opening to closing"
    )
    trajectory_summary: str = OutputField(
        desc="One or two sentences describing the emotional journey of this call"
    )

    # ── Resolution ────────────────────────────────────────────────────────────

    was_resolved: bool = OutputField(
        desc="True if the customer's core issue was addressed and closed by the end of the call"
    )
    resolution_quality: Literal[
        "fully_resolved", "partially_resolved", "unresolved", "escalated"
    ] = OutputField(
        desc=(
            "fully_resolved: issue closed and customer acknowledged; "
            "partially_resolved: progress made but follow-up required; "
            "unresolved: no tangible progress; "
            "escalated: transferred to another team or tier"
        )
    )
    resolution_reason: str = OutputField(
        desc="Concrete evidence from the transcript that supports the resolution verdict"
    )

    # ── CSAT Prediction ───────────────────────────────────────────────────────

    csat_prediction: int = OutputField(
        desc="Predicted CSAT score the customer would give (1 = very dissatisfied, 5 = very satisfied)"
    )
```

---

## Step 2 — Agent and Wrapper

`_Analyzer` is the raw agent — it takes `{"transcript": text}` and returns the fused CoT output. `CallAnalyzer` wraps it: `forward` unwraps the `final_answer` envelope and merges `reasoning` into the returned dict so all fields are accessible at the top level.

```python
class _Analyzer(nn.Agent):
    model = model
    signature = CallAnalysisSignature
    generation_schema = ChainOfThought
    config = {"verbose": True}


class CallAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent = _Analyzer()

    def forward(self, transcript: str) -> dict:
        raw = self.agent({"transcript": transcript})
        return {**raw.get("final_answer", raw), "reasoning": raw.get("reasoning", "")}

    async def aforward(self, transcript: str) -> dict:
        raw = await self.agent.acall({"transcript": transcript})
        return {**raw.get("final_answer", raw), "reasoning": raw.get("reasoning", "")}
```

---

## Step 3 — Run It

```python
TRANSCRIPT_RESOLVED = """
[Customer]: Hi there, I placed an order five days ago and it still hasn't shown up.
[Agent]: I'm sorry about that. Could I get your order number?
[Customer]: It's 8842-B. This is really frustrating, I needed it for a presentation yesterday.
[Agent]: I completely understand. Let me pull up the tracking... it looks like there was a carrier delay. I can express-ship a replacement today at no charge and it will arrive tomorrow morning.
[Customer]: Oh, that's actually really helpful. So I'll get it tomorrow for sure?
[Agent]: Yes, guaranteed by 10 AM. I'll send the tracking link to your email right now.
[Customer]: Great, thank you. That's exactly what I needed.
[Agent]: Perfect! Is there anything else I can help with today?
[Customer]: No, that's all. I really appreciate how quickly you sorted this out.
"""

TRANSCRIPT_UNRESOLVED = """
[Customer]: I've been charged twice for the same subscription this month.
[Agent]: I see the issue. I'll need to escalate this to our billing team.
[Customer]: I've been waiting two weeks already. Can't you just refund it now?
[Agent]: Unfortunately I don't have access to billing systems directly.
[Customer]: This is unacceptable. I want to speak to a manager.
[Agent]: I understand your frustration. Let me transfer you to our billing department.
[Customer]: Fine, but this is the third time I've called about this. It's ridiculous.
[Agent]: I'm transferring you now. Your reference number is REF-2291.
[Customer]: Whatever.
"""


def print_report(a: dict) -> None:
    trajectory_icon = {
        "improved": "📈", "stable_positive": "✅", "stable_neutral": "➡️",
        "stable_negative": "⚠️", "worsened": "📉", "volatile": "〰️",
    }.get(a["sentiment_trajectory"], "❓")
    resolution_icon = "✅" if a["was_resolved"] else "❌"

    print("=" * 60)
    print("CALL ANALYSIS REPORT")
    print("=" * 60)
    print("\n── Sentiment by Phase ──────────────────────────────────")
    print(f"  Opening  [{a['opening_sentiment']:>10}]  {a['opening_reason']}")
    print(f"  Middle   [{a['middle_sentiment']:>10}]  {a['middle_reason']}")
    print(f"  Closing  [{a['closing_sentiment']:>10}]  {a['closing_reason']}")
    print(f"\n── Trajectory {trajectory_icon} ────────────────────────────────────")
    print(f"  {a['sentiment_trajectory'].upper()}: {a['trajectory_summary']}")
    print(f"\n── Resolution {resolution_icon} ────────────────────────────────────")
    print(f"  Quality : {a['resolution_quality']}")
    print(f"  Reason  : {a['resolution_reason']}")
    print(f"\n── CSAT Prediction ({'⭐' * a['csat_prediction']}) ({a['csat_prediction']}/5)")
    print("\n── Reasoning Trace ─────────────────────────────────────")
    print(f"  {a['reasoning']}")
    print("=" * 60)


analyzer = CallAnalyzer()

for label, transcript in [
    ("RESOLVED CALL", TRANSCRIPT_RESOLVED),
    ("UNRESOLVED CALL", TRANSCRIPT_UNRESOLVED),
]:
    print(f"\n\n{'#' * 60}\n# {label}\n{'#' * 60}")
    analysis = analyzer(transcript)
    print_report(analysis)
```

---

## Examples

???+ example

    === "Resolved call"

        ```python
        analyzer = CallAnalyzer()
        analysis = analyzer(TRANSCRIPT_RESOLVED)

        print(f"Trajectory : {analysis['sentiment_trajectory']}")
        print(f"Resolution : {analysis['resolution_quality']}")
        print(f"CSAT       : {analysis['csat_prediction']}/5")
        print(f"Reasoning  : {analysis['reasoning']}")
        ```

    === "Escalated call"

        ```python
        analyzer = CallAnalyzer()
        analysis = analyzer(TRANSCRIPT_UNRESOLVED)

        print(f"Trajectory : {analysis['sentiment_trajectory']}")
        print(f"Resolution : {analysis['resolution_quality']}")
        print(f"CSAT       : {analysis['csat_prediction']}/5")
        print(f"Opening    : {analysis['opening_sentiment']} — {analysis['opening_reason']}")
        print(f"Closing    : {analysis['closing_sentiment']} — {analysis['closing_reason']}")
        ```

    === "Async batch"

        Score a queue of transcripts concurrently — each runs an independent model call:

        ```python
        import asyncio
        import msgflux.nn.functional as F


        async def main():
            analyzer = CallAnalyzer()
            transcripts = [TRANSCRIPT_RESOLVED, TRANSCRIPT_UNRESOLVED]

            results = await F.amap_gather(
                analyzer,
                args_list=[(t,) for t in transcripts],
            )

            for i, analysis in enumerate(results, 1):
                print(
                    f"Call {i}: "
                    f"trajectory={analysis['sentiment_trajectory']}, "
                    f"resolved={analysis['was_resolved']}, "
                    f"csat={analysis['csat_prediction']}/5"
                )


        asyncio.run(main())
        ```

---

## Extending

### Adding agent quality metrics

Add output fields to `CallAnalysisSignature` to turn this into a full QA scorecard:

```python
agent_empathy: Literal["high", "medium", "low"] = OutputField(
    desc="Degree to which the agent acknowledged and validated the customer's feelings"
)
protocol_compliance: bool = OutputField(
    desc="True if the agent followed standard greeting, verification, and closing procedures"
)
```

The `reasoning` step will naturally incorporate these new fields into its analysis before filling them.

### Routing by resolution quality

Act immediately on the result without any extra infrastructure:

```python
analysis = analyzer(transcript)

if analysis["resolution_quality"] == "escalated":
    flag_for_supervisor(analysis)
elif analysis["csat_prediction"] <= 2:
    schedule_callback(analysis)
```

### Building a service health dashboard

Run `amap_gather` over a day's call batch and aggregate:

```python
results = await F.amap_gather(analyzer, args_list=[(t,) for t in transcripts])

from collections import Counter

trajectories = Counter(r["sentiment_trajectory"] for r in results)
avg_csat     = sum(r["csat_prediction"] for r in results) / len(results)
resolved_pct = sum(r["was_resolved"] for r in results) / len(results) * 100

print(f"Resolved: {resolved_pct:.1f}%  |  Avg CSAT: {avg_csat:.2f}")
print(f"Trajectories: {dict(trajectories)}")
```

---

## Complete Script

```python
import asyncio
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Signature, InputField, OutputField
from msgflux.generation.reasoning import ChainOfThought
from typing import Literal

mf.load_dotenv()


# ── Model ─────────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


# ── Signature ─────────────────────────────────────────────────────────────────

class CallAnalysisSignature(Signature):
    """
    Analyze a customer service call transcript across three conversational
    phases and evaluate how well the issue was resolved.
    """

    transcript: str = InputField(
        desc=(
            "Full conversation transcript with speaker labels. "
            "Example format:\n"
            "[Customer]: Hello, my order hasn't arrived...\n"
            "[Agent]: I'm sorry to hear that, let me check..."
        )
    )

    opening_sentiment: Literal["positive", "neutral", "frustrated", "angry"] = OutputField(
        desc="Customer sentiment in the opening phase (roughly the first third)"
    )
    opening_reason: str = OutputField(
        desc="Specific words, tone, or cues from the opening that justify this sentiment"
    )
    middle_sentiment: Literal["positive", "neutral", "frustrated", "angry"] = OutputField(
        desc="Customer sentiment in the middle phase (roughly the central third)"
    )
    middle_reason: str = OutputField(
        desc="Specific words, tone, or cues from the middle that justify this sentiment"
    )
    closing_sentiment: Literal["positive", "neutral", "satisfied", "frustrated", "angry"] = OutputField(
        desc="Customer sentiment in the closing phase (roughly the final third)"
    )
    closing_reason: str = OutputField(
        desc="Specific words, tone, or cues from the closing that justify this sentiment"
    )
    sentiment_trajectory: Literal[
        "improved", "stable_positive", "stable_neutral", "stable_negative", "worsened", "volatile"
    ] = OutputField(
        desc="Overall arc of the customer's emotional state from opening to closing"
    )
    trajectory_summary: str = OutputField(
        desc="One or two sentences describing the emotional journey of this call"
    )
    was_resolved: bool = OutputField(
        desc="True if the customer's core issue was addressed and closed by the end of the call"
    )
    resolution_quality: Literal[
        "fully_resolved", "partially_resolved", "unresolved", "escalated"
    ] = OutputField(
        desc=(
            "fully_resolved: issue closed and customer acknowledged; "
            "partially_resolved: progress made but follow-up required; "
            "unresolved: no tangible progress; "
            "escalated: transferred to another team or tier"
        )
    )
    resolution_reason: str = OutputField(
        desc="Concrete evidence from the transcript that supports the resolution verdict"
    )
    csat_prediction: int = OutputField(
        desc="Predicted CSAT score the customer would give (1 = very dissatisfied, 5 = very satisfied)"
    )


# ── Agent and Wrapper ─────────────────────────────────────────────────────────

class _Analyzer(nn.Agent):
    model = model
    signature = CallAnalysisSignature
    generation_schema = ChainOfThought
    config = {"verbose": True}


class CallAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent = _Analyzer()

    def forward(self, transcript: str) -> dict:
        raw = self.agent({"transcript": transcript})
        return {**raw.get("final_answer", raw), "reasoning": raw.get("reasoning", "")}

    async def aforward(self, transcript: str) -> dict:
        raw = await self.agent.acall({"transcript": transcript})
        return {**raw.get("final_answer", raw), "reasoning": raw.get("reasoning", "")}


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(a: dict) -> None:
    trajectory_icon = {
        "improved": "📈", "stable_positive": "✅", "stable_neutral": "➡️",
        "stable_negative": "⚠️", "worsened": "📉", "volatile": "〰️",
    }.get(a["sentiment_trajectory"], "❓")
    resolution_icon = "✅" if a["was_resolved"] else "❌"

    print("=" * 60)
    print("CALL ANALYSIS REPORT")
    print("=" * 60)
    print("\n── Sentiment by Phase ──────────────────────────────────")
    print(f"  Opening  [{a['opening_sentiment']:>10}]  {a['opening_reason']}")
    print(f"  Middle   [{a['middle_sentiment']:>10}]  {a['middle_reason']}")
    print(f"  Closing  [{a['closing_sentiment']:>10}]  {a['closing_reason']}")
    print(f"\n── Trajectory {trajectory_icon} ────────────────────────────────────")
    print(f"  {a['sentiment_trajectory'].upper()}: {a['trajectory_summary']}")
    print(f"\n── Resolution {resolution_icon} ────────────────────────────────────")
    print(f"  Quality : {a['resolution_quality']}")
    print(f"  Reason  : {a['resolution_reason']}")
    print(f"\n── CSAT Prediction ({'⭐' * a['csat_prediction']}) ({a['csat_prediction']}/5)")
    print("\n── Reasoning Trace ─────────────────────────────────────")
    print(f"  {a['reasoning']}")
    print("=" * 60)


# ── Transcripts ───────────────────────────────────────────────────────────────

TRANSCRIPT_RESOLVED = """
[Customer]: Hi there, I placed an order five days ago and it still hasn't shown up.
[Agent]: I'm sorry about that. Could I get your order number?
[Customer]: It's 8842-B. This is really frustrating, I needed it for a presentation yesterday.
[Agent]: I completely understand. Let me pull up the tracking... it looks like there was a carrier delay. I can express-ship a replacement today at no charge and it will arrive tomorrow morning.
[Customer]: Oh, that's actually really helpful. So I'll get it tomorrow for sure?
[Agent]: Yes, guaranteed by 10 AM. I'll send the tracking link to your email right now.
[Customer]: Great, thank you. That's exactly what I needed.
[Agent]: Perfect! Is there anything else I can help with today?
[Customer]: No, that's all. I really appreciate how quickly you sorted this out.
"""

TRANSCRIPT_UNRESOLVED = """
[Customer]: I've been charged twice for the same subscription this month.
[Agent]: I see the issue. I'll need to escalate this to our billing team.
[Customer]: I've been waiting two weeks already. Can't you just refund it now?
[Agent]: Unfortunately I don't have access to billing systems directly.
[Customer]: This is unacceptable. I want to speak to a manager.
[Agent]: I understand your frustration. Let me transfer you to our billing department.
[Customer]: Fine, but this is the third time I've called about this. It's ridiculous.
[Agent]: I'm transferring you now. Your reference number is REF-2291.
[Customer]: Whatever.
"""


# ── Run ───────────────────────────────────────────────────────────────────────

analyzer = CallAnalyzer()

for label, transcript in [
    ("RESOLVED CALL", TRANSCRIPT_RESOLVED),
    ("UNRESOLVED CALL", TRANSCRIPT_UNRESOLVED),
]:
    print(f"\n\n{'#' * 60}\n# {label}\n{'#' * 60}")
    analysis = analyzer(transcript)
    print_report(analysis)
```

---

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — signatures, message fields, and structured output
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) — `ChainOfThought` and structured output
- [Functional API](../learn/nn/functional.md) — `amap_gather` and parallel execution
