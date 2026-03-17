# Call Transcript Analysis

Analyze customer service transcripts across three conversational phases — opening, middle, and closing — extracting per-phase sentiments with their rationale, and a resolution verdict with its justification. Uses `Signature` combined with `generation_schema=ChainOfThought` so the model reasons holistically over the conversation arc before producing structured outputs.

## What You'll Build

```
Transcript
    │
    ▼
CallAnalyzer (Agent)
  ├── generation_schema = ChainOfThought  →  msg.reasoning
  └── signature = CallAnalysisSignature   →  msg.final_answer
                                                 │
                              ┌──────────────────┼──────────────────┐
                              ▼                  ▼                  ▼
                         Sentiments         Trajectory         Resolution
                     opening / middle    improved / stable    was_resolved
                       / closing          / worsened           + quality
                     + reason each        + summary            + reason
```

The reasoning trace (`msg.reasoning`) records *how* the model interpreted the conversation before committing to each label — invaluable for auditing disputed cases.

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Define the Signature

The `Signature` encodes the full analytical contract: what goes in, and every structured field that comes out. Separate `reason` fields for sentiments and resolution force the model to provide evidence, not just labels:

```python
import msgflux as mf
from msgflux import Signature, InputField, OutputField
from typing import Literal

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

    sentiment_trajectory: Literal["improved", "stable_positive", "stable_neutral", "stable_negative", "worsened", "volatile"] = OutputField(
        desc="Overall arc of the customer's emotional state from opening to closing"
    )
    trajectory_summary: str = OutputField(
        desc="One or two sentences describing the emotional journey of this call"
    )

    # ── Resolution ────────────────────────────────────────────────────────────

    was_resolved: bool = OutputField(
        desc="True if the customer's core issue was addressed and closed by the end of the call"
    )
    resolution_quality: Literal["fully_resolved", "partially_resolved", "unresolved", "escalated"] = OutputField(
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

    # ── Prediction ────────────────────────────────────────────────────────────

    csat_prediction: int = OutputField(
        desc="Predicted CSAT score the customer would give (1 = very dissatisfied, 5 = very satisfied)"
    )
```

---

## Step 2 — Combine with ChainOfThought

Adding `generation_schema=ChainOfThought` makes msgFlux fuse the two schemas: the model first fills a `reasoning` field (step-by-step analysis of the transcript), then populates `final_answer` with every field from the signature:

```python
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class CallAnalyzer(nn.Agent):
    """Analyzes a customer service transcript across phases and evaluates resolution."""
    model = model
    signature = CallAnalysisSignature
    generation_schema = ChainOfThought
    config = {"verbose": True}
```

The fused schema that msgFlux builds internally:

```
Output
  ├── reasoning     — "Let's think step by step …" (ChainOfThought)
  └── final_answer
        ├── opening_sentiment / opening_reason
        ├── middle_sentiment  / middle_reason
        ├── closing_sentiment / closing_reason
        ├── sentiment_trajectory / trajectory_summary
        ├── was_resolved / resolution_quality / resolution_reason
        └── csat_prediction
```

---

## Step 3 — Build the Pipeline

A thin `Module` wraps the agent and prints a formatted report:

```python
from msgflux import Message


class CallAnalysisPipeline(nn.Module):
    """Runs the full analysis and formats the result."""

    def __init__(self):
        super().__init__()
        self.analyzer = CallAnalyzer()

    def forward(self, msg: Message) -> Message:
        self.analyzer(msg)
        return msg


def print_report(msg: Message) -> None:
    """Print a formatted analysis report from a processed Message."""
    trajectory_icon = {
        "improved": "📈",
        "stable_positive": "✅",
        "stable_neutral": "➡️",
        "stable_negative": "⚠️",
        "worsened": "📉",
        "volatile": "〰️",
    }.get(msg.sentiment_trajectory, "❓")

    resolution_icon = "✅" if msg.was_resolved else "❌"

    print("=" * 60)
    print("CALL ANALYSIS REPORT")
    print("=" * 60)

    print("\n── Sentiment by Phase ──────────────────────────────────")
    print(f"  Opening  [{msg.opening_sentiment:>10}]  {msg.opening_reason}")
    print(f"  Middle   [{msg.middle_sentiment:>10}]  {msg.middle_reason}")
    print(f"  Closing  [{msg.closing_sentiment:>10}]  {msg.closing_reason}")

    print(f"\n── Trajectory {trajectory_icon} ────────────────────────────────────")
    print(f"  {msg.sentiment_trajectory.upper()}: {msg.trajectory_summary}")

    print(f"\n── Resolution {resolution_icon} ────────────────────────────────────")
    print(f"  Quality : {msg.resolution_quality}")
    print(f"  Reason  : {msg.resolution_reason}")

    print(f"\n── CSAT Prediction {'⭐' * msg.csat_prediction} ({'⭐' * msg.csat_prediction}/{5 * '⭐'})")

    print("\n── Reasoning Trace ─────────────────────────────────────")
    print(f"  {msg.reasoning}")
    print("=" * 60)
```

---

## Step 4 — Run Against a Sample Transcript

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

pipeline = CallAnalysisPipeline()

for label, transcript in [("RESOLVED", TRANSCRIPT_RESOLVED), ("UNRESOLVED", TRANSCRIPT_UNRESOLVED)]:
    print(f"\n\n{'#' * 60}")
    print(f"# SCENARIO: {label}")
    print(f"{'#' * 60}")
    msg = Message(transcript=transcript)
    pipeline(msg)
    print_report(msg)
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from msgflux.generation.reasoning import ChainOfThought
from typing import Literal


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

    sentiment_trajectory: Literal["improved", "stable_positive", "stable_neutral", "stable_negative", "worsened", "volatile"] = OutputField(
        desc="Overall arc of the customer's emotional state from opening to closing"
    )
    trajectory_summary: str = OutputField(
        desc="One or two sentences describing the emotional journey of this call"
    )

    was_resolved: bool = OutputField(
        desc="True if the customer's core issue was addressed and closed by the end of the call"
    )
    resolution_quality: Literal["fully_resolved", "partially_resolved", "unresolved", "escalated"] = OutputField(
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


# ── Model + Agent ─────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class CallAnalyzer(nn.Agent):
    """Analyzes a customer service transcript across phases and evaluates resolution."""
    model = model
    signature = CallAnalysisSignature
    generation_schema = ChainOfThought
    config = {"verbose": True}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class CallAnalysisPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = CallAnalyzer()

    def forward(self, msg: Message) -> Message:
        self.analyzer(msg)
        return msg


def print_report(msg: Message) -> None:
    trajectory_icon = {
        "improved": "📈", "stable_positive": "✅", "stable_neutral": "➡️",
        "stable_negative": "⚠️", "worsened": "📉", "volatile": "〰️",
    }.get(msg.sentiment_trajectory, "❓")
    resolution_icon = "✅" if msg.was_resolved else "❌"

    print("=" * 60)
    print("CALL ANALYSIS REPORT")
    print("=" * 60)
    print("\n── Sentiment by Phase ──────────────────────────────────")
    print(f"  Opening  [{msg.opening_sentiment:>10}]  {msg.opening_reason}")
    print(f"  Middle   [{msg.middle_sentiment:>10}]  {msg.middle_reason}")
    print(f"  Closing  [{msg.closing_sentiment:>10}]  {msg.closing_reason}")
    print(f"\n── Trajectory {trajectory_icon} ────────────────────────────────────")
    print(f"  {msg.sentiment_trajectory.upper()}: {msg.trajectory_summary}")
    print(f"\n── Resolution {resolution_icon} ────────────────────────────────────")
    print(f"  Quality : {msg.resolution_quality}")
    print(f"  Reason  : {msg.resolution_reason}")
    print(f"\n── CSAT Prediction {'⭐' * msg.csat_prediction} ({msg.csat_prediction}/5)")
    print("\n── Reasoning Trace ─────────────────────────────────────")
    print(f"  {msg.reasoning}")
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

pipeline = CallAnalysisPipeline()

for label, transcript in [
    ("RESOLVED CALL", TRANSCRIPT_RESOLVED),
    ("UNRESOLVED CALL", TRANSCRIPT_UNRESOLVED),
]:
    print(f"\n\n{'#' * 60}\n# SCENARIO: {label}\n{'#' * 60}")
    msg = Message(transcript=transcript)
    pipeline(msg)
    print_report(msg)
```

---

## Batch Analysis

Analyze a queue of transcripts in parallel with `ascatter_gather`:

```python
import asyncio
import msgflux.nn.functional as F
from msgflux import Message


class AsyncCallAnalysisPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = CallAnalyzer()

    async def aforward(self, msg: Message) -> Message:
        await self.analyzer.acall(msg)
        return msg


async def analyze_batch(transcripts: list[str]) -> list[Message]:
    pipeline = AsyncCallAnalysisPipeline()
    messages = [Message(transcript=t) for t in transcripts]

    return await F.ascatter_gather(
        [pipeline.acall] * len(messages),
        args_list=[(msg,) for msg in messages],
    )


async def main():
    transcripts = [TRANSCRIPT_RESOLVED, TRANSCRIPT_UNRESOLVED]
    results = await analyze_batch(transcripts)

    for i, msg in enumerate(results, 1):
        print(f"\nCall {i}: trajectory={msg.sentiment_trajectory}, "
              f"resolved={msg.was_resolved}, csat={msg.csat_prediction}/5")


asyncio.run(main())
```

---

## Why ChainOfThought improves phase analysis

Sentiment classification across temporal phases is a *reasoning task*, not a lookup task. The model must:

1. Locate where each phase begins and ends
2. Identify sentiment-carrying language in each zone
3. Weigh the trajectory across all three before committing to labels
4. Decide whether resolution evidence is present or absent

Without `ChainOfThought`, the model answers each field independently and can produce incoherent combinations (e.g., `closing_sentiment = "satisfied"` but `was_resolved = false`). With CoT, the `reasoning` step builds a shared context that keeps all fields internally consistent — and leaves an audit trail in `msg.reasoning` for every decision.

---

## Next Steps

- **Agent quality scoring**: Add output fields like `agent_empathy`, `response_speed`, and `protocol_compliance` to turn this into a full QA scorecard.
- **Topic extraction**: Add a `main_topic: str` and `issue_category: Literal[...]` field to route reports by department automatically.
- **Multi-call trend**: Run `analyze_batch` over a week of calls and aggregate `sentiment_trajectory` and `csat_prediction` to build a daily service health dashboard.
