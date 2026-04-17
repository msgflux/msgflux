# Call Transcript Analysis

<span class="tag tag-purple">Intermediate</span><span class="tag tag-gray">Generation Schema</span><span class="tag tag-gray">Reasoning</span><span class="tag tag-gray">Multimodal</span>

Customer service teams still review call quality manually. A supervisor listens to a recording, fills out a scorecard, and writes notes. At 5 to 10 minutes per call, a team handling hundreds of calls a day can review only a small sample.

## The Problem

The usual quality review loop looks like this:

```
Call transcript
       |
       v
+------------------------------------------+
|             SupervisorAgent              |
|                                          |
|   read transcript  <-->  score it        |
+------------------------------------------+
       |
       v
   quality report
```

- One pass without structure produces inconsistent results.
- Phase-level detail gets lost.
- There is no clear evidence trail for disputes.
- Review capacity does not scale with call volume.

## The Plan

We will build an analyzer that can take a transcript or an audio recording and return:

- sentiment for the opening, middle, and closing stages
- a short reason for each stage
- the overall sentiment trajectory
- a resolution assessment
- a predicted CSAT score
- a reasoning trace

Instead of a large `Signature` with repeated fields like `opening_sentiment`, `middle_sentiment`, and `closing_sentiment`, we will model the repeated unit once with `msgspec.Struct`. The output becomes a list of `CallPhase` objects, which fits the domain better and avoids forcing the schema into fake one-off field names.

## Architecture

```
Call input (text or audio)
       |
       +-- audio? -> Transcriber (Whisper) -> transcript
       |
       v
  CallAnalyzer (nn.Module)
       |
       v
   _Analyzer (nn.Agent)
       |
       +-- generation_schema = CallAnalysis
              |
              +-- reasoning
              +-- phases[opening, middle, closing]
              +-- sentiment_trajectory
              +-- trajectory_summary
              +-- resolution
              +-- csat_prediction
```

The key point is the `phases` list. We define one `CallPhase` struct and reuse it three times, instead of spelling out six near-duplicate fields for the three stages.

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

## Step 1 - Models

```python
from typing import Annotated, Literal

import msgspec

import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F

mf.load_dotenv()

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model = mf.Model.speech_to_text("openai/whisper-1")
```

## Step 2 - Output Schema

`CallPhase` models the repeated part of the output: one conversation stage, one sentiment, one reason. `CallAnalysis` then wraps the full report, including the reasoning trace.

```python
class CallPhase(msgspec.Struct):
    stage: Annotated[
        Literal["opening", "middle", "closing"],
        msgspec.Meta(description="Conversation stage. Return opening, middle, and closing exactly once."),
    ]
    sentiment: Annotated[
        Literal["positive", "neutral", "satisfied", "frustrated", "angry"],
        msgspec.Meta(
            description=(
                "Customer sentiment in this stage. Use satisfied only when the call clearly ends well."
            )
        ),
    ]
    reason: Annotated[
        str,
        msgspec.Meta(description="Short evidence from the transcript that justifies the sentiment."),
    ]


class ResolutionAssessment(msgspec.Struct):
    was_resolved: Annotated[
        bool,
        msgspec.Meta(description="True if the customer's core issue was addressed by the end of the call."),
    ]
    quality: Annotated[
        Literal["fully_resolved", "partially_resolved", "unresolved", "escalated"],
        msgspec.Meta(
            description=(
                "fully_resolved: issue closed and customer acknowledged it; "
                "partially_resolved: progress made but follow-up still required; "
                "unresolved: no meaningful progress; "
                "escalated: transferred to another team or tier."
            )
        ),
    ]
    reason: Annotated[
        str,
        msgspec.Meta(description="Concrete evidence from the transcript that supports the resolution verdict."),
    ]


class CallAnalysis(msgspec.Struct):
    reasoning: Annotated[
        str,
        msgspec.Meta(
            description="Let's think step by step in order to analyze the transcript consistently before filling the fields."
        ),
    ]
    phases: Annotated[
        list[CallPhase],
        msgspec.Meta(description="Exactly three entries in this order: opening, middle, closing."),
    ]
    sentiment_trajectory: Annotated[
        Literal["improved", "stable_positive", "stable_neutral", "stable_negative", "worsened", "volatile"],
        msgspec.Meta(description="Overall emotional arc of the customer from opening to closing."),
    ]
    trajectory_summary: Annotated[
        str,
        msgspec.Meta(description="One or two sentences describing the emotional journey across the call."),
    ]
    resolution: ResolutionAssessment
    csat_prediction: Annotated[
        int,
        msgspec.Meta(description="Predicted customer satisfaction score from 1 to 5."),
    ]
```

This shape is easier to maintain than a large `Signature`. If you ever want to add another phase later, you update the `stage` values and the instructions, instead of creating a new set of repeated output fields.

## Step 3 - Transcriber

If the input is audio, we transcribe it first and write the text into `msg.call.transcript`.

```python
class CallTranscriber(nn.Transcriber):
    """Transcribes call audio into msg.call.transcript."""

    model = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode = "call.transcript"
```

## Step 4 - Analyzer

`_Analyzer` is the raw agent. It receives a transcript and returns a `CallAnalysis` object directly.

```python
class _Analyzer(nn.Agent):
    model = chat_model
    system_message = """
    You are a call quality analyst for customer support teams.
    """
    instructions = """
    Analyze the transcript across the opening, middle, and closing stages.

    Rules:
    - Return exactly three items in phases, in this order: opening, middle, closing.
    - Ground every sentiment and resolution judgment in transcript evidence.
    - Use satisfied only when the closing stage clearly ends positive after progress or resolution.
    - Mark quality as escalated when the issue is handed to another team or tier.
    - Predict csat_prediction on a 1 to 5 scale.
    """
    generation_schema = CallAnalysis
    templates = {"task": "Transcript:\n{{ transcript }}"}
    config = {"verbose": True}


class CallAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = CallTranscriber()
        self.agent = _Analyzer()

    def forward(self, transcript: str | None = None, audio: bytes | None = None) -> CallAnalysis:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            self.transcriber(msg)
            transcript = msg.call.transcript
        return self.agent(transcript=transcript)

    async def aforward(self, transcript: str | None = None, audio: bytes | None = None) -> CallAnalysis:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            await self.transcriber.acall(msg)
            transcript = msg.call.transcript
        return await self.agent.acall(transcript=transcript)
```

Because the output is already a `msgspec.Struct`, there is no need to unwrap `final_answer` or flatten fields by hand.

## Step 5 - Run It

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


def get_phase(analysis: CallAnalysis, stage: str) -> CallPhase:
    return next(phase for phase in analysis.phases if phase.stage == stage)


def print_report(analysis: CallAnalysis) -> None:
    opening = get_phase(analysis, "opening")
    middle = get_phase(analysis, "middle")
    closing = get_phase(analysis, "closing")

    print("=" * 60)
    print("CALL ANALYSIS REPORT")
    print("=" * 60)
    print("\n-- Sentiment by Phase ----------------------------------")
    print(f"  Opening  [{opening.sentiment:>10}]  {opening.reason}")
    print(f"  Middle   [{middle.sentiment:>10}]  {middle.reason}")
    print(f"  Closing  [{closing.sentiment:>10}]  {closing.reason}")
    print("\n-- Trajectory ------------------------------------------")
    print(f"  {analysis.sentiment_trajectory.upper()}: {analysis.trajectory_summary}")
    print("\n-- Resolution ------------------------------------------")
    print(f"  Quality : {analysis.resolution.quality}")
    print(f"  Resolved: {analysis.resolution.was_resolved}")
    print(f"  Reason  : {analysis.resolution.reason}")
    print(f"\n-- CSAT Prediction -------------------------------------")
    print(f"  {analysis.csat_prediction}/5")
    print("\n-- Reasoning -------------------------------------------")
    print(f"  {analysis.reasoning}")
    print("=" * 60)


analyzer = CallAnalyzer()

for label, transcript in [
    ("RESOLVED CALL", TRANSCRIPT_RESOLVED),
    ("UNRESOLVED CALL", TRANSCRIPT_UNRESOLVED),
]:
    print(f"\n\n{'#' * 60}\n# {label}\n{'#' * 60}")
    analysis = analyzer(transcript=transcript)
    print_report(analysis)
```

## Examples

!!! example

    === "Resolved call"

        ```python
        analyzer = CallAnalyzer()
        analysis = analyzer(transcript=TRANSCRIPT_RESOLVED)
        closing = get_phase(analysis, "closing")

        print(f"Trajectory : {analysis.sentiment_trajectory}")
        print(f"Resolution : {analysis.resolution.quality}")
        print(f"CSAT       : {analysis.csat_prediction}/5")
        print(f"Closing    : {closing.sentiment} - {closing.reason}")
        ```

        Possible Output:

        ```text
        Trajectory : improved
        Resolution : fully_resolved
        CSAT       : 5/5
        Closing    : satisfied - The customer thanks the agent and says the issue was sorted out quickly.
        ```

    === "Escalated call"

        ```python
        analyzer = CallAnalyzer()
        analysis = analyzer(transcript=TRANSCRIPT_UNRESOLVED)
        opening = get_phase(analysis, "opening")
        closing = get_phase(analysis, "closing")

        print(f"Trajectory : {analysis.sentiment_trajectory}")
        print(f"Resolution : {analysis.resolution.quality}")
        print(f"CSAT       : {analysis.csat_prediction}/5")
        print(f"Opening    : {opening.sentiment} - {opening.reason}")
        print(f"Closing    : {closing.sentiment} - {closing.reason}")
        ```

        Possible Output:

        ```text
        Trajectory : worsened
        Resolution : escalated
        CSAT       : 1/5
        Opening    : frustrated - The customer opens the call complaining about a duplicate charge and a long wait.
        Closing    : angry - The customer says the situation is ridiculous and ends with "Whatever."
        ```

    === "Audio recording"

        ```python
        analyzer = CallAnalyzer()
        analysis = analyzer(audio=open("call.mp3", "rb").read())

        print(f"Trajectory : {analysis.sentiment_trajectory}")
        print(f"Resolution : {analysis.resolution.quality}")
        print(f"CSAT       : {analysis.csat_prediction}/5")
        ```

        Possible Output:

        ```text
        Trajectory : stable_negative
        Resolution : partially_resolved
        CSAT       : 2/5
        ```

    === "Async batch"

        ```python
        import asyncio
        import msgflux.nn.functional as F


        async def main():
            analyzer = CallAnalyzer()
            transcripts = [TRANSCRIPT_RESOLVED, TRANSCRIPT_UNRESOLVED]

            results = await F.amap_gather(
                analyzer,
                kwargs_list=[{"transcript": t} for t in transcripts],
            )

            for i, analysis in enumerate(results, 1):
                print(
                    f"Call {i}: "
                    f"trajectory={analysis.sentiment_trajectory}, "
                    f"resolved={analysis.resolution.was_resolved}, "
                    f"csat={analysis.csat_prediction}/5"
                )


        asyncio.run(main())
        ```

        Possible Output:

        ```text
        Call 1: trajectory=improved, resolved=True, csat=5/5
        Call 2: trajectory=worsened, resolved=False, csat=1/5
        ```

## Extending

### Adding agent quality metrics

Add another nested struct when you want to score the agent as well as the customer outcome:

```python
class AgentQuality(msgspec.Struct):
    empathy: Literal["high", "medium", "low"]
    protocol_compliance: bool


class CallAnalysis(msgspec.Struct):
    ...
    agent_quality: AgentQuality
```

### Routing by resolution quality

Act on the result immediately:

```python
analysis = analyzer(transcript=TRANSCRIPT_UNRESOLVED)

if analysis.resolution.quality == "escalated":
    flag_for_supervisor(analysis)
elif analysis.csat_prediction <= 2:
    schedule_callback(analysis)
```

### Building a service health dashboard

Run the analyzer over a queue and aggregate the outputs:

```python
results = await F.amap_gather(
    analyzer,
    kwargs_list=[{"transcript": t} for t in transcripts],
)

from collections import Counter

trajectories = Counter(r.sentiment_trajectory for r in results)
avg_csat = sum(r.csat_prediction for r in results) / len(results)
resolved_pct = sum(r.resolution.was_resolved for r in results) / len(results) * 100

print(f"Resolved: {resolved_pct:.1f}%  |  Avg CSAT: {avg_csat:.2f}")
print(f"Trajectories: {dict(trajectories)}")
```

## Complete Script

??? example "Expand full script"
    ```python
    # /// script
    # dependencies = []
    # ///

    from typing import Annotated, Literal

    import msgspec

    import msgflux as mf
    import msgflux.nn as nn

    mf.load_dotenv()


    chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    stt_model = mf.Model.speech_to_text("openai/whisper-1")


    class CallPhase(msgspec.Struct):
        stage: Annotated[
            Literal["opening", "middle", "closing"],
            msgspec.Meta(description="Conversation stage. Return opening, middle, and closing exactly once."),
        ]
        sentiment: Annotated[
            Literal["positive", "neutral", "satisfied", "frustrated", "angry"],
            msgspec.Meta(
                description=(
                    "Customer sentiment in this stage. Use satisfied only when the call clearly ends well."
                )
            ),
        ]
        reason: Annotated[
            str,
            msgspec.Meta(description="Short evidence from the transcript that justifies the sentiment."),
        ]


    class ResolutionAssessment(msgspec.Struct):
        was_resolved: Annotated[
            bool,
            msgspec.Meta(description="True if the customer's core issue was addressed by the end of the call."),
        ]
        quality: Annotated[
            Literal["fully_resolved", "partially_resolved", "unresolved", "escalated"],
            msgspec.Meta(
                description=(
                    "fully_resolved: issue closed and customer acknowledged it; "
                    "partially_resolved: progress made but follow-up still required; "
                    "unresolved: no meaningful progress; "
                    "escalated: transferred to another team or tier."
                )
            ),
        ]
        reason: Annotated[
            str,
            msgspec.Meta(description="Concrete evidence from the transcript that supports the resolution verdict."),
        ]


    class CallAnalysis(msgspec.Struct):
        reasoning: Annotated[
            str,
            msgspec.Meta(
                description="Let's think step by step in order to analyze the transcript consistently before filling the fields."
            ),
        ]
        phases: Annotated[
            list[CallPhase],
            msgspec.Meta(description="Exactly three entries in this order: opening, middle, closing."),
        ]
        sentiment_trajectory: Annotated[
            Literal["improved", "stable_positive", "stable_neutral", "stable_negative", "worsened", "volatile"],
            msgspec.Meta(description="Overall emotional arc of the customer from opening to closing."),
        ]
        trajectory_summary: Annotated[
            str,
            msgspec.Meta(description="One or two sentences describing the emotional journey across the call."),
        ]
        resolution: ResolutionAssessment
        csat_prediction: Annotated[
            int,
            msgspec.Meta(description="Predicted customer satisfaction score from 1 to 5."),
        ]


    class CallTranscriber(nn.Transcriber):
        """Transcribes call audio into msg.call.transcript."""

        model = stt_model
        message_fields = {"task_multimodal": {"audio": "audio_content"}}
        response_mode = "call.transcript"


    class _Analyzer(nn.Agent):
        model = chat_model
        system_message = """
        You are a call quality analyst for customer support teams.
        """
        instructions = """
        Analyze the transcript across the opening, middle, and closing stages.

        Rules:
        - Return exactly three items in phases, in this order: opening, middle, closing.
        - Ground every sentiment and resolution judgment in transcript evidence.
        - Use satisfied only when the closing stage clearly ends positive after progress or resolution.
        - Mark quality as escalated when the issue is handed to another team or tier.
        - Predict csat_prediction on a 1 to 5 scale.
        """
        generation_schema = CallAnalysis
        templates = {"task": "Transcript:\n{{ transcript }}"}
        config = {"verbose": True}


    class CallAnalyzer(nn.Module):
        def __init__(self):
            super().__init__()
            self.transcriber = CallTranscriber()
            self.agent = _Analyzer()

        def forward(self, transcript: str | None = None, audio: bytes | None = None) -> CallAnalysis:
            if audio:
                msg = mf.Message()
                msg.audio_content = audio
                self.transcriber(msg)
                transcript = msg.call.transcript
            return self.agent(transcript=transcript)

        async def aforward(self, transcript: str | None = None, audio: bytes | None = None) -> CallAnalysis:
            if audio:
                msg = mf.Message()
                msg.audio_content = audio
                await self.transcriber.acall(msg)
                transcript = msg.call.transcript
            return await self.agent.acall(transcript=transcript)


    def get_phase(analysis: CallAnalysis, stage: str) -> CallPhase:
        return next(phase for phase in analysis.phases if phase.stage == stage)


    def print_report(analysis: CallAnalysis) -> None:
        opening = get_phase(analysis, "opening")
        middle = get_phase(analysis, "middle")
        closing = get_phase(analysis, "closing")

        print("=" * 60)
        print("CALL ANALYSIS REPORT")
        print("=" * 60)
        print("\n-- Sentiment by Phase ----------------------------------")
        print(f"  Opening  [{opening.sentiment:>10}]  {opening.reason}")
        print(f"  Middle   [{middle.sentiment:>10}]  {middle.reason}")
        print(f"  Closing  [{closing.sentiment:>10}]  {closing.reason}")
        print("\n-- Trajectory ------------------------------------------")
        print(f"  {analysis.sentiment_trajectory.upper()}: {analysis.trajectory_summary}")
        print("\n-- Resolution ------------------------------------------")
        print(f"  Quality : {analysis.resolution.quality}")
        print(f"  Resolved: {analysis.resolution.was_resolved}")
        print(f"  Reason  : {analysis.resolution.reason}")
        print(f"\n-- CSAT Prediction -------------------------------------")
        print(f"  {analysis.csat_prediction}/5")
        print("\n-- Reasoning -------------------------------------------")
        print(f"  {analysis.reasoning}")
        print("=" * 60)


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


    if __name__ == "__main__":
        import sys

        analyzer = CallAnalyzer()
        mode = sys.argv[1] if len(sys.argv) > 1 else "text"

        if mode == "audio":
            print("=== Audio demo ===")
            analysis = analyzer(audio=open("call.mp3", "rb").read())
            print_report(analysis)
        else:
            for label, transcript in [
                ("RESOLVED CALL", TRANSCRIPT_RESOLVED),
                ("UNRESOLVED CALL", TRANSCRIPT_UNRESOLVED),
            ]:
                print(f"\n\n{'#' * 60}\n# {label}\n{'#' * 60}")
                analysis = analyzer(transcript=transcript)
                print_report(analysis)
    ```

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) - agent configuration and execution
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) - `msgspec.Struct` and reasoning-oriented output
- [Reasoning](../learn/nn/agent/reasoning.md) - model-level and schema-level reasoning
- [nn.Transcriber](../learn/nn/transcriber.md) - multimodal transcription pipelines
- [Functional API](../learn/nn/functional.md) - `amap_gather` and parallel execution
