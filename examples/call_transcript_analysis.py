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
