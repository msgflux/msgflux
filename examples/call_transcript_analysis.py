from typing import Literal

import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Signature, InputField, OutputField
from msgflux.generation.reasoning import ChainOfThought

mf.load_dotenv()


# ── Models ────────────────────────────────────────────────────────────────────

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model  = mf.Model.speech_to_text("openai/whisper-1")


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


# ── Transcriber ───────────────────────────────────────────────────────────────

class CallTranscriber(nn.Transcriber):
    """Transcribes call audio into msg.call.transcript."""
    model          = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode  = "call.transcript"


# ── Analyzer ──────────────────────────────────────────────────────────────────

class _Analyzer(nn.Agent):
    model             = chat_model
    signature         = CallAnalysisSignature
    generation_schema = ChainOfThought
    config            = {"verbose": True}


class CallAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = CallTranscriber()
        self.agent       = _Analyzer()

    def forward(self, transcript: str | None = None, audio: bytes | None = None) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            self.transcriber(msg)
            transcript = msg.call.transcript
        raw = self.agent({"transcript": transcript})
        return {**raw.get("final_answer", raw), "reasoning": raw.get("reasoning", "")}

    async def aforward(self, transcript: str | None = None, audio: bytes | None = None) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            await self.transcriber.acall(msg)
            transcript = msg.call.transcript
        raw = await self.agent.acall({"transcript": transcript})
        return {**raw.get("final_answer", raw), "reasoning": raw.get("reasoning", "")}


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(a: dict) -> None:
    print("=" * 60)
    print("CALL ANALYSIS REPORT")
    print("=" * 60)
    print("\n── Sentiment by Phase ──────────────────────────────────")
    print(f"  Opening  [{a['opening_sentiment']:>10}]  {a['opening_reason']}")
    print(f"  Middle   [{a['middle_sentiment']:>10}]  {a['middle_reason']}")
    print(f"  Closing  [{a['closing_sentiment']:>10}]  {a['closing_reason']}")
    print(f"\n── Trajectory: {a['sentiment_trajectory'].upper()}")
    print(f"  {a['trajectory_summary']}")
    print(f"\n── Resolution: {a['resolution_quality']} ({'resolved' if a['was_resolved'] else 'unresolved'})")
    print(f"  {a['resolution_reason']}")
    print(f"\n── CSAT Prediction: {a['csat_prediction']}/5")
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
