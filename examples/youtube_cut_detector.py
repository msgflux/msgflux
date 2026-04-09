# /// script
# dependencies = [
#   "youtube-transcript-api",
# ]
# ///

import re
from typing import Annotated

import msgspec

import msgflux as mf
import msgflux.nn as nn
from youtube_transcript_api import YouTubeTranscriptApi

mf.load_dotenv()

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")


def fetch_transcript(url: str) -> list:
    """Download transcript snippets for a YouTube URL."""
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    return list(YouTubeTranscriptApi().fetch(match.group(1)))


def format_transcript(snippets: list, max_chars: int = 12_000) -> str:
    """Render snippets as a timestamped string."""
    lines = []
    for snippet in snippets:
        minutes, seconds = divmod(int(snippet.start), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {snippet.text}")
    return "\n".join(lines)[:max_chars]


class VideoCut(msgspec.Struct):
    start_seconds: Annotated[
        int,
        msgspec.Meta(description="Start time of the clip in seconds."),
    ]
    end_seconds: Annotated[
        int,
        msgspec.Meta(description="End time of the clip in seconds."),
    ]
    title: Annotated[
        str,
        msgspec.Meta(description="Short, punchy title for the clip."),
    ]
    hook: Annotated[
        str,
        msgspec.Meta(description="Opening line or angle that makes the clip immediately compelling."),
    ]
    viral_score: Annotated[
        int,
        msgspec.Meta(description="Viral potential score from 1 to 10."),
    ]


class VideoCutAnalysis(msgspec.Struct):
    reasoning: Annotated[
        str,
        msgspec.Meta(
            description="Let's think step by step in order to choose the strongest short-form cuts from the full transcript."
        ),
    ]
    cuts: Annotated[
        list[VideoCut],
        msgspec.Meta(description="Return up to max_cuts clips, ordered from strongest to weakest."),
    ]
    strategy: Annotated[
        str,
        msgspec.Meta(description="Short summary of the overall cutting strategy."),
    ]


class CutAnalyzer(nn.Agent):
    """Analyzes a transcript and returns the strongest cut candidates."""

    model = chat_model
    system_message = """
    You are a short-form video editor who turns long YouTube videos into strong clip candidates.
    """
    instructions = """
    Read the full transcript before choosing clips.

    Rules:
    - Pick moments that stand alone as shorts.
    - Prefer hooks, punchlines, reveals, strong opinions, concise stories, and clear payoffs.
    - Avoid overlapping clips.
    - Keep each cut long enough to make sense without the rest of the video.
    - Score each cut from 1 to 10 for viral potential.
    """
    generation_schema = VideoCutAnalysis
    templates = {
        "task": "Select up to {{ max_cuts }} short-form clips from this transcript.\n\nTranscript:\n{{ transcript }}"
    }
    config = {"verbose": True}


class VideoCutPipeline(nn.Module):
    """Fetches a YouTube transcript and detects the best cut intervals."""

    def __init__(self, max_cuts: int = 5):
        super().__init__()
        self.max_cuts = max_cuts
        self.analyzer = CutAnalyzer()

    def _remove_overlaps(self, cuts: list[VideoCut]) -> list[VideoCut]:
        """Keep the strongest non-overlapping cuts in model-returned order."""
        accepted: list[VideoCut] = []

        for cut in cuts:
            if cut.end_seconds <= cut.start_seconds:
                continue

            overlaps = any(
                cut.start_seconds < kept.end_seconds
                and cut.end_seconds > kept.start_seconds
                for kept in accepted
            )
            if overlaps:
                continue

            accepted.append(cut)

            if len(accepted) >= self.max_cuts:
                break

        return accepted

    def forward(self, url: str) -> VideoCutAnalysis:
        transcript = format_transcript(fetch_transcript(url))
        result = self.analyzer(transcript=transcript, max_cuts=self.max_cuts)
        result.cuts = self._remove_overlaps(result.cuts)
        return result

    async def aforward(self, url: str) -> VideoCutAnalysis:
        transcript = format_transcript(fetch_transcript(url))
        result = await self.analyzer.acall(transcript=transcript, max_cuts=self.max_cuts)
        result.cuts = self._remove_overlaps(result.cuts)
        return result


if __name__ == "__main__":
    pipeline = VideoCutPipeline(max_cuts=5)
    result = pipeline.forward("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    print("=== STRATEGY ===")
    print(result.strategy)
    print()
    print("=== CUTS ===")
    for i, clip in enumerate(result.cuts, 1):
        print(
            f"{i}. [{clip.start_seconds}s -> {clip.end_seconds}s] "
            f"{clip.title} (score: {clip.viral_score}/10)"
        )
        print(f"   Hook: {clip.hook}")
    print()
    print("=== REASONING ===")
    print(result.reasoning)
