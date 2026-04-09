# YouTube Cut Detector

<span class="tag tag-green">Beginner</span><span class="tag tag-gray">Generation Schema</span><span class="tag tag-gray">Reasoning</span>

Every long-form video contains a handful of moments that could stand alone as short clips. Finding them manually means watching the whole video, marking timestamps, and guessing which moments will still work out of context.

## The Problem

The naive approach is to paste a transcript into one prompt and ask for clips. It works until the model starts picking fragments that look interesting alone but ignore the flow of the video.

- A buildup gets cut before the payoff.
- Repeated moments get picked at random.
- Strong lines get chosen without enough context to make sense as a short.
- When a clip underperforms, there is no clear record of why it was selected.

## The Plan

We will build a pipeline that:

- downloads a YouTube transcript
- formats each line with a timestamp
- asks a model to find the strongest clip candidates
- returns a structured list of cuts, plus the reasoning and overall strategy

Instead of using a `Signature` with parallel lists like `start_seconds`, `end_seconds`, `titles`, and `hooks`, we will model one clip once with `msgspec.Struct` and return `cuts: list[VideoCut]`. This matches the domain better and avoids keeping several lists aligned by index.

## Architecture

```
YouTube URL
      |
      v
fetch_transcript() -> snippets
      |
      v
format_transcript() -> "[MM:SS] text\n..."
      |
      v
CutAnalyzer
  generation_schema = VideoCutAnalysis
      |
      +-- reasoning
      +-- cuts[list[VideoCut]]
      +-- strategy
      |
      v
VideoCutPipeline -> structured result
```

## Setup

```bash
pip install youtube-transcript-api
```

--8<-- "docs/_includes/init_chat_completion_model.md"

## Step 1 - Fetch and Format the Transcript

`YouTubeTranscriptApi` returns transcript snippets with text and start time. We convert them into one string with `[MM:SS]` markers so the model can point to exact moments.

```python
import re
from youtube_transcript_api import YouTubeTranscriptApi


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
```

!!! tip
    `max_chars=12_000` is a simple guardrail for long videos. If the transcript is much longer, lower the limit or chunk the video before analysis.

## Step 2 - Output Schema

`VideoCut` models one clip. `VideoCutAnalysis` wraps the full answer: the reasoning trace, the chosen cuts, and the overall strategy.

```python
from typing import Annotated

import msgspec


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
```

This shape is easier to work with than several parallel lists. Each cut carries its own timestamps, title, hook, and score in one object.

## Step 3 - CutAnalyzer

The agent reads the full transcript, reasons over the full video, and returns a `VideoCutAnalysis` object.

```python
import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")


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
```

Because the schema already includes `reasoning`, there is no need for `Signature` or for a `final_answer` wrapper.

## Step 4 - VideoCutPipeline

The pipeline fetches the transcript, formats it, passes it to the analyzer, and then removes overlapping cuts before returning the result.

```python
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


pipeline = VideoCutPipeline(max_cuts=5)
```

This makes the example safer in practice: the model proposes candidates, and the pipeline enforces the non-overlap rule afterward.

## Examples

!!! example

    === "Single video"

        ```python
        pipeline = VideoCutPipeline(max_cuts=5)
        result = pipeline.forward("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        print(result.strategy)
        print()
        for i, clip in enumerate(result.cuts, 1):
            print(
                f"{i}. [{clip.start_seconds}s -> {clip.end_seconds}s] "
                f"{clip.title} (score: {clip.viral_score}/10)"
            )
            print(f"   Hook: {clip.hook}")
        ```

        Possible Output:

        ```text
        Focus on moments with a clear emotional turn, a memorable line, or a payoff that works without extra setup.

        1. [42s -> 68s] The Line Everyone Waits For (score: 9/10)
           Hook: This is the exact moment the whole video pays off.
        2. [95s -> 126s] The Unexpected Turn (score: 8/10)
           Hook: What sounds routine suddenly becomes the most surprising part of the story.
        ```

    === "Inspecting the reasoning trace"

        ```python
        result = pipeline.forward("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        print(result.reasoning)
        ```

        Possible Output:

        ```text
        The strongest clips are the segments with a complete setup and payoff in a short window.
        I avoided repeated chorus moments and prioritized lines that can hook a viewer immediately.
        ```

    === "Multiple videos (async)"

        ```python
        import asyncio
        import msgflux.nn.functional as F


        async def main():
            pipeline = VideoCutPipeline(max_cuts=3)
            urls = [
                "https://www.youtube.com/watch?v=VIDEO_1",
                "https://www.youtube.com/watch?v=VIDEO_2",
                "https://www.youtube.com/watch?v=VIDEO_3",
            ]

            results = await F.amap_gather(
                pipeline,
                kwargs_list=[{"url": url} for url in urls],
            )

            for url, result in zip(urls, results):
                print(f"\n{url}")
                for clip in result.cuts:
                    print(f"  [{clip.start_seconds}s -> {clip.end_seconds}s] {clip.title}")


        asyncio.run(main())
        ```

        Possible Output:

        ```text
        https://www.youtube.com/watch?v=VIDEO_1
          [12s -> 35s] The Fastest Explanation
          [77s -> 104s] The Strong Opinion

        https://www.youtube.com/watch?v=VIDEO_2
          [24s -> 51s] The Story Twist
          [131s -> 160s] The Best Punchline
        ```

## Extending

### Filtering by viral score

```python
result = pipeline.forward(url)
top_cuts = [clip for clip in result.cuts if clip.viral_score >= 8]
```

### Adding a topic label

If you want to categorize clips by theme, add one more field to `VideoCut`:

```python
class VideoCut(msgspec.Struct):
    ...
    topic: str
```

### Exporting timestamps for a video editor

```python
def to_edl(cuts: list[VideoCut]) -> str:
    lines = ["TITLE: Auto Cuts", "FCM: NON-DROP FRAME", ""]
    for i, clip in enumerate(cuts, 1):
        lines.append(
            f"{i:03d}  AX  V  C  {clip.start_seconds}s {clip.end_seconds}s"
        )
    return "\n".join(lines)


print(to_edl(result.cuts))
```

## Complete Script

??? example "Expand full script"
    ```python
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
    ```

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) - agent configuration and execution
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) - `msgspec.Struct` and structured outputs
- [Reasoning](../learn/nn/agent/reasoning.md) - reasoning traces and schema-level reasoning
- [Functional API](../learn/nn/functional.md) - `amap_gather` for concurrent execution
