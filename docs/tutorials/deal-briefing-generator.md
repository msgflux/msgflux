# Deal Briefing Generator

---

## The Problem

Sales reps finish a call and move on to the next one. Context fades fast — the specific objection raised, the budget number mentioned in passing, the follow-up that was implicitly agreed to. Voice recordings exist but no one has time to re-listen before the next meeting. The rep walks in underprepared, asks questions already answered in the previous call, and loses credibility.

The information is there. It just never gets turned into anything actionable.

---

## The Plan

We will build a pipeline that takes a sales call recording and produces a structured briefing the rep can review — or listen to — before the next meeting.

The call is transcribed first. From there, an extractor identifies the key signals that a rep actually needs: what pain points the prospect raised, what objections came up, what was agreed on as a next step, and whether a date or budget was mentioned. A drafting agent turns those signals into a concise briefing document with clearly labeled sections.

The same briefing is then narrated via text-to-speech so the rep can listen on the way to the next call instead of reading. The audio file is saved alongside the text output.

Few-shot examples anchor the extraction to a consistent format — showing the model what "pain point" means in a sales context versus a casual complaint, and what counts as a committed next step versus a vague "let's reconnect."

---

## Architecture

```
Call recording (audio)
        │
        ▼
   Transcriber (Whisper)
        │ transcript text
        ▼
   Extractor ─── mf.Example × N (labeled call excerpts)
        │
        │  pain_points, objections, next_step, budget, timeline
        ▼
   Drafter
        │ briefing text
        ▼
   Speaker (TTS)
        │
        ▼
   briefing.md + briefing.mp3
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"
