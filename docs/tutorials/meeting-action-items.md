# Meeting Action Items Tracker

---

## The Problem

Meetings generate commitments that never become tasks. Someone says "let's look into that", someone else says "I'll handle it" — and three days later, no one remembers who was supposed to do what. The recording exists. The transcript could be parsed. But without a quality bar, everything sounds like an action item.

The harder problem is noise. A model without guidance treats "we should probably revisit this at some point" the same as "Ana will send the contract by Friday." Both are future-oriented statements. Only one is a real action item.

---

## The Plan

We will build a pipeline that takes a meeting transcript — or an audio recording — and produces a structured checklist of action items, each with an assignee and a deadline when one was stated.

The transcript is the primary input; if audio is provided, it is transcribed first and the rest of the pipeline runs identically. An extractor reads the transcript and identifies action items. Few-shot examples teach it the distinction that matters: a committed action ("Pedro will send the proposal by Thursday") versus a vague intention ("we'll figure that out later") versus an implicit assignment ("so that's on Ana's side"). The extractor captures all three types differently — confirmed, vague, and implicit — so the output reflects the actual level of commitment.

A formatter agent turns the extracted items into a clean checklist, grouping by assignee and flagging items with no deadline or no owner.

---

## Architecture

```
Meeting input (text or audio)
        │
        ├── audio? → Transcriber (Whisper) → transcript
        │
        ▼
   Extractor ─── mf.Example × N (labeled transcript excerpts)
        │
        │  action_items: [{assignee, task, deadline, confidence}]
        ▼
   Formatter
        │
        ▼
   msg.checklist  (markdown checklist grouped by assignee)
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"
