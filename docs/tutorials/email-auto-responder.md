# Email Auto Responder

Classify an incoming email, draft a contextually appropriate reply, review it for quality and tone, and keep revising until it passes — using the `Inline` DSL's `@{while}` loop to manage the review cycle declaratively.

## What You'll Build

```
Incoming email
       │
       ▼
  Classifier ──── Signature: email_body → intent, urgency, tone, sender_name
       │
       ▼
  Drafter ──────── Signature: email_body, intent, urgency, tone → draft: str
       │
       ▼
  Reviewer ─────── Signature: email_body, draft → approved: bool, feedback: str, score: float
       │
  @{ approved == False }
       │  ↓ revise with feedback
       └─ Reviser ── Signature: draft, feedback → draft: str (overwrite)
              │
              ▼
          Reviewer (again)
              │ approved == True
              ▼
         msg.final_reply
```

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Signatures

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField, Inline
from typing import Literal


class ClassifyEmail(Signature):
    """Classify the incoming email to inform the reply strategy."""

    email_body: str = InputField(desc="The full text of the incoming email")

    intent: Literal[
        "question", "complaint", "request", "follow_up", "cancellation", "praise"
    ] = OutputField(desc="Primary intent of the email")
    urgency: Literal["low", "medium", "high"] = OutputField(
        desc="How urgently this email needs a response"
    )
    tone: Literal["formal", "neutral", "informal"] = OutputField(
        desc="Appropriate reply tone based on sender style"
    )
    sender_name: str = OutputField(desc="Sender's first name extracted from the email")


class DraftReply(Signature):
    """Draft a professional reply to the email."""

    email_body: str = InputField(desc="The original email")
    intent: str = InputField(desc="Classified intent")
    urgency: str = InputField(desc="Urgency level")
    tone: str = InputField(desc="Reply tone to use")

    draft: str = OutputField(
        desc="A complete, ready-to-send reply addressing all points raised"
    )


class ReviewDraft(Signature):
    """Review a draft reply for quality, accuracy, and tone before sending."""

    email_body: str = InputField(desc="The original email")
    draft: str = InputField(desc="The draft reply to review")

    approved: bool = OutputField(
        desc="True if the draft is ready to send, False if it needs revision"
    )
    feedback: str = OutputField(
        desc="Specific, actionable feedback if not approved; empty string if approved"
    )
    score: float = OutputField(
        desc="Quality score from 0.0 to 1.0 (approved when >= 0.8)"
    )


class ReviseDraft(Signature):
    """Revise a draft based on reviewer feedback."""

    draft: str = InputField(desc="The draft that needs improvement")
    feedback: str = InputField(desc="Specific feedback from the reviewer")

    draft: str = OutputField(desc="Improved version of the draft")
```

!!! note
    `ReviseDraft` uses `draft` as both input and output, so the revised reply
    overwrites `msg.draft` in place — the `Inline` loop always reads the latest version.

---

## Step 2 — Agents

```python
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class Classifier(nn.Agent):
    model = model
    signature = ClassifyEmail
    config = {"verbose": True}


class Drafter(nn.Agent):
    model = model
    signature = DraftReply
    config = {"verbose": True}


class Reviewer(nn.Agent):
    model = model
    signature = ReviewDraft
    config = {"verbose": True}


class Reviser(nn.Agent):
    model = model
    signature = ReviseDraft
    config = {"verbose": True}
```

---

## Step 3 — Wiring with `Inline`

The `@{condition}: actions;` node runs `actions` repeatedly while `condition` is true.
Here the loop keeps drafting and reviewing until `approved` is `True`:

```python
pipeline = Inline(
    "classifier -> drafter -> reviewer -> @{approved == False}: reviser -> reviewer;",
    {
        "classifier": Classifier(),
        "drafter":    Drafter(),
        "reviewer":   Reviewer(),
        "reviser":    Reviser(),
    },
)
```

!!! tip
    Set `max_iterations` on `Inline` to cap the number of revision cycles and avoid infinite loops:

    ```python
    pipeline = Inline("...", {...}, max_iterations=5)
    ```

---

## Step 4 — Running the Pipeline

```python
msg = Message()
msg.email_body = """
Hi there,

I placed an order three weeks ago (order #ORD-9921) and it still hasn't arrived.
The tracking page just says "processing". This is really frustrating — I needed
this for a trip that already happened. I'd like a refund or an explanation.

Thanks,
Maria
"""

# Seed: not yet approved
msg.approved = False

pipeline(msg)

print(f"Intent:       {msg.intent}")
print(f"Urgency:      {msg.urgency}")
print(f"Final score:  {msg.score:.2f}")
print(f"\nFinal reply:\n{msg.draft}")
```

Sample output (abbreviated):

```
[classifier][response] {'intent': 'complaint', 'urgency': 'high', 'tone': 'neutral', ...}
[drafter][response]    {'draft': 'Dear Maria, ...'}
[reviewer][response]   {'approved': False, 'score': 0.62, 'feedback': 'Add empathy ...'}
[reviser][response]    {'draft': 'Dear Maria, I sincerely apologize ...'}
[reviewer][response]   {'approved': True, 'score': 0.91, 'feedback': ''}

Intent:       complaint
Urgency:      high
Final score:  0.91

Final reply:
Dear Maria, I sincerely apologize for the inconvenience...
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField, Inline
from typing import Literal


# ── Signatures ────────────────────────────────────────────────────────────────

class ClassifyEmail(Signature):
    """Classify the incoming email to inform the reply strategy."""

    email_body: str = InputField(desc="The full text of the incoming email")
    intent: Literal[
        "question", "complaint", "request", "follow_up", "cancellation", "praise"
    ] = OutputField(desc="Primary intent of the email")
    urgency: Literal["low", "medium", "high"] = OutputField(
        desc="How urgently this email needs a response"
    )
    tone: Literal["formal", "neutral", "informal"] = OutputField(
        desc="Appropriate reply tone based on sender style"
    )
    sender_name: str = OutputField(desc="Sender's first name")


class DraftReply(Signature):
    """Draft a professional reply to the email."""

    email_body: str = InputField(desc="The original email")
    intent: str = InputField(desc="Classified intent")
    urgency: str = InputField(desc="Urgency level")
    tone: str = InputField(desc="Reply tone to use")
    draft: str = OutputField(desc="Complete, ready-to-send reply")


class ReviewDraft(Signature):
    """Review a draft reply for quality, accuracy, and tone."""

    email_body: str = InputField(desc="The original email")
    draft: str = InputField(desc="The draft reply")
    approved: bool = OutputField(desc="True if ready to send")
    feedback: str = OutputField(desc="Actionable feedback if not approved")
    score: float = OutputField(desc="Quality score 0.0-1.0")


class ReviseDraft(Signature):
    """Revise a draft based on reviewer feedback."""

    draft: str = InputField(desc="Draft to improve")
    feedback: str = InputField(desc="Specific feedback")
    draft: str = OutputField(desc="Improved draft")


# ── Agents ────────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class Classifier(nn.Agent):
    model = model
    signature = ClassifyEmail
    config = {"verbose": True}


class Drafter(nn.Agent):
    model = model
    signature = DraftReply
    config = {"verbose": True}


class Reviewer(nn.Agent):
    model = model
    signature = ReviewDraft
    config = {"verbose": True}


class Reviser(nn.Agent):
    model = model
    signature = ReviseDraft
    config = {"verbose": True}


# ── Pipeline ──────────────────────────────────────────────────────────────────

pipeline = Inline(
    "classifier -> drafter -> reviewer -> @{approved == False}: reviser -> reviewer;",
    {
        "classifier": Classifier(),
        "drafter":    Drafter(),
        "reviewer":   Reviewer(),
        "reviser":    Reviser(),
    },
    max_iterations=5,
)


# ── Run ───────────────────────────────────────────────────────────────────────

emails = [
    """Hi, I placed order #ORD-9921 three weeks ago and it hasn't arrived.
    The tracking just says "processing". I needed it for a trip that already happened.
    I'd like a refund or explanation. — Maria""",

    """Hey! Quick question — does your Premium plan include API access?
    I'm evaluating options for our startup. Thanks! — Jake""",

    """To whom it may concern,
    I wish to formally cancel my subscription effective immediately.
    Please confirm cancellation and refund the current billing period.
    Regards, Dr. Chen""",
]

for email in emails:
    msg = Message()
    msg.email_body = email
    msg.approved = False

    pipeline(msg)

    print(f"\n{'─' * 60}")
    print(f"Intent: {msg.intent} | Urgency: {msg.urgency} | Score: {msg.score:.2f}")
    print(f"\nReply:\n{msg.draft}")
```

---

## Async Version

```python
import asyncio

pipeline_async = Inline(
    "classifier -> drafter -> reviewer -> @{approved == False}: reviser -> reviewer;",
    {
        "classifier": Classifier(),
        "drafter":    Drafter(),
        "reviewer":   Reviewer(),
        "reviser":    Reviser(),
    },
    max_iterations=5,
)

async def main():
    msg = Message()
    msg.email_body = "Your invoice for $1,200 is attached. Payment due in 30 days."
    msg.approved = False
    await pipeline_async.acall(msg)
    print(msg.draft)

asyncio.run(main())
```

---

## DSL At a Glance

| Syntax | Meaning |
|---|---|
| `a -> b -> c` | Sequential execution |
| `[a, b, c]` | Parallel execution (same message) |
| `{cond?a,b}` | Conditional branch |
| `@{cond}: a -> b;` | While loop — run `a -> b` while condition holds |
