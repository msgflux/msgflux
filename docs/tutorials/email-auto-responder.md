# Email Auto Responder

---

## The Problem

Here is how most teams handle incoming email at first.

```
Incoming email
       │
       ▼
┌──────────────────────────────────────────┐
│           GeneralAgent                   │
│                                          │
│   read email  ←──→  write reply          │
└──────────────────────────────────────────┘
         │ sends whatever comes out
         ▼
      reply (unreviewed)
```

- The agent replies without understanding intent. A complaint gets the same treatment as a quick question.
- Tone is inconsistent. A formal cancellation request might get a casual response.
- There is no quality gate. A poorly drafted reply goes out as-is.
- When a reply is bad, you rewrite the prompt and hope. You have no record of why it failed.

You are shipping unreviewed text.

---

## The Plan

We will build a flux based on the Reflection architecture that classifies the email before responding, drafts a calibrated reply, and runs it through a reviewer before sending.

A `Classifier` reads the email and extracts `intent`, `urgency`, and `tone`. A `Drafter` uses those signals to write a context-aware reply. A `Reviewer` scores the draft and decides whether it's ready. If not, a `Reviser` incorporates the feedback — and the `Reviewer` runs again. The revision cycle is expressed declaratively with `Inline`'s `@{while}` construct, so the flux keeps iterating until the draft passes.

---

## Architecture

```
Incoming email
       │
       ▼
  Classifier ──── Signature: email_body → intent, urgency, tone, sender_name
       │
       ▼
  Drafter ──────── Signature: email_body, intent, urgency, tone → draft
       │
       ▼
  Reviewer ─────── Signature: email_body, draft → approved, feedback, score
       │
  @{ approved == False }
       │  ↺ revise with feedback
       └─ Reviser ── Signature: draft, feedback → draft
              │
              ▼
          Reviewer (again)
              │ approved == True
              ▼
         msg.draft  (ready to send)
```

The flux is:

- **Adaptive** — tone and depth are driven by the classified intent, not a fixed prompt
- **Self-correcting** — the revision loop runs until quality passes, capped by `max_iterations`
- **Observable** — every classification decision and reviewer score is structured data on `msg`

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 — Classifying the Email

Before drafting anything, the flux needs to understand what kind of email arrived. The classifier extracts four signals that the rest of the flux depends on: the sender's primary `intent`, how urgently they need a response, what `tone` the reply should use, and the sender's name for personalization.

```python
import msgflux as mf
import msgflux.nn as nn
from typing import Literal


class ClassifyEmail(mf.Signature):
    """Classify the incoming email to inform the reply strategy."""

    email_body: str = mf.InputField(desc="The full text of the incoming email")

    intent: Literal[
        "question", "complaint", "request", "follow_up", "cancellation", "praise"
    ] = mf.OutputField(desc="Primary intent of the email")
    urgency: Literal["low", "medium", "high"] = mf.OutputField(
        desc="How urgently this email needs a response"
    )
    tone: Literal["formal", "neutral", "informal"] = mf.OutputField(
        desc="Appropriate reply tone based on sender style"
    )
    sender_name: str = mf.OutputField(desc="Sender's first name extracted from the email")
```

---

## Step 2 — Drafting and the Review Loop

Three signatures drive the quality loop. `DraftReply` consumes the classifier's output to write a calibrated first reply. `ReviewDraft` scores it and decides whether it's ready to send. `ReviseDraft` takes the reviewer's feedback and rewrites the draft — its output goes to the same `rsp` namespace as `DraftReply`, so `msg.rsp.draft` is updated in place on each iteration.

**Draft:**

```python
class DraftReply(mf.Signature):
    """Draft a professional reply to the email."""

    email_body: str = mf.InputField(desc="The original email")
    intent: str = mf.InputField(desc="Classified intent")
    urgency: str = mf.InputField(desc="Urgency level")
    tone: str = mf.InputField(desc="Reply tone to use")

    draft: str = mf.OutputField(
        desc="A complete, ready-to-send reply addressing all points raised"
    )
```

**Review:**

```python
class ReviewDraft(mf.Signature):
    """Review a draft reply for quality, accuracy, and tone before sending."""

    email_body: str = mf.InputField(desc="The original email")
    draft: str = mf.InputField(desc="The draft reply to review")

    feedback: str = mf.OutputField(
        desc="Specific, actionable feedback if not approved; empty string if approved"
    )
    approved: bool = mf.OutputField(
        desc="True if the draft is ready to send, False if it needs revision"
    )    
    score: float = mf.OutputField(
        desc="Quality score from 0.0 to 1.0 (approved when >= 0.8)"
    )
```

**Revise:**

```python
class ReviseDraft(mf.Signature):
    """Revise a draft based on reviewer feedback."""

    current_draft: str = mf.InputField(desc="The current draft that needs improvement")
    feedback: str = mf.InputField(desc="Specific feedback from the reviewer")

    draft: str = mf.OutputField(desc="Improved version of the draft")
```

---

## Step 3 — Agents

Each agent declares `message_fields` to read its inputs from the shared `msg` object and `response_mode` to write its outputs back to a dedicated namespace. This keeps each agent's output isolated — `cls` for classification, `rsp` for the current draft, `rev` for the review — and makes every field addressable with dotted paths like `cls.intent` or `rev.approved`.

```python
class Classifier(nn.Agent):
    model = model
    signature = ClassifyEmail
    message_fields = {"task": {"email_body": "email_body"}}
    response_mode = "cls"
    config = {"verbose": True}


class Drafter(nn.Agent):
    model = model
    signature = DraftReply
    message_fields = {
        "task": {
            # Map input_name to msg field
            "email_body": "email_body",
            "intent": "cls.intent",
            "urgency": "cls.urgency",
            "tone": "cls.tone"
        }
    }
    response_mode = "rsp"
    config = {"verbose": True}


class Reviewer(nn.Agent):
    model = model
    signature = ReviewDraft
    message_fields = {"task": {"email_body": "email_body", "draft": "rsp.draft"}}
    response_mode = "rev"
    config = {"verbose": True}


class Reviser(nn.Agent):
    model = model
    signature = ReviseDraft
    message_fields = {"task": {"current_draft": "rsp.draft", "feedback": "rev.feedback"}}
    response_mode = "rsp"
    config = {"verbose": True}
```

---

## Step 4 — Wiring the Pipeline

`Inline` composes the agents into a single flux. The `@{rev.approved == False}: reviser -> reviewer;` node runs the revision cycle while the reviewer has not approved the draft — then exits when `rev.approved` is `True`. The dotted path `rev.approved` resolves to `msg.rev.approved`, which is written in-place by the `Reviewer`'s `response_mode`.

```python
flux = mf.Inline(
    "classifier -> drafter -> reviewer -> @{rev.approved == False}: reviser -> reviewer;",
    {
        "classifier": Classifier(),
        "drafter":    Drafter(),
        "reviewer":   Reviewer(),
        "reviser":    Reviser(),
    },
    max_iterations=5,
)
```

!!! tip
    `max_iterations` caps the revision loop. Without it, a consistently failing draft would
    run indefinitely. Five iterations is a safe upper bound for most cases.

---

## Step 5 — Running the Flux

Pass the email in and let the flux run. Each agent writes to its own namespace on `msg`; results are accessed via dotted paths after the flux returns.

???+ example

    === "Sync"

        ```python
        msg = mf.Message()
        msg.email_body = """
        Hi there,

        I placed an order three weeks ago (order #ORD-9921) and it still hasn't arrived.
        The tracking page just says "processing". This is really frustrating — I needed
        this for a trip that already happened. I'd like a refund or an explanation.

        Thanks,
        Maria
        """

        flux(msg)

        print(f"Intent:      {msg.cls.intent}")
        print(f"Urgency:     {msg.cls.urgency}")
        print(f"Final score: {msg.rev.score:.2f}")
        print(f"\nFinal reply:\n{msg.rsp.draft}")
        ```

        ```
        [classifier][response] {'intent': 'complaint', 'urgency': 'high', 'tone': 'neutral', ...}
        [drafter][response]    {'draft': 'Dear Maria, ...'}
        [reviewer][response]   {'approved': False, 'score': 0.62, 'feedback': 'Add empathy ...'}
        [reviser][response]    {'draft': 'Dear Maria, I sincerely apologize ...'}
        [reviewer][response]   {'approved': True, 'score': 0.91, 'feedback': ''}

        Intent:      complaint
        Urgency:     high
        Final score: 0.91

        Final reply:
        Dear Maria, I sincerely apologize for the inconvenience...
        ```

    === "Async"

        ```python
        import asyncio

        async def main():
            msg = mf.Message()
            msg.email_body = """
            Hi there,

            I placed an order three weeks ago (order #ORD-9921) and it still hasn't arrived.
            The tracking page just says "processing". This is really frustrating — I needed
            this for a trip that already happened. I'd like a refund or an explanation.

            Thanks,
            Maria
            """

            await flux.acall(msg)

            print(f"Intent:      {msg.cls.intent}")
            print(f"Urgency:     {msg.cls.urgency}")
            print(f"Final score: {msg.rev.score:.2f}")
            print(f"\nFinal reply:\n{msg.rsp.draft}")

        asyncio.run(main())
        ```

---

## Complete Script

```python
import msgflux as mf
import msgflux.nn as nn
from typing import Literal

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class ClassifyEmail(mf.Signature):
    """Classify the incoming email to inform the reply strategy."""

    email_body: str = mf.InputField(desc="The full text of the incoming email")
    intent: Literal[
        "question", "complaint", "request", "follow_up", "cancellation", "praise"
    ] = mf.OutputField(desc="Primary intent of the email")
    urgency: Literal["low", "medium", "high"] = mf.OutputField(
        desc="How urgently this email needs a response"
    )
    tone: Literal["formal", "neutral", "informal"] = mf.OutputField(
        desc="Appropriate reply tone based on sender style"
    )
    sender_name: str = mf.OutputField(desc="Sender's first name extracted from the email")


class DraftReply(mf.Signature):
    """Draft a professional reply to the email."""

    email_body: str = mf.InputField(desc="The original email")
    intent: str = mf.InputField(desc="Classified intent")
    urgency: str = mf.InputField(desc="Urgency level")
    tone: str = mf.InputField(desc="Reply tone to use")
    draft: str = mf.OutputField(desc="A complete, ready-to-send reply addressing all points raised")


class ReviewDraft(mf.Signature):
    """Review a draft reply for quality, accuracy, and tone before sending."""

    email_body: str = mf.InputField(desc="The original email")
    draft: str = mf.InputField(desc="The draft reply to review")
    feedback: str = mf.OutputField(desc="Specific, actionable feedback if not approved")
    approved: bool = mf.OutputField(desc="True if the draft is ready to send")    
    score: float = mf.OutputField(desc="Quality score from 0.0 to 1.0")


class ReviseDraft(mf.Signature):
    """Revise a draft based on reviewer feedback."""

    current_draft: str = mf.InputField(desc="The current draft that needs improvement")
    feedback: str = mf.InputField(desc="Specific feedback from the reviewer")
    draft: str = mf.OutputField(desc="Improved version of the draft")


class Classifier(nn.Agent):
    model = model
    signature = ClassifyEmail
    message_fields = {"task": {"email_body": "email_body"}}
    response_mode = "cls"
    config = {"verbose": True}


class Drafter(nn.Agent):
    model = model
    signature = DraftReply
    message_fields = {"task": {"email_body": "email_body", "intent": "cls.intent", "urgency": "cls.urgency", "tone": "cls.tone"}}
    response_mode = "rsp"
    config = {"verbose": True}


class Reviewer(nn.Agent):
    model = model
    signature = ReviewDraft
    message_fields = {"task": {"email_body": "email_body", "draft": "rsp.draft"}}
    response_mode = "rev"
    config = {"verbose": True}


class Reviser(nn.Agent):
    model = model
    signature = ReviseDraft
    message_fields = {"task": {"current_draft": "rsp.draft", "feedback": "rev.feedback"}}
    response_mode = "rsp"
    config = {"verbose": True}


flux = mf.Inline(
    "classifier -> drafter -> reviewer -> @{rev.approved == False}: reviser -> reviewer;",
    {
        "classifier": Classifier(),
        "drafter":    Drafter(),
        "reviewer":   Reviewer(),
        "reviser":    Reviser(),
    },
    max_iterations=5,
)
```

---

## Further Reading

- [Inline DSL](../learn/inline.md) — flux syntax, branching, and while loops
- [Signatures](../learn/nn/agent/signatures.md) — declarative input/output contracts for agents
- [Async](../learn/nn/agent/async.md) — running fluxs asynchronously with `acall`
