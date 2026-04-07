# Customer Support Ticket Router

<span class="tag tag-teal">Beginner</span><span class="tag tag-gray">Signature</span><span class="tag tag-gray">Inline</span>

---

## The Problem

Every support inbox has the same latent tension: tickets arrive in a single stream, but the right response depends entirely on what the ticket actually is.

A billing dispute and a feature request are not the same problem. One needs an empathetic acknowledgment and a clear resolution path. The other needs a brief confirmation that the feedback was logged. Treating them the same way — with a generic reply and a default queue — frustrates the customer who needed urgency and wastes the team's time routing manually after the fact.

The deeper issue is consistency. Without a structured decision process, two tickets that describe the same situation get different outcomes depending on which agent handles them, how the customer phrased the request, or what else was in the queue that day. An API outage reported in a calm, professional tone can be misread as a low-priority question. A billing confusion that mentions the subscription cycle gets sent to the billing team when the real problem is account access.

What we actually want is for every incoming ticket to pass through the same decision process — one that reads the text, understands the intent, and routes accordingly — before a single word of the reply is drafted.

---

## The Plan

We want two things from this system: correct routing and calibrated responses.

Correct routing means each ticket lands with the right team at the right priority, regardless of how it was written. Calibrated responses mean the reply matches the emotional register and urgency of the ticket — direct and factual for a general question, empathetic and action-oriented for a frustrated customer.

To get both, we build a two-step pipeline. A **Router** reads the ticket and produces four signals: the category, the priority level, the team best suited to handle it, and the customer's sentiment. A **Drafter** uses those signals to write a reply that is calibrated to the situation — not a generic template, but a response shaped by what the Router found.

A set of labeled examples anchors the Router to past decisions, so routing reflects established patterns rather than ad-hoc interpretation. The examples include deliberate edge cases — a ticket that mentions subscription renewal but is really an access problem, and a critical outage reported in a calm tone — which are exactly the cases where a model without labeled reference diverges from expected behavior.

The two steps are wired together with `Inline`: a single string declares the pipeline order, and async is handled automatically.

---

## Architecture

```
Ticket text
      │
      ▼
  Router ─── mf.Example × 7 (labeled past tickets)
      │
      │  category, priority, assigned_team, sentiment → msg.routing
      ▼
  Drafter
      │
      ▼
  msg.rsp.response  (ready-to-send reply)
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 — Synthetic Tickets

A handful of representative tickets covers each category — including an edge case that sits between `billing` and `account` to stress-test the classifier.

```python
import msgflux as mf

mf.load_dotenv()
model = mf.Model.chat_completion("openai/gpt-4.1-mini")

TICKETS = {
    "billing_double_charge": "I was charged twice this month — $49 on the 3rd and again on the 5th. Order #ORD-2847.",
    "technical_api_outage":  "All API calls returning 503 since 14:00 UTC. Our entire product is down. This is urgent.",
    "account_login":         "Can't log in after renewing my subscription. It says my account is inactive.",
    "account_billing_edge":  "Can't access my account since the subscription renewed. Shows active but I get 'access denied'.",
    "feature_request":       "Would love a dark mode. The white background is really hard on the eyes during long sessions.",
    "general":               "Hi, quick question — does the Pro plan include API access?",
}
```

---

## Step 2 — Signatures

Two signatures define the contract for each stage. `RouteTicket` produces the routing metadata. `DraftResponse` consumes it to write a calibrated reply.

```python
from typing import Literal

class RouteTicket(mf.Signature):
    """Classify the support ticket to determine routing and response strategy."""

    ticket: str = mf.InputField(desc="The full text of the support ticket")

    category: Literal["billing", "technical", "account", "feature_request", "general"] = mf.OutputField(
        desc="Primary category of the ticket"
    )
    priority: Literal["low", "medium", "high", "critical"] = mf.OutputField(
        desc=(
            "Urgency level — critical for outages or data loss, "
            "high for blocking issues, medium for degraded service, "
            "low for questions and requests"
        )
    )
    assigned_team: Literal["billing", "engineering", "account_management", "product", "support"] = mf.OutputField(
        desc="Team best suited to handle this ticket"
    )
    sentiment: Literal["neutral", "frustrated", "angry", "satisfied"] = mf.OutputField(
        desc="Customer's emotional tone"
    )


class DraftResponse(mf.Signature):
    """Draft a support response calibrated to the ticket classification."""

    ticket: str = mf.InputField(desc="The original ticket text")
    category: str = mf.InputField(desc="Ticket category")
    priority: str = mf.InputField(desc="Priority level")
    sentiment: str = mf.InputField(desc="Customer sentiment")

    response: str = mf.OutputField(
        desc=(
            "A ready-to-send support response. "
            "Use an empathetic tone for frustrated or angry customers. "
            "Be brief and direct for general questions."
        )
    )
```

---

## Step 3 — Few-shot Examples

Seven labeled examples cover each category and include two edge cases. The fifth example — "access denied after renewal" — is the critical one: it mentions the subscription cycle (billing vocabulary) but the correct category is `account`. Without this labeled reference, the classifier routes it to `billing` inconsistently.

```python
examples = [
    mf.Example(
        inputs="I was charged twice this month. My card was billed $49 on the 3rd and again on the 5th.",
        labels={
            "category": "billing",
            "priority": "high",
            "assigned_team": "billing",
            "sentiment": "frustrated",
        },
        title="Double charge",
    ),
    mf.Example(
        inputs="The export button does nothing when I click it. Tried Firefox and Chrome, same issue.",
        labels={
            "category": "technical",
            "priority": "medium",
            "assigned_team": "engineering",
            "sentiment": "neutral",
        },
        title="Broken export button",
    ),
    mf.Example(
        inputs="All API calls have been returning 503 since 14:00 UTC. Our entire product is down.",
        labels={
            "category": "technical",
            "priority": "critical",
            "assigned_team": "engineering",
            "sentiment": "angry",
        },
        title="API outage",
    ),
    mf.Example(
        inputs="I can't log in. It says my account doesn't exist but I've been a customer for 2 years.",
        labels={
            "category": "account",
            "priority": "high",
            "assigned_team": "account_management",
            "sentiment": "frustrated",
        },
        title="Login failure — existing customer",
    ),
    mf.Example(
        inputs="Can't access my account since the subscription renewed. Shows active but I get 'access denied'.",
        labels={
            "category": "account",
            "priority": "high",
            "assigned_team": "account_management",
            "sentiment": "frustrated",
        },
        title="Access denied after renewal",  # edge case: billing vocabulary → account category
    ),
    mf.Example(
        inputs="Would it be possible to add CSV import? Right now we enter all data manually.",
        labels={
            "category": "feature_request",
            "priority": "low",
            "assigned_team": "product",
            "sentiment": "neutral",
        },
        title="CSV import request",
    ),
    mf.Example(
        inputs="Hi, quick question — does the Pro plan include API access?",
        labels={
            "category": "general",
            "priority": "low",
            "assigned_team": "support",
            "sentiment": "neutral",
        },
        title="Plan inquiry",
    ),
]
```

---

## Step 4 — Router and Drafter

`Router` reads the ticket from `msg`, writes the four routing fields to `msg.routing`, and carries the labeled examples. `Drafter` reads both the ticket and the routing signals, and writes the final reply to `msg.rsp`.

```python
import msgflux.nn as nn

class Router(nn.Agent):
    model = model
    examples = examples
    signature = RouteTicket
    message_fields = {"task": {"ticket": "ticket"}}
    response_mode = "routing"    
    config = {"verbose": True}


class Drafter(nn.Agent):
    model = model
    signature = DraftResponse
    message_fields = {
        "task": {
            "ticket":    "ticket",
            "category":  "routing.category",
            "priority":  "routing.priority",
            "sentiment": "routing.sentiment",
        }
    }
    response_mode = "rsp"
    config = {"verbose": True}
```

---

## Step 5 — Wiring the Pipeline

`Inline` composes the two agents into a single callable. The string declares the order — router always runs first because the drafter depends on `msg.routing` being populated. Async is handled automatically via `acall`.

```python
flux = mf.Inline(
    "router -> drafter",
    {
        "router":  Router(),
        "drafter": Drafter(),
    },
)
```

---

## Examples

???+ example

    === "Billing — high priority"

        ```python
        msg = mf.Message()
        msg.ticket = TICKETS["billing_double_charge"]
        flux(msg)

        print(f"Category:      {msg.routing.category}")
        print(f"Priority:      {msg.routing.priority}")
        print(f"Assigned team: {msg.routing.assigned_team}")
        print(f"Sentiment:     {msg.routing.sentiment}")
        print(f"\nResponse:\n{msg.rsp.response}")
        ```

        ```
        [Router][response]  {'category': 'billing', 'priority': 'high', 'assigned_team': 'billing', 'sentiment': 'frustrated'}
        [Drafter][response] {'response': 'Hi, I sincerely apologize for the duplicate charge on your account...'}

        Category:      billing
        Priority:      high
        Assigned team: billing
        Sentiment:     frustrated

        Response:
        Hi, I sincerely apologize for the duplicate charge on your account. I can see order #ORD-2847...
        ```

    === "Edge case — access denied after renewal"

        Without the fifth example in the list, this ticket routes to `billing` because it mentions the subscription cycle. The labeled reference for "access denied after renewal" anchors the decision to `account`.

        ```python
        msg = mf.Message()
        msg.ticket = TICKETS["account_billing_edge"]
        flux(msg)

        print(f"Category:      {msg.routing.category}")     # account (not billing)
        print(f"Assigned team: {msg.routing.assigned_team}")  # account_management
        print(f"\nResponse:\n{msg.rsp.response}")
        ```

        ```
        Category:      account
        Assigned team: account_management

        Response:
        Thank you for reaching out. I can see your subscription is showing as active,
        so this looks like an account access issue rather than a billing problem...
        ```

    === "Technical — critical outage"

        ```python
        msg = mf.Message()
        msg.ticket = TICKETS["technical_api_outage"]
        flux(msg)

        print(f"Category: {msg.routing.category}")
        print(f"Priority: {msg.routing.priority}")
        print(f"Team:     {msg.routing.assigned_team}")
        print(f"\nResponse:\n{msg.rsp.response}")
        ```

        ```
        Category: technical
        Priority: critical
        Team:     engineering

        Response:
        We have escalated this to our engineering team as a critical incident.
        We are actively investigating the 503 errors and will provide an update within 15 minutes...
        ```

    === "Async"

        ```python
        import asyncio

        async def main():
            msg = mf.Message()
            msg.ticket = TICKETS["billing_double_charge"]
            await flux.acall(msg)
            print(f"Category: {msg.routing.category}")
            print(f"Response:\n{msg.rsp.response}")

        asyncio.run(main())
        ```

---

## Complete Script

```python
import msgflux as mf
import msgflux.nn as nn
from typing import Literal

mf.load_dotenv()
model = mf.Model.chat_completion("openai/gpt-4.1-mini")

TICKETS = {
    "billing_double_charge": "I was charged twice this month — $49 on the 3rd and again on the 5th. Order #ORD-2847.",
    "technical_api_outage":  "All API calls returning 503 since 14:00 UTC. Our entire product is down. This is urgent.",
    "account_login":         "Can't log in after renewing my subscription. It says my account is inactive.",
    "account_billing_edge":  "Can't access my account since the subscription renewed. Shows active but I get 'access denied'.",
    "feature_request":       "Would love a dark mode. The white background is really hard on the eyes during long sessions.",
    "general":               "Hi, quick question — does the Pro plan include API access?",
}


class RouteTicket(mf.Signature):
    """Classify the support ticket to determine routing and response strategy."""

    ticket: str = mf.InputField(desc="The full text of the support ticket")
    category: Literal["billing", "technical", "account", "feature_request", "general"] = mf.OutputField(
        desc="Primary category of the ticket"
    )
    priority: Literal["low", "medium", "high", "critical"] = mf.OutputField(
        desc="critical for outages/data loss, high for blocking issues, medium for degraded service, low for questions"
    )
    assigned_team: Literal["billing", "engineering", "account_management", "product", "support"] = mf.OutputField(
        desc="Team best suited to handle this ticket"
    )
    sentiment: Literal["neutral", "frustrated", "angry", "satisfied"] = mf.OutputField(
        desc="Customer's emotional tone"
    )


class DraftResponse(mf.Signature):
    """Draft a support response calibrated to the ticket classification."""

    ticket: str = mf.InputField(desc="The original ticket text")
    category: str = mf.InputField(desc="Ticket category")
    priority: str = mf.InputField(desc="Priority level")
    sentiment: str = mf.InputField(desc="Customer sentiment")
    response: str = mf.OutputField(
        desc="A ready-to-send reply. Empathetic for frustrated/angry customers, direct for general questions."
    )


examples = [
    mf.Example(
        inputs="I was charged twice this month. My card was billed $49 on the 3rd and again on the 5th.",
        labels={"category": "billing", "priority": "high", "assigned_team": "billing", "sentiment": "frustrated"},
        title="Double charge",
    ),
    mf.Example(
        inputs="The export button does nothing when I click it. Tried Firefox and Chrome, same issue.",
        labels={"category": "technical", "priority": "medium", "assigned_team": "engineering", "sentiment": "neutral"},
        title="Broken export button",
    ),
    mf.Example(
        inputs="All API calls have been returning 503 since 14:00 UTC. Our entire product is down.",
        labels={"category": "technical", "priority": "critical", "assigned_team": "engineering", "sentiment": "angry"},
        title="API outage",
    ),
    mf.Example(
        inputs="I can't log in. It says my account doesn't exist but I've been a customer for 2 years.",
        labels={"category": "account", "priority": "high", "assigned_team": "account_management", "sentiment": "frustrated"},
        title="Login failure — existing customer",
    ),
    mf.Example(
        inputs="Can't access my account since the subscription renewed. Shows active but I get 'access denied'.",
        labels={"category": "account", "priority": "high", "assigned_team": "account_management", "sentiment": "frustrated"},
        title="Access denied after renewal",
    ),
    mf.Example(
        inputs="Would it be possible to add CSV import? Right now we enter all data manually.",
        labels={"category": "feature_request", "priority": "low", "assigned_team": "product", "sentiment": "neutral"},
        title="CSV import request",
    ),
    mf.Example(
        inputs="Hi, quick question — does the Pro plan include API access?",
        labels={"category": "general", "priority": "low", "assigned_team": "support", "sentiment": "neutral"},
        title="Plan inquiry",
    ),
]


class Router(nn.Agent):
    model = model
    examples = examples
    signature = RouteTicket
    message_fields = {"task": {"ticket": "ticket"}}
    response_mode = "routing"    
    config = {"verbose": True}


class Drafter(nn.Agent):
    model = model
    signature = DraftResponse
    message_fields = {
        "task": {
            "ticket":    "ticket",
            "category":  "routing.category",
            "priority":  "routing.priority",
            "sentiment": "routing.sentiment",
        }
    }
    response_mode = "rsp"
    config = {"verbose": True}


flux = mf.Inline(
    "router -> drafter",
    {
        "router":  Router(),
        "drafter": Drafter(),
    },
)
```

---

## Further Reading

- [System Prompt & Examples](../learn/nn/agent/system-prompt.md) — few-shot examples and how they are formatted in the system prompt
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts for agents
- [Task and Context](../learn/nn/agent/task-and-context.md) — reading inputs from `msg` via `message_fields`
- [Inline DSL](../learn/inline.md) — flux syntax and pipeline composition
