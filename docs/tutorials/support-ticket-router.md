# Customer Support Ticket Router

---

## The Problem

Support inboxes without classification treat every ticket the same way. An API outage and a feature request arrive in the same queue, get the same priority, and receive the same generic response. As the volume grows, the model's decisions become inconsistent — a ticket that mentions both a billing cycle and a login failure gets routed differently each time.

The problem is not the model's ability to classify. It is the absence of reference. Without labeled examples of how similar tickets were handled before, the model improvises. Two tickets phrased slightly differently but describing the same issue land with different teams.

---

## The Plan

We will build a router that classifies each incoming ticket before drafting a response. A set of labeled examples anchors the classifier to past decisions, so routing reflects established patterns rather than ad-hoc interpretation.

A classifier reads the ticket and produces four signals: the category, the priority level, the team best suited to handle it, and the customer's sentiment. A drafter uses those signals to write a calibrated response — brief and factual for a general question, empathetic and action-oriented for a frustrated customer reporting a billing error.

The examples include deliberate edge cases: a ticket that mentions subscription renewal but is really an access problem, and a critical outage reported in a calm tone. These are exactly the cases where a model without labeled reference diverges from expected behavior.

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
import msgflux.nn as nn
from typing import Literal

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

Two signatures define the contract for each stage. `RouteTicket` produces the routing metadata; `DraftResponse` consumes it to write a calibrated reply.

```python
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
class Router(nn.Agent):
    """Classify the ticket and determine routing metadata."""
    model = model
    signature = RouteTicket
    message_fields = {"task": {"ticket": "ticket"}}
    response_mode = "routing"
    examples = examples
    config = {"verbose": True}


class Drafter(nn.Agent):
    """Draft a calibrated response based on the routing signals."""
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

`TicketRouter` runs both agents sequentially. The router always runs first — the drafter depends on `msg.routing` being populated.

```python
class TicketRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = Router()
        self.drafter = Drafter()

    def forward(self, msg: mf.Message) -> mf.Message:
        self.router(msg)
        self.drafter(msg)
        return msg

    async def aforward(self, msg: mf.Message) -> mf.Message:
        await self.router.acall(msg)
        await self.drafter.acall(msg)
        return msg


ticket_router = TicketRouter()
```

---

## Examples

???+ example

    === "Billing — high priority"

        ```python
        msg = mf.Message()
        msg.ticket = TICKETS["billing_double_charge"]
        ticket_router(msg)

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
        ticket_router(msg)

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
        ticket_router(msg)

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
            await ticket_router.acall(msg)
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
    """Classify the ticket and determine routing metadata."""
    model = model
    signature = RouteTicket
    message_fields = {"task": {"ticket": "ticket"}}
    response_mode = "routing"
    examples = examples
    config = {"verbose": True}


class Drafter(nn.Agent):
    """Draft a calibrated response based on the routing signals."""
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


class TicketRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = Router()
        self.drafter = Drafter()

    def forward(self, msg: mf.Message) -> mf.Message:
        self.router(msg)
        self.drafter(msg)
        return msg

    async def aforward(self, msg: mf.Message) -> mf.Message:
        await self.router.acall(msg)
        await self.drafter.acall(msg)
        return msg


ticket_router = TicketRouter()

msg = mf.Message()
msg.ticket = TICKETS["billing_double_charge"]
ticket_router(msg)
print(f"Category: {msg.routing.category}")
print(f"Priority: {msg.routing.priority}")
print(f"Response:\n{msg.rsp.response}")
```

---

## Further Reading

- [System Prompt & Examples](../learn/nn/agent/system-prompt.md) — few-shot examples and how they are formatted in the system prompt
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts for agents
- [Task and Context](../learn/nn/agent/task-and-context.md) — reading inputs from `msg` via `message_fields`
