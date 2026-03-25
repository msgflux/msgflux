# Visit Report Assistant

Forms with many fields have high abandonment rates. Users have to recall every detail, open fields one by one, and the experience is tedious. This tutorial shows how to replace that flow with a conversation: the user narrates what happened and the assistant extracts, validates, and submits the report — using `inject_vars` to keep the form state available to tools on every turn.

---

## The Problem

A typical visit report requires:

| Field | Required |
|---|---|
| `company` — company visited | ✅ |
| `location` — location of the visit | ✅ |
| `participants` — list of attendees | ✅ |
| `purpose` — objective of the visit | ✅ |
| `next_steps` — agreed follow-up actions | ✅ |
| `competitors` — competitors mentioned | optional |
| `closing_deadline` — expected deal closure date | optional |
| `notes` — additional observations | optional |

Instead of filling field by field, the user simply narrates:

> *"I visited Acme Corp at their office in Austin with John Smith and Sarah Lee. We discussed the ERP proposal. The client mentioned they're also evaluating SAP. They want to close by October."*

The assistant extracts what it can, identifies what's missing, and asks only for what's needed.

---

## Architecture

```
User: "I visited Acme in Austin with John and Sarah..."
              │
              ▼
    VisitReportAgent
    (system: required and optional fields)
    (message_fields: {"vars": "form_data"})
              │
              ├── extracts fields from the narrative
              │   calls: fill_fields(company, location, ...)
              │               ↓ inject_vars=True
              │          modifies msg.form_data in place
              │
              ├── calls: validate_report()
              │               ↓ inject_vars=True
              │          reads msg.form_data, returns missing fields
              │
              │   (loop: asks the user for missing fields)
              │
              └── calls: submit_report()
                              ↓ inject_vars=True + return_direct=True
                         validates and returns the report ID
```

**`inject_vars=True`** causes the `msg.form_data` dict to be injected as `vars` into each tool call — no need to pass fields manually. Every tool reads from or writes to the same dict, which persists across conversation turns.

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Define the Form Schema

We keep the required and optional fields in a single place so that both tools and the system prompt are derived from the same source.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import tool_config, Message
from datetime import datetime

# Report fields
REQUIRED_FIELDS = ["company", "location", "participants", "purpose", "next_steps"]
OPTIONAL_FIELDS = ["competitors", "closing_deadline", "notes"]
ALL_FIELDS      = REQUIRED_FIELDS + OPTIONAL_FIELDS


def empty_form() -> dict:
    """Return an empty dict with all report fields set to None."""
    return {field: None for field in ALL_FIELDS}
```

---

## Step 2 — Tools with `inject_vars`

Three tools, each with a single clear responsibility:

```python
@tool_config(inject_vars=True)
def fill_fields(vars: dict, **updates) -> dict:
    """Fill visit report fields with extracted values.

    Call this tool whenever you extract new information from the user's message.
    Only pass the fields that were mentioned; the rest will be preserved.
    Valid fields: company, location, participants, purpose, next_steps,
    competitors, closing_deadline, notes.
    """
    for field, value in updates.items():
        if field in ALL_FIELDS and value is not None:
            vars[field] = value

    return {
        "filled":           [f for f in ALL_FIELDS if vars.get(f)],
        "missing_required": [f for f in REQUIRED_FIELDS if not vars.get(f)],
    }


@tool_config(inject_vars=True)
def validate_report(vars: dict) -> dict:
    """Check which required report fields are still missing.

    Use this tool to know what to ask the user before submitting.
    """
    missing = [f for f in REQUIRED_FIELDS if not vars.get(f)]
    filled  = {f: vars[f] for f in ALL_FIELDS if vars.get(f)}
    return {
        "complete":         len(missing) == 0,
        "missing_required": missing,
        "filled":           filled,
    }


@tool_config(inject_vars=True, return_direct=True)
def submit_report(vars: dict) -> dict:
    """Submit the visit report once all required fields are filled.

    Only call this tool after validate_report confirms the form is complete.
    Returns the created report ID.
    """
    missing = [f for f in REQUIRED_FIELDS if not vars.get(f)]
    if missing:
        return {"error": f"Cannot submit: missing required fields: {missing}"}

    report_id = f"VISIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # In production: persist to the database here
    return {
        "success":   True,
        "report_id": report_id,
        "report":    {f: vars[f] for f in ALL_FIELDS if vars.get(f)},
    }
```

**Why `inject_vars` works here:**

- `fill_fields` and `validate_report` receive `vars` as a reference to the `msg.form_data` dict
- Modifying `vars` inside `fill_fields` modifies `msg.form_data` directly — state persists across turns
- `submit_report` with `return_direct=True` returns its result straight to `BriefingHub`, bypassing the LLM

---

## Step 3 — The Report Agent

`message_fields={"vars": "form_data"}` tells the agent which `Message` field
to expose as `vars` when calling tools.

```python
model = mf.Model.chat_completion("openai/gpt-4.1-mini")

SYSTEM_PROMPT = f"""
You are a CRM assistant that helps sales reps fill out client visit reports.

REQUIRED fields (all must be filled before submitting):
{chr(10).join(f'  - {f}' for f in REQUIRED_FIELDS)}

Optional fields (fill if the user mentions them):
{chr(10).join(f'  - {f}' for f in OPTIONAL_FIELDS)}

Instructions:
1. Read the user's account and extract every field you can.
2. Use fill_fields to save the extracted information.
3. Use validate_report to check what is still missing.
4. Ask the user ONLY for the missing required fields.
5. Once everything is complete, confirm with the user and call submit_report.
6. Be direct and concise. Do not ask unnecessary questions.
"""


class VisitReportAgent(nn.Agent):
    """Assistant for filling out commercial visit reports."""

    model          = model
    system_message = SYSTEM_PROMPT
    tools          = [fill_fields, validate_report, submit_report]
    message_fields = {"vars": "form_data"}   # ← msg.form_data becomes vars in tools
    config         = {"verbose": True}
```

---

## Step 4 — Orchestrator

`BriefingHub` initialises the empty form, runs the conversation, and detects
when `submit_report` (return_direct) has been called to close the loop.

```python
class BriefingHub(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent = VisitReportAgent()

    def forward(self, msg):
        # Ensure the form exists before the first turn
        if not hasattr(msg, "form_data") or msg.form_data is None:
            msg.form_data = empty_form()

        response = self.agent(msg.content, vars=msg.form_data)

        # submit_report with return_direct=True returns a dict directly
        if isinstance(response, dict) and "report_id" in response:
            msg.report    = response["report"]
            msg.report_id = response["report_id"]
            msg.response  = (
                f"Report **{response['report_id']}** submitted successfully!\n"
                f"Company: {msg.report.get('company')}\n"
                f"Location: {msg.report.get('location')}\n"
                f"Participants: {msg.report.get('participants')}"
            )
        else:
            msg.response = response

        return msg
```

---

## Running the Conversation

```python
hub = BriefingHub()
msg = Message()

# ── Turn 1: user narrates the visit ──────────────────────────────────────────
msg.content = (
    "I visited Acme Corp at their Austin office today. "
    "John Smith (Commercial Director) and Sarah Lee (IT) were present. "
    "We discussed the ERP proposal. They mentioned they're also evaluating SAP. "
    "They want to close by October."
)
hub(msg)
print(msg.response)
# "Got it! I captured: company, location, participants, purpose,
#  competitors, and deadline. Just missing: what are the agreed next steps?"

print(msg.form_data)
# {
#   'company': 'Acme Corp',
#   'location': 'Austin',
#   'participants': ['John Smith', 'Sarah Lee'],
#   'purpose': 'ERP proposal',
#   'next_steps': None,              ← still missing
#   'competitors': ['SAP'],
#   'closing_deadline': 'October',
#   'notes': None
# }

# ── Turn 2: user fills in the missing field ───────────────────────────────────
msg.content = "Next steps: send revised proposal by Friday and schedule a technical demo."
hub(msg)
print(msg.response)
# "Report VISIT-20260317-143022 submitted successfully!
#  Company: Acme Corp
#  Location: Austin
#  Participants: ['John Smith', 'Sarah Lee']"

print(msg.report_id)   # "VISIT-20260317-143022"
print(msg.report)      # dict with all filled fields
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import tool_config, Message
from datetime import datetime


# ── Schema ────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = ["company", "location", "participants", "purpose", "next_steps"]
OPTIONAL_FIELDS = ["competitors", "closing_deadline", "notes"]
ALL_FIELDS      = REQUIRED_FIELDS + OPTIONAL_FIELDS


def empty_form() -> dict:
    return {field: None for field in ALL_FIELDS}


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool_config(inject_vars=True)
def fill_fields(vars: dict, **updates) -> dict:
    """Fill visit report fields with extracted values.

    Call this tool whenever you extract new information from the user's message.
    Valid fields: company, location, participants, purpose, next_steps,
    competitors, closing_deadline, notes.
    """
    for field, value in updates.items():
        if field in ALL_FIELDS and value is not None:
            vars[field] = value
    return {
        "filled":           [f for f in ALL_FIELDS if vars.get(f)],
        "missing_required": [f for f in REQUIRED_FIELDS if not vars.get(f)],
    }


@tool_config(inject_vars=True)
def validate_report(vars: dict) -> dict:
    """Check which required report fields are still missing."""
    return {
        "complete":         len([f for f in REQUIRED_FIELDS if not vars.get(f)]) == 0,
        "missing_required": [f for f in REQUIRED_FIELDS if not vars.get(f)],
        "filled":           {f: vars[f] for f in ALL_FIELDS if vars.get(f)},
    }


@tool_config(inject_vars=True, return_direct=True)
def submit_report(vars: dict) -> dict:
    """Submit the report once all required fields are filled."""
    missing = [f for f in REQUIRED_FIELDS if not vars.get(f)]
    if missing:
        return {"error": f"Missing required fields: {missing}"}

    return {
        "success":   True,
        "report_id": f"VISIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "report":    {f: vars[f] for f in ALL_FIELDS if vars.get(f)},
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are a CRM assistant that helps fill out visit reports.

REQUIRED fields: {', '.join(REQUIRED_FIELDS)}
Optional fields: {', '.join(OPTIONAL_FIELDS)}

1. Read the account and use fill_fields to save what you extract.
2. Use validate_report to check what's missing.
3. Ask only for missing required fields.
4. Once complete, confirm and call submit_report.
"""


class VisitReportAgent(nn.Agent):
    """Assistant for filling out commercial visit reports."""
    model          = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = SYSTEM_PROMPT
    tools          = [fill_fields, validate_report, submit_report]
    message_fields = {"vars": "form_data"}
    config         = {"verbose": True}


# ── Orchestrator ──────────────────────────────────────────────────────────────

class BriefingHub(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent = VisitReportAgent()

    def forward(self, msg):
        if not hasattr(msg, "form_data") or msg.form_data is None:
            msg.form_data = empty_form()

        response = self.agent(msg.content, vars=msg.form_data)

        if isinstance(response, dict) and "report_id" in response:
            msg.report    = response["report"]
            msg.report_id = response["report_id"]
            msg.response  = (
                f"Report {response['report_id']} submitted!\n"
                + "\n".join(f"{k}: {v}" for k, v in msg.report.items())
            )
        else:
            msg.response = response

        return msg


# ── Run ───────────────────────────────────────────────────────────────────────

hub = BriefingHub()
msg = Message()

# Turn 1 — initial account
msg.content = (
    "I visited Acme Corp at their Austin office today. "
    "John Smith (Commercial Director) and Sarah Lee (IT) were present. "
    "We discussed the ERP proposal. They mentioned they're also evaluating SAP. "
    "They want to close by October."
)
hub(msg)
print("Assistant:", msg.response)
print("Form state:", msg.form_data)

# Turn 2 — fill in the missing field
msg.content = "Next steps: send revised proposal by Friday and schedule a technical demo."
hub(msg)
print("Assistant:", msg.response)
print("Report ID:", msg.report_id)
```

---

## Why `inject_vars` shines here

Without `inject_vars`, every tool would need the form data as an explicit parameter — and the agent would have to pass everything on every call:

```python
# ❌ without inject_vars: agent must manually re-pass the entire form
def fill_fields(company: str, location: str, participants: list, ...) -> dict: ...
```

With `inject_vars=True`, the form lives in `msg.form_data` and is injected automatically into every tool that declares `vars: dict` — no data is lost between turns, no field needs to be re-passed in the call:

```python
# ✅ with inject_vars: vars arrives automatically, tools write in place
@tool_config(inject_vars=True)
def fill_fields(vars: dict, **updates) -> dict:
    vars.update(...)  # modifies msg.form_data directly
```

The result: the agent focuses on **what to ask**, not on **how to transport data**.

---

## Next Steps

- **[Quickstart — PIX Assistant](../learn/nn/agent/quickstart.md)** — `return_direct` for structured extraction
- **[Intent Router](intent-router.md)** — multi-agent routing with typed Signatures
