# Visit Report Assistant

<span class="tag tag-orange">Advanced</span><span class="tag tag-gray">Multimodal</span><span class="tag tag-gray">Retrivers</span>

---

## The Problem

A sales rep just left a client meeting. They need to log the visit before context fades — who was there, what was discussed, what happens next. Instead they open a CRM form with eight blank fields and start typing from memory.

- Names get abbreviated or misspelled. "Jon" goes in instead of "Jonathan Davies".
- Half the fields are left empty because the rep can't remember or doesn't have time.
- Voice notes exist but no one transcribes them into the form.
- A confirmed name needs to be enriched with data from external systems — IDs, emails, roles — but feeding that data to the model creates new problems: it may reproduce it incorrectly, confuse it with the conversation, or include it in the wrong field.
- Reps sometimes visit multiple clients in the same day and need to switch between drafts mid-conversation.

By the time the data reaches the pipeline, it is incomplete, inconsistent, or simply wrong.

---

## The Plan

We will build a conversational assistant that captures visit report fields through a multi-turn dialogue. The user narrates the visit — the assistant extracts, validates, and fills.

A rep may visit several clients in the same day. Each report lives in its own draft. The user can create a new one or switch between existing ones at any point — the assistant always knows which report is currently active, so the user never has to repeat it.

Sensitive fields — participants, companies, and competitors — are never stored as raw text. Before writing any of them, a dedicated resolver agent looks up the name in the company database. When the first attempt fails, it tries alternative forms: normalized spelling, first name only, phonetic variants. This covers transcription errors and nicknames alike. The conversation agent only ever sees resolved candidates — it never knows how many attempts were needed.

If the user sends an audio note, it is transcribed first. Downstream, everything works exactly as with text.

Once a name is confirmed, the assistant quietly fetches the enriched record — IDs, email, role, company website — and attaches it to the draft. The conversation agent never sees these values, which means it cannot reproduce them incorrectly or mix them into the dialogue. They surface only in the final report, returned directly to the caller when requested.

---

## Architecture

```
User Input (text or audio)
          │
          ├── audio? → STT (Whisper) → user.text
          │
          ▼
  VisitAssistant
  (task: user.text, vars: msg.vars)
          │
          ├── activate_report(report_id?)  [inject_vars]
          │          │
          │          ├── "" → create draft_1, draft_2, …
          │          └── "draft_N" → switch active context
          │
          ├── query_entity(domain, query)
          │          │
          │          ├── "participant" → ParticipantResolver
          │          ├── "company"     → CompanyResolver
          │          └── "competitor"  → CompetitorResolver
          │                   │
          │          [ReAct + Searcher-as-tool]
          │          attempt 1: exact query  → no result
          │          attempt 2: normalize    → no result
          │          attempt 3: first token  → match found
          │                   │
          │          response template → resolved name or candidates
          │
          ├── fill_fields(updates)     [inject_vars]
          │          │
          │          ├── writes field to vars.drafts[active_report]
          │          └── sensitive field? → _fetch_*() → draft._*_data (silent)
          │
          ├── validate_report()        [inject_vars] → {complete, missing_required}
          │
          ├── submit_report()          [inject_vars]
          │          │
          │          ├── writes to vars.reports list
          │          └── removes draft from vars.drafts, clears active_report
          │
          └── get_report(report_id?)   [inject_vars, return_direct]
                     │
                     ├── active draft → formatted draft preview
                     └── submitted → full report with enriched data
```

`query_entity` is the single interface for all entity resolution. The root agent passes a `domain` and the raw query — the sub-agent handles the uncertainty. `get_report` is the only tool that bypasses the Assistant on return — it delivers the enriched report directly to the orchestrator.

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 — Models

```python
import msgflux as mf

mf.load_dotenv()
chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model  = mf.Model.speech_to_text("openai/whisper-1")
```

---

## Step 2 — Corpora and Fuzzy Retrievers

Three independent fuzzy retrievers cover the three sensitive domains. Install the dependencies:

```bash
pip install rapidfuzz faker
```

Word-level fuzzy matching handles partial names and transcription variants. A query like `"Jon"` surfaces `"Jonathan Davies"` through token matching even without a clean prefix — this is the key difference from BM25, which requires lexical overlap.

```python
from faker import Faker

fake = Faker("en_US")

participants_corpus = [fake.name() for _ in range(60)]
companies_corpus    = [fake.company() for _ in range(40)]
competitors_corpus  = [
    "SAP", "Oracle", "Salesforce", "Microsoft Dynamics",
    "TOTVS", "Linx", "Senior Sistemas", "Protheus", "Sankhya",
]

participants_fuzzy = mf.Retriever.fuzzy("rapidfuzz")
companies_fuzzy    = mf.Retriever.fuzzy("rapidfuzz")
competitors_fuzzy  = mf.Retriever.fuzzy("rapidfuzz")

participants_fuzzy.add(participants_corpus)
companies_fuzzy.add(companies_corpus)
competitors_fuzzy.add(competitors_corpus)
```


---

## Step 3 — Resolvers

Each resolver is a ReAct agent with a `nn.Searcher` as its only tool. On each turn the agent calls the Searcher, reads the formatted result, and decides whether to refine the query or finalize.

The `signature` types the `final_answer`: the root agent receives `candidates` and `resolved`.

**Searchers:**

The shared response template renders the top matches with their scores. `return_score: True` is required because the template references `r.score`. When no result passes the threshold, the template returns `"No matches found."` — the agent knows to retry with a different query.

```python
import msgflux.nn as nn

SEARCH_TEMPLATE = (
    "{% if results %}"
    "Matches:\n"
    "{% for r in results %}{{ loop.index }}. {{ r.data }} (score: {{ r.score | round(2) }})\n{% endfor %}"
    "{% else %}"
    "No matches found."
    "{% endif %}"
)


class ParticipantSearcher(nn.Searcher):
    """Search the participant registry by name."""
    retriever = participants_fuzzy
    config    = {"top_k": 5, "threshold": 0.5, "return_score": True}
    templates = {"response": SEARCH_TEMPLATE}


class CompanySearcher(nn.Searcher):
    """Search the company registry."""
    retriever = companies_fuzzy
    config    = {"top_k": 5, "threshold": 0.5, "return_score": True}
    templates = {"response": SEARCH_TEMPLATE}


class CompetitorSearcher(nn.Searcher):
    """Search the known competitor list."""
    retriever = competitors_fuzzy
    config    = {"top_k": 5, "threshold": 0.5, "return_score": True}
    templates = {"response": SEARCH_TEMPLATE}
```

**Resolvers:**

`EntityResolver` defines the shared behaviour. The three specialisations inherit from it and each declares only their `tools` — that is the only difference between them.

```python
from msgflux.generation.reasoning import ReAct

RESOLVER_INSTRUCTIONS = """
You receive a name query that may contain typos, nicknames, or transcription errors.
Your goal: find the closest real name in the registry.

Strategy:
1. Search with the exact query.
2. If no result, try a normalized variant (lowercase, remove accents).
3. If still no result, try only the first name or a prefix.
4. After all attempts, return the best candidates found.

Always call the search tool — never return a name without searching first.
"""


class EntityResolver(nn.Agent):
    instructions      = RESOLVER_INSTRUCTIONS
    model             = chat_model
    generation_schema = ReAct
    signature         = "query: str -> candidates: list[str], resolved: bool"
    templates         = {
        "response": (
            "{% if final_answer.resolved and final_answer.candidates %}"
            "Resolved: {{ final_answer.candidates[0] }}"
            "{% elif final_answer.candidates %}"
            "Candidates: {{ final_answer.candidates | join(', ') }}. Ask the user to confirm."
            "{% else %}"
            "No match found. Ask the user to spell the full name."
            "{% endif %}"
        )
    }


class ParticipantResolver(EntityResolver):
    tools = [ParticipantSearcher]


class CompanyResolver(EntityResolver):
    tools = [CompanySearcher]


class CompetitorResolver(EntityResolver):
    tools = [CompetitorSearcher]
```

---

## Step 4 — `query_entity` Tool

`QueryEntity` routes by `domain` and returns a single formatted string to the root agent. The root agent never sees the fuzzy scores or retry iterations — only whether the name was `resolved` with confidence and what the `candidates` are.

The `query` argument is passed as a keyword argument (`query=query`) so the agent's `signature` maps it correctly to the template variable.

```python
from typing import Literal

class QueryEntity(nn.Module):
    """Validate and resolve an entity name against the registry.

    Call this tool before filling any participant, companies, or competitor field.
    Pass the name exactly as heard — the resolver handles typos and transcription errors.
    """

    def __init__(self):
        super().__init__()
        self.set_name("query_entity")
        self.set_annotations({
            "domain": Literal["participant", "company", "competitor"],
            "query":  str,
            "return": str,
        })
        self.resolvers = nn.ModuleDict({
            "participant": ParticipantResolver(),
            "company":     CompanyResolver(),
            "competitor":  CompetitorResolver(),
        })

    def forward(self, domain: str, query: str) -> str:
        return self.resolvers[domain](query=query)

    async def aforward(self, domain: str, query: str) -> str:
        return await self.resolvers[domain].acall(query=query)
```

---

## Step 5 — Report State and Tools

### State layout

All report state lives inside `msg.vars`:

| Key | Type | Description |
|-----|------|-------------|
| `active_report` | `str \| None` | ID of the draft currently being edited |
| `drafts` | `dict[str, dict]` | All in-progress drafts (`draft_1`, `draft_2`, …) |
| `reports` | `list[dict]` | Submitted reports |

Each draft dict holds the eight form fields (`companies`, `location`, `participants`, `purpose`, `next_steps`, `competitors`, `closing_deadline`, `notes`) plus silent enriched data under `_participants_data`, `_companies_data`, and `_competitors_data`.

The helper `_active_draft(vars)` resolves the current draft. Every tool calls it so the root agent never has to pass a `report_id` on each call.

### `activate_report`

The single context-switching primitive. Calling it without an argument creates the next draft ID automatically (`draft_1`, `draft_2`, …). Passing an existing draft ID switches to it. The return value summarises the current state so the agent knows what drafts and submitted reports exist.

The counter is derived from `len(drafts)` at the moment of the call — it reflects the number of drafts currently open, not how many have ever existed. If `draft_1` is submitted and removed from `drafts`, calling `activate_report()` again produces a fresh `draft_1`. There is no collision: submitted records use a timestamp ID (`VISIT-YYYYMMDD-HHMMSS`) and are kept in `vars.reports`, not in `vars.drafts`.

The pointer `vars.active_report` is the only piece of shared state the tools read. Every subsequent tool (`fill_fields`, `validate_report`, `submit_report`, `get_report`) calls `_active_draft(vars)`, which reads this pointer and returns the mutable dict for that draft. The agent never has to pass the draft ID on follow-up calls — the context is already there.

```python
from typing import Union
from datetime import datetime


REQUIRED_FIELDS = ["companies", "location", "participants", "purpose", "next_steps"]
OPTIONAL_FIELDS = ["competitors", "closing_deadline", "notes"]
ALL_FIELDS      = REQUIRED_FIELDS + OPTIONAL_FIELDS


def empty_form() -> dict:
    return {field: None for field in ALL_FIELDS}


def _active_draft(vars: dict) -> dict:
    """Return the mutable dict for the currently active draft."""
    active = vars.get("active_report")
    if not active:
        raise ValueError("No active report. Call activate_report first.")
    drafts = vars.setdefault("drafts", {})
    return drafts.setdefault(active, empty_form())


@mf.tool_config(inject_vars=True)
def activate_report(report_id: str = "", **kwargs) -> dict:
    """Create a new report draft or switch to an existing one.

    If report_id is empty, a new draft is created automatically (draft_1, draft_2, …).
    If report_id matches an existing draft, that context becomes active.

    Always call this before starting to fill a report.
    Returns the active report_id and a summary of all drafts and submitted reports.
    """
    vars   = kwargs["vars"]
    drafts = vars.setdefault("drafts", {})

    if not report_id:
        n         = len(drafts) + 1
        report_id = f"draft_{n}"
        drafts.setdefault(report_id, empty_form())
    elif report_id not in drafts:
        drafts[report_id] = empty_form()

    vars["active_report"] = report_id
    return {
        "active_report": report_id,
        "drafts":        list(drafts.keys()),
        "submitted":     [r["id"] for r in (vars.get("reports") or [])],
    }
```

### `fill_fields`, `validate_report`, `submit_report`

All three tools use `inject_vars=True` and resolve state through `_active_draft`. The root agent never passes the draft ID — it is read from `vars.active_report` automatically.

`fill_fields` writes only non-null valid fields. For sensitive list fields it also triggers silent enrichment: participant emails and roles, company websites, competitor segments are stored inside the draft without appearing in the tool response.

`submit_report` copies the draft into `vars.reports`, removes it from `vars.drafts`, and clears `active_report`. The conversation can immediately move to a new report.

```python
@mf.tool_config(inject_vars=True)
def fill_fields(updates: dict[str, Union[str, list[str], None]], **kwargs) -> dict:
    """Fill visit report fields with validated values.

    Call after query_entity resolves a name, or directly for non-sensitive fields.
    Only pass fields that changed — the rest are preserved.
    Valid fields: companies, location, participants, purpose, next_steps,
    competitors, closing_deadline, notes.

    Operates on the active report draft. Call activate_report first if needed.
    """
    vars  = kwargs["vars"]
    draft = _active_draft(vars)
    for field, value in updates.items():
        if field not in ALL_FIELDS or value is None:
            continue
        draft[field] = value
        if field == "participants":
            names = value if isinstance(value, list) else [value]
            draft["_participants_data"] = [_fetch_participant_data(n) for n in names]
        elif field == "companies":
            names = value if isinstance(value, list) else [value]
            draft["_companies_data"] = [_fetch_company_data(n) for n in names]
        elif field == "competitors":
            names = value if isinstance(value, list) else [value]
            draft["_competitors_data"] = [_fetch_competitor_data(n) for n in names]
    return {
        "active_report":    vars["active_report"],
        "filled":           [f for f in ALL_FIELDS if draft.get(f)],
        "missing_required": [f for f in REQUIRED_FIELDS if not draft.get(f)],
    }


@mf.tool_config(inject_vars=True)
def validate_report(**kwargs) -> dict:
    """Check which required fields are still missing in the active report draft."""
    vars  = kwargs["vars"]
    draft = _active_draft(vars)
    missing = [f for f in REQUIRED_FIELDS if not draft.get(f)]
    return {
        "active_report":    vars["active_report"],
        "complete":         len(missing) == 0,
        "missing_required": missing,
        "filled":           {f: draft[f] for f in ALL_FIELDS if draft.get(f)},
    }


@mf.tool_config(inject_vars=True)
def submit_report(**kwargs) -> dict:
    """Submit the active report draft. Only call after validate_report returns complete=True."""
    vars  = kwargs["vars"]
    draft = _active_draft(vars)
    missing = [f for f in REQUIRED_FIELDS if not draft.get(f)]
    if missing:
        return {"error": f"Cannot submit: missing required fields: {missing}"}
    report_id  = f"VISIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report_rec = {
        "id":                report_id,
        "date":              datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status":            "submitted",
        "fields":            {f: draft[f] for f in ALL_FIELDS if draft.get(f)},
        "_participants_data": draft.get("_participants_data", []),
        "_companies_data":   draft.get("_companies_data", []),
        "_competitors_data": draft.get("_competitors_data", []),
    }
    drafts = vars.get("drafts", {})
    drafts.pop(vars["active_report"], None)
    vars["active_report"] = None
    reports = list(vars.get("reports") or [])
    reports.append(report_rec)
    vars["reports"] = reports
    return {"success": True, "report_id": report_id}
```

### `get_report`

Uses `return_direct=True` — the formatted string goes directly to the orchestrator without returning to the model. When `report_id` is empty, it shows the active draft first; if there is no active draft, it shows the most recent submitted report.

```python
@mf.tool_config(inject_vars=True, return_direct=True)
def get_report(report_id: str = "", **kwargs) -> str:
    """Return a report by its ID.

    Priority when report_id is empty:
      1. Active report draft (if any)
      2. Most recent submitted report

    Pass a specific report_id to retrieve a draft or submitted report by name.
    """
    vars    = kwargs["vars"]
    drafts  = vars.get("drafts") or {}
    reports = list(vars.get("reports") or [])

    if report_id:
        if report_id in drafts:
            return _format_draft(report_id, drafts[report_id])
        rec = next((r for r in reports if r["id"] == report_id), None)
        if rec is None:
            return f"Report {report_id!r} not found."
    else:
        active = vars.get("active_report")
        if active and active in drafts:
            return _format_draft(active, drafts[active])
        if not reports:
            return "No reports found."
        rec = reports[-1]

    fields = rec.get("fields", {})
    lines  = [f"# Visit Report {rec['id']}  ({rec['date']})"]
    for label, value in fields.items():
        lines.append(f"{label}: {', '.join(value) if isinstance(value, list) else value}")
    _append_enriched_lines(lines, rec)
    return "\n".join(lines)


def _format_draft(draft_id: str, draft: dict) -> str:
    fields = {f: draft[f] for f in ALL_FIELDS if draft.get(f)}
    lines  = [f"# Draft Report {draft_id}  (in progress)"]
    for label, value in fields.items():
        lines.append(f"{label}: {', '.join(value) if isinstance(value, list) else value}")
    _append_enriched_lines(lines, draft)
    return "\n".join(lines)


def _append_enriched_lines(lines: list, source: dict) -> None:
    participants = source.get("_participants_data", [])
    companies    = source.get("_companies_data", [])
    competitors  = source.get("_competitors_data", [])
    if participants:
        lines.append("\nParticipants:")
        for p in participants:
            lines.append(f"  - {p['name']} ({p['role']}) — {p['email']}")
    if companies:
        lines.append("\nCompanies:")
        for c in companies:
            lines.append(f"  - {c['name']} — {c['website']}")
    if competitors:
        lines.append("\nCompetitors:")
        for c in competitors:
            lines.append(f"  - {c['name']}: {c['segment']}")
```

**`_format_draft`** renders the current state of an in-progress draft. It uses the same field iteration as the submitted report view but adds `(in progress)` in the header and omits the timestamp and status. Calling `_append_enriched_lines` at the end means the rep sees the enriched participant and company details as soon as names are confirmed — not only after submission.

**`_append_enriched_lines`** is the shared rendering step used by both the draft view (`_format_draft`) and the submitted report view (`get_report`). It reads the three underscore-prefixed data keys from the source dict and appends them as structured sections after the form fields. The underscore prefix is the convention that keeps this data invisible to the model: the `system_extra_message` template skips keys starting with `_` when rendering the draft summary — the values live in the dict but the model never sees them.

**`_participants_data`** is a list of dicts, one per confirmed participant. Each entry has three keys:

| Key | Description |
|-----|-------------|
| `name` | Canonical name from the participant registry |
| `role` | Job title (e.g. `"CTO"`, `"IT Manager"`) |
| `email` | Work email address |

`_companies_data` follows `{name, website}` and `_competitors_data` follows `{name, segment}`. All three are populated inside `fill_fields` the moment the field is written — not lazily on `get_report`. When a draft is submitted, `submit_report` copies the three dicts into the report record so the enriched data travels with the report and is always available for rendering.

In production, `_fetch_participant_data`, `_fetch_company_data`, and `_fetch_competitor_data` would query your CRM or identity directory. Here they return synthetic records generated by Faker, which is enough to verify the full pipeline end-to-end.

---

## Step 6 — STT and Root Agent

`STT` transcribes audio into `msg.user`. From there the agent reads `user.text` identically to a typed message.

```python
class STT(nn.Transcriber):
    model          = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode  = "user"
```

!!! tip
    `Transcriber` returns a dict with a `text` key. Setting `response_mode = "user"` stores that dict at `msg.user`, making `msg.user.text` accessible. Using `"user.text"` would nest one level deeper — `msg.user.text.text`.

`VisitAssistant` drives the entire conversation. Step 0 in the instructions ensures it always calls `activate_report` before any other tool when there is no active draft.

The `system_extra_message` exposes the current context state — active draft, open drafts, and submitted reports — so the agent can switch between them on demand. The context variables (`active_report`, `drafts`, `reports`) are the top-level keys of `msg.vars`, accessible directly in the Jinja template without a `vars.` prefix.

```python

class VisitAssistant(nn.Agent):
    model          = chat_model
    system_message = """
    You are a CRM assistant integrated into the sales team's workflow.
    Your job is to help reps log client visit reports quickly and accurately
    through a natural conversation — no forms, no manual field entry.
    """
    instructions = """
    The report has required fields that must all be present before submitting:
    companies, location, participants, purpose, next_steps.

    Optional fields — fill if mentioned:
    competitors, closing_deadline, notes.

    Follow this flow on each turn:
    0. If there is no active report, call activate_report() first to create one.
    1. Extract every field you can from the user's message.
    2. For participants, companies, and competitors: always call query_entity first.
       Never fill these fields with an unvalidated name.
    3. Call fill_fields to save validated values into the active draft.
    4. Call validate_report to check what is still missing.
    5. Ask only for missing required fields — one question at a time.
    6. Once validate_report returns complete=True, confirm with the user and call submit_report.
    7. After submitting, inform the user (report_id) and offer to show the full report.
       If they confirm, call get_report.

    Multiple drafts:
    - If the user wants to start a new report, call activate_report() — a new draft is created automatically.
    - If the user wants to resume a previous draft, call activate_report(report_id="draft_N").
    - All fill/validate/submit operations always target the currently active draft.
    """
    system_extra_message = (
        "<customer_info>\n"
        "Rep name: {{ user_name }}\n"
        "Active report: {{ active_report or 'none' }}\n"
        "{% if drafts %}"
        "\nDrafts in progress:\n"
        "{% for id, draft in drafts.items() %}"
        "  - {{ id }}: {% set filled = [] %}{% for f, v in draft.items() %}"
        "{% if v and not f.startswith('_') %}{% set _ = filled.append(f) %}{% endif %}"
        "{% endfor %}{{ filled | join(', ') or 'empty' }}\n"
        "{% endfor %}"
        "{% endif %}"
        "{% if reports %}"
        "\nSubmitted reports:\n"
        "{% for r in reports %}"
        "  - {{ r.id }} ({{ r.date }}) — status: {{ r.status }}\n"
        "{% endfor %}"
        "{% endif %}"
        "</customer_info>"
    )
    message_fields = {
        "task": "user.text",
        "vars": "vars",
    }
    tools         = [
        QueryEntity, activate_report, fill_fields,
        validate_report, submit_report, get_report
    ]
    response_mode = "response"
    config        = {"include_date": True}
```

!!! note "Template context"
    `system_extra_message` is rendered in a second Jinja pass where the context is `msg.vars` directly. Variables are the **keys** of `msg.vars` — use `{{ active_report }}`, not `{{ vars.active_report }}`.

---

## Step 7 — Orchestrator

`VisitReportAssistant` is the entry point. It initialises `vars.drafts`, `vars.active_report`, and `vars.reports` on the first turn, optionally runs STT, then calls the agent. Because `get_report` uses `return_direct=True`, the raw response can be a tool-responses dict — `_extract_direct_result` unwraps the string before returning it to the caller.

```python
class VisitReportAssistant(nn.Module):
    """Entry point for the visit report conversation."""

    def __init__(self):
        super().__init__()
        self.agent = VisitAssistant()
        self.stt   = STT()

    def _init_form(self, msg: mf.Message) -> None:
        """Initialise draft/report containers on the first turn."""
        if msg.get("vars.drafts") is None:
            msg.set("vars.drafts", {})
        if msg.get("vars.active_report") is None:
            msg.set("vars.active_report", None)
        if msg.get("vars.reports") is None:
            msg.set("vars.reports", [])

    @staticmethod
    def _extract_direct_result(response) -> any:
        """Unwrap the string from a return_direct tool_responses payload."""
        if isinstance(response, dict) and "tool_responses" in response:
            calls = response["tool_responses"].get("tool_calls", [])
            if calls:
                return calls[0].get("result", response)
        return response

    def forward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._init_form(msg)
        if msg.get("audio_content"):
            self.stt(msg)
        self.agent(msg, messages=history)
        msg.response = self._extract_direct_result(msg.response)
        return msg

    async def aforward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._init_form(msg)
        if msg.get("audio_content"):
            await self.stt.acall(msg)
        await self.agent.acall(msg, messages=history)
        msg.response = self._extract_direct_result(msg.response)
        return msg
```

---

## Examples

???+ example

    === "Single report (text)"

        ```python
        assistant = VisitReportAssistant()
        msg       = mf.Message()
        history   = []

        msg.set("vars.user_name", "Alex Johnson")

        _company    = companies_corpus[0]
        _p1, _p2    = participants_corpus[0], participants_corpus[1]
        _competitor = competitors_corpus[0]

        # Turn 1 — narrate the visit
        msg.set("user.text", (
            f"Start a new visit. I was at {_company} in Austin. "
            f"{_p1} and {_p2} from IT were there. "
            f"We discussed ERP — they're also evaluating {_competitor}. "
            f"They want to close by October."
        ))
        assistant(msg, history=history)
        history.append(mf.ChatBlock.assist(msg.response))
        print(msg.response)
        # activate_report() → draft_1
        # query_entity resolves all names via fuzzy retry
        # fill_fields(companies, location, participants, competitors, closing_deadline)
        # → "Got it. What was the purpose and what are the next steps?"

        # Turn 2 — fill remaining fields
        msg.set("user.text", "Purpose was ERP renewal. Next steps: send revised proposal by Friday.")
        assistant(msg, history=history)
        history.append(mf.ChatBlock.assist(msg.response))
        print(msg.response)
        # → "All fields filled. Shall I submit the report?"

        # Turn 3 — confirm
        msg.set("user.text", "Yes, save it.")
        assistant(msg, history=history)
        history.append(mf.ChatBlock.assist(msg.response))
        print(msg.response)
        # → "Report VISIT-20260404-143022 submitted."

        # Turn 4 — show the report (return_direct — string goes straight to caller)
        msg.set("user.text", "Show me the report.")
        assistant(msg, history=history)
        print(msg.response)
        # # Visit Report VISIT-20260404-143022  (2026-04-04 14:30)
        # companies: Acme Corp
        # location: Austin
        # ...
        ```

    === "Multiple drafts"

        ```python
        assistant = VisitReportAssistant()
        msg       = mf.Message()
        history   = []

        msg.set("vars.user_name", "Alex Johnson")

        def turn(text):
            msg.set("user.text", text)
            assistant(msg, history=history)
            history.append(mf.ChatBlock.assist(msg.response))
            return msg.response

        _c1, _c2 = companies_corpus[0], companies_corpus[1]
        _p1, _p2, _p3 = participants_corpus[0], participants_corpus[1], participants_corpus[2]

        # Report 1
        turn(f"New visit: {_c1} in Austin. {_p1} and {_p2} from IT. ERP proposal, also evaluating SAP.")
        turn("Purpose: ERP renewal. Next steps: send proposal by Friday.")
        turn("Yes, submit.")

        # Report 2 — activate_report() automatically creates draft_2
        turn(f"New visit: {_c2} in São Paulo. {_p3} from procurement. Contract renewal. Follow-up next week.")
        turn("Yes, submit.")

        # See last report
        turn("Show me the last submitted report.")
        ```

    === "Audio input"

        ```python
        assistant = VisitReportAssistant()
        msg       = mf.Message()
        history   = []

        msg.set("vars.user_name", "Alex Johnson")

        # audio note replaces typed text — STT writes to user.text before the agent runs
        msg.audio_content = open("visit_note.ogg", "rb").read()
        assistant(msg, history=history)
        history.append(mf.ChatBlock.assist(msg.response))
        print("Assistant:", msg.response)
        ```

---

## Complete Script

??? example "Expand full script"
    ```python
    # /// script
    # dependencies = [
    #   "faker",
    #   "rapidfuzz",
    # ]
    # ///

    import msgflux as mf
    import msgflux.nn as nn
    from datetime import datetime
    from faker import Faker
    from msgflux.generation.reasoning import ReAct
    from typing import Literal, Union


    chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    stt_model  = mf.Model.speech_to_text("openai/whisper-1")

    fake = Faker("en_US")


    participants_corpus = [fake.name() for _ in range(60)]
    companies_corpus    = [fake.company() for _ in range(40)]
    competitors_corpus  = [
        "SAP", "Oracle", "Salesforce", "Microsoft Dynamics",
        "TOTVS", "Linx", "Senior Sistemas", "Protheus", "Sankhya",
    ]

    participants_fuzzy = mf.Retriever.fuzzy("rapidfuzz")
    companies_fuzzy    = mf.Retriever.fuzzy("rapidfuzz")
    competitors_fuzzy  = mf.Retriever.fuzzy("rapidfuzz")

    participants_fuzzy.add(participants_corpus)
    companies_fuzzy.add(companies_corpus)
    competitors_fuzzy.add(competitors_corpus)


    SEARCH_TEMPLATE = (
        "{% if results %}"
        "Matches:\n"
        "{% for r in results %}{{ loop.index }}. {{ r.data }} (score: {{ r.score | round(2) }})\n{% endfor %}"
        "{% else %}"
        "No matches found."
        "{% endif %}"
    )


    class ParticipantSearcher(nn.Searcher):
        """Search the participant registry by name."""
        retriever = participants_fuzzy
        config    = {"top_k": 5, "threshold": 0.5, "return_score": True}
        templates = {"response": SEARCH_TEMPLATE}


    class CompanySearcher(nn.Searcher):
        """Search the company registry."""
        retriever = companies_fuzzy
        config    = {"top_k": 5, "threshold": 0.5, "return_score": True}
        templates = {"response": SEARCH_TEMPLATE}


    class CompetitorSearcher(nn.Searcher):
        """Search the known competitor list."""
        retriever = competitors_fuzzy
        config    = {"top_k": 5, "threshold": 0.5, "return_score": True}
        templates = {"response": SEARCH_TEMPLATE}


    RESOLVER_INSTRUCTIONS = """
    You receive a name query that may contain typos, nicknames, or transcription errors.
    Your goal: find the closest real name in the registry.

    Strategy:
    1. Search with the exact query.
    2. If no result, try a normalized variant (lowercase, remove accents).
    3. If still no result, try only the first name or a prefix.
    4. After all attempts, return the best candidates found.

    Always call the search tool — never return a name without searching first.
    """


    class EntityResolver(nn.Agent):
        instructions      = RESOLVER_INSTRUCTIONS
        model             = chat_model
        generation_schema = ReAct
        signature         = "query: str -> candidates: list[str], resolved: bool"
        config            = {"verbose": True}
        templates         = {
            "response": (
                "{% if final_answer.resolved and final_answer.candidates %}"
                "Resolved: {{ final_answer.candidates[0] }}"
                "{% elif final_answer.candidates %}"
                "Candidates: {{ final_answer.candidates | join(', ') }}. Ask the user to confirm."
                "{% else %}"
                "No match found. Ask the user to spell the full name."
                "{% endif %}"
            )
        }


    class ParticipantResolver(EntityResolver):
        tools = [ParticipantSearcher]


    class CompanyResolver(EntityResolver):
        tools = [CompanySearcher]


    class CompetitorResolver(EntityResolver):
        tools = [CompetitorSearcher]


    class QueryEntity(nn.Module):
        """Validate and resolve an entity name against the registry.

        Call this tool before filling any participant, companies, or competitor field.
        Pass the name exactly as heard — the resolver handles typos and transcription errors.
        """

        def __init__(self):
            super().__init__()
            self.set_name("query_entity")
            self.set_annotations({
                "domain": Literal["participant", "company", "competitor"],
                "query":  str,
                "return": str,
            })
            self.resolvers = nn.ModuleDict({
                "participant": ParticipantResolver(),
                "company":     CompanyResolver(),
                "competitor":  CompetitorResolver(),
            })

        def forward(self, domain: str, query: str) -> str:
            return self.resolvers[domain](query=query)

        async def aforward(self, domain: str, query: str) -> str:
            return await self.resolvers[domain].acall(query=query)


    REQUIRED_FIELDS = ["companies", "location", "participants", "purpose", "next_steps"]
    OPTIONAL_FIELDS = ["competitors", "closing_deadline", "notes"]
    ALL_FIELDS      = REQUIRED_FIELDS + OPTIONAL_FIELDS


    def empty_form() -> dict:
        return {field: None for field in ALL_FIELDS}


    def _active_draft(vars: dict) -> dict:
        """Return the mutable dict for the currently active draft."""
        active = vars.get("active_report")
        if not active:
            raise ValueError("No active report. Call activate_report first.")
        drafts = vars.setdefault("drafts", {})
        return drafts.setdefault(active, empty_form())


    def _fetch_participant_data(name: str) -> dict:
        return {
            "id":    fake.uuid4()[:8].upper(),
            "name":  name,
            "email": fake.email(),
            "role":  fake.job(),
        }


    def _fetch_company_data(name: str) -> dict:
        return {
            "id":      fake.uuid4()[:8].upper(),
            "name":    name,
            "website": fake.url(),
        }


    def _fetch_competitor_data(name: str) -> dict:
        return {
            "id":      fake.uuid4()[:8].upper(),
            "name":    name,
            "segment": fake.catch_phrase(),
        }


    @mf.tool_config(inject_vars=True)
    def activate_report(report_id: str = "", **kwargs) -> dict:
        """Create a new report draft or switch to an existing one.

        If report_id is empty, a new draft is created automatically (draft_1, draft_2, …).
        If report_id matches an existing draft or submitted report, that context becomes active.

        Always call this before starting to fill a report.
        Returns the active report_id and a summary of all drafts and submitted reports.
        """
        vars   = kwargs["vars"]
        drafts = vars.setdefault("drafts", {})

        if not report_id:
            # Auto-generate next draft ID
            n         = len(drafts) + 1
            report_id = f"draft_{n}"
            drafts.setdefault(report_id, empty_form())
        elif report_id not in drafts:
            # Could be a submitted report being re-activated as a new draft slot
            submitted = [r["id"] for r in (vars.get("reports") or [])]
            if report_id not in submitted:
                drafts[report_id] = empty_form()

        vars["active_report"] = report_id
        return {
            "active_report": report_id,
            "drafts":        list(drafts.keys()),
            "submitted":     [r["id"] for r in (vars.get("reports") or [])],
        }


    @mf.tool_config(inject_vars=True)
    def fill_fields(updates: dict[str, Union[str, list[str], None]], **kwargs) -> dict:
        """Fill visit report fields with validated values.

        Call after query_entity resolves a name, or directly for non-sensitive fields.
        Only pass fields that changed — the rest are preserved.
        Valid fields: companies, location, participants, purpose, next_steps,
        competitors, closing_deadline, notes.

        Operates on the active report draft. Call activate_report first if needed.

        For list fields (participants, companies, competitors), passing a new value
        replaces the entire current list and rebuilds all enriched data from scratch.
        Use this to correct a previously filled field.
        """
        vars  = kwargs["vars"]
        draft = _active_draft(vars)
        for field, value in updates.items():
            if field not in ALL_FIELDS or value is None:
                continue
            draft[field] = value
            if field == "participants":
                names = value if isinstance(value, list) else [value]
                draft["_participants_data"] = [_fetch_participant_data(n) for n in names]
            elif field == "companies":
                names = value if isinstance(value, list) else [value]
                draft["_companies_data"] = [_fetch_company_data(n) for n in names]
            elif field == "competitors":
                names = value if isinstance(value, list) else [value]
                draft["_competitors_data"] = [_fetch_competitor_data(n) for n in names]
        return {
            "active_report":    vars["active_report"],
            "filled":           [f for f in ALL_FIELDS if draft.get(f)],
            "missing_required": [f for f in REQUIRED_FIELDS if not draft.get(f)],
        }


    @mf.tool_config(inject_vars=True)
    def validate_report(**kwargs) -> dict:
        """Check which required fields are still missing in the active report draft."""
        vars  = kwargs["vars"]
        draft = _active_draft(vars)
        missing = [f for f in REQUIRED_FIELDS if not draft.get(f)]
        return {
            "active_report":    vars["active_report"],
            "complete":         len(missing) == 0,
            "missing_required": missing,
            "filled":           {f: draft[f] for f in ALL_FIELDS if draft.get(f)},
        }


    @mf.tool_config(inject_vars=True)
    def submit_report(**kwargs) -> dict:
        """Submit the active report draft. Only call after validate_report returns complete=True."""
        vars  = kwargs["vars"]
        draft = _active_draft(vars)
        missing = [f for f in REQUIRED_FIELDS if not draft.get(f)]
        if missing:
            return {"error": f"Cannot submit: missing required fields: {missing}"}
        report_id  = f"VISIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        report_rec = {
            "id":     report_id,
            "date":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "submitted",
            "fields": {f: draft[f] for f in ALL_FIELDS if draft.get(f)},
            "_participants_data": draft.get("_participants_data", []),
            "_companies_data":    draft.get("_companies_data", []),
            "_competitors_data":  draft.get("_competitors_data", []),
        }
        # Remove draft from active drafts after submission
        drafts = vars.get("drafts", {})
        drafts.pop(vars["active_report"], None)
        vars["active_report"] = None

        reports = list(vars.get("reports") or [])
        reports.append(report_rec)
        vars["reports"] = reports
        return {"success": True, "report_id": report_id}


    @mf.tool_config(inject_vars=True, return_direct=True)
    def get_report(report_id: str = "", **kwargs) -> str:
        """Return a report by its ID.

        Priority when report_id is empty:
          1. Active report draft (if any)
          2. Most recent submitted report

        Pass a specific report_id to retrieve a draft or submitted report by name.
        """
        vars    = kwargs["vars"]
        drafts  = vars.get("drafts") or {}
        reports = list(vars.get("reports") or [])

        # Resolve which report to show
        if report_id:
            if report_id in drafts:
                draft  = drafts[report_id]
                fields = {f: draft[f] for f in ALL_FIELDS if draft.get(f)}
                lines  = [f"# Draft Report {report_id}  (in progress)"]
                for label, value in fields.items():
                    lines.append(f"{label}: {', '.join(value) if isinstance(value, list) else value}")
                _append_enriched_lines(lines, draft)
                return "\n".join(lines)
            rec = next((r for r in reports if r["id"] == report_id), None)
            if rec is None:
                return f"Report {report_id!r} not found."
        else:
            active = vars.get("active_report")
            if active and active in drafts:
                draft  = drafts[active]
                fields = {f: draft[f] for f in ALL_FIELDS if draft.get(f)}
                lines  = [f"# Draft Report {active}  (in progress)"]
                for label, value in fields.items():
                    lines.append(f"{label}: {', '.join(value) if isinstance(value, list) else value}")
                _append_enriched_lines(lines, draft)
                return "\n".join(lines)
            if not reports:
                return "No reports found."
            rec = reports[-1]

        fields = rec.get("fields", {})
        lines  = [f"# Visit Report {rec['id']}  ({rec['date']})"]
        for label, value in fields.items():
            lines.append(f"{label}: {', '.join(value) if isinstance(value, list) else value}")
        _append_enriched_lines(lines, rec)
        return "\n".join(lines)


    def _append_enriched_lines(lines: list, source: dict) -> None:
        participants = source.get("_participants_data", [])
        companies    = source.get("_companies_data", [])
        competitors  = source.get("_competitors_data", [])
        if participants:
            lines.append("\nParticipants:")
            for p in participants:
                lines.append(f"  - {p['name']} ({p['role']}) — {p['email']}")
        if companies:
            lines.append("\nCompanies:")
            for c in companies:
                lines.append(f"  - {c['name']} — {c['website']}")
        if competitors:
            lines.append("\nCompetitors:")
            for c in competitors:
                lines.append(f"  - {c['name']}: {c['segment']}")


    class STT(nn.Transcriber):
        model          = stt_model
        message_fields = {"task_multimodal": {"audio": "audio_content"}}
        response_mode  = "user"


    class VisitAssistant(nn.Agent):
        model          = chat_model
        system_message = """
        You are a CRM assistant integrated into the sales team's workflow.
        Your job is to help reps log client visit reports quickly and accurately
        through a natural conversation — no forms, no manual field entry.
        """
        instructions = """
        The report has required fields that must all be present before submitting:
        companies, location, participants, purpose, next_steps.

        Optional fields — fill if mentioned:
        competitors, closing_deadline, notes.

        Follow this flow on each turn:
        0. If there is no active report, call activate_report() first to create one.
        1. Extract every field you can from the user's message.
        2. For participants, companies, and competitors: always call query_entity first.
           Never fill these fields with an unvalidated name.
        3. Call fill_fields to save validated values into the active draft.
        4. Call validate_report to check what is still missing.
        5. Ask only for missing required fields — one question at a time.
        6. Once validate_report returns complete=True, confirm with the user and call submit_report.
        7. After submitting, inform the user (report_id) and offer to show the full report.
           If they confirm, call get_report.

        Multiple drafts:
        - If the user wants to start a new report, call activate_report() — a new draft is created automatically.
        - If the user wants to resume a previous draft, call activate_report(report_id="draft_N").
        - All fill/validate/submit operations always target the currently active draft.
        """
        system_extra_message = (
            "<customer_info>\n"
            "Rep name: {{ user_name }}\n"
            "Active report: {{ active_report or 'none' }}\n"
            "{% if drafts %}"
            "\nDrafts in progress:\n"
            "{% for id, draft in drafts.items() %}"
            "  - {{ id }}: {% set filled = [] %}{% for f, v in draft.items() %}{% if v and not f.startswith('_') %}{% set _ = filled.append(f) %}{% endif %}{% endfor %}{{ filled | join(', ') or 'empty' }}\n"
            "{% endfor %}"
            "{% endif %}"
            "{% if reports %}"
            "\nSubmitted reports:\n"
            "{% for r in reports %}"
            "  - {{ r.id }} ({{ r.date }}) — status: {{ r.status }}\n"
            "{% endfor %}"
            "{% endif %}"
            "</customer_info>"
        )
        message_fields = {
            "task": "user.text",
            "vars": "vars",
        }
        tools         = [QueryEntity, activate_report, fill_fields, validate_report, submit_report, get_report]
        response_mode = "response"
        config        = {"verbose": True, "include_date": True}


    class VisitReportAssistant(nn.Module):
        """Entry point for the visit report conversation.

        Manages the full turn lifecycle: form initialisation, audio transcription,
        and agent execution with history.
        """

        def __init__(self):
            super().__init__()
            self.agent = VisitAssistant()
            self.stt   = STT()

        def _init_form(self, msg: mf.Message) -> None:
            """Seed msg.vars with draft/report containers on the first turn."""
            if msg.get("vars.drafts") is None:
                msg.set("vars.drafts", {})
            if msg.get("vars.active_report") is None:
                msg.set("vars.active_report", None)
            if msg.get("vars.reports") is None:
                msg.set("vars.reports", [])

        @staticmethod
        def _extract_direct_result(response: any) -> any:
            """Extract the tool result string from a return_direct tool_responses payload."""
            if isinstance(response, dict) and "tool_responses" in response:
                calls = response["tool_responses"].get("tool_calls", [])
                if calls:
                    return calls[0].get("result", response)
            return response

        def forward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
            self._init_form(msg)

            if msg.get("audio_content"):
                self.stt(msg)

            self.agent(msg, messages=history)

            msg.response = self._extract_direct_result(msg.response)
            return msg

        async def aforward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
            self._init_form(msg)

            if msg.get("audio_content"):
                await self.stt.acall(msg)

            await self.agent.acall(msg, messages=history)

            if isinstance(msg.response, dict) and "report_id" in msg.response:
                msg.report_id = msg.response["report_id"]

            return msg


    if __name__ == "__main__":
        assistant = VisitReportAssistant()
        msg       = mf.Message()
        history   = []

        msg.set("vars.user_name", "Alex Johnson")

        # Use real names from the generated corpora so the resolvers can find them.
        _company1   = companies_corpus[0]
        _company2   = companies_corpus[1]
        _p1, _p2    = participants_corpus[0], participants_corpus[1]
        _p3         = participants_corpus[2]
        _competitor = competitors_corpus[0]  # e.g. "SAP"

        def turn(text: str, label: str = "Assistant") -> None:
            msg.set("user.text", text)
            assistant(msg, history=history)
            history.append(mf.ChatBlock.assist(msg.response))
            print(f"\n[{label}]: {msg.response}")

        # -- Report 1 --
        turn(
            f"Start a new visit report. I visited {_company1} in Austin today. "
            f"{_p1} and {_p2} from IT were there. "
            f"We discussed the ERP proposal — they're also evaluating {_competitor}. "
            f"They want to close by October.",
            "Turn 1",
        )
        turn("Purpose was the ERP renewal. Next steps: send revised proposal by Friday.", "Turn 2")
        turn("Yes, save it.", "Turn 3")

        # -- Report 2 --
        turn(
            f"New visit: I was at {_company2} in São Paulo. "
            f"{_p3} from procurement. Purpose: contract renewal. "
            f"Next steps: follow-up call next week.",
            "Turn 4",
        )
        turn("Yes, go ahead and submit.", "Turn 5")

        # -- See both reports --
        turn("Show me the submitted reports.", "Turn 6")
    ```

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — tools, message fields, and inject_vars
- [nn.Searcher](../learn/nn/searcher.md) — retrieval modules and response templates
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) — ReAct and structured output
- [Async](../learn/nn/agent/async.md) — running assistants asynchronously with `acall`
