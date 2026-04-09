---
hide:
  - toc
---

# Tutorials

Learn good AI system design through real problems. Each tutorial turns an industry challenge into production-ready code — showing how msgFlux modules compose, how Signatures enforce typed contracts, and how pipelines stay testable from the first line.

<div class="tutorial-section" markdown>

## :material-credit-card: Payments & Commerce

<div class="grid cards" markdown>

-   [**PIX Assistant**](./pix-assistant.md) <span class="tag tag-orange">Advanced</span>

    ---

    Multimodal PIX payment assistant: accepts text, voice notes, and images (bills, menus, tickets with embedded keys), resolves contacts via fuzzy lookup, and routes to a payment tool with guardrails.

    `Signature` · `Multimodal` · `Retriever` · `Guardrails`

-   [**Food Delivery Assistant**](./food-delivery-assistant.md) <span class="tag tag-purple">Intermediate</span>

    ---

    iFood-style conversational assistant: enriches a raw dish catalog with parallel agents, builds fuzzy indexes, handles vague requests and dietary restrictions across turns, and places the order after confirmation.

    `Generation Schema` · `Multimodal` · `Retriever` · `Guardrails`

-   [**Restaurant Supply Assistant**](./restaurant-supply-assistant.md) <span class="tag tag-orange">Advanced</span>

    ---

    Multimodal purchasing assistant for restaurant kitchens: accepts text, audio, or shelf photos, extracts items via a structured schema, and matches them to a supplier catalog via fuzzy search.

    `Signature` · `Multimodal` · `Retriever` · `Guardrails`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-handshake: Sales

<div class="grid cards" markdown>

-   [**Lead Scoring**](./lead-scoring.md) <span class="tag tag-purple">Intermediate</span>

    ---

    Score inbound leads across four dimensions simultaneously — demographic fit, engagement, budget, and timing — with parallel agents and a typed Signature per dimension.

    `Signature` · `parallel`

-   [**Deal Briefing Generator**](./deal-briefing-generator.md) <span class="tag tag-purple">Intermediate</span>

    ---

    Turns a sales call recording into a structured pre-meeting briefing: transcribes the audio, extracts pain points, objections, and the agreed next step with few-shot examples, drafts the briefing, and narrates it via TTS.

    `Signature` · `Few-shot` · `Multimodal`

-   [**Visit Report Assistant**](./visit-report.md) <span class="tag tag-orange">Advanced</span>

    ---

    Generate structured field visit reports from salesperson voice notes and photos. Retrieves client history and formats observations into CRM-ready output.

    `Multimodal` · `Retriever`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-headset: Customer Service

<div class="grid cards" markdown>

-   [**Support Ticket Router**](./support-ticket-router.md) <span class="tag tag-teal">Beginner</span>

    ---

    Classify incoming support tickets by category and urgency with a typed Signature and route each to the right team — no agent loop needed.

    `Signature` · `Inline`

-   [**Email Auto Responder**](./email-auto-responder.md) <span class="tag tag-purple">Intermediate</span>

    ---

    Classify incoming emails by intent and urgency, then draft context-aware replies. Handles escalation routing and triage with typed outputs.

    `Signature`

-   [**Call Transcript Analysis**](./call-transcript-analysis.md) <span class="tag tag-purple">Intermediate</span>

    ---

    Analyze call recordings or transcripts for sentiment across three phases, resolution quality, and predicted CSAT — with a full reasoning trace for QA audits.

    `Signature` · `Reasoning` · `Multimodal`

-   [**Streaming Support Triage**](./streaming-support-triage.md) <span class="tag tag-green">Beginner</span>

    ---

    Stream a support reply while the agent checks order status and decides whether to answer directly or escalate. A compact example of real-time output plus tool calls.

    `Streaming` · `Tools` · `Signature`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-calendar-check: Meetings & Productivity

<div class="grid cards" markdown>

-   [**Meeting Assistant**](./meeting-assistant.md) <span class="tag tag-purple">Intermediate</span>

    ---

    Transcribe a meeting recording and extract structured notes: decisions, action items with owner and deadline, open questions, sentiment, and a follow-up flag — all with typed outputs from a single Signature.

    `Signature` · `Multimodal`

-   [**Meeting Action Items Tracker**](./meeting-action-items.md) <span class="tag tag-purple">Intermediate</span>

    ---

    Extract action items from meeting transcripts or audio recordings. Few-shot examples teach the model the difference between a committed task and a vague intention, a named assignee and an implicit one.

    `Signature` · `Few-shot` · `Multimodal`

-   [**YouTube Video Cut Detector**](./youtube-cut-detector.md) <span class="tag tag-green">Beginner</span>

    ---

    Find the best clips in a long video by analyzing its transcript for key moments, topic shifts, and audience hooks.

    `Inline`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-palette: Marketing & Creative

<div class="grid cards" markdown>

-   [**Product Poster Generator**](./product-poster.md)

    ---

    Combine a product photo with a reference style image to produce polished marketing posters via image-to-image generation.

    `MediaMaker` · `Vision`

-   [**Ad Focus Group Simulator**](./ad-focus-group.md) <span class="tag tag-orange">Advanced</span>

    ---

    Simulate a diverse group of personas that evaluate ad concepts in parallel, provide ratings, and surface strategic insights across demographic angles.

    `parallel` · `ModuleList`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-transit-connection: Routing & Classification

<div class="grid cards" markdown>

-   [**Intent Router**](./intent-router.md) <span class="tag tag-purple">Intermediate</span>

    ---

    Stop tool sprawl: route user queries to specialized handlers using typed Signatures and observable intent-based orchestration instead of a single overloaded agent.

    `Signature` · `Reasoning`

-   [**Query Router with Signatures**](./signature-router.md) <span class="tag tag-purple">Intermediate</span>

    ---

    Dispatch queries across multiple backends — SQL, vector DB, knowledge base — using a typed routing layer with one Signature per specialist.

    `Signature`

-   [**Oracle Specialist Tool**](./oracle-specialist-tool.md) <span class="tag tag-green">Beginner</span>

    ---

    Build a root assistant that delegates product and policy questions to an Oracle specialist agent. A compact example of agent-as-tool, ChainOfThought, and response templating.

    `Tools` · `ChainOfThought` · `Templates`

-   [**Plan Tool with Root Context**](./plan-tool-with-root-context.md) <span class="tag tag-green">Beginner</span>

    ---

    Delegate planning to a specialist tool that receives both the root task and the full conversation history. A compact example of `inject_messages=True`.

    `Tools` · `ChainOfThought` · `Templates`

</div>
</div>
