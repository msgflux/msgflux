---
hide:
  - toc
---

# Tutorials

Learn good AI system design through real problems. Each tutorial shows how msgFlux modules compose and how Inline expresses dynamic workflows, turning an industry challenge into production-ready code.

<div class="tutorial-section" markdown>

## :material-bank: Finance & Payments

<div class="grid cards" markdown>

-   [**PIX Assistant**](./pix-assistant.md) <span class="tag tag-orange">Advanced</span>

    ---

    PIX multimodal pipeline: accepts text, voice notes, or images with embedded keys, resolves contacts via BM25 lookup, and routes to a payment tool with guardrails.

    `multimodal` · `BM25` · `tools` · `guardrails`

-   [**Restaurant Supply Assistant**](./restaurant-supply-assistant.md)

    ---

    Multimodal purchasing assistant for restaurant kitchens: accepts text, audio, shelf photos, or combinations, identifies products with a VLM, and matches them to a supplier catalog via BM25.

    `multimodal` · `VLM` · `Transcriber` · `BM25` · `tools`

-   [**Food Delivery Assistant**](./food-delivery-assistant.md)

    ---

    iFood-style conversational assistant: searches dishes and restaurants via BM25, handles vague requests and dietary restrictions across multiple turns, and submits the order after confirmation.

    `BM25` · `tools` · `multi-turn` · `bcast_gather`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-transit-connection: Routing & Classification

<div class="grid cards" markdown>

-   [**Intent Router**](./intent-router.md) <span class="tag tag-purple">Intermediate</span>    

    Stop tool sprawl: route user queries to specialized handlers using typed signatures and observable intent-based orchestration.

    `Signature` · `routing` · `ChainOfThought`

-   [**Query Router with Signatures**](./signature-router.md)

    ---

    Dispatch queries across multiple backends — SQL, vector DB, knowledge base — using a typed routing layer.

    `Signature` · `routing` · `multi-backend`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-microphone: Audio & Video

<div class="grid cards" markdown>

-   [**Meeting Assistant**](./meeting-assistant.md)

    ---

    Record, transcribe, and summarize meetings. Produces structured notes, action items, and owner assignments.

    `Transcriber` · `Agent` · `audio`

-   [**Call Transcript Analysis**](./call-transcript-analysis.md)

    ---

    Analyze sales or support call recordings for sentiment, talk time, objections, and follow-up tasks.

    `Transcriber` · `structured output` · `audio`

-   [**YouTube Cut Detector**](./youtube-cut-detector.md)

    ---

    Find the best clips in a long video by analyzing the transcript for key moments, topics, and audience hooks.

    `Transcriber` · `inline` · `video`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-magnify: Research & Analysis

<div class="grid cards" markdown>

-   [**Research Scholar Agent**](./research-scholar.md)

    ---

    Multi-hop research agent that decomposes complex questions, retrieves from Wikipedia, and synthesizes structured answers.

    `Searcher` · `Wikipedia` · `ReAct`

-   [**Legal Document Review**](./legal-document-review.md)

    ---

    Extract obligations, deadlines, and risk clauses from contracts. Flags high-risk provisions with structured output.

    `Signature` · `structured output` · `PDF`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-cow: Agro & Industrial

<div class="grid cards" markdown>

-   [**Herd Monitor**](./herd-monitor.md)

    ---

    Overlays a chess-like grid on farm camera images to give the VLM spatial references, tracks herd state across observations, and fires structured alerts for missing animals, crowding, or isolation.

    `VLM` · `grid overlay` · `stateful Module` · `Inline`

-   [**Smart Collar Analytics**](./smart-collar-analytics.md)

    ---

    Individual bovine biometrics from smart collars: detects estrus, illness, and lameness via parallel sensor analysis, and enforces virtual fences with GPS geofencing and collar alerts.

    `map_gather` · `Signature` · `geofencing` · `Inline`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-traffic-light: Safety & Infrastructure

<div class="grid cards" markdown>

-   [**Road Collision Detector**](./road-collision-detector.md)

    ---

    VLM pipeline that reasons step by step through road camera frames to detect collisions, returns a structured report with severity and confidence, and dispatches emergency alerts on confirmation.

    `VLM` · `CoT` · `structured output` · `Inline` · `map_gather`

-   [**PPE Compliance Monitor**](./ppe-compliance-monitor.md)

    ---

    Analyzes construction site and factory camera frames to detect missing or incorrectly worn protective equipment per worker, using a grid overlay for spatial reference and structured violation alerts.

    `VLM` · `grid overlay` · `structured output` · `Inline` · `map_gather`

-   [**Workplace Incident Reporter**](./workplace-incident-reporter.md)

    ---

    Accepts a worker's incident description, classifies type and severity via a typed Signature, matches relevant safety regulations with BM25, and generates a formal boletim ready for CIPA submission.

    `Signature` · `BM25` · `structured output` · `Inline`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-chart-line: Business & Sales

<div class="grid cards" markdown>

-   [**Visit Report Assistant**](./visit-report.md)

    ---

    Generate structured field visit reports from salesperson notes. Formats observations into standardized CRM-ready output.

    `inline` · `structured output`

-   [**Lead Scoring**](./lead-scoring.md)

    ---

    Score inbound leads across four dimensions simultaneously — demographic fit, engagement, budget, and timing — with parallel agents.

    `bcast_gather` · `parallel` · `Signature`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-email-fast: Automation

<div class="grid cards" markdown>

-   [**Email Auto Responder**](./email-auto-responder.md)

    ---

    Classify incoming emails by intent and urgency, then draft context-aware replies. Handles escalation and triage.

    `inline` · `classification` · `generation`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-code-tags: Developer Tools

<div class="grid cards" markdown>

-   [**Code Generation & Debugging Agent**](./code-debug-agent.md)

    ---

    Iterative coding agent: generates, executes, and fixes code in a loop until tests pass.

    `Agent` · `tools` · `ReAct`

-   [**README Generator**](./readme-generator.md)

    ---

    Analyze a codebase and generate comprehensive documentation: overview, quickstart, API reference, and badges.

    `Agent` · `tools` · `code analysis`

</div>
</div>

<div class="tutorial-section" markdown>

## :material-palette: Marketing & Creative

<div class="grid cards" markdown>

-   [**Product Poster Generator**](./product-poster.md)

    ---

    Combine a product photo with a reference style image to produce polished marketing posters via image-to-image generation.

    `MediaMaker` · `Vision` · `image-to-image`

-   [**Ad Focus Group Simulator**](./ad-focus-group.md)

    ---

    Simulate a diverse group of personas that evaluate ad concepts, provide ratings, and surface strategic insights.

    `bcast_gather` · `parallel` · `ModuleList`

</div>
</div>
