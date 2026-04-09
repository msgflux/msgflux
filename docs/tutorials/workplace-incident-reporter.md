# Workplace Incident Reporter

Every manufacturing and industrial facility is required to document workplace incidents formally. The challenge is not the paperwork itself — it is the gap between how workers describe what happened and what a CIPA (Internal Accident Prevention Commission) report or an eSocial notification actually requires.

Workers dictate or type incident descriptions in plain, first-person language. The formal boletim needs a classified incident type, a root cause, the applicable NR safety regulations, corrective actions, and a decision on whether an eSocial notification is mandatory. Bridging that gap manually costs safety officers hours per incident and introduces inconsistency.

This tutorial builds a **Workplace Incident Reporter** that accepts a raw incident description, classifies it with a structured analyzer, looks up the most relevant safety regulations via BM25, and produces a complete `IncidentBoletim` ready for CIPA review and eSocial submission.

---

## Architecture

```
Incident description (text)
      │
      ▼
IncidentAnalyzer  (Agent + Signature)
extracts: type, severity, body part, immediate cause,
          root cause, involved parties, PPE gaps
      │
      ▼
NRLookup  (BM25 Searcher)
queries with description + incident type
returns top-3 applicable safety regulations
      │
      ▼
ReportWriter  (Agent + generation_schema)
generates formal IncidentBoletim
{corrective_actions, preventive_actions,
 requires_cipa_investigation,
 requires_esocial_notification,
 formal_report}
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

```bash
pip install rank-bm25 msgspec
```

---

## Step 1 — Models and Imports

```python
import msgspec
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from dataclasses import dataclass
from typing import Optional
```

A single chat completion model drives all three modules. `gpt-4.1-mini` is well-suited here: the classification and report generation tasks are text-only and benefit from speed and low cost at scale.

```python
mf.load_dotenv()
chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
```

---

## Step 2 — Incident Input and Simulated Data

The pipeline receives an `IncidentInput` dataclass containing everything the worker provided at the time of reporting. Real deployments attach this data from a mobile form or voice transcription; here it is simulated with eight realistic first-person narratives.

```python
@dataclass
class IncidentInput:
    id:          str
    reporter:    str
    role:        str        # e.g. "Electrician", "Forklift Operator"
    sector:      str
    timestamp:   str
    description: str        # 3-5 sentences of first-person narrative
```

The descriptions below are written at the level of detail a real worker would provide when filling in a digital incident form immediately after the event. They name the specific activity, the exact body part, what PPE was or was not worn, and what was done immediately afterward.

```python
SIMULATED_INCIDENTS = [
    IncidentInput(
        id="INC-001",
        reporter="Roberto Almeida",
        role="Electrician",
        sector="Packaging Line – Electrical Panel Room",
        timestamp="2026-03-10T09:14:00",
        description=(
            "I was performing preventive maintenance on the packaging area panel. "
            "I assumed the circuit was de-energized but it wasn't. "
            "When I grabbed the contactor terminal I got a shock. "
            "I felt tingling in my right arm and have a small burn on my index and middle fingers. "
            "I went to the infirmary for dressing. I was not wearing insulating gloves at the time."
        ),
    ),
    IncidentInput(
        id="INC-002",
        reporter="Marcos Souza",
        role="Forklift Operator",
        sector="Raw Materials Warehouse",
        timestamp="2026-03-11T14:32:00",
        description=(
            "I was reversing the forklift toward Bay 7 to pick up a pallet of resin drums. "
            "The spotter had stepped away for a moment and I didn't realize a co-worker, Luiz, "
            "was walking behind the vehicle. The rear wheel ran over his left foot. "
            "Luiz was wearing safety boots with steel toe cap. "
            "He was taken to the hospital by ambulance and is receiving treatment for a possible fracture. "
            "The backup alarm on the forklift was working. I had my seatbelt on."
        ),
    ),
    IncidentInput(
        id="INC-003",
        reporter="Carla Mendes",
        role="Maintenance Technician",
        sector="Finished Goods Storage – Mezzanine Level",
        timestamp="2026-03-12T11:05:00",
        description=(
            "I was replacing a burned-out fluorescent fixture on the mezzanine, "
            "standing on a portable aluminum ladder about 3.5 meters above the floor. "
            "The ladder shifted sideways on the smooth concrete and I lost my balance. "
            "I fell and landed on my left shoulder and hip. "
            "I was wearing a hard hat but no fall harness because no one told me it was required "
            "for this height in this area. "
            "I was taken to the clinic and X-rays showed a hairline fracture on the left shoulder blade."
        ),
    ),
    IncidentInput(
        id="INC-004",
        reporter="Fernanda Costa",
        role="Assembly Line Operator",
        sector="Electronics Assembly – Station 12",
        timestamp="2026-03-13T15:48:00",
        description=(
            "Over the past three weeks my right wrist has been getting progressively more painful. "
            "I work eight hours a day fitting circuit board connectors using a pneumatic driver. "
            "The motion is the same every cycle — grip, insert, drive, release — about 900 times per shift. "
            "I reported it to my supervisor last week and finally went to occupational health today. "
            "The doctor diagnosed me with incipient carpal tunnel syndrome. "
            "I have not been rotating stations or taking micro-breaks as the ergonomics program recommends."
        ),
    ),
    IncidentInput(
        id="INC-005",
        reporter="Diego Farias",
        role="Press Operator",
        sector="Metal Stamping Department",
        timestamp="2026-03-14T08:22:00",
        description=(
            "I was removing a jammed sheet from the 400-ton mechanical press. "
            "I used a metal rod to try to free the sheet without locking out the machine "
            "because I thought it would only take a second. "
            "The press cycled unexpectedly and trapped three fingers of my left hand between the die plates. "
            "I was wearing cut-resistant gloves but they provided no protection against this kind of force. "
            "I lost the tip of the index finger and sustained deep lacerations on the middle and ring fingers. "
            "I was taken directly to the hospital by a supervisor."
        ),
    ),
    IncidentInput(
        id="INC-006",
        reporter="Tatiane Rocha",
        role="Chemical Process Operator",
        sector="Surface Treatment – Tank Room",
        timestamp="2026-03-17T10:55:00",
        description=(
            "I was transferring sodium hydroxide solution from the bulk drum to the treatment tank "
            "using a hand pump. The hose connection on the drum was loose and it disconnected "
            "while the pump was running. About two liters of solution sprayed onto my face and chest. "
            "I was wearing safety glasses but not a full face shield. "
            "I immediately went to the emergency eyewash station and flushed for fifteen minutes. "
            "My eyes are red and burning and there are irritation marks on my chin and neck. "
            "The nurse has referred me to the ophthalmologist."
        ),
    ),
    IncidentInput(
        id="INC-007",
        reporter="Eduardo Pires",
        role="Welder",
        sector="Fabrication Shop – Bay 3",
        timestamp="2026-03-18T13:20:00",
        description=(
            "I have been doing continuous MIG welding on galvanized steel frames for the past four days. "
            "The local exhaust ventilation hood above my station was reported broken last Monday "
            "and has not been fixed yet. "
            "Today I started feeling a headache, dizziness, and tightness in my chest about two hours "
            "into the shift. A colleague noticed I looked pale and took me to the medical center. "
            "The occupational nurse suspects metal fume fever from zinc oxide exposure. "
            "I was wearing a half-face respirator with P100 filters, which are not rated for metal fumes."
        ),
    ),
    IncidentInput(
        id="INC-008",
        reporter="André Lima",
        role="Carpenter",
        sector="Maintenance Workshop",
        timestamp="2026-03-19T16:03:00",
        description=(
            "I was cutting a 50mm pine board on the bench circular saw to make a replacement shelf. "
            "The wood kicked back when it contacted a knot in the grain and my right hand "
            "slid forward into the blade. "
            "The blade guard had been removed by someone earlier in the day and I did not check before starting. "
            "I was not wearing cut-resistant gloves. "
            "The cut is about 6 cm long across the palm, deep but not reaching the tendons. "
            "I applied direct pressure and a colleague drove me to the emergency room."
        ),
    ),
]
```

---

## Step 3 — NR Safety Regulation Catalog

The BM25 index is built from this catalog. Each entry combines a formal description with keyword terms that overlap naturally with incident narratives — the retriever does not need to understand semantics, only token overlap.

```python
NR_CATALOG = [
    {
        "nr":          "NR-6",
        "title":       "Personal Protective Equipment (PPE)",
        "description": "Mandates employer provision and worker use of PPE. Defines responsibilities, selection criteria, hygiene, and disposal procedures.",
        "keywords":    "ppe helmet gloves boots vest protective equipment personal",
    },
    {
        "nr":          "NR-9",
        "title":       "Occupational Exposure Assessment and Control",
        "description": "Requires companies to develop and implement a program identifying and controlling physical, chemical, and biological agents in the workplace.",
        "keywords":    "noise heat radiation chemical biological environmental risk exposure fumes",
    },
    {
        "nr":          "NR-10",
        "title":       "Safety in Electrical Installations and Services",
        "description": "Establishes minimum requirements to ensure safety of workers interacting with electrical installations and services.",
        "keywords":    "electric shock electricity panel wire conductor voltage energy burn",
    },
    {
        "nr":          "NR-12",
        "title":       "Safety in Machinery and Equipment",
        "description": "Defines technical references and protective measures to ensure health and physical integrity of workers operating machinery and equipment.",
        "keywords":    "machine equipment press guillotine protection device safety maintenance entrapment",
    },
    {
        "nr":          "NR-17",
        "title":       "Ergonomics",
        "description": "Establishes parameters to adapt working conditions to workers' psychophysiological characteristics, covering load lifting, transport, and repetitive tasks.",
        "keywords":    "ergonomics posture lifting weight load repetition effort lumbar strain repetitive",
    },
    {
        "nr":          "NR-18",
        "title":       "Conditions and Work Environment in Construction",
        "description": "Establishes administrative, planning, and organizational guidelines to implement control measures and preventive systems in civil construction.",
        "keywords":    "construction scaffolding fall height ladder site civil building",
    },
    {
        "nr":          "NR-35",
        "title":       "Work at Height",
        "description": "Establishes minimum requirements and protective measures for work at height, covering planning, organization, and execution to ensure worker safety.",
        "keywords":    "height fall harness safety belt scaffold roof ladder elevated anchor",
    },
]
```

---

## Step 4 — BM25 Index and NRLookup

The BM25 corpus is built by concatenating each NR entry's description and keywords into a single string. This gives the retriever both the formal regulatory language and the informal vocabulary workers use.

```python
_nr_corpus = [
    f"{entry['nr']} | {entry['title']} | {entry['description']} {entry['keywords']}"
    for entry in NR_CATALOG
]
_nr_by_index = {i: entry for i, entry in enumerate(NR_CATALOG)}

nr_retriever = mf.Retriever.lexical("rank_bm25")
nr_retriever.add(_nr_corpus)


class NRSearcher(nn.Searcher):
    retriever = nr_retriever
    config    = {"top_k": 3}


class NRLookup(nn.Module):
    """
    Retrieves the top-3 applicable NR regulations for an incident
    by querying BM25 with the incident description combined with
    the classified incident type.
    """

    def __init__(self):
        super().__init__()
        self.searcher = NRSearcher()

    def forward(self, msg: mf.Message) -> mf.Message:
        query   = f"{msg.incident.description} {msg.analysis['incident_type']}"
        results = self.searcher(query)
        hits    = results[0]["results"] if results else []

        applicable = []
        for hit in hits:
            # Extract NR code from corpus string "NR-10 | Title | ..."
            nr_code = hit["data"].split(" | ")[0].strip()
            # Resolve to catalog entry by matching code
            entry   = next((e for e in NR_CATALOG if e["nr"] == nr_code), None)
            if entry:
                applicable.append(f"{entry['nr']} – {entry['title']}")

        msg.applicable_nrs = applicable
        return msg
```

---

## Step 5 — IncidentAnalyzer

`IncidentAnalyzer` extracts a structured safety assessment from the raw narrative. The `Signature` defines every field the downstream report writer will need — classification, severity, affected body part, causes, and PPE gaps.

```python
class IncidentAnalysis(mf.Signature):
    """Analyze a workplace incident report and extract structured safety information."""
    description:   mf.InputField(desc="Raw incident description from the worker")
    reporter_role: mf.InputField(desc="Job role of the person reporting")
    sector:        mf.InputField(desc="Work sector or area where the incident occurred")

    incident_type:      mf.OutputField(desc="Type: electrical, fall, ergonomic, chemical, mechanical, collision, other")
    severity:           mf.OutputField(desc="near_miss | minor | moderate | severe")
    body_part_affected: mf.OutputField(desc="Body part(s) affected, or 'none' for near-miss")
    immediate_cause:    mf.OutputField(desc="Direct cause of the incident")
    root_cause:         mf.OutputField(desc="Underlying systemic cause")
    involved_parties:   mf.OutputField(desc="Names or roles of people involved")
    ppe_gaps:           mf.OutputField(desc="PPE that was missing or incorrectly used")


class IncidentAnalyzer(nn.Agent):
    """
    Classifies a workplace incident from a raw worker narrative.
    Extracts type, severity, causal chain, and PPE gaps for the report writer.
    """
    model          = chat_model
    system_message = """
    You are an occupational safety specialist analyzing workplace incident reports.
    Read the worker's first-person narrative carefully and extract factual,
    specific information — do not infer beyond what is stated.
    For severity: near_miss = no injury; minor = first aid only;
    moderate = medical treatment beyond first aid; severe = hospitalization or permanent impairment.
    """
    signature      = IncidentAnalysis

    def forward(self, msg: mf.Message) -> mf.Message:
        result      = self(
            description=msg.incident.description,
            reporter_role=msg.incident.role,
            sector=msg.incident.sector,
        )
        msg.analysis = result
        return msg
```

---

## Step 6 — IncidentBoletim and ReportWriter

`IncidentBoletim` is a `msgspec.Struct` that defines the formal output. The `requires_cipa_investigation` and `requires_esocial_notification` flags are computed by the model based on severity and whether an actual injury occurred — the agent applies Brazilian regulatory thresholds.

```python
class IncidentBoletim(msgspec.Struct):
    incident_id:                   str
    date:                          str
    reporter:                      str
    sector:                        str
    severity:                      str
    incident_type:                 str
    body_part_affected:            str
    immediate_cause:               str
    root_cause:                    str
    applicable_regulations:        list[str]
    corrective_actions:            list[str]   # immediate fixes
    preventive_actions:            list[str]   # systemic changes
    requires_cipa_investigation:   bool        # moderate or severe
    requires_esocial_notification: bool        # any injury
    formal_report:                 str         # 150-200 word formal narrative
```

`ReportWriter` receives the full context — raw narrative, structured analysis, and applicable NRs — and produces the final boletim. The `generation_schema` ensures the output conforms to `IncidentBoletim` without manual JSON parsing.

```python
class ReportWriter(nn.Agent):
    """
    Generates a formal CIPA/eSocial incident boletim from the structured analysis.
    Acts as a CIPA safety officer drafting the official record.
    """
    model          = chat_model
    system_message = """
    You are a CIPA safety officer writing a formal incident report (Boletim de Ocorrência)
    for Brazilian workplace safety compliance.

    Rules:
    - requires_cipa_investigation must be True for severity 'moderate' or 'severe'.
    - requires_esocial_notification must be True whenever a worker suffered a physical injury,
      regardless of severity level.
    - corrective_actions are immediate fixes (lock-out, PPE replacement, area isolation).
    - preventive_actions are systemic changes (training, procedure update, equipment inspection program).
    - formal_report must be 150-200 words in professional, third-person language.
    """
    generation_schema = IncidentBoletim

    def forward(self, msg: mf.Message) -> mf.Message:
        context = (
            f"Incident ID: {msg.incident.id}\n"
            f"Date: {msg.incident.timestamp}\n"
            f"Reporter: {msg.incident.reporter} ({msg.incident.role})\n"
            f"Sector: {msg.incident.sector}\n\n"
            f"Worker narrative:\n{msg.incident.description}\n\n"
            f"Analysis:\n"
            f"  Type: {msg.analysis['incident_type']}\n"
            f"  Severity: {msg.analysis['severity']}\n"
            f"  Body part affected: {msg.analysis['body_part_affected']}\n"
            f"  Immediate cause: {msg.analysis['immediate_cause']}\n"
            f"  Root cause: {msg.analysis['root_cause']}\n"
            f"  Involved parties: {msg.analysis['involved_parties']}\n"
            f"  PPE gaps: {msg.analysis['ppe_gaps']}\n\n"
            f"Applicable regulations:\n"
            + "\n".join(f"  - {nr}" for nr in msg.applicable_nrs)
        )
        msg.boletim = self(context)
        return msg
```

---

## Step 7 — IncidentPipeline

`IncidentPipeline` wires the three modules together with `mf.Inline`. The pipeline expression `"analyze -> lookup_nrs -> write_report"` conveys the data flow clearly — each step reads from and writes to a shared `mf.Message`.

```python
class IncidentPipeline(nn.Module):
    """
    End-to-end pipeline: raw incident description → formal CIPA/eSocial boletim.

    Steps:
      analyze     — IncidentAnalyzer extracts structured safety fields
      lookup_nrs  — NRLookup retrieves top-3 applicable regulations via BM25
      write_report — ReportWriter generates the formal IncidentBoletim
    """

    def __init__(self):
        super().__init__()
        self.analyzer     = IncidentAnalyzer()
        self.nr_lookup    = NRLookup()
        self.report_writer = ReportWriter()

        self._flux = mf.Inline(
            "analyze -> lookup_nrs -> write_report",
            {
                "analyze":      self.analyzer,
                "lookup_nrs":   self.nr_lookup,
                "write_report": self.report_writer,
            },
        )

    def forward(self, incident: IncidentInput) -> mf.Message:
        msg          = mf.Message()
        msg.incident = incident
        return self._flux(msg)
```

---

## Running the pipeline

The pipeline is instantiated once and reused for every incident. Each call is stateless — the shared `mf.Message` is created fresh inside `forward`.

```python
pipeline = IncidentPipeline()
```

Process a single incident and inspect the boletim:

```python
result = pipeline(SIMULATED_INCIDENTS[0])   # INC-001 — electrical shock
b      = result.boletim

print(f"ID:          {b.incident_id}")
print(f"Severity:    {b.severity}")
print(f"Type:        {b.incident_type}")
print(f"Body part:   {b.body_part_affected}")
print(f"Root cause:  {b.root_cause}")
print(f"Regulations: {b.applicable_regulations}")
print(f"CIPA inv.:   {b.requires_cipa_investigation}")
print(f"eSocial:     {b.requires_esocial_notification}")
print()
print("Corrective actions:")
for action in b.corrective_actions:
    print(f"  - {action}")
print()
print("Preventive actions:")
for action in b.preventive_actions:
    print(f"  - {action}")
print()
print("Formal report:")
print(b.formal_report)
```

Process two more incidents to see how severity and regulation matching vary across incident types:

```python
# INC-005 — mechanical entrapment (press)
result2 = pipeline(SIMULATED_INCIDENTS[4])
b2      = result2.boletim
print(f"\n[{b2.incident_id}] {b2.incident_type} | {b2.severity}")
print(f"  NRs: {b2.applicable_regulations}")
print(f"  CIPA: {b2.requires_cipa_investigation} | eSocial: {b2.requires_esocial_notification}")

# INC-007 — metal fume inhalation (welder)
result3 = pipeline(SIMULATED_INCIDENTS[6])
b3      = result3.boletim
print(f"\n[{b3.incident_id}] {b3.incident_type} | {b3.severity}")
print(f"  NRs: {b3.applicable_regulations}")
print(f"  CIPA: {b3.requires_cipa_investigation} | eSocial: {b3.requires_esocial_notification}")
```

---

## Batch processing

`F.map_gather` runs the pipeline over all eight incidents concurrently. Each incident is fully independent, so the entire batch completes in roughly the time of the slowest single call.

```python
def _process(incident: IncidentInput) -> mf.Message:
    return pipeline(incident)

results = F.map_gather(_process, args_list=[(inc,) for inc in SIMULATED_INCIDENTS])

print(f"\n{'ID':<10} {'Type':<14} {'Severity':<10} {'CIPA':<6} {'eSocial'}")
print("-" * 60)
for msg in results:
    b = msg.boletim
    print(
        f"{b.incident_id:<10} {b.incident_type:<14} {b.severity:<10} "
        f"{'Yes' if b.requires_cipa_investigation else 'No':<6} "
        f"{'Yes' if b.requires_esocial_notification else 'No'}"
    )
```

---

## Going further

### Voice input

Workers often submit incident reports by voice note rather than typing. Wrapping the pipeline with `nn.Transcriber` adds a transcription step before the rest of the pipeline.

```python
class TranscribingIncidentPipeline(nn.Module):
    """
    Accepts an audio file path, transcribes it, and runs the full incident pipeline.
    """

    def __init__(self):
        super().__init__()
        self.transcriber = nn.Transcriber(model=mf.Model.speech_to_text("openai/whisper-1"))
        self.pipeline    = IncidentPipeline()

    def forward(self, audio_path: str, incident_meta: IncidentInput) -> mf.Message:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        transcript = self.transcriber(audio_bytes)

        # Replace the text description with the transcript
        from dataclasses import replace
        incident_with_transcript = replace(incident_meta, description=transcript)

        return self.pipeline(incident_with_transcript)


# Usage
voice_pipeline = TranscribingIncidentPipeline()
result = voice_pipeline(
    audio_path="./incident_report.wav",
    incident_meta=IncidentInput(
        id="INC-009",
        reporter="Paulo Neves",
        role="Warehouse Worker",
        sector="Receiving Dock",
        timestamp="2026-03-20T07:45:00",
        description="",          # filled in from audio
    ),
)
print(result.boletim.formal_report)
```

The `nn.Transcriber` handles the audio bytes and returns a plain string. The rest of the pipeline is unchanged — the transcript is treated identically to a typed description.

### Photo attachment

When a worker can photograph the accident scene, a VLM step can describe it before the analyzer runs. This adds spatial context — the position of a machine guard, the condition of a floor surface, the absence of signage — that the worker may not think to mention.

```python
class SceneDescriber(nn.Agent):
    """
    Describes an accident scene image to add visual context to the incident analysis.
    Uses a vision-capable model.
    """
    model          = mf.Model.chat_completion("openai/gpt-4.1")
    system_message = """
    You are an occupational safety inspector examining a workplace accident scene photograph.
    Describe what you see objectively: equipment state, floor conditions, PPE visible on workers,
    signage present or absent, apparent hazards. Be specific and factual. Maximum 150 words.
    """

    def forward(self, msg: mf.Message) -> mf.Message:
        description      = self(image=msg.scene_image)
        msg.scene_description = str(description)
        return msg


class PhotoIncidentPipeline(nn.Module):
    """
    Extends the pipeline with a VLM scene description step before analysis.
    Pipeline: describe_scene -> analyze -> lookup_nrs -> write_report
    """

    def __init__(self):
        super().__init__()
        self.scene_describer  = SceneDescriber()
        self.analyzer         = IncidentAnalyzer()
        self.nr_lookup        = NRLookup()
        self.report_writer    = ReportWriter()

        self._flux = mf.Inline(
            "describe_scene -> analyze -> lookup_nrs -> write_report",
            {
                "describe_scene": self.scene_describer,
                "analyze":        self._analyze_with_scene,
                "lookup_nrs":     self.nr_lookup,
                "write_report":   self.report_writer,
            },
        )

    def _analyze_with_scene(self, msg: mf.Message) -> mf.Message:
        # Prepend scene description to the worker narrative before analysis
        augmented_description = (
            f"{msg.incident.description}\n\n"
            f"Scene photograph description: {msg.scene_description}"
        )
        result      = self.analyzer(
            description=augmented_description,
            reporter_role=msg.incident.role,
            sector=msg.incident.sector,
        )
        msg.analysis = result
        return msg

    def forward(self, incident: IncidentInput, scene_image: bytes) -> mf.Message:
        msg             = mf.Message()
        msg.incident    = incident
        msg.scene_image = scene_image
        return self._flux(msg)


# Usage
photo_pipeline = PhotoIncidentPipeline()

with open("./accident_scene.jpg", "rb") as f:
    image_bytes = f.read()

result = photo_pipeline(
    incident=SIMULATED_INCIDENTS[2],   # INC-003 — fall from ladder
    scene_image=image_bytes,
)
print(result.boletim.formal_report)
```

### Multi-turn clarification for vague near-miss reports

Near-miss reports are the most valuable safety data — they reveal hazards before an injury occurs — but they are often the shortest and vaguest. When a report is classified as `near_miss` and the description is under 100 characters, routing to a clarification agent before generating the boletim produces a far more useful record.

```python
class ClarificationAgent(nn.Agent):
    """
    Asks targeted follow-up questions when a near-miss report is too vague
    to generate a meaningful corrective action plan.
    """
    model          = chat_model
    system_message = """
    You are a CIPA safety officer conducting a structured near-miss interview.
    Ask exactly three specific follow-up questions to gather the information
    needed to write a formal incident report.
    Focus on: what activity was being performed, what specifically went wrong,
    and what PPE or safeguard was or was not in place.
    """
    signature      = """
    description: str, role: str, sector: str ->
    questions:        list[str],
    clarification_prompt: str
    """


class SmartIncidentPipeline(nn.Module):
    """
    Routes vague near-miss reports through a clarification step before analysis.
    Clear reports go directly to the standard pipeline.
    """

    def __init__(self):
        super().__init__()
        self.clarifier  = ClarificationAgent()
        self.pipeline   = IncidentPipeline()

    def forward(self, incident: IncidentInput, answers: Optional[str] = None) -> dict:
        is_vague = len(incident.description.strip()) < 100

        if is_vague and answers is None:
            # First pass — ask for clarification
            clarification = self.clarifier(
                description=incident.description,
                role=incident.role,
                sector=incident.sector,
            )
            return {
                "status":    "needs_clarification",
                "questions": clarification["questions"],
                "prompt":    clarification["clarification_prompt"],
            }

        # Second pass — incorporate answers and run full pipeline
        from dataclasses import replace
        enriched_description = (
            f"{incident.description}\n\nWorker follow-up answers:\n{answers}"
            if answers
            else incident.description
        )
        enriched = replace(incident, description=enriched_description)
        msg      = self.pipeline(enriched)
        return {"status": "complete", "boletim": msg.boletim}


# Usage
smart_pipeline = SmartIncidentPipeline()

vague_incident = IncidentInput(
    id="INC-010",
    reporter="Camila Torres",
    role="Lab Technician",
    sector="Quality Control Laboratory",
    timestamp="2026-03-21T10:00:00",
    description="Almost had an accident with a chemical today. No injury.",
)

# First call — returns clarification questions
result = smart_pipeline(vague_incident)
if result["status"] == "needs_clarification":
    print("Questions for the worker:")
    for q in result["questions"]:
        print(f"  {q}")

    # Simulate the worker answering
    worker_answers = (
        "I was pipetting hydrochloric acid into a beaker and the pipette tip came loose. "
        "The acid dripped onto the bench surface about 5 cm from my ungloved left hand. "
        "I was wearing safety glasses and a lab coat but no chemical-resistant gloves."
    )

    # Second call — generates the full boletim
    final = smart_pipeline(vague_incident, answers=worker_answers)
    print(f"\nBoletim generated: {final['boletim'].incident_id}")
    print(f"Severity: {final['boletim'].severity}")
    print(final["boletim"].formal_report)
```

The two-pass pattern avoids over-engineering the first call into a full multi-turn agent loop. Safety officers can send the clarification questions to the worker via any channel — SMS, email, app push notification — and call the pipeline again when the answers arrive.

---

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — generation schemas and structured output
- [nn.Module](../learn/nn/module/index.md) — building stateful modules with `forward`
- [nn.Searcher](../learn/nn/searcher.md) — BM25 and semantic retrieval
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Inline](../learn/inline.md) — pipeline expressions and conditional routing
- [Functional API](../learn/nn/functional.md) — `map_gather` for parallel batch processing
- [nn.Transcriber](../learn/nn/transcriber.md) — speech-to-text for voice input
