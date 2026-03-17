# Legal Document Review

Parse a contract PDF, extract the parties and key clauses, analyze risk exposure, and produce a structured legal review — using the data layer's PDF parser and a multi-pass agent pipeline.

## What You'll Build

```
contract.pdf
       │
       ▼
  Parser (PyPDF) ─── mf.Parser.pdf("pypdf")
       │  text: str
       ▼
  Extractor ───────── Signature:
                        text → parties, effective_date, governing_law,
                                termination_clauses, payment_terms, key_clauses
       │
       ▼
  RiskAnalyzer ────── Signature:
                        text, key_clauses → risks: list[dict],
                                            risk_level: Literal[...],
                                            missing_protections: list[str]
       │
       ▼
  Summarizer ─────── Signature:
                        text, risks → executive_summary, red_flags,
                                      recommendations, negotiation_points
       │
       ▼
  Structured review on msg
```

---

## Setup

```bash
pip install msgflux[openai] pypdf
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Parse the PDF

`mf.Parser.pdf()` extracts text from any PDF. Pass a file path or raw `bytes`:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from typing import Literal, List, Optional


parser = mf.Parser.pdf("pypdf")

# From a file path
response = parser("contract.pdf")
contract_text = response.data["text"]

print(f"Extracted {len(contract_text)} characters")
print(contract_text[:500])
```

!!! tip
    For scanned PDFs (images), use an OCR-enabled parser or a vision model to extract text before passing to the agents.

---

## Step 2 — Extraction Signature

```python
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class ExtractContract(Signature):
    """Extract the key structural elements from a contract."""

    text: str = InputField(desc="Full text of the contract")

    parties: List[dict] = OutputField(
        desc=(
            "All parties to the contract. Each entry: "
            "{'name': str, 'role': str, 'jurisdiction': str | None}"
        )
    )
    effective_date: Optional[str] = OutputField(
        desc="Contract effective date (ISO format if possible, else as written)"
    )
    governing_law: Optional[str] = OutputField(
        desc="Jurisdiction whose law governs the contract"
    )
    termination_clauses: List[str] = OutputField(
        desc="Conditions under which either party may terminate"
    )
    payment_terms: Optional[str] = OutputField(
        desc="Payment schedule, amounts, and late-payment penalties if present"
    )
    key_clauses: List[str] = OutputField(
        desc=(
            "10-15 most important clauses or provisions, each as a brief summary "
            "(one sentence each)"
        )
    )
```

---

## Step 3 — Risk Analysis Signature

```python
class AnalyzeRisk(Signature):
    """Identify legal and business risks in the contract."""

    text: str = InputField(desc="Full contract text")
    key_clauses: List[str] = InputField(desc="Extracted key clauses to focus analysis on")

    risks: List[dict] = OutputField(
        desc=(
            "Identified risks. Each entry: "
            "{'clause': str, 'risk': str, 'severity': 'low' | 'medium' | 'high' | 'critical'}"
        )
    )
    risk_level: Literal["low", "medium", "high", "critical"] = OutputField(
        desc="Overall risk level of the contract"
    )
    missing_protections: List[str] = OutputField(
        desc="Standard protections typically expected in this type of contract but absent here"
    )
```

---

## Step 4 — Executive Summary Signature

```python
class SummarizeReview(Signature):
    """Produce an executive summary of the legal review."""

    text: str = InputField(desc="Full contract text")
    risks: List[dict] = InputField(desc="Identified risks from analysis")

    executive_summary: str = OutputField(
        desc="2-3 paragraph summary of the contract and its key issues"
    )
    red_flags: List[str] = OutputField(
        desc="Critical issues that must be addressed before signing"
    )
    recommendations: List[str] = OutputField(
        desc="Actionable changes to request in negotiation"
    )
    negotiation_points: List[str] = OutputField(
        desc="Clauses where there is room to negotiate better terms"
    )
```

---

## Step 5 — Pipeline

```python
class Extractor(nn.Agent):
    model = model
    signature = ExtractContract
    config = {"verbose": True}


class RiskAnalyzer(nn.Agent):
    model = model
    signature = AnalyzeRisk
    config = {"verbose": True}


class Summarizer(nn.Agent):
    model = model
    signature = SummarizeReview


class LegalReviewer(nn.Module):
    def __init__(self):
        super().__init__()
        self.parser    = mf.Parser.pdf("pypdf")
        self.extractor = Extractor()
        self.analyzer  = RiskAnalyzer()
        self.summarizer = Summarizer()

    def forward(self, msg):
        # 1. Parse PDF (skip if text already provided)
        if not msg.get("text"):
            response = self.parser(msg.pdf_path)
            msg.text = response.data["text"]

        # 2. Extract structure
        self.extractor(msg)

        # 3. Analyze risks
        self.analyzer(msg)

        # 4. Summarize
        self.summarizer(msg)

        return msg


reviewer = LegalReviewer()

msg = Message(pdf_path="service_agreement.pdf")
reviewer(msg)

print(f"Risk level: {msg.risk_level.upper()}")
print(f"\nParties: {[p['name'] for p in msg.parties]}")
print(f"Governing law: {msg.governing_law}")

print("\nRed flags:")
for flag in msg.red_flags:
    print(f"  ⚠ {flag}")

print("\nRecommendations:")
for rec in msg.recommendations:
    print(f"  → {rec}")
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from typing import Literal, List, Optional


# ── Signatures ────────────────────────────────────────────────────────────────

class ExtractContract(Signature):
    """Extract the key structural elements from a contract."""

    text: str = InputField(desc="Full text of the contract")
    parties: List[dict] = OutputField(
        desc="All parties: [{'name': str, 'role': str, 'jurisdiction': str | None}, ...]"
    )
    effective_date: Optional[str] = OutputField(
        desc="Contract effective date"
    )
    governing_law: Optional[str] = OutputField(
        desc="Jurisdiction whose law governs the contract"
    )
    termination_clauses: List[str] = OutputField(
        desc="Conditions under which either party may terminate"
    )
    payment_terms: Optional[str] = OutputField(
        desc="Payment schedule, amounts, and late-payment penalties"
    )
    key_clauses: List[str] = OutputField(
        desc="10-15 most important provisions, each as a one-sentence summary"
    )


class AnalyzeRisk(Signature):
    """Identify legal and business risks in the contract."""

    text: str = InputField(desc="Full contract text")
    key_clauses: List[str] = InputField(desc="Key clauses to focus analysis on")
    risks: List[dict] = OutputField(
        desc="Risks: [{'clause': str, 'risk': str, 'severity': 'low'|'medium'|'high'|'critical'}, ...]"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = OutputField(
        desc="Overall contract risk level"
    )
    missing_protections: List[str] = OutputField(
        desc="Standard protections absent from this contract"
    )


class SummarizeReview(Signature):
    """Produce an executive summary of the legal review."""

    text: str = InputField(desc="Full contract text")
    risks: List[dict] = InputField(desc="Identified risks")
    executive_summary: str = OutputField(desc="2-3 paragraph summary")
    red_flags: List[str] = OutputField(
        desc="Critical issues that must be addressed before signing"
    )
    recommendations: List[str] = OutputField(
        desc="Actionable changes to request in negotiation"
    )
    negotiation_points: List[str] = OutputField(
        desc="Clauses where better terms can be negotiated"
    )


# ── Agents ────────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class Extractor(nn.Agent):
    model = model
    signature = ExtractContract
    config = {"verbose": True}


class RiskAnalyzer(nn.Agent):
    model = model
    signature = AnalyzeRisk
    config = {"verbose": True}


class Summarizer(nn.Agent):
    model = model
    signature = SummarizeReview


# ── Pipeline ──────────────────────────────────────────────────────────────────

class LegalReviewer(nn.Module):
    def __init__(self):
        super().__init__()
        self.parser     = mf.Parser.pdf("pypdf")
        self.extractor  = Extractor()
        self.analyzer   = RiskAnalyzer()
        self.summarizer = Summarizer()

    def forward(self, msg):
        if not msg.get("text"):
            response = self.parser(msg.pdf_path)
            msg.text = response.data["text"]

        self.extractor(msg)
        self.analyzer(msg)
        self.summarizer(msg)
        return msg


# ── Run ───────────────────────────────────────────────────────────────────────

reviewer = LegalReviewer()

# ── Option A: from PDF ────────────────────────────────────────────────────────
# msg = Message(pdf_path="service_agreement.pdf")

# ── Option B: from text (for testing) ────────────────────────────────────────
msg = Message()
msg.text = """
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 1, 2026
between Acme Corp ("Client") and DevShop LLC ("Service Provider").

1. SERVICES. Service Provider agrees to develop a custom CRM system as
   specified in Schedule A. Delivery is expected within 90 days.

2. PAYMENT. Client shall pay $50,000 upon signing and $50,000 upon delivery.
   No penalties for late payment are specified.

3. INTELLECTUAL PROPERTY. All work product created under this Agreement
   shall be owned exclusively by Service Provider. Client receives a
   non-exclusive license to use the delivered software.

4. TERMINATION. Either party may terminate this Agreement with 30 days
   written notice. No provisions for termination for cause are included.

5. LIMITATION OF LIABILITY. Service Provider's liability is limited to the
   fees paid in the preceding 30 days.

6. GOVERNING LAW. This Agreement is governed by the laws of Delaware.
"""

reviewer(msg)

print("=" * 60)
print(f"Risk Level: {msg.risk_level.upper()}")
print(f"Parties:    {[p['name'] for p in msg.parties]}")
print(f"Law:        {msg.governing_law}")

print("\nKey Clauses:")
for clause in msg.key_clauses[:5]:
    print(f"  • {clause}")

print(f"\nRisks ({len(msg.risks)} found):")
for risk in sorted(msg.risks, key=lambda r: ["low","medium","high","critical"].index(r["severity"]), reverse=True):
    print(f"  [{risk['severity'].upper()}] {risk['risk']}")

print("\nMissing Protections:")
for mp in msg.missing_protections:
    print(f"  • {mp}")

print("\nRed Flags:")
for flag in msg.red_flags:
    print(f"  ⚠ {flag}")

print("\nRecommendations:")
for rec in msg.recommendations:
    print(f"  → {rec}")

print("\nExecutive Summary:")
print(msg.executive_summary)
```

---

## Async Version

```python
import asyncio


class LegalReviewer(nn.Module):
    def __init__(self):
        super().__init__()
        self.parser     = mf.Parser.pdf("pypdf")
        self.extractor  = Extractor()
        self.analyzer   = RiskAnalyzer()
        self.summarizer = Summarizer()

    async def aforward(self, msg):
        if not msg.get("text"):
            response = self.parser(msg.pdf_path)
            msg.text = response.data["text"]
        await self.extractor.acall(msg)
        await self.analyzer.acall(msg)
        await self.summarizer.acall(msg)
        return msg


async def main():
    reviewer = LegalReviewer()
    msg = Message(pdf_path="nda.pdf")
    await reviewer.acall(msg)
    print(f"Risk: {msg.risk_level}")
    print(msg.executive_summary)

asyncio.run(main())
```

---

## Multi-document Review

Review a folder of contracts in parallel with `ascatter_gather`:

```python
import asyncio
import msgflux.nn.functional as F
from pathlib import Path


async def review_folder(folder: str):
    reviewer = LegalReviewer()
    pdfs = list(Path(folder).glob("*.pdf"))

    messages = [Message(pdf_path=str(p)) for p in pdfs]

    results = await F.ascatter_gather(
        [reviewer.acall] * len(messages),
        args_list=[(msg,) for msg in messages],
    )

    for pdf, msg in zip(pdfs, results):
        print(f"\n{pdf.name}: {msg.risk_level.upper()} — {len(msg.red_flags)} red flags")

asyncio.run(review_folder("./contracts/"))
```
