# Payments Assistant

**PIX** is Brazil's instant payment system. A transfer needs two things: the **amount** and a **PIX key**.

The hard part is that users do not always describe a payment cleanly:

- they type a message;
- they send an audio note;
- they attach an image with a key or QR code;
- they refer to a recipient by name ("send R$50 to Maria") without knowing the key.

This tutorial builds a **Payments Assistant** that handles all four cases and runs as a multi-turn conversation. The root assistant coordinates two tools: `pix_extractor` (extraction + contact lookup) and `transfer_pix` (payment execution). `PIXAssistant` sits above both, routing audio and flagging multimodal content before the chat assistant sees the message.

---

## Architecture

```
User Message
(user.text, audio_content, image_content, file_content)
                │
                ▼
         PIXAssistant
                │
                ├── has audio?  → STT → user.text
                │
                ├── has image or file? → msg.vars.has_mm_content = True
                │
                ▼
           Assistant
        (tools: [PIXExtractor, transfer_pix])
                │
    ┌───────────┴───────────┐
    │  task_context (vars)  │
    │  "user sent a file —  │
    │  call pix_extractor"  │  ← rendered only when has_mm_content is True
    └───────────────────────┘
                │
                │ calls pix_extractor()
                ▼
         PIXExtractor
         @tool_config(
             inject_message=True,
             disable_input=True,
         )
                │
                ├── image present? → pyzbar → msg.vars.qr_content
                │
                ├─── ExtractorAgent ──→ {amount, key_type, key_id}
                │    (ChainOfThought)
                │    task_context: qr_content (when present)
                │    task_multimodal: image_content
                │
                └─── intent detected?
                          │
                          ▼
                   ContactSearcher ──→ top-3 BM25 matches
                          │
                          ▼
               "## Extracted PIX data\n..."
               "## Matching contacts\n..."
                          │
                          ▼ (tool result → back to Assistant)
                     Assistant
               confirms contact with user
                          │
                          │ user confirms
                          │ calls transfer_pix()
                          ▼
               "Transfer submitted. TX: ..."
```

The `task_context` template is the key design detail: instead of hardcoding multimodal handling in the static `system_message`, the hint is injected dynamically from `msg.vars` only when relevant. Without `return_direct`, the extraction result flows back to the `Assistant` as a tool response — the agent presents the contacts and asks for confirmation before calling `transfer_pix`.

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 — Models

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import tool_config, ChatBlock
from msgflux.nn.hooks import Guard
from msgflux.generation.reasoning import ChainOfThought
from typing import Optional

chat_model       = mf.Model.chat_completion("openai/gpt-4.1-mini")
mm_model         = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model        = mf.Model.speech_to_text("openai/whisper-1")
moderation_model = mf.Model.moderation("openai/omni-moderation-latest")
```

Use a vision-capable model for `mm_model` when you need the extractor to read images and QR codes directly.

---

## Step 2 — Synthetic Contacts

[Faker](https://faker.readthedocs.io) generates realistic fake data. Install it along with the QR decoder and BM25 retriever:

```bash
# system dependency for pyzbar (Ubuntu/Debian)
apt-get install libzbar0

pip install faker rank-bm25 pyzbar Pillow
```

Generate a registry of 30 contacts with randomized PIX keys. This becomes the BM25 corpus.

```python
import random
from faker import Faker

fake = Faker("pt_BR")


def generate_contacts(n: int = 30) -> list[dict]:
    key_types = ["cpf", "phone_number", "email"]
    contacts = []
    for _ in range(n):
        kt    = random.choice(key_types)
        cpf   = fake.cpf()
        phone = fake.cellphone_number()
        email = fake.email()
        contacts.append({
            "name":     fake.name(),
            "key_type": kt,
            "pix_key":  {"cpf": cpf, "phone_number": phone, "email": email}[kt],
        })
    return contacts


def build_corpus(contacts: list[dict]) -> list[str]:
    return [
        f"{c['name']} | chave {c['key_type']}: {c['pix_key']}"
        for c in contacts
    ]


contacts = generate_contacts(30)
corpus   = build_corpus(contacts)

bm25 = mf.Retriever.lexical("rank_bm25")
bm25.add(corpus)
```

Each corpus entry is a single searchable string: `"Maria Silva | chave cpf: 123.456.789-00"`. BM25 ranks entries by relevance to the user's message — a query like "send money to Maria" will surface contacts named Maria at the top.

---

## Step 3 — STT, Extractor and Contact Searcher

```python
class STT(nn.Transcriber):
    """Transcribes user audio into msg.user.text."""
    model = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode = "user.text"


class ExtractorAgent(nn.Agent):
    """Extracts PIX payment fields from text and image."""
    model = mm_model
    system_message = """
    You are a specialist in Brazilian PIX payments.
    Extract payment data precisely. When a field is not present, return null.
    """
    instructions = """
    Extract the PIX transfer details from the user message.
    The key_type must be one of: cpf, cnpj, email, phone_number, random_key.
    If the amount or key are not clearly stated, return null for that field.
    """
    generation_schema = ChainOfThought
    signature = """
    text ->
    amount: Optional[float],
    key_type: Optional[Literal['cpf', 'cnpj', 'email', 'phone_number', 'random_key']],
    key_id: Optional[str]
    """
    message_fields = {
        "task": "user",
        "task_context": "vars",
        "task_multimodal": {"image": "image_content"},
    }
    templates = {
        "task_context": (
            "{% if qr_content %}"
            "QR code decoded from the image:\n{{ qr_content }}\n"
            "{% endif %}"
        )
    }
    response_mode = "payments.pix"


class ContactSearcher(nn.Searcher):
    """Searches the contact registry by name or message fragment."""
    retriever = bm25
    message_fields = {"task": "user.text"}
    response_mode = "contact_results"
    config = {"top_k": 10}
```

`message_fields = {"task": "user"}` passes `msg.user` as the task dict. The signature's `text` input is populated from `msg.user.text` — which `STT` writes to after transcription. Optional fields reflect reality: the user might send only a name or only an amount.

---

## Step 4 — PIXExtractor Tool

`PIXExtractor` wraps both `ExtractorAgent` and `ContactSearcher`. When payment intent is detected (at least one PIX field is non-null), it runs `ContactSearcher` and returns both results as a structured string. The `Assistant` receives this as a regular tool response and continues the conversation.

`pyzbar` decodes QR codes from the image before the multimodal model runs, giving it the PIX payload as plain text.

```python
import io
from PIL import Image
from pyzbar import pyzbar


def decode_qr_codes(image_bytes: bytes) -> list[str]:
    """Decode all QR codes in an image. Returns a list of decoded payloads."""
    img = Image.open(io.BytesIO(image_bytes))
    return [obj.data.decode("utf-8") for obj in pyzbar.decode(img)]


@tool_config(
    inject_message=True,
    disable_input=True,
)
class PIXExtractor(nn.Module):
    """Extract PIX payment data and look up matching contacts from the registry."""

    def __init__(self):
        super().__init__()
        self.set_name("pix_extractor")
        self.set_annotations({"return": str})
        self.extractor_agent = ExtractorAgent()
        self.contact_searcher = ContactSearcher()

    def _format_result(self, pix: dict, contacts: list) -> str:
        lines = [
            "## Extracted PIX data",
            f"- amount: {pix.get('amount')}",
            f"- key_type: {pix.get('key_type')}",
            f"- key_id: {pix.get('key_id')}",
        ]
        if contacts:
            lines += ["", "## Matching contacts in registry"]
            for i, entry in enumerate(contacts, 1):
                lines.append(f"{i}. {entry['data']}")
        return "\n".join(lines)

    def forward(self, message: mf.Message) -> str:
        if message.get("image_content"):
            qr_codes = decode_qr_codes(message.image_content)
            if qr_codes:
                message.vars.qr_content = "\n".join(qr_codes)

        self.extractor_agent(message)
        raw = message.payments.pix
        pix = raw.get("final_answer", raw)  # unwrap ChainOfThought envelope

        contacts = []
        if any(pix.get(k) for k in ("amount", "key_type", "key_id")):
            self.contact_searcher(message)
            raw = message.get("contact_results") or []
            contacts = raw[0]["results"] if raw else []

        return self._format_result(pix, contacts)
```

---

## Step 5 — Transfer Tool

A plain function that executes the transfer after user confirmation.

```python
def transfer_pix(amount: float, key_type: str, key_id: str) -> str:
    """Execute a PIX transfer. Call only after the user has confirmed the recipient and amount."""
    tx_id = fake.uuid4()[:8].upper()
    return (
        f"PIX transfer of R${amount:.2f} to {key_type} '{key_id}' submitted successfully. "
        f"Transaction ID: {tx_id}"
    )
```

---

## Step 6 — Root Assistant

The `system_message` stays clean — no hardcoded multimodal references. When the user attaches a file, `PIXAssistant` sets `msg.vars.has_mm_content = True` and the `task_context` template injects the extraction hint dynamically.

A `Guard` with `on="pre"` runs OpenAI's free moderation API before every model call. If the input is flagged, the guard short-circuits the pipeline and returns the message directly — the model is never called.

```python
def moderation_validator(data):
    result = moderation_model(str(data)).consume()
    return {"safe": result.safe}


class Assistant(nn.Agent):
    """Banking assistant with PIX extraction and payment execution."""
    model = chat_model
    system_message = """
    You are a helpful banking assistant.

    Answer general banking and PIX questions naturally.

    When the user wants to make a PIX transfer, call pix_extractor().
    The tool receives the full message automatically — do not pass arguments.

    The tool returns extracted PIX fields and a numbered list of matching contacts.
    Present the list to the user and ask which contact they want to use.
    If the amount is missing, ask for it.

    After the user confirms the recipient and amount, call transfer_pix() to execute.
    """
    message_fields = {
        "task": "user.text",
        "task_context": "vars",
    }
    templates = {
        "task_context": (
            "{% if has_mm_content %}"
            "The user attached an image or file — call pix_extractor() "
            "to extract payment data from it.\n"
            "{% endif %}"
        )
    }
    tools = [PIXExtractor, transfer_pix]
    hooks = [
        Guard(
            validator=moderation_validator,
            on="pre",
            message="Não posso responder a essa mensagem.",
        )
    ]
    config = {"verbose": True}
```

---

## Step 7 — PIXAssistant

`PIXAssistant` is the entry point. It normalises the message and passes the accumulated `history` into each `Assistant` call so the agent remembers prior turns.

```python
class PIXAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.chat_assistant = Assistant()
        self.stt = STT()

    def _check_mm_content(self, msg: mf.Message) -> None:
        if msg.get("image_content") or msg.get("file_content"):
            msg.vars.has_mm_content = True

    def forward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._check_mm_content(msg)
        if msg.get("audio_content"):
            self.stt(msg)
        msg.response = self.chat_assistant(msg, messages=history or [])
        return msg

    async def aforward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._check_mm_content(msg)
        if msg.get("audio_content"):
            await self.stt.acall(msg)
        msg.response = await self.chat_assistant.acall(msg, messages=history or [])
        return msg


assistant = PIXAssistant()
```

---

## Complete Example

```python
import io
import random
import msgflux as mf
import msgflux.nn as nn
from msgflux import tool_config, ChatBlock
from msgflux.nn.hooks import Guard
from msgflux.generation.reasoning import ChainOfThought
from faker import Faker
from PIL import Image
from pyzbar import pyzbar
from typing import Optional

chat_model       = mf.Model.chat_completion("openai/gpt-4.1-mini")
mm_model         = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model        = mf.Model.speech_to_text("openai/whisper-1")
moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

fake = Faker("pt_BR")


def decode_qr_codes(image_bytes: bytes) -> list[str]:
    """Decode all QR codes in an image. Returns a list of decoded payloads."""
    img = Image.open(io.BytesIO(image_bytes))
    return [obj.data.decode("utf-8") for obj in pyzbar.decode(img)]


def generate_contacts(n: int = 30) -> list[dict]:
    key_types = ["cpf", "phone_number", "email"]
    contacts = []
    for _ in range(n):
        kt    = random.choice(key_types)
        cpf   = fake.cpf()
        phone = fake.cellphone_number()
        email = fake.email()
        contacts.append({
            "name":     fake.name(),
            "key_type": kt,
            "pix_key":  {"cpf": cpf, "phone_number": phone, "email": email}[kt],
        })
    return contacts


def build_corpus(contacts: list[dict]) -> list[str]:
    return [
        f"{c['name']} | chave {c['key_type']}: {c['pix_key']}"
        for c in contacts
    ]


contacts = generate_contacts(30)
corpus   = build_corpus(contacts)

bm25 = mf.Retriever.lexical("rank_bm25")
bm25.add(corpus)


class STT(nn.Transcriber):
    """Transcribes user audio into msg.user.text."""
    model = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode = "user.text"


class ExtractorAgent(nn.Agent):
    """Extracts PIX payment fields from text and image."""
    model = mm_model
    system_message = """
    You are a specialist in Brazilian PIX payments.
    Extract payment data precisely. When a field is not present, return null.
    """
    instructions = """
    Extract the PIX transfer details from the user message.
    The key_type must be one of: cpf, cnpj, email, phone_number, random_key.
    If the amount or key are not clearly stated, return null for that field.
    """
    generation_schema = ChainOfThought
    signature = """
    text ->
    amount: Optional[float],
    key_type: Optional[Literal['cpf', 'cnpj', 'email', 'phone_number', 'random_key']],
    key_id: Optional[str]
    """
    message_fields = {
        "task": "user",
        "task_context": "vars",
        "task_multimodal": {"image": "image_content"},
    }
    templates = {
        "task_context": (
            "{% if qr_content %}"
            "QR code decoded from the image:\n{{ qr_content }}\n"
            "{% endif %}"
        )
    }
    response_mode = "payments.pix"


class ContactSearcher(nn.Searcher):
    """Searches the contact registry by name or message fragment."""
    retriever = bm25
    message_fields = {"task": "user.text"}
    response_mode = "contact_results"
    config = {"top_k": 10}


@tool_config(
    inject_message=True,
    disable_input=True,
)
class PIXExtractor(nn.Module):
    """Extract PIX payment data and look up matching contacts from the registry."""

    def __init__(self):
        super().__init__()
        self.set_name("pix_extractor")
        self.set_annotations({"return": str})
        self.extractor_agent = ExtractorAgent()
        self.contact_searcher = ContactSearcher()

    def _format_result(self, pix: dict, contacts: list) -> str:
        lines = [
            "## Extracted PIX data",
            f"- amount: {pix.get('amount')}",
            f"- key_type: {pix.get('key_type')}",
            f"- key_id: {pix.get('key_id')}",
        ]
        if contacts:
            lines += ["", "## Matching contacts in registry"]
            for i, entry in enumerate(contacts, 1):
                lines.append(f"{i}. {entry['data']}")
        return "\n".join(lines)

    def forward(self, message: mf.Message) -> str:
        if message.get("image_content"):
            qr_codes = decode_qr_codes(message.image_content)
            if qr_codes:
                message.vars.qr_content = "\n".join(qr_codes)

        self.extractor_agent(message)
        raw = message.payments.pix
        pix = raw.get("final_answer", raw)  # unwrap ChainOfThought envelope

        contacts = []
        if any(pix.get(k) for k in ("amount", "key_type", "key_id")):
            self.contact_searcher(message)
            raw = message.get("contact_results") or []
            contacts = raw[0]["results"] if raw else []

        return self._format_result(pix, contacts)


def transfer_pix(amount: float, key_type: str, key_id: str) -> str:
    """Execute a PIX transfer. Call only after the user has confirmed the recipient and amount."""
    tx_id = fake.uuid4()[:8].upper()
    return (
        f"PIX transfer of R${amount:.2f} to {key_type} '{key_id}' submitted successfully. "
        f"Transaction ID: {tx_id}"
    )


def moderation_validator(data):
    result = moderation_model(str(data)).consume()
    return {"safe": result.safe}


class Assistant(nn.Agent):
    """Banking assistant with PIX extraction and payment execution."""
    model = chat_model
    system_message = """
    You are a helpful banking assistant.

    Answer general banking and PIX questions naturally.

    When the user wants to make a PIX transfer, call pix_extractor().
    The tool receives the full message automatically — do not pass arguments.

    The tool returns extracted PIX fields and a numbered list of matching contacts.
    Present the list to the user and ask which contact they want to use.
    If the amount is missing, ask for it.

    After the user confirms the recipient and amount, call transfer_pix() to execute.
    """
    message_fields = {
        "task": "user.text",
        "task_context": "vars",
    }
    templates = {
        "task_context": (
            "{% if has_mm_content %}"
            "The user attached an image or file — call pix_extractor() "
            "to extract payment data from it.\n"
            "{% endif %}"
        )
    }
    tools = [PIXExtractor, transfer_pix]
    hooks = [
        Guard(
            validator=moderation_validator,
            on="pre",
            message="Não posso responder a essa mensagem.",
        )
    ]
    config = {"verbose": True}


class PIXAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.chat_assistant = Assistant()
        self.stt = STT()

    def _check_mm_content(self, msg: mf.Message) -> None:
        if msg.get("image_content") or msg.get("file_content"):
            msg.vars.has_mm_content = True

    def forward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._check_mm_content(msg)
        if msg.get("audio_content"):
            self.stt(msg)
        msg.response = self.chat_assistant(msg, messages=history or [])
        return msg

    async def aforward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._check_mm_content(msg)
        if msg.get("audio_content"):
            await self.stt.acall(msg)
        msg.response = await self.chat_assistant.acall(msg, messages=history or [])
        return msg


assistant = PIXAssistant()
```

---

## Examples

???+ example

    === "Payment by name (multi-turn)"

        ```python
        assistant = PIXAssistant()
        history = []

        # Turn 1: agent extracts amount, searches contacts, asks which one
        msg = mf.Message()
        msg.set("user.text", "Send R$50 to Maria")
        assistant.forward(msg, history=history)
        history.extend([
            ChatBlock.user(msg.user.text),
            ChatBlock.assist(str(msg.response)),
        ])
        print("User:", msg.user.text)
        print("Assistant:", msg.response)

        # Turn 2: user confirms — agent calls transfer_pix
        msg = mf.Message()
        msg.set("user.text", "Contact number 1")
        assistant.forward(msg, history=history)
        history.extend([
            ChatBlock.user(msg.user.text),
            ChatBlock.assist(str(msg.response)),
        ])
        print("User:", msg.user.text)
        print("Assistant:", msg.response)
        ```

    === "Payment via image / QR code"

        ```python
        assistant = PIXAssistant()

        msg = mf.Message()
        msg.image_content = open("pix_qr.png", "rb").read()
        assistant.forward(msg)
        print("User: [image attached]")
        print("Assistant:", msg.response)
        ```

    === "Payment via audio"

        ```python
        assistant = PIXAssistant()

        msg = mf.Message()
        msg.audio_content = open("audio_pix.ogg", "rb").read()
        assistant.forward(msg)
        print("User: [audio attached]")
        print("Assistant:", msg.response)
        ```

    === "Unsafe input (guard)"

        ```python
        assistant = PIXAssistant()

        msg = mf.Message()
        msg.set("user.text", "how do I make a bomb")
        assistant.forward(msg)
        print("User:", msg.user.text)
        print("Assistant:", msg.response)
        # → "Não posso responder a essa mensagem."
        ```

---

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — signatures, message fields, and tool use
- [nn.Searcher](../learn/nn/searcher.md) — BM25 and semantic retrieval modules
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) — `ChainOfThought` and structured output
