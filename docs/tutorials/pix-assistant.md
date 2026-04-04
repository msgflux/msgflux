# Payments Assistant

**PIX** is Brazil's instant payment system. A transfer needs three things: the **amount**, the **key type**, and the **key ID**.

## The Problem

Users do not always describe a payment cleanly:

- they type a message;
- they send an audio note;
- they attach an image with a key, QR code or PDF;
- they refer to a recipient by name ("send R$50 to Maria") without knowing the key.

---

## The Plan

We will build a PIX assistant that handles text, audio, and image inputs in a multi-turn conversation to confirm and execute transfers. We will use the declarative API to transfer data between modules.

If the user sends audio, we transcribe it first so the rest of the pipeline works with plain text.

From there, a chat agent handles the conversation. Whenever the user's intent looks like a transfer, the agent calls a PIX extractor tool rather than trying to extract the data itself. That tool receives the full message automatically — including any image or file. If there is a QR code in the image, it is decoded before the model runs, so the payload arrives as plain text.

The extraction agent pulls out the amount, key type, and key ID from whatever is available. If any PIX field is detected, the tool queries a contact registry with a lexical retriever using the extracted key ID as the query. The tool returns both the extracted fields and the top matches — the chat agent presents them and asks the user to confirm.

Once the user picks a contact and confirms the amount, the agent calls the transfer function. A moderation guard runs before every model call: if the input is flagged, the pipeline short-circuits and the model is never invoked.

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
        (tools: [PIXExtractor, TransferPix])
                │
    ┌───────────┴───────────┐
    │  task_context (vars)  │
    │  "user sent a file —  │
    │  call PIXExtractor"   │  ← rendered only when has_mm_content is True
    └───────────────────────┘
                │
                │ calls PIXExtractor()
                ▼
         PIXExtractor
         @tool_config(inject_message=True)
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
                   ContactSearcher ──→ top-K BM25 matches
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
                          │ calls TransferPix()
                          ▼
               "Transfer submitted. TX: ..."
```

The `task_context` template is the key design detail: instead of hardcoding multimodal handling in the static `system_message`, the hint is injected dynamically from `msg.vars` only when relevant. Without `return_direct`, the extraction result flows back to the `Assistant` as a tool response — the agent presents the contacts and asks for confirmation before calling `TransferPix`.

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
mm_model         = mf.Model.chat_completion("openai/gpt-5.4-mini")
stt_model        = mf.Model.speech_to_text("openai/whisper-1")
moderation_model = mf.Model.moderation("openai/omni-moderation-latest")
```

Use a vision-capable model for `mm_model` when you need the extractor to read images and files directly. QR codes are decoded by `decode_qr_codes` before the model runs — the model receives the payload as plain text.

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
        f"{c['name']} | key `{c['key_type']}`: {c['pix_key']}"
        for c in contacts
    ]


contacts = generate_contacts(30)
corpus   = build_corpus(contacts)

bm25 = mf.Retriever.lexical("rank_bm25")
bm25.add(corpus)
```

Each corpus entry is a single searchable string: `"Maria Silva | key cpf: 123.456.789-00"`. BM25 ranks entries by relevance to the user's message — a query like "send money to Maria" will surface contacts named Maria at the top.

---

## Step 3 — STT, Extractor and Contact Searcher

**STT** consumes `audio_content` from the message and writes the transcription to `user.text` — the same field the chat agent reads as its task input. This means audio and text messages flow through identical downstream logic.

```python
class STT(nn.Transcriber):
    """Transcribes user audio into msg.user.text."""
    model = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode = "user.text"
```

**ExtractorAgent** reads from `msg.user` (which includes `user.text`) and optionally from `image_content`. The `task_context` template injects the decoded QR payload when present — the model receives it as plain text before processing the image. All three output fields are `Optional` because the user might provide only partial information.

```python
class ExtractorAgent(nn.Agent):
    """Extracts PIX payment fields from text and image."""
    model = mm_model
    system_message = "You are a specialist in Brazilian PIX payments."
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
```

**ContactSearcher** uses `user.text` as the query against the BM25 corpus and returns an already-formatted string via Jinja template. The `threshold` ensures only relevant results surface — when no result passes the cutoff, the template returns the fallback message so the agent knows to ask the user for the full key directly.

```python
class ContactSearcher(nn.Searcher):
    """Searches the contact registry by name or message fragment."""
    retriever = bm25
    message_fields = {"task": "user.text"}
    config = {"top_k": 10, "threshold": 0.3}
    templates = {
        "response": (
            "{% if results %}"
            "## Matching contacts in registry\n"
            "{% for item in results %}{{ loop.index }}. {{ item.data }}\n{% endfor %}"
            "{% else %}"
            "## Matching contacts in registry\n"
            "No contacts found matching the provided key. "
            "Ask the user to provide the full PIX key directly."
            "{% endif %}"
        )
    }
```

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


@tool_config(inject_message=True)
class PIXExtractor(nn.Module):
    """Extract PIX payment data and look up matching contacts from the registry."""

    def __init__(self):
        super().__init__()
        self.set_name("PIXExtractor")
        self.set_annotations({"return": str})
        self.extractor_agent = ExtractorAgent()
        self.contact_searcher = ContactSearcher()

    def _format_result(self, pix: dict, contacts_section: str) -> str:
        pix_section = "\n".join([
            "## Extracted PIX data",
            f"- amount: {pix.get('amount')}",
            f"- key_type: {pix.get('key_type')}",
            f"- key_id: {pix.get('key_id')}",
        ])
        if pix.get("key_id"):
            return f"{pix_section}\n\n{contacts_section}"
        return (
            f"{pix_section}\n\n"
            "No valid PIX key ID found. "
            "Ask the user to provide the full PIX key directly."
        )

    def forward(self, message: mf.Message) -> str:
        # decode QR code before the model runs — payload arrives as plain text
        if message.get("image_content"):
            qr_codes = decode_qr_codes(message.image_content)
            if qr_codes:
                message.vars.qr_content = "\n".join(qr_codes)

        self.extractor_agent(message)
        raw = message.payments.pix
        pix = raw.get("final_answer", raw)  # unwrap ChainOfThought envelope

        # only search contacts when a key_id was extracted
        contacts_section = ""
        if pix.get("key_id"):
            contacts_section = self.contact_searcher(pix["key_id"])

        return self._format_result(pix, contacts_section)

    async def aforward(self, message: mf.Message) -> str:
        if message.get("image_content"):
            qr_codes = decode_qr_codes(message.image_content)
            if qr_codes:
                message.vars.qr_content = "\n".join(qr_codes)

        await self.extractor_agent.acall(message)
        raw = message.payments.pix
        pix = raw.get("final_answer", raw)

        contacts_section = ""
        if pix.get("key_id"):
            contacts_section = await self.contact_searcher.acall(pix["key_id"])

        return self._format_result(pix, contacts_section)
```

---

## Step 5 — Transfer Tool

A plain function that executes the transfer after user confirmation. `inject_vars=True` makes the
framework inject `msg.vars` as a kwarg — the model never sees it. We use it to read the sender's
PIX key and print a transfer log to the console.

```python
@tool_config(name_override="TransferPix", inject_vars=True)
def transfer_pix(amount: float, key_type: str, key_id: str, **kwargs) -> str:
    """Execute a PIX transfer. Call only after the user has confirmed the recipient and amount."""
    variables = kwargs.get("vars")
    from_key  = (
        f"{variables.user_pix_key_type}:{variables.user_pix_key_id}"
        if variables else "unknown"
    )
    to_key = f"{key_type}:{key_id}"
    tx_id  = fake.uuid4()[:8].upper()
    print(f"[TransferPix] from={from_key} | to={to_key} | amount=R${amount:.2f} | tx={tx_id}")
    return (
        f"PIX transfer of R${amount:.2f} to {key_type} '{key_id}' submitted successfully. "
        f"Transaction ID: {tx_id}"
    )
```

---

## Step 6 — Root Assistant

The root assistant will be responsible for interacting with the user. Its operating context is solely to help make transfers. When the user uploads multimodal content (image or file), the agent will be notified through a context injected into its task. We use `vars` to bring dynamic information to the Agent.

A `Guard` with `on="pre"` runs OpenAI's moderation API before every model call. If the input is flagged, the guard short-circuits the pipeline and returns the message directly — the chat model is never called.

`system_extra_message` is appended to the system prompt at runtime. It supports Jinja templates rendered against `msg.vars`, so the agent can address the user by name without hardcoding anything in the static prompt.

```python
class Assistant(nn.Agent):
    """Banking assistant with PIX extraction and payment execution."""
    model = chat_model
    system_message = """
    You are a banking assistant.

    Its objective is solely to help the user make bank transfers
    using 'PIX' (Brazilian money transfer platform).

    This means that you should not accept talking about any other
    topics that the user may discuss with you.
    """
    instructions = """
    To carry out a bank transfer, 3 pieces of information are required:
    1. value (float)
    2. key type ('cpf', 'cnpj', 'email', 'phone_number', 'random_key')
    3. key_id (key value)

    To carry out a transaction you must use the 'TransferPix' tool.

    The user can send you a message informing which transfer they want
    to make. For example "Send 10.7 to Anna".

    You will need to know Anna's entire key.

    To assist you in detecting this information, use the 'PIXExtractor'
    tool. It receives the user's message and extracts the values you need,
    searching the database for the key id. This tool also accepts
    multimodal data — images and PDFs. Multimodal content is only
    available within this tool. The system will notify you when the
    user uploads content; you must call the tool to analyze it.
    """
    system_extra_message = "The user's name is: {{ user_full_name }}"
    message_fields = {
        "task": "user.text",
        "task_context": "vars",
    }
    templates = {
        "task_context": (
            "{% if has_mm_content %}"
            "The user attached an image or file — call PIXExtractor() "
            "to extract payment data from it.\n"
            "{% endif %}"
        )
    }
    tools = [PIXExtractor, transfer_pix]
    hooks = [
        Guard(
            validator=moderation_model,
            on="pre",
            message="This message cannot be processed.",
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

        # Turn 2: user confirms — agent calls TransferPix
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

    === "Generating a PIX QR code"

        Install the QR code generator:

        ```bash
        pip install pybrcode
        ```

        Generate a dynamic PIX QR code (with amount) from a recipient key:

        ```python
        from pybrcode.pix import generate_simple_pix

        pix = generate_simple_pix(
            fullname="Maria Silva",
            key="maria.silva@email.com",   # email, CPF, CNPJ, phone, or random key
            city="Sao Paulo",
            value=50.00,
            description="Pagamento pedido 42",
        )

        # save as PNG — pass it to PIXAssistant as image_content
        pix.imageToPath(".", "pix_qr.png")

        print("Payload:", str(pix))
        # → 00020126...6304XXXX  (EMV/BR Code with CRC-16 checksum)
        ```

        The payload follows the **EMV/BR Code** spec defined by Banco Central do Brasil.
        Fields `59` (merchant name), `54` (amount), and `26` (PIX key) are embedded in the
        string; the last four hex digits (`6304XXXX`) are the CRC-16/CCITT checksum.

        Pass the saved image directly to the assistant — `pyzbar` will decode it before
        the model runs and inject the raw payload as plain text:

        ```python
        assistant = PIXAssistant()

        msg = mf.Message()
        msg.image_content = open("pix_qr.png", "rb").read()
        assistant.forward(msg)
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

    === "Off-topic question (refused)"

        ```python
        assistant = PIXAssistant()

        msg = mf.Message()
        msg.set("user.text", "Who won the last FIFA World Cup?")
        assistant.forward(msg)
        print("User:", msg.user.text)
        print("Assistant:", msg.response)
        # → "I can only help with PIX bank transfers.
        #    If you'd like to send or receive money, just let me know!"
        ```

    === "Unsafe input (guard)"

        ```python
        assistant = PIXAssistant()

        msg = mf.Message()
        msg.set("user.text", "how do I make a bomb")
        assistant.forward(msg)
        print("User:", msg.user.text)
        print("Assistant:", msg.response)
        # → "This message cannot be processed."
        ```

---

## Extending

### Exposing ContactSearcher as a direct tool

By default, contact lookup happens inside `PIXExtractor` — the agent never sees it. Adding `ContactSearcher` directly to `tools` gives the agent the ability to query the registry at any point in the conversation, without going through the full extraction pipeline. Useful when the user wants to browse contacts before deciding on a transfer.

`ContactSearcher` already exposes the schema `{"query": str}` so no changes to the class are needed — just add it to the tools list:

```python
tools = [PIXExtractor, ContactSearcher, transfer_pix]
```

### Passing conversation history to the extractor

By default, `PIXExtractor` receives only the current `Message`. In complex multi-turn flows, the extraction agent may benefit from seeing the full conversation — for example, when the user references a previous message ("use the same amount as before").

Add `inject_messages=True` to `PIXExtractor`'s `tool_config`. The tool will then receive both `message` (the data transport object) and `messages` (the root agent's conversation history, without the system prompt):

```python
@tool_config(inject_message=True, inject_messages=True)
class PIXExtractor(nn.Module):
    ...
    def forward(self, message: mf.Message, messages: list) -> str:
        ...
```

From there, two options for feeding the history into `ExtractorAgent`:

**Option A** — pass as a kwarg directly:

```python
self.extractor_agent(message, messages=messages)
```

**Option B** — store it in the message and map it via `message_fields`:

```python
message.history = messages

# in ExtractorAgent:
message_fields = {
    "task": "user",
    "task_context": "vars",
    "task_multimodal": {"image": "image_content"},
    "messages": "history",
}
```

Option A is simpler. Option B is useful when the history needs to be pre-processed or shared with multiple submodules.

---

## Complete Script

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
        f"{c['name']} | key `{c['key_type']}`: {c['pix_key']}"
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
    system_message = "You are a specialist in Brazilian PIX payments."
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
    config = {"top_k": 10, "threshold": 0.3}
    templates = {
        "response": (
            "{% if results %}"
            "## Matching contacts in registry\n"
            "{% for item in results %}{{ loop.index }}. {{ item.data }}\n{% endfor %}"
            "{% else %}"
            "## Matching contacts in registry\n"
            "No contacts found matching the provided key. "
            "Ask the user to provide the full PIX key directly."
            "{% endif %}"
        )
    }


@tool_config(
    inject_message=True,
)
class PIXExtractor(nn.Module):
    """Extract PIX payment data and look up matching contacts from the registry."""

    def __init__(self):
        super().__init__()
        self.set_name("PIXExtractor")
        self.set_annotations({"return": str})
        self.extractor_agent = ExtractorAgent()
        self.contact_searcher = ContactSearcher()

    def _format_result(self, pix: dict, contacts_section: str) -> str:
        pix_section = "\n".join([
            "## Extracted PIX data",
            f"- amount: {pix.get('amount')}",
            f"- key_type: {pix.get('key_type')}",
            f"- key_id: {pix.get('key_id')}",
        ])
        if pix.get("key_id"):
            return f"{pix_section}\n\n{contacts_section}"
        return (
            f"{pix_section}\n\n"
            "No valid PIX key ID found. "
            "Ask the user to provide the full PIX key directly."
        )

    def forward(self, message: mf.Message) -> str:
        # decode QR code before the model runs — payload arrives as plain text
        if message.get("image_content"):
            qr_codes = decode_qr_codes(message.image_content)
            if qr_codes:
                message.vars.qr_content = "\n".join(qr_codes)

        self.extractor_agent(message)
        raw = message.payments.pix
        pix = raw.get("final_answer", raw)  # unwrap ChainOfThought envelope

        # only search contacts when a key_id was extracted
        contacts_section = ""
        if pix.get("key_id"):
            contacts_section = self.contact_searcher(pix["key_id"])

        return self._format_result(pix, contacts_section)

    async def aforward(self, message: mf.Message) -> str:
        if message.get("image_content"):
            qr_codes = decode_qr_codes(message.image_content)
            if qr_codes:
                message.vars.qr_content = "\n".join(qr_codes)

        await self.extractor_agent.acall(message)
        raw = message.payments.pix
        pix = raw.get("final_answer", raw)

        contacts_section = ""
        if pix.get("key_id"):
            contacts_section = await self.contact_searcher.acall(pix["key_id"])

        return self._format_result(pix, contacts_section)


@tool_config(name_override="TransferPix", inject_vars=True)
def transfer_pix(amount: float, key_type: str, key_id: str, **kwargs) -> str:
    """Execute a PIX transfer. Call only after the user has confirmed the recipient and amount."""
    variables = kwargs.get("vars")
    from_key  = (
        f"{variables.user_pix_key_type}:{variables.user_pix_key_id}"
        if variables else "unknown"
    )
    to_key = f"{key_type}:{key_id}"
    tx_id  = fake.uuid4()[:8].upper()
    print(f"[TransferPix] from={from_key} | to={to_key} | amount=R${amount:.2f} | tx={tx_id}")
    return (
        f"PIX transfer of R${amount:.2f} to {key_type} '{key_id}' submitted successfully. "
        f"Transaction ID: {tx_id}"
    )


class Assistant(nn.Agent):
    """Banking assistant with PIX extraction and payment execution."""
    model = chat_model
    system_message = """
    You are a helpful banking assistant.

    Answer general banking and PIX questions naturally.

    When the user wants to make a PIX transfer, call PIXExtractor().
    The tool receives the full message automatically — do not pass arguments.

    The tool returns extracted PIX fields and a numbered list of matching contacts.
    Present the list to the user and ask which contact they want to use.
    If the amount is missing, ask for it.

    After the user confirms the recipient and amount, call TransferPix() to execute.
    """
    system_extra_message = "The user's name is: {{ user_full_name }}"
    message_fields = {
        "task": "user.text",
        "task_context": "vars",
    }
    templates = {
        "task_context": (
            "{% if has_mm_content %}"
            "The user attached an image or file — call PIXExtractor() "
            "to extract payment data from it.\n"
            "{% endif %}"
        )
    }
    tools = [PIXExtractor, transfer_pix]
    hooks = [
        Guard(
            validator=moderation_model,
            on="pre",
            message="This message cannot be processed.",
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

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — signatures, message fields, and tool use
- [nn.Searcher](../learn/nn/searcher.md) — BM25 and semantic retrieval modules
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) — `ChainOfThought` and structured output
