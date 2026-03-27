# Payments Assistant

**PIX** is Brazil's instant payment system. A transfer usually needs two things:
the **amount** and a **PIX key**.

The hard part is not the extraction itself. The hard part is that users do not
always describe a payment as clean text:

- they type a message;
- they send an audio note;
- they attach an image with a key or QR code;
- they carry useful metadata in the same `Message`.

This tutorial builds a **Payments Assistant** that keeps the root agent simple
while delegating extraction to a specialist tool that receives the **original
`Message` envelope**.

That is the key design change in this version:

- the model sees a small tool schema: `collect_pix_data()`;
- the runtime injects the full `Message` into that tool;
- the specialist tool can run its own submodules on top of the same `Message`.

You end up with a tool that is **zero-input for the model** and
**message-aware at runtime**.

---

## Architecture

```text
User Message
(content, user_audio, user_image, payments.*, extra.*)
                │
                ▼
        BankingAssistant
        (tools: [collect_pix_data])
                │
                │ detects payment intent
                │ calls collect_pix_data()
                ▼
   CollectPIXData                      ← submodule that is also a tool
   @tool_config(
       return_direct=True,
       inject_message=True,
       disable_input=True,
   )
                │
                │ receives the original Message
                ▼
        ┌───────────────────────┐
        │  if audio:            │
        │     Transcriber       │
        │  Extractor Agent      │
        │  writes payments.pix  │
        └──────────┬────────────┘
                   │
                   ▼
      {"amount": 50.0, "key_type": "phone_number", "key_id": "..."}
                   │
                   │ return_direct=True
                   ▼
             BankingHub
                   │
                   │ calls assistant again with:
                   │ - the same Message
                   │ - a new explicit task
                   ▼
    "I'll transfer R$ 50,00 to ... Confirm?"
```

The important point is that the specialist does **not** receive a flattened
string like `user_message: str`. It receives the same `Message` object the root
assistant received, including audio, image, extracted outputs, and any
bank-specific metadata.

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Build a Specialist Tool That Works on `Message`

Instead of wrapping extraction in a plain function like
`collect_pix_data(user_message: str)`, we will create a **tool module**.

This module does three jobs:

1. optionally transcribes audio into `message.content`;
2. extracts PIX fields from text and image;
3. writes the structured result to `message.payments.pix`.

Because it is also a tool, the root agent can call it directly.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import tool_config


chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")


pix_signature = """
text ->
amount: float,
key_type: Literal['cpf', 'cnpj', 'email', 'phone_number', 'random_key'],
key_id: str
"""


@tool_config(
    return_direct=True,
    inject_message=True,
    disable_input=True,
)
class CollectPIXData(nn.Module):
    """Extract PIX payment data from the current Message."""

    name = "collect_pix_data"
    annotations = {"return": dict}

    def __init__(self):
        super().__init__()
        self.transcriber = nn.Transcriber(
            name="transcriber",
            model=stt_model,
            message_fields={"task_multimodal": {"audio": "user_audio"}},
            response_mode="content",
        )
        self.extractor = nn.Agent(
            name="pix_extractor",
            model=chat_model,
            signature=pix_signature,
            message_fields={
                "task": "content",
                "task_multimodal": {"image": "user_image"},
            },
            response_mode="payments.pix",
        )

    def forward(self, message: mf.Message) -> dict:
        if message.get("user_audio") is not None:
            self.transcriber(message)

        self.extractor(message)
        return message.payments.pix
```

### Why this shape matters

- `disable_input=True` removes public tool parameters from the schema.
- `inject_message=True` still passes the original `Message` at runtime.
- `return_direct=True` returns the structured extraction result without another
  LLM pass in between.

So the model sees this:

```text
collect_pix_data()
```

But the implementation still receives:

```python
def forward(self, message: mf.Message) -> dict:
    ...
```

That is exactly the sweet spot for multimodal banking flows.

---

## Step 2 — Root Agent With a Small Tool Surface

Now create the general assistant.

It should:

- answer normal banking questions;
- call `collect_pix_data()` when the user wants to transfer via PIX;
- later reuse the extracted data from the same `Message`.

To make the second step clean, we map `payments.pix` as `task_context`.

```python
class BankingAssistant(nn.Agent):
    """General banking assistant."""

    model = chat_model
    message_fields = {
        "task": "content",
        "task_context": "payments.pix",
    }
    templates = {
        "context": (
            "PIX data already extracted:\n"
            "- amount: {{amount}}\n"
            "- key_type: {{key_type}}\n"
            "- key_id: {{key_id}}"
        )
    }
    system_message = """
    You are a helpful banking assistant.

    Answer general banking and PIX questions naturally.

    When the user wants to make a PIX payment or transfer,
    call collect_pix_data().

    The tool already receives the original Message envelope.
    Do not try to serialize the user message into tool arguments.
    """
    tools = [CollectPIXData]
    config = {"verbose": True}
```

Notice the model instruction: the agent should call `collect_pix_data()`, not
`collect_pix_data(user_message=...)`.

That is one of the main benefits of the new design: the tool schema stays
minimal, while the runtime still has the complete `Message`.

---

## Step 3 — BankingHub Orchestrates the Two-Step Flow

The assistant still behaves like a normal agent. The difference is that a
`return_direct=True` tool produces a structured `tool_responses` object.

`BankingHub` intercepts that, stores the extracted PIX data on the `Message`,
and asks the assistant to generate a confirmation.

The second call is where the current API becomes especially useful:

- `message` is still the same `Message` envelope;
- `task=...` overrides the original `message_fields["task"]`;
- `payments.pix` is still available through `message_fields["task_context"]`.

```python
class BankingHub(nn.Module):
    def __init__(self):
        super().__init__()
        self.assistant = BankingAssistant()

    def forward(self, message: mf.Message) -> mf.Message:
        response = self.assistant(message)

        if isinstance(response, dict) and "tool_responses" in response:
            tool_call = response.tool_responses.tool_calls[0]
            pix_data = tool_call["result"]

            message.set("payments.pix", pix_data)
            message.response = self.assistant(
                message,
                task=(
                    "Confirm the PIX transfer to the user. "
                    "Use the extracted PIX data from context, "
                    "format the amount in BRL, and ask for explicit confirmation. "
                    "Do not call collect_pix_data again."
                ),
            )
        else:
            message.response = response

        return message
```

This is a good pattern whenever you want:

- deterministic structured extraction;
- natural language confirmation;
- reuse of the same `Message` across multiple agent passes.

---

## Running

```python
hub = BankingHub()


# 1. Normal banking conversation
msg = mf.Message(content="What's the PIX transfer limit at night?")
hub(msg)
print(msg.response)


# 2. Payment via text
msg = mf.Message(content="Transfer R$ 22,40 to CPF 123.456.789-00")
hub(msg)
print(msg.payments.pix)
# {'amount': 22.4, 'key_type': 'cpf', 'key_id': '123.456.789-00'}
print(msg.response)
# "I'll transfer R$ 22,40 to CPF 123.456.789-00 via PIX. Confirm?"


# 3. Payment via audio
msg = mf.Message(content="Please process the transfer from this audio.")
msg.user_audio = "audio_pix.ogg"
hub(msg)
print(msg.payments.pix)
print(msg.response)


# 4. Payment via image or QR code
msg = mf.Message(content="Pay this PIX QR code")
msg.user_image = "pix_qr.png"
hub(msg)
print(msg.payments.pix)
print(msg.response)
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import tool_config


# ── Models ────────────────────────────────────────────────────────────────────

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")


# ── Specialist Tool ───────────────────────────────────────────────────────────

pix_signature = """
text ->
amount: float,
key_type: Literal['cpf', 'cnpj', 'email', 'phone_number', 'random_key'],
key_id: str
"""


@tool_config(
    return_direct=True,
    inject_message=True,
    disable_input=True,
)
class CollectPIXData(nn.Module):
    """Extract PIX payment data from the current Message."""

    name = "collect_pix_data"
    annotations = {"return": dict}

    def __init__(self):
        super().__init__()
        self.transcriber = nn.Transcriber(
            name="transcriber",
            model=stt_model,
            message_fields={"task_multimodal": {"audio": "user_audio"}},
            response_mode="content",
        )
        self.extractor = nn.Agent(
            name="pix_extractor",
            model=chat_model,
            signature=pix_signature,
            message_fields={
                "task": "content",
                "task_multimodal": {"image": "user_image"},
            },
            response_mode="payments.pix",
        )

    def forward(self, message: mf.Message) -> dict:
        if message.get("user_audio") is not None:
            self.transcriber(message)

        self.extractor(message)
        return message.payments.pix


# ── Root Assistant ────────────────────────────────────────────────────────────

class BankingAssistant(nn.Agent):
    """General banking assistant."""

    model = chat_model
    message_fields = {
        "task": "content",
        "task_context": "payments.pix",
    }
    templates = {
        "context": (
            "PIX data already extracted:\n"
            "- amount: {{amount}}\n"
            "- key_type: {{key_type}}\n"
            "- key_id: {{key_id}}"
        )
    }
    system_message = """
    You are a helpful banking assistant.

    Answer general banking and PIX questions naturally.

    When the user wants to make a PIX payment or transfer,
    call collect_pix_data().

    The tool already receives the original Message envelope.
    Do not try to serialize the user message into tool arguments.
    """
    tools = [CollectPIXData]
    config = {"verbose": True}


# ── Orchestrator ──────────────────────────────────────────────────────────────

class BankingHub(nn.Module):
    def __init__(self):
        super().__init__()
        self.assistant = BankingAssistant()

    def forward(self, message: mf.Message) -> mf.Message:
        response = self.assistant(message)

        if isinstance(response, dict) and "tool_responses" in response:
            tool_call = response.tool_responses.tool_calls[0]
            pix_data = tool_call["result"]

            message.set("payments.pix", pix_data)
            message.response = self.assistant(
                message,
                task=(
                    "Confirm the PIX transfer to the user. "
                    "Use the extracted PIX data from context, "
                    "format the amount in BRL, and ask for explicit confirmation. "
                    "Do not call collect_pix_data again."
                ),
            )
        else:
            message.response = response

        return message


# ── Run ───────────────────────────────────────────────────────────────────────

hub = BankingHub()

msg = mf.Message(content="Transfer R$ 22,40 to CPF 123.456.789-00")
hub(msg)
print(msg.payments.pix)
print(msg.response)

msg = mf.Message(content="Please process the transfer from this audio.")
msg.user_audio = "audio_pix.ogg"
hub(msg)
print(msg.payments.pix)
print(msg.response)
```

---

## Why This Version Is Better

The old shape of this tutorial usually looked like this:

```python
def collect_pix_data(user_message: str) -> dict:
    msg = mf.Message(content=user_message)
    ...
```

That works for simple text, but it quietly throws away most of what makes
`Message` useful:

- audio attachments;
- images and QR codes;
- extracted intermediate outputs;
- extra state and metadata.

The new design keeps those pieces intact.

You get three important properties at once:

1. **Small tool schema for the model**
   `collect_pix_data()` is easier for the model to call than a large envelope.
2. **Rich runtime context for the specialist**
   the tool still receives the full `Message`.
3. **Clean multi-step orchestration**
   the same `Message` flows from extraction to confirmation.

This pattern is especially strong for:

- multimodal banking assistants;
- specialist subagents used as tools;
- workflows where structured data must stay exact before confirmation.

---

## Next Steps

- **[Tutorials](tutorials.md)** — more complete examples
- **[Product Poster Generator](product-poster.md)** — multimodal pipeline with image generation
- **[Intent Router](intent-router.md)** — multi-agent routing with Signatures
