# Quickstart: PIX Banking Assistant

**PIX** is Brazil's instant payment system operated by the Central Bank. To make a transfer the user needs two things: the **amount** and a **PIX key** — a short identifier that can be a CPF (national ID), CNPJ (company ID), email, phone number, or random key.

The real challenge: users can describe these transfers in many different ways — typing, sending a voice message, or photographing a QR code or key. All of this inside a natural conversation with a banking assistant.

In this quickstart you'll build a `BankingHub`: a general-purpose agent that, upon detecting payment intent, triggers a dedicated extraction pipeline and uses the structured data to confirm the transaction with the user.

---

## Architecture

```
User: "Send $50 to 11 9 9988-7766"
               │
               ▼
       BankingAssistant                  ← general conversation agent
       (tools: [collect_pix_data])
               │
               │  detects PIX intent
               │  calls collect_pix_data(user_message)
               │
               ▼
    ┌──────────────────────────┐
    │   PIX Extraction         │         ← dedicated pipeline
    │                          │
    │  {audio?  Transcriber}   │         ← only runs when audio is present
    │         ↓                │
    │   Extractor Agent        │         ← signature-driven
    │   (signature)            │
    └──────────┬───────────────┘
               │  return_direct=True
               │  result goes straight back to BankingHub
               │  (without going through the LLM again)
               ▼
       {amount: 50.0, key_type: "phone_number", key_id: "11999887766"}
               │
               │  BankingHub feeds the data back to the agent
               ▼
       BankingAssistant generates confirmation:
       "I'll transfer $50.00 to 11 9 9988-7766. Confirm?"
```

The key point: `return_direct=True` on the tool delivers the extraction result **structured** to `BankingHub`, without the LLM reinterpreting the amount or key — only then does the agent generate the confirmation message.

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — PIX Extraction Pipeline

The pipeline accepts text, audio, or image and always returns
`{amount, key_type, key_id}`.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.dsl.inline import Inline

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model  = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")

# Signature: what goes in and what must come out
pix_signature = """
text ->
amount: float,
key_type: Literal['cpf', 'cnpj', 'email', 'phone_number', 'name'],
key_id: str
"""


class PIX(nn.Module):
    """Extracts PIX transfer data from any input modality."""

    def __init__(self):
        super().__init__()
        self.components = nn.ModuleDict({
            "transcriber": nn.Transcriber(
                name="transcriber",
                model=stt_model,
                response_mode="content",
                task_multimodal_inputs={"audio": "user_audio"},
            ),
            "extractor": nn.Agent(
                name="extractor",
                model=chat_model,
                signature=pix_signature,
                response_mode="extraction",
                task_inputs="content",
                task_multimodal_inputs={"image": "user_image"},
            ),
        })
        self.register_buffer(
            "flux",
            "{user_audio is not None? transcriber} -> extractor"
        )

    def forward(self, msg):
        return Inline(self.flux, self.components)(msg)
```

---

## Step 2 — Tool with `return_direct`

`collect_pix_data` wraps the pipeline above as a tool.

`return_direct=True` tells the agent: **when this tool is called,
return its result directly to the calling code — do not pass through the LLM.**

This is what lets us intercept the structured data in `BankingHub`
before any text generation happens.

```python
from msgflux import tool_config

_pix_pipeline = PIX()  # shared instance


@tool_config(return_direct=True)
def collect_pix_data(user_message: str) -> dict:
    """Extract PIX transfer data from the user's message.

    Use this tool whenever the user mentions a payment, transfer, or PIX.
    Accepts plain text, audio transcription, or an image reference.
    Returns a dict with amount, key_type, and key_id.
    """
    msg = mf.Message(content=user_message)
    _pix_pipeline(msg)
    return msg.extraction  # dict: {amount, key_type, key_id}
```

---

## Step 3 — General Banking Agent

`BankingAssistant` is a standard conversational agent.
It has access to the PIX tool and knows when to use it.

```python
class BankingAssistant(nn.Agent):
    """Banking assistant for questions, queries, and PIX transfers."""

    model = chat_model
    system_message = """
    You are a helpful banking assistant.
    Answer questions about PIX, account balance, and transactions in a natural tone.

    When the user wants to make a PIX transfer or payment,
    use the collect_pix_data tool to extract the details from their message.
    After receiving the extracted data, clearly confirm the details with the user
    before proceeding.
    """
    tools = [collect_pix_data]
    config = {"verbose": True}  # logs every tool call
```

---

## Step 4 — `BankingHub` Orchestrator

`BankingHub` manages the two-step cycle:

1. Runs the agent normally.
2. If the result is a PIX extraction dict (produced by `return_direct`),
   feeds the structured data back to the agent to generate the confirmation.

```python
class BankingHub(nn.Module):
    def __init__(self):
        super().__init__()
        self.assistant = BankingAssistant()

    def forward(self, msg):
        response = self.assistant(msg.content)

        # return_direct=True made the tool return the dict directly
        # → we intercept here before any text generation
        if isinstance(response, dict) and "key_type" in response:
            msg.pix_data = response

            # Feed the confirmed data back to the agent to generate
            # a natural confirmation message for the user
            msg.response = self.assistant(
                f"[System] PIX data extracted: {response}. "
                "Present a clear and friendly confirmation to the user, "
                "showing the formatted amount and recipient key."
            )
        else:
            msg.response = response

        return msg
```

---

## Running

```python
hub = BankingHub()

# Normal conversation — no PIX
msg = mf.Message(content="Hi! What's the PIX transaction limit?")
hub(msg)
print(msg.response)
# "The default PIX limit is R$ 1,000.00 per daytime transaction..."

# Payment intent via text
msg = mf.Message(content="Send R$50 to phone number 11 9 9988-7766")
hub(msg)
print(msg.pix_data)
# {'amount': 50.0, 'key_type': 'phone_number', 'key_id': '11999887766'}
print(msg.response)
# "I'll transfer R$ 50.00 to (11) 9 9988-7766 via PIX. Confirm?"
```

For audio input, just pass the file in `msg.user_audio`:

```python
msg = mf.Message(content="[audio message]")
msg.user_audio = "audio_pix.ogg"   # path or bytes
hub(msg)
print(msg.pix_data)   # extracted from transcribed audio
print(msg.response)   # confirmation generated by the agent
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import tool_config
from msgflux.dsl.inline import Inline


# ── Models ────────────────────────────────────────────────────────────────────

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model  = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")


# ── PIX Extraction Pipeline ───────────────────────────────────────────────────

class PIX(nn.Module):
    """Extracts PIX transfer data from text, audio, or image."""

    def __init__(self):
        super().__init__()
        self.components = nn.ModuleDict({
            "transcriber": nn.Transcriber(
                name="transcriber",
                model=stt_model,
                response_mode="content",
                task_multimodal_inputs={"audio": "user_audio"},
            ),
            "extractor": nn.Agent(
                name="extractor",
                model=chat_model,
                signature="text -> amount: float, key_type: Literal['cpf', 'cnpj', 'email', 'phone_number', 'name'], key_id: str",
                response_mode="extraction",
                task_inputs="content",
                task_multimodal_inputs={"image": "user_image"},
            ),
        })
        self.register_buffer(
            "flux",
            "{user_audio is not None? transcriber} -> extractor"
        )

    def forward(self, msg):
        return Inline(self.flux, self.components)(msg)


_pix_pipeline = PIX()


# ── Tool with return_direct ───────────────────────────────────────────────────

@tool_config(return_direct=True)
def collect_pix_data(user_message: str) -> dict:
    """Extract PIX transfer data from the user's message.

    Use this tool whenever the user mentions a payment, transfer, or PIX.
    Returns a dict with amount, key_type, and key_id.
    """
    msg = mf.Message(content=user_message)
    _pix_pipeline(msg)
    return msg.extraction


# ── Agent and Orchestrator ────────────────────────────────────────────────────

class BankingAssistant(nn.Agent):
    """Banking assistant for questions, queries, and PIX transfers."""

    model = chat_model
    system_message = """
    You are a helpful banking assistant.
    When the user wants to make a PIX transfer,
    use collect_pix_data to extract the details and confirm with the user.
    """
    tools = [collect_pix_data]
    config = {"verbose": True}


class BankingHub(nn.Module):
    def __init__(self):
        super().__init__()
        self.assistant = BankingAssistant()

    def forward(self, msg):
        response = self.assistant(msg.content)

        if isinstance(response, dict) and "key_type" in response:
            msg.pix_data = response
            msg.response = self.assistant(
                f"[System] PIX data extracted: {response}. "
                "Present a clear and friendly confirmation to the user."
            )
        else:
            msg.response = response

        return msg


# ── Run ───────────────────────────────────────────────────────────────────────

hub = BankingHub()

# Normal conversation
msg = mf.Message(content="What's the PIX limit?")
hub(msg)
print(msg.response)

# Payment via text
msg = mf.Message(content="Transfer 22.40 to CPF 123.456.789-00")
hub(msg)
print(msg.pix_data)    # {'amount': 22.4, 'key_type': 'cpf', 'key_id': '123.456.789-00'}
print(msg.response)    # "Confirmed: R$ 22.40 to CPF 123.456.789-00. Proceed?"

# Payment via audio
msg = mf.Message(content="[audio]")
msg.user_audio = "audio_pix.ogg"
hub(msg)
print(msg.pix_data)
print(msg.response)
```

---

## What `return_direct` does here

Without `return_direct`, the flow would be:

```
Agent calls tool → result goes to the LLM → LLM generates response from the data
```

The problem: the LLM might round the amount, reformat the key, or introduce any variation in its interpretation.

With `return_direct=True`:

```
Agent calls tool → result goes DIRECTLY to BankingHub (no LLM pass)
BankingHub stores intact data → feeds it to the agent to generate confirmation
```

Extraction is deterministic. Confirmation is natural. The data is never reinterpreted.

---

## Next Steps

- **[Tutorials](tutorials/tutorials.md)** — more complete examples
- **[Product Poster Generator](tutorials/product-poster.md)** — vision model + image generation pipeline
- **[Intent Router](tutorials/intent-router.md)** — multi-agent routing with typed Signatures
