import io
import re
import urllib.request
import uuid

import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought
from msgflux.nn.hooks import Guard
from PIL import Image
from pyzbar import pyzbar
from typing import Optional, Literal

mf.load_dotenv()
chat_model       = mf.Model.chat_completion("openai/gpt-4.1-mini")
mm_model         = mf.Model.chat_completion("openai/gpt-4o-mini")
stt_model        = mf.Model.speech_to_text("openai/whisper-1")
moderation_model = mf.Model.moderation("openai/omni-moderation-latest")


def decode_qr_codes(image_bytes: bytes) -> list[str]:
    """Decode all QR codes from image bytes."""
    img = Image.open(io.BytesIO(image_bytes))
    return [obj.data.decode("utf-8") for obj in pyzbar.decode(img)]


_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)


def extract_pix_key(qr_value: str) -> str:
    """Extract the PIX key from a QR code value.

    For dynamic PIX QR codes (URL-based), extracts the embedded UUID and
    labels it as random_key so the model sets key_type and key_id correctly.
    For static QR codes (CPF, email, phone, etc.), returns the raw value
    for the model to interpret.
    """
    m = _UUID_RE.search(qr_value)
    if m:
        return f"random_key: {m.group(0)}"
    return qr_value


def build_corpus(contacts: list[dict]) -> list[str]:
    return [
        f"{c['name']} | key `{c['key_type']}`: {c['pix_key']}"
        for c in contacts
    ]


contacts = [
    {"name": "Ana Souza",        "key_type": "cpf",          "pix_key": "123.456.789-00"},
    {"name": "Bruno Oliveira",   "key_type": "phone_number",  "pix_key": "+55 11 91234-5678"},
    {"name": "Carlos Mendes",    "key_type": "email",         "pix_key": "carlos.mendes@email.com"},
    {"name": "Daniela Rocha",    "key_type": "random_key",    "pix_key": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"},
    {"name": "Eduardo Ferreira", "key_type": "cpf",           "pix_key": "987.654.321-00"},
    {"name": "Fernanda Lima",    "key_type": "email",         "pix_key": "fernanda.lima@email.com"},
    {"name": "Gabriel Costa",    "key_type": "phone_number",  "pix_key": "+55 21 99876-5432"},
    {"name": "Helena Martins",   "key_type": "email",         "pix_key": "helena.martins@mail.com"},
    {"name": "Igor Pereira",     "key_type": "cpf",           "pix_key": "111.222.333-44"},
    {"name": "Julia Almeida",    "key_type": "random_key",    "pix_key": "f9e8d7c6-b5a4-3210-fedc-ba9876543210"},
]
corpus = build_corpus(contacts)

fuzzy = mf.Retriever.fuzzy("rapidfuzz")
fuzzy.add(corpus)


class STT(nn.Transcriber):
    """Transcribes user audio into msg.user.text."""
    model          = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode  = "user"


class ExtractorAgent(nn.Agent):
    """Extracts PIX payment fields from text and image."""
    model          = mm_model
    system_message = "You are a specialist in Brazilian PIX payments."
    instructions   = """
    Extract the PIX transfer details from the user message.
    The key_type must be one of: cpf, cnpj, email, phone_number, random_key.
    recipient_name is the person's name when mentioned (e.g. "send to Fernanda").
    key_id is the actual PIX key value (email, CPF, phone, etc.) — not a name.
    If a field is not clearly stated, return null for that field.
    """
    generation_schema = ChainOfThought
    signature = """
    text ->
    amount: Optional[float],
    key_type: Optional[Literal['cpf', 'cnpj', 'email', 'phone_number', 'random_key']],
    key_id: Optional[str],
    recipient_name: Optional[str]
    """
    message_fields = {
        "task":           "user",
        "task_context":   "vars",
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
    """Checks the contact agenda for recipients matching the extracted key."""
    retriever      = fuzzy
    message_fields = {"query": "user.text"}
    config         = {"top_k": 10, "threshold": 60.0}
    templates = {
        "response": (
            "{% if results %}"
            "## Contacts in your agenda\n"
            "{% for item in results %}{{ loop.index }}. {{ item.data }}\n{% endfor %}"
            "{% else %}"
            "## Contacts in your agenda\n"
            "This recipient is not in your contact list. "
            "You can still proceed — ask the user for the full PIX key directly."
            "{% endif %}"
        )
    }


@mf.tool_config(inject_message=True)
class PIXExtractor(nn.Module):
    """Extract PIX payment data and look up matching contacts from the registry."""

    def __init__(self):
        super().__init__()
        self.set_name("PIXExtractor")
        self.set_annotations({"return": str})
        self.extractor_agent   = ExtractorAgent()
        self.contact_searcher  = ContactSearcher()

    def _format_result(self, pix: dict, contacts_section: str) -> str:
        pix_section = "\n".join([
            "## Extracted PIX data",
            f"- amount: {pix.get('amount')}",
            f"- key_type: {pix.get('key_type')}",
            f"- key_id: {pix.get('key_id')}",
        ])
        if contacts_section:
            return f"{pix_section}\n\n{contacts_section}"
        return (
            f"{pix_section}\n\n"
            "No valid PIX key ID found. "
            "Ask the user to provide the full PIX key directly."
        )

    def _search_query(self, pix: dict) -> str | None:
        """Return the best query for contact lookup.

        Prefer the extracted key_id (exact key); fall back to recipient_name
        so that partial names like 'Fernand' still surface fuzzy matches.
        """
        return pix.get("key_id") or pix.get("recipient_name") or None

    def _resolve_image(self, message: mf.Message) -> None:
        """Download URL → bytes so the vision model receives base64, not a remote URL."""
        img = message.image_content
        if isinstance(img, str):
            req = urllib.request.Request(img, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                message.image_content = resp.read()

    def forward(self, message: mf.Message) -> str:
        if message.get("image_content"):
            self._resolve_image(message)
            qr_codes = decode_qr_codes(message.image_content)
            if qr_codes:
                message.vars.qr_content = "\n".join(extract_pix_key(q) for q in qr_codes)

        self.extractor_agent(message)
        raw = message.payments.pix
        pix = raw.get("final_answer", raw)

        contacts_section = ""
        query = self._search_query(pix)
        if query:
            contacts_section = self.contact_searcher(query)

        return self._format_result(pix, contacts_section)

    async def aforward(self, message: mf.Message) -> str:
        if message.get("image_content"):
            self._resolve_image(message)
            qr_codes = decode_qr_codes(message.image_content)
            if qr_codes:
                message.vars.qr_content = "\n".join(extract_pix_key(q) for q in qr_codes)

        await self.extractor_agent.acall(message)
        raw = message.payments.pix
        pix = raw.get("final_answer", raw)

        contacts_section = ""
        query = self._search_query(pix)
        if query:
            contacts_section = await self.contact_searcher.acall(query)

        return self._format_result(pix, contacts_section)


@mf.tool_config(name_override="TransferPix", inject_vars=True)
def transfer_pix(amount: float, key_type: str, key_id: str, **kwargs) -> str:
    """Execute a PIX transfer. Call only after the user has confirmed the recipient and amount."""
    variables = kwargs.get("vars")
    from_key  = f"{variables['user_pix_key_type']}:{variables['user_pix_key_id']}"
    to_key    = f"{key_type}:{key_id}"
    tx_id     = str(uuid.uuid4())[:8].upper()
    print(f"[TransferPix] from={from_key} | to={to_key} | amount=R${amount:.2f} | tx={tx_id}")
    return (
        f"PIX transfer of R${amount:.2f} to {key_type} '{key_id}' submitted successfully. "
        f"Transaction ID: {tx_id}"
    )


class Assistant(nn.Agent):
    """Banking assistant with PIX extraction and payment execution."""
    model          = chat_model
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
        "task":         "user.text",
        "task_context": "vars",
        "vars":         "vars",
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
    def __init__(self, user_pix_key_type: str = "email", user_pix_key_id: str = "user@bank.com"):
        super().__init__()
        self.chat_assistant    = Assistant()
        self.stt               = STT()
        self._user_pix_key_type = user_pix_key_type
        self._user_pix_key_id   = user_pix_key_id

    def _setup_vars(self, msg: mf.Message) -> None:
        msg.set("vars.user_pix_key_type", self._user_pix_key_type)
        msg.set("vars.user_pix_key_id",   self._user_pix_key_id)
        if msg.get("image_content") or msg.get("file_content"):
            msg.set("vars.has_mm_content", True)
            if not msg.get("user.text"):
                msg.set("user.text", "[image attached]")

    def forward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._setup_vars(msg)
        if msg.get("audio_content"):
            self.stt(msg)
        msg.response = self.chat_assistant(msg, messages=history or [])
        return msg

    async def aforward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._setup_vars(msg)
        if msg.get("audio_content"):
            await self.stt.acall(msg)
        msg.response = await self.chat_assistant.acall(msg, messages=history or [])
        return msg


if __name__ == "__main__":
    assistant = PIXAssistant()

    # pick the first name from the generated corpus so the fuzzy search always hits
    first_contact = contacts[0]
    first_name    = first_contact["name"].split()[0]   # e.g. "Ana"

    print(f"\n--- Fuzzy contact lookup: 'Send R$60 to {first_name}' ---")
    print(f"    (full contact: {first_contact['name']} | {first_contact['key_type']}: {first_contact['pix_key']})")
    history = []

    msg = mf.Message()
    msg.set("vars.user_full_name", "Test User")
    msg.set("user.text", f"Send R$60 to {first_name}")
    assistant.forward(msg, history=history)
    history.extend([
        mf.ChatBlock.user(msg.user.text),
        mf.ChatBlock.assist(str(msg.response)),
    ])
    print("User:", msg.user.text)
    print("Assistant:", msg.response)

    msg = mf.Message()
    msg.set("vars.user_full_name", "Test User")
    msg.set("user.text", "Contact 1")
    assistant.forward(msg, history=history)
    print("User:", msg.user.text)
    print("Assistant:", msg.response)
