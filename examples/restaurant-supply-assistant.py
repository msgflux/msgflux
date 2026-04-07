import uuid
from typing import List, Optional

from msgspec import Meta, Struct
from typing_extensions import Annotated

import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()
chat_model   = mf.Model.chat_completion("openai/gpt-4.1-mini")
vision_model = mf.Model.chat_completion("openai/gpt-5.4")
stt_model    = mf.Model.speech_to_text("openai/whisper-1")


class OrderItem(Struct):
    name:     Annotated[str,            Meta(description="Product name")]
    quantity: Annotated[Optional[str],  Meta(description="Requested quantity, null if not specified")] = None
    unit:     Annotated[Optional[str],  Meta(description="Unit of measure (kg, lt, un, etc.), null if not specified")] = None


class ExtractedItems(Struct):
    items: Annotated[List[OrderItem], Meta(description="All products the user wants to order")]


CATALOG = [
    {"id": "CAR001", "name": "Beef top sirloin",         "unit": "kg",   "price": 45.90, "category": "Carnes"},
    {"id": "CAR002", "name": "Whole chilled chicken",    "unit": "kg",   "price": 12.50, "category": "Carnes"},
    {"id": "CAR003", "name": "Chicken fillet",           "unit": "kg",   "price": 18.90, "category": "Carnes"},
    {"id": "CAR004", "name": "Pork ribs",                "unit": "kg",   "price": 22.00, "category": "Carnes"},
    {"id": "CAR005", "name": "Tuscan sausage",           "unit": "kg",   "price": 19.50, "category": "Carnes"},
    {"id": "CAR006", "name": "Frozen medium shrimp",     "unit": "kg",   "price": 65.00, "category": "Carnes"},
    {"id": "CAR007", "name": "Tilapia fillet",           "unit": "kg",   "price": 28.00, "category": "Carnes"},
    {"id": "HOR001", "name": "Onion",                    "unit": "kg",   "price":  4.50, "category": "Hortifruti"},
    {"id": "HOR002", "name": "Tomato",                   "unit": "kg",   "price":  6.90, "category": "Hortifruti"},
    {"id": "HOR003", "name": "Iceberg lettuce",          "unit": "un",   "price":  3.50, "category": "Hortifruti"},
    {"id": "HOR004", "name": "Potato",                   "unit": "kg",   "price":  5.80, "category": "Hortifruti"},
    {"id": "HOR005", "name": "Carrot",                   "unit": "kg",   "price":  4.20, "category": "Hortifruti"},
    {"id": "HOR006", "name": "Red bell pepper",          "unit": "kg",   "price":  8.90, "category": "Hortifruti"},
    {"id": "HOR007", "name": "Garlic bulb",              "unit": "kg",   "price": 22.00, "category": "Hortifruti"},
    {"id": "HOR008", "name": "Persian lime",             "unit": "kg",   "price":  5.50, "category": "Hortifruti"},
    {"id": "HOR009", "name": "Parsley",                  "unit": "bunch","price":  2.50, "category": "Hortifruti"},
    {"id": "HOR010", "name": "Cilantro",                 "unit": "bunch","price":  2.50, "category": "Hortifruti"},
    {"id": "LAT001", "name": "UHT whole milk",           "unit": "lt",   "price":  4.90, "category": "Laticínios"},
    {"id": "LAT002", "name": "Salted butter",            "unit": "kg",   "price": 42.00, "category": "Laticínios"},
    {"id": "LAT003", "name": "Mozzarella cheese",        "unit": "kg",   "price": 38.00, "category": "Laticínios"},
    {"id": "LAT004", "name": "Heavy cream",              "unit": "lt",   "price":  9.90, "category": "Laticínios"},
    {"id": "LAT005", "name": "Cream cheese spread",      "unit": "kg",   "price": 28.00, "category": "Laticínios"},
    {"id": "GRA001", "name": "Long grain white rice",    "unit": "kg",   "price":  6.50, "category": "Grãos"},
    {"id": "GRA002", "name": "Pinto beans",              "unit": "kg",   "price":  8.90, "category": "Grãos"},
    {"id": "GRA003", "name": "Spaghetti pasta",          "unit": "kg",   "price":  7.50, "category": "Grãos"},
    {"id": "GRA004", "name": "Wheat flour",              "unit": "kg",   "price":  5.20, "category": "Grãos"},
    {"id": "GRA005", "name": "Fine cornmeal",            "unit": "kg",   "price":  4.80, "category": "Grãos"},
    {"id": "BEB001", "name": "Still mineral water",      "unit": "cx",   "price": 28.00, "category": "Bebidas"},
    {"id": "BEB002", "name": "Cola can soda",            "unit": "cx",   "price": 72.00, "category": "Bebidas"},
    {"id": "BEB003", "name": "Whole orange juice",       "unit": "lt",   "price": 12.00, "category": "Bebidas"},
    {"id": "TEM001", "name": "Soybean oil",              "unit": "lt",   "price":  8.90, "category": "Temperos"},
    {"id": "TEM002", "name": "Refined salt",             "unit": "kg",   "price":  3.50, "category": "Temperos"},
    {"id": "TEM003", "name": "Ground black pepper",      "unit": "kg",   "price": 45.00, "category": "Temperos"},
    {"id": "TEM004", "name": "Refined sugar",            "unit": "kg",   "price":  5.80, "category": "Temperos"},
    {"id": "OUT001", "name": "White eggs",               "unit": "cx",   "price": 24.00, "category": "Outros"},
    {"id": "OUT002", "name": "Sliced sandwich bread",    "unit": "un",   "price":  8.50, "category": "Outros"},
]


def build_corpus(catalog: list[dict]) -> list[str]:
    return [
        f"{p['id']} | {p['name']} | {p['category']} | R${p['price']:.2f}/{p['unit']}"
        for p in catalog
    ]


corpus = build_corpus(CATALOG)
fuzzy  = mf.Retriever.fuzzy("rapidfuzz")
fuzzy.add(corpus)


class STT(nn.Transcriber):
    model          = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode  = "user"


class ShelfScanner(nn.Agent):
    model        = vision_model
    instructions = """
    Look at this image and list every food product you can identify.
    For each one, estimate the approximate remaining quantity if visible.
    Be concise — one line per item.
    """
    message_fields = {"task_multimodal": {"image": "image_content"}}
    templates      = {"task": "Identify the food products visible in the image."}
    response_mode  = "vars.image_description"


class ItemExtractor(nn.Agent):
    model             = chat_model
    instructions      = """
    Extract every product the user is requesting.
    For each item fill in name, quantity, and unit.
    Use null for quantity or unit when not specified.
    """
    generation_schema = ExtractedItems
    message_fields    = {
        "task":         "user.text",
        "task_context": "vars",
    }
    templates = {
        "task_context": (
            "{% if image_description %}"
            "Products identified in the image:\n{{ image_description }}\n"
            "{% endif %}"
        )
    }
    response_mode = "extracted_items"


class ProductSearcher(nn.Searcher):
    retriever = fuzzy
    config    = {"top_k": 5}


@mf.tool_config(inject_message=True)
class ProductFinder(nn.Module):
    """Find catalog matches for the products the user wants to order."""

    def __init__(self):
        super().__init__()
        self.set_name("product_finder")
        self.set_annotations({"return": str})
        self.shelf_scanner    = ShelfScanner()
        self.item_extractor   = ItemExtractor()
        self.product_searcher = ProductSearcher()

    def _format_result(self, items: list, matches: dict) -> str:
        lines = ["## Identified items and catalog matches"]
        for item in items:
            name = item.get("name", "unknown")
            qty  = item.get("quantity") or "?"
            unit = item.get("unit") or ""
            lines.append(f"\n**{name}** — {qty} {unit}".rstrip())
            for i, match in enumerate(matches.get(name, []), 1):
                lines.append(f"  {i}. {match}")
            if not matches.get(name):
                lines.append("  (no catalog match found)")
        return "\n".join(lines)

    def _to_items(self, message: mf.Message) -> list:
        extracted = message.get("extracted_items")
        raw = extracted.get("items", []) if extracted else []
        return [
            {"name": i.get("name", ""), "quantity": i.get("quantity") or "", "unit": i.get("unit") or ""}
            for i in raw
        ]

    def forward(self, message: mf.Message) -> str:
        if message.get("image_content"):
            self.shelf_scanner(message)

        self.item_extractor(message)
        items = self._to_items(message)

        matches = {}
        for item in items:
            name    = item["name"]
            results = self.product_searcher(name)
            top     = results[0]["results"] if results else []
            matches[name] = [r["data"] for r in top]

        return self._format_result(items, matches)

    async def aforward(self, message: mf.Message) -> str:
        if message.get("image_content"):
            await self.shelf_scanner.acall(message)

        await self.item_extractor.acall(message)
        items = self._to_items(message)

        matches = {}
        for item in items:
            name    = item["name"]
            results = await self.product_searcher.acall(name)
            top     = results[0]["results"] if results else []
            matches[name] = [r["data"] for r in top]

        return self._format_result(items, matches)


@mf.tool_config(inject_vars=True)
def add_item(
    product_id: str,
    name: str,
    quantity: float,
    unit: str,
    unit_price: float,
    **kwargs,
) -> str:
    """Add one confirmed catalog item to the pending order.
    Call once per item after the user selects a catalog option and confirms the quantity.
    """
    vars  = kwargs["vars"]
    items = vars.setdefault("order_items", [])
    items.append({
        "product_id": product_id,
        "name":       name,
        "quantity":   quantity,
        "unit":       unit,
        "unit_price": unit_price,
    })
    running_total = sum(i["unit_price"] * i["quantity"] for i in items)
    return (
        f"Added {quantity} {unit} of {name} (ID: {product_id}, "
        f"R${unit_price:.2f}/{unit}). "
        f"{len(items)} item(s) queued — running total: R${running_total:.2f}."
    )


@mf.tool_config(inject_vars=True)
def submit_order(**kwargs) -> str:
    """Submit all queued items as a purchase order.
    Call only after the user has confirmed all selections and quantities.
    """
    vars  = kwargs["vars"]
    items = vars.get("order_items") or []
    if not items:
        return "No items queued. Use add_item to add confirmed selections first."
    order_id = str(uuid.uuid4())[:8].upper()
    total    = sum(i["unit_price"] * i["quantity"] for i in items)
    vars["order_items"] = []
    return (
        f"Order {order_id} submitted successfully. "
        f"{len(items)} item(s), total: R${total:.2f}."
    )


class Assistant(nn.Agent):
    model          = chat_model
    system_message = """
    You are a purchasing assistant for restaurant kitchens.

    Help the team place orders with suppliers.

    When the user describes what they need — by text, audio transcript, or image —
    call product_finder(). The tool receives the full message automatically.

    The tool returns identified items and numbered catalog matches.
    Present the list to the user, confirm quantities, and ask which catalog
    entry they want for each item. Note items with no match as unavailable.

    Once the user confirms a selection, call add_item() for each confirmed item
    with the product_id, name, quantity, unit, and unit_price from the catalog.
    After all items are added, call submit_order() to finalize.
    """
    message_fields = {
        "task":         "user.text",
        "task_context": "vars",
    }
    templates = {
        "task_context": (
            "{% if has_image %}"
            "The user sent an image of the pantry or shelf — "
            "call product_finder() to identify what needs restocking.\n"
            "{% endif %}"
        )
    }
    tools  = [ProductFinder, add_item, submit_order]
    config = {"verbose": True}


class SupplyAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.assistant = Assistant()
        self.stt       = STT()

    def _prepare(self, msg: mf.Message) -> None:
        if msg.get("image_content"):
            msg.set("vars.has_image", True)
        if not msg.get("user.text"):
            msg.set("user.text", "[image attached]")

    def forward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._prepare(msg)
        if msg.get("audio_content"):
            self.stt(msg)
        msg.response = self.assistant(msg, messages=history or [])
        return msg

    async def aforward(self, msg: mf.Message, history: list | None = None) -> mf.Message:
        self._prepare(msg)
        if msg.get("audio_content"):
            await self.stt.acall(msg)
        msg.response = await self.assistant.acall(msg, messages=history or [])
        return msg


class VoiceNote(nn.Speaker):
    model           = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_format = "mp3"
    config          = {"voice": "nova"}


SHELF_IMAGE_URL = (
    "https://en-chatelaine.mblycdn.com/ench/resized/2017/02/w1534/"
    "tomatoes-deniz-altindas.jpg"
)


if __name__ == "__main__":
    import sys

    assistant = SupplyAssistant()
    mode = sys.argv[1] if len(sys.argv) > 1 else "text"

    if mode == "audio":
        print("=== Audio demo ===")
        voice      = VoiceNote()
        audio_path = voice("I need 10 kilos of chicken fillet and 5 kilos of onion")
        msg = mf.Message()
        msg.audio_content = audio_path  # path only — STT transcribes on entry
        assistant.forward(msg)
        print("Transcription:", msg.user.text)
        print("Assistant:", msg.response)

    elif mode == "image":
        print("=== Shelf photo demo ===")
        msg = mf.Message()
        msg.image_content = SHELF_IMAGE_URL
        assistant.forward(msg)
        print("Shelf scan:", msg.vars.get("image_description"))
        print("Assistant:", msg.response)

    else:
        history: list = []
        print("Restaurant Supply Assistant — text mode (type 'quit' to exit)\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                break
            msg = mf.Message()
            msg.set("user.text", user_input)
            assistant.forward(msg, history=history)
            history.extend([
                mf.ChatBlock.user(user_input),
                mf.ChatBlock.assist(str(msg.response)),
            ])
            print(f"\nAssistant: {msg.response}\n")