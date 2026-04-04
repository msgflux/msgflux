# Food Delivery Assistant

<span class="tag tag-orange">Advanced</span>

An iFood or UberEats-style assistant that recommends dishes and places orders through a natural conversation.

## The Problem

Food catalogs are large and users describe what they want in natural language — vague, implicit, sometimes contradictory:

- `"quero sushi"` — clear cuisine preference;
- `"algo rápido e barato pra almoço"` — implicit preferences;
- `"sem glúten, até R$40"` — restriction plus price filter;
- `"o mesmo de sempre"` — relies on history.

A keyword search fails on vague queries. A model without a structured catalog hallucinates dishes. The challenge is pairing semantic understanding with a real, searchable catalog.

---

## The Plan

This tutorial is split into two phases:

- **Phase 1 — Ingestion**: generate raw catalog data with Faker, enrich every dish in parallel using a `ProductClassifier` agent, and build two BM25 indexes — one for dishes, one for restaurants.
- **Phase 2 — Assistant**: a multi-turn conversational agent with four tools (`search_dishes`, `search_restaurants`, `get_menu`, `place_order`) that searches the BM25 indexes and places orders.

---

## Architecture

```
Phase 1 — Ingestion
──────────────────────────────────────────────────────
Faker → raw restaurants + raw dish names
                    │
                    ▼
          ProductClassifier (Agent + Signature)
          map_gather: all dishes in parallel
                    │
                    ▼  {name, description, category,
                        price, tags, dietary}
          Enriched catalog
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   dish_bm25            restaurant_bm25
   (name + desc          (name + cuisine
    + tags)               + tags)

Phase 2 — Assistant
──────────────────────────────────────────────────────
User message
      │
      ▼
 FoodAssistant
 (tools: [search_dishes, search_restaurants,
          get_menu, place_order])
      │
      ├── search_dishes / search_restaurants → BM25
      ├── get_menu → catalog lookup
      └── place_order → order confirmation
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

```bash
pip install rank-bm25 faker
```

---

# Phase 1 — Ingestion

The ingestion phase generates raw data and enriches it using a `ProductClassifier`. Raw dishes contain only an informal name and the restaurant cuisine. The classifier infers description, category, price, tags, and dietary restrictions.

---

## Step 1 — Models

```python
import uuid
import random
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import ChatBlock
from faker import Faker
from typing import Literal

mf.load_dotenv()
chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
fake       = Faker("pt_BR")
```

---

## Step 2 — Raw Data

Raw restaurants carry only identity fields. Faker generates the operational metadata — rating, delivery time, minimum order. Raw dishes carry only an informal name and a reference to the restaurant.

```python
RAW_RESTAURANTS = [
    {"id": "REST001", "name": "Pizzaria Bella Napoli", "cuisine": "pizza"},
    {"id": "REST002", "name": "Sushi Sakura",          "cuisine": "japanese"},
    {"id": "REST003", "name": "Burger Bros",            "cuisine": "burger"},
    {"id": "REST004", "name": "Sabor Mineiro",          "cuisine": "brazilian"},
    {"id": "REST005", "name": "Shawarma Palace",        "cuisine": "arabic"},
    {"id": "REST006", "name": "Verde & Bom",            "cuisine": "vegan"},
    {"id": "REST007", "name": "Wok House",              "cuisine": "chinese"},
    {"id": "REST008", "name": "Tacos & Co",             "cuisine": "mexican"},
    {"id": "REST009", "name": "Frango & Cia",           "cuisine": "brazilian"},
    {"id": "REST010", "name": "Pasta Mia",              "cuisine": "italian"},
]

RAW_DISHES = [
    # Pizzaria Bella Napoli
    {"restaurant_id": "REST001", "raw_name": "margherita"},
    {"restaurant_id": "REST001", "raw_name": "calabresa"},
    {"restaurant_id": "REST001", "raw_name": "quatro queijos"},
    {"restaurant_id": "REST001", "raw_name": "frango catupiry"},
    {"restaurant_id": "REST001", "raw_name": "calzone de presunto"},
    {"restaurant_id": "REST001", "raw_name": "pão de alho"},
    # Sushi Sakura
    {"restaurant_id": "REST002", "raw_name": "combinado 20 peças"},
    {"restaurant_id": "REST002", "raw_name": "temaki salmão"},
    {"restaurant_id": "REST002", "raw_name": "uramaki philadelphia"},
    {"restaurant_id": "REST002", "raw_name": "sashimi salmão"},
    {"restaurant_id": "REST002", "raw_name": "yakisoba frango"},
    {"restaurant_id": "REST002", "raw_name": "missoshiru"},
    # Burger Bros
    {"restaurant_id": "REST003", "raw_name": "classic burger"},
    {"restaurant_id": "REST003", "raw_name": "double smash"},
    {"restaurant_id": "REST003", "raw_name": "chicken crispy"},
    {"restaurant_id": "REST003", "raw_name": "veggie burger"},
    {"restaurant_id": "REST003", "raw_name": "batata frita"},
    {"restaurant_id": "REST003", "raw_name": "milk shake chocolate"},
    # Sabor Mineiro
    {"restaurant_id": "REST004", "raw_name": "frango com quiabo"},
    {"restaurant_id": "REST004", "raw_name": "tutu de feijão"},
    {"restaurant_id": "REST004", "raw_name": "costelinha com mandioca"},
    {"restaurant_id": "REST004", "raw_name": "feijão tropeiro"},
    {"restaurant_id": "REST004", "raw_name": "arroz com pequi"},
    # Shawarma Palace
    {"restaurant_id": "REST005", "raw_name": "shawarma de frango"},
    {"restaurant_id": "REST005", "raw_name": "shawarma de carne"},
    {"restaurant_id": "REST005", "raw_name": "falafel no pão"},
    {"restaurant_id": "REST005", "raw_name": "kafta grelhada"},
    {"restaurant_id": "REST005", "raw_name": "prato árabe completo"},
    # Verde & Bom
    {"restaurant_id": "REST006", "raw_name": "bowl proteico"},
    {"restaurant_id": "REST006", "raw_name": "hambúrguer de grão de bico"},
    {"restaurant_id": "REST006", "raw_name": "açaí na tigela"},
    {"restaurant_id": "REST006", "raw_name": "wrap vegano"},
    {"restaurant_id": "REST006", "raw_name": "salada caesar vegana"},
    # Wok House
    {"restaurant_id": "REST007", "raw_name": "frango ao molho agridoce"},
    {"restaurant_id": "REST007", "raw_name": "yakisoba misto"},
    {"restaurant_id": "REST007", "raw_name": "arroz chop suey"},
    {"restaurant_id": "REST007", "raw_name": "camarão ao alho e óleo"},
    # Tacos & Co
    {"restaurant_id": "REST008", "raw_name": "tacos de carne"},
    {"restaurant_id": "REST008", "raw_name": "tacos de frango"},
    {"restaurant_id": "REST008", "raw_name": "burrito de carne"},
    {"restaurant_id": "REST008", "raw_name": "quesadilla de queijo"},
    {"restaurant_id": "REST008", "raw_name": "nachos com guacamole"},
    # Frango & Cia
    {"restaurant_id": "REST009", "raw_name": "meio frango grelhado"},
    {"restaurant_id": "REST009", "raw_name": "porção de frango frito"},
    {"restaurant_id": "REST009", "raw_name": "frango à passarinho"},
    {"restaurant_id": "REST009", "raw_name": "sanduíche de frango"},
    {"restaurant_id": "REST009", "raw_name": "aipim frito"},
    # Pasta Mia
    {"restaurant_id": "REST010", "raw_name": "spaghetti bolognese"},
    {"restaurant_id": "REST010", "raw_name": "fettuccine alfredo"},
    {"restaurant_id": "REST010", "raw_name": "penne all'arrabbiata"},
    {"restaurant_id": "REST010", "raw_name": "lasanha de carne"},
    {"restaurant_id": "REST010", "raw_name": "risoto de funghi"},
    {"restaurant_id": "REST010", "raw_name": "tiramisu"},
]
```

Restaurant operational metadata is generated with Faker. Delivery times and minimum orders follow realistic ranges per cuisine.

```python
_DELIVERY_BY_CUISINE = {
    "burger":   (20, 30), "brazilian": (30, 45), "pizza":    (30, 40),
    "japanese": (35, 50), "arabic":    (25, 35), "vegan":    (30, 40),
    "chinese":  (25, 35), "mexican":   (20, 30), "italian":  (30, 45),
}

def generate_restaurant_metadata(raw: list[dict]) -> list[dict]:
    enriched = []
    for r in raw:
        low, high = _DELIVERY_BY_CUISINE.get(r["cuisine"], (30, 45))
        enriched.append({
            **r,
            "rating":       round(random.uniform(4.1, 4.9), 1),
            "delivery_min": random.randint(low, high),
            "min_order":    random.choice([20.0, 25.0, 30.0, 40.0]),
            "tags":         [],  # populated after dish enrichment
        })
    return enriched

RESTAURANTS = generate_restaurant_metadata(RAW_RESTAURANTS)
_rest_by_id = {r["id"]: r for r in RESTAURANTS}
```

---

## Step 3 — ProductClassifier

`ProductClassifier` takes an informal dish name and the restaurant's cuisine and returns a fully structured entry. The `Signature` enforces the output shape — price, category, tags, and dietary restrictions are all inferred from context.

```python
class ProductClassifier(nn.Agent):
    """
    Classifies a raw dish name into a structured catalog entry.
    Infers description, price, category, tags, and dietary restrictions
    from the dish name and restaurant cuisine.
    """
    model        = chat_model
    system_message = """
    You are a food catalog specialist for a Brazilian food delivery platform.
    Generate realistic, appetizing catalog entries in Brazilian Portuguese.
    Price estimates should reflect typical restaurant prices in Brazil (BRL).
    """
    signature = """
    raw_name: str, cuisine: str ->
    name:        str,
    description: str,
    category:    Literal['main course', 'starter', 'dessert', 'drink', 'side dish'],
    price:       float,
    tags:        list[str],
    dietary:     list[str]
    """

classifier = ProductClassifier()
```

The `dietary` field carries restriction tags like `vegetariano`, `vegano`, `sem glúten`, `picante` — the search layer uses them to filter results later.

---

## Step 4 — Parallel Enrichment

`F.map_gather` runs the classifier over all raw dishes concurrently. Each call is independent, so the entire catalog is enriched in a single parallel batch instead of sequentially.

```python
def _classify(raw_name: str, cuisine: str) -> dict:
    return classifier(raw_name=raw_name, cuisine=cuisine)

raw_inputs = [
    (d["raw_name"], _rest_by_id[d["restaurant_id"]]["cuisine"])
    for d in RAW_DISHES
]

enriched = F.map_gather(_classify, args_list=raw_inputs)
```

Assemble the final catalog and assign sequential IDs:

```python
DISHES = []
for i, (raw, result) in enumerate(zip(RAW_DISHES, enriched), start=1):
    DISHES.append({
        "id":            f"D{i:03d}",
        "restaurant_id": raw["restaurant_id"],
        **result,
    })

_dish_by_id = {d["id"]: d for d in DISHES}
```

Backfill restaurant tags from their dishes so the restaurant index carries cuisine signals too:

```python
for rest in RESTAURANTS:
    dish_tags = [
        tag
        for d in DISHES
        if d["restaurant_id"] == rest["id"]
        for tag in d.get("tags", [])
    ]
    rest["tags"] = list(dict.fromkeys(dish_tags))[:8]  # deduplicated, top 8
```

---

## Step 5 — Building the Indexes

Two BM25 indexes. The dish corpus packs name, description, price, and tags. The restaurant corpus packs name, cuisine, rating, delivery time, and tags.

```python
def _dish_corpus(dishes: list[dict]) -> list[str]:
    entries = []
    for d in dishes:
        rest = _rest_by_id.get(d["restaurant_id"], {})
        tags = " ".join(d.get("tags", []) + d.get("dietary", []))
        entries.append(
            f"{d['id']} | {d['name']} | {rest.get('name', '')} | "
            f"{d.get('description', '')} | R${d['price']:.2f} | {tags}"
        )
    return entries


def _restaurant_corpus(restaurants: list[dict]) -> list[str]:
    return [
        f"{r['id']} | {r['name']} | {r['cuisine']} | "
        f"rating: {r['rating']} | {r['delivery_min']}min | "
        f"mín: R${r['min_order']:.0f} | {' '.join(r['tags'])}"
        for r in restaurants
    ]


dish_bm25 = mf.Retriever.lexical("rank_bm25")
dish_bm25.add(_dish_corpus(DISHES))

restaurant_bm25 = mf.Retriever.lexical("rank_bm25")
restaurant_bm25.add(_restaurant_corpus(RESTAURANTS))
```

The ingestion phase is complete. `DISHES`, `RESTAURANTS`, `dish_bm25`, and `restaurant_bm25` are ready for Phase 2.

---

# Phase 2 — Conversational Assistant

---

## Step 6 — Searchers and Tools

Two `nn.Searcher` instances are used imperatively inside tool functions.

```python
class DishSearcher(nn.Searcher):
    retriever = dish_bm25
    config    = {"top_k": 5}


class RestaurantSearcher(nn.Searcher):
    retriever = restaurant_bm25
    config    = {"top_k": 5}


_dish_searcher       = DishSearcher()
_restaurant_searcher = RestaurantSearcher()
```

### search_dishes

```python
def search_dishes(query: str) -> str:
    """
    Search for dishes by name, description, ingredients, cuisine, or dietary tag.
    Include price constraints and dietary restrictions directly in the query
    (e.g. "vegan under R$35", "gluten-free japanese").
    """
    raw     = _dish_searcher(query)
    results = raw[0]["results"] if raw else []

    lines = []
    for r in results:
        dish_id = r["data"].split(" | ")[0]
        dish    = _dish_by_id.get(dish_id)
        if not dish:
            continue
        rest    = _rest_by_id.get(dish["restaurant_id"], {})
        dietary = ", ".join(dish.get("dietary", []))
        lines.append(
            f"{dish_id} | {dish['name']} | {rest.get('name', '?')} | "
            f"R${dish['price']:.2f}"
            + (f" | {dietary}" if dietary else "")
        )

    return "\n".join(lines) if lines else "No dishes found."
```

### search_restaurants

```python
def search_restaurants(query: str) -> str:
    """
    Search for restaurants by name, cuisine type, or tags.
    Include delivery time constraints directly in the query
    (e.g. "japanese fast delivery", "pizza 30 minutes").
    """
    raw     = _restaurant_searcher(query)
    results = raw[0]["results"] if raw else []

    lines = []
    for r in results:
        rest_id = r["data"].split(" | ")[0]
        rest    = _rest_by_id.get(rest_id)
        if not rest:
            continue
        lines.append(
            f"{rest_id} | {rest['name']} | {rest['cuisine']} | "
            f"⭐ {rest['rating']} | {rest['delivery_min']}min | mín: R${rest['min_order']:.0f}"
        )

    return "\n".join(lines) if lines else "No restaurants found."
```

### get_menu

```python
def get_menu(restaurant_id: str) -> str:
    """
    Get the full menu of a restaurant, including prices, descriptions, and dietary tags.
    Use when the user wants details about a specific restaurant.
    """
    rest = _rest_by_id.get(restaurant_id)
    if not rest:
        return f"Restaurant {restaurant_id} not found."

    dishes = [d for d in DISHES if d["restaurant_id"] == restaurant_id]
    lines  = [
        f"# {rest['name']} ({rest['cuisine']})",
        f"⭐ {rest['rating']} | {rest['delivery_min']}min | mín: R${rest['min_order']:.0f}",
        "",
    ]
    for d in dishes:
        dietary = f" [{', '.join(d['dietary'])}]" if d.get("dietary") else ""
        lines.append(f"{d['id']} | {d['name']} — R${d['price']:.2f}{dietary}")
        lines.append(f"     {d.get('description', '')}")
    return "\n".join(lines)
```

### place_order

```python
def place_order(
    restaurant_id: str,
    dish_ids:      list[str],
    names:         list[str],
    quantities:    list[int],
) -> str:
    """
    Place a food order. Call only after the user has confirmed their selection.
    dish_ids, names, and quantities are parallel lists — index i describes one item.
    """
    rest = _rest_by_id.get(restaurant_id)
    if not rest:
        return f"Restaurant {restaurant_id} not found."

    order_id = str(uuid.uuid4())[:8].upper()
    total    = 0.0
    lines    = [f"Order {order_id} confirmed at {rest['name']}", ""]

    for dish_id, name, qty in zip(dish_ids, names, quantities):
        dish   = _dish_by_id.get(dish_id)
        price  = dish["price"] if dish else 0.0
        total += price * qty
        lines.append(f"  {qty}x {name} — R${price * qty:.2f}")

    lines += ["", f"Total: R${total:.2f}", f"Estimated delivery: {rest['delivery_min']}min"]
    return "\n".join(lines)
```

---

## Step 7 — FoodAssistant

```python
class FoodAssistant(nn.Agent):
    """Food delivery assistant with restaurant and dish search."""
    model          = chat_model
    system_message = """
    You are a food delivery assistant, similar to iFood or UberEats.

    Help the user find and order food through a natural conversation.

    Available tools:
    - search_dishes: search by name, ingredient, cuisine, or tag (e.g. "vegan", "gluten-free", "spicy")
    - search_restaurants: search by name or cuisine type
    - get_menu: get the full menu of a specific restaurant
    - place_order: submit the order after user confirmation

    Guidelines:
    - When the request is vague, search both dishes and restaurants and present the best options.
    - Always show dish ID, name, restaurant, price, and dietary tags.
    - Ask clarifying questions when the user has restrictions (gluten-free, vegetarian, etc.).
    - Before calling place_order, confirm the exact items and quantities with the user.
    - If nothing matches, suggest the closest alternatives.
    """
    tools  = [search_dishes, search_restaurants, get_menu, place_order]
    config = {"verbose": True}


assistant = FoodAssistant()
```

---

## Complete Example

```python
import uuid
import random
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import ChatBlock
from faker import Faker
from typing import Literal

mf.load_dotenv()
chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
fake       = Faker("pt_BR")

# ── Phase 1: paste Steps 2–5 here ──────────────────────────────────────────

# ── Phase 2: paste Steps 6–7 here ──────────────────────────────────────────

assistant = FoodAssistant()
```

---

## Examples

???+ example

    === "Vague request (multi-turn)"

        ```python
        assistant = FoodAssistant()
        history   = []

        # Turn 1 — vague
        response = assistant("Quero comer algo japonês hoje", messages=history)
        history += [ChatBlock.user("Quero comer algo japonês hoje"), ChatBlock.assist(str(response))]
        print("Assistant:", response)

        # Turn 2 — refinement
        response = assistant("Tem algo sem glúten e até R$40?", messages=history)
        history += [ChatBlock.user("Tem algo sem glúten e até R$40?"), ChatBlock.assist(str(response))]
        print("Assistant:", response)

        # Turn 3 — confirm
        response = assistant("Quero o sashimi. Confirma o pedido.", messages=history)
        print("Assistant:", response)
        ```

    === "Dietary restriction"

        ```python
        assistant = FoodAssistant()
        history   = []

        response = assistant("Quero algo vegano, até R$35, entrega rápida", messages=history)
        history += [ChatBlock.user("Quero algo vegano, até R$35, entrega rápida"), ChatBlock.assist(str(response))]
        print("Assistant:", response)

        response = assistant("Quero o bowl proteico. Pode pedir.", messages=history)
        print("Assistant:", response)
        ```

    === "Browse restaurant"

        ```python
        assistant = FoodAssistant()
        history   = []

        response = assistant("Mostra o cardápio da Pizzaria Bella Napoli", messages=history)
        history += [ChatBlock.user("Mostra o cardápio da Pizzaria Bella Napoli"), ChatBlock.assist(str(response))]
        print("Assistant:", response)

        response = assistant("Quero uma Margherita e um Pão de Alho", messages=history)
        history += [ChatBlock.user("Quero uma Margherita e um Pão de Alho"), ChatBlock.assist(str(response))]
        print("Assistant:", response)

        response = assistant("Confirma.", messages=history)
        print("Assistant:", response)
        ```

---

## Extending

### Strategy B — Structured Recommender

Use **Strategy A** for a chat interface where the user refines the request interactively, and **Strategy B** for a feed or push recommendation where a single message needs to produce a ranked list immediately.

```python
class PreferenceExtractor(nn.Agent):
    """Extracts structured food preferences from a natural language message."""
    model     = chat_model
    signature = """
    message ->
    cuisine:          Optional[str],
    max_price:        Optional[float],
    max_delivery_min: Optional[int],
    restrictions:     list[str],
    keywords:         list[str]
    """


class Ranker(nn.Agent):
    """Ranks search results and produces the top 3 recommendations with reasoning."""
    model          = chat_model
    system_message = "You are a food recommendation engine. Be concise and opinionated."
    instructions   = """
    Given user preferences and search results, select the 3 best options.
    For each one explain in one sentence why it matches.
    """
    signature = """
    preferences, dish_results, restaurant_results ->
    recommendations: list[dict],
    reasoning:       str
    """


class StructuredRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = PreferenceExtractor()
        self.ranker    = Ranker()

    def forward(self, message: str) -> dict:
        preferences = self.extractor(message)

        keywords = preferences.get("keywords") or []
        cuisine  = preferences.get("cuisine") or ""
        query    = " ".join(keywords + ([cuisine] if cuisine else []))

        dish_results, restaurant_results = F.bcast_gather(
            [
                lambda q: _dish_searcher(q),
                lambda q: _restaurant_searcher(q),
            ],
            query,
        )

        return self.ranker(
            preferences=preferences,
            dish_results=dish_results,
            restaurant_results=restaurant_results,
        )


# Usage
recommender = StructuredRecommender()
result      = recommender("Quero algo vegano, leve, até R$35")

print("Recommendations:", result["recommendations"])
print("Reasoning:", result["reasoning"])
```

---

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — tools, multi-turn, system messages
- [nn.Searcher](../learn/nn/searcher.md) — BM25 and semantic retrieval
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Functional API](../learn/nn/functional.md) — `map_gather` and `bcast_gather`
