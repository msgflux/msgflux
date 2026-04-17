# Food Delivery Assistant

<span class="tag tag-purple">Intermediate</span><span class="tag tag-gray">Generation Schema</span><span class="tag tag-gray">Multimodal</span><span class="tag tag-gray">Retrivers</span><span class="tag tag-gray">Guardrails</span>

A food ordering assistant that recommends dishes and places orders through a natural conversation.

## The Problem

Food catalogs are large and users rarely describe what they want in a way that maps cleanly to a menu item. The same intent arrives in dozens of forms:

- a clear cuisine preference — "I want sushi";
- an implicit constraint — "something quick and cheap for lunch";
- a restriction plus a budget — "gluten-free, under US$40";
- a reference to habit — "the same as always."

Keyword search breaks on vague queries. A model without a real catalog invents dishes that do not exist. Matching prices and dietary restrictions requires structured data that most pipelines never produce in the first place.

---

## The Plan

The tutorial is split into two phases.

**Phase 1 — Ingestion.** The catalog has two kinds of data with very different origins. Restaurant records are operational: ratings, delivery times, and minimum orders come straight from a simulated database — no model needed. Dish records are the opposite: each entry starts as nothing more than an informal name (`"margherita"`, `"salmon sashimi"`). A `DishEnricher` agent takes each raw name together with the restaurant's cuisine type and generates the full catalog entry — a proper name, description, category, realistic price, flavor tags, and dietary restrictions. Every dish is enriched in parallel, so the entire catalog is ready in one batch. Once enriched, the restaurant records are updated with the tags inferred from their dishes, and two separate indexes are built — one for dishes, one for restaurants — so both can be searched independently.

**Phase 2 — Assistant.** A conversational agent handles the multi-turn dialogue. When the user describes what they want, the agent searches the dish or restaurant index and presents the closest matches. The user can browse a full restaurant menu, filter by dietary tag or price, and refine across turns. Once the selection is confirmed, the agent places the order and returns the total with the estimated delivery time.

---

## Architecture

```
Phase 1 — Ingestion
random → restaurants (rating, delivery_min, min_order)
raw dish names (name only)
                    │
                    ▼
          DishEnricher (Agent + generation_schema)
          map_gather: all dishes in parallel
                    │
                    ▼  {name, description, category,
                        price, tags, dietary}
          Enriched catalog
          + backfill restaurant tags
                    │
          ▼                   ▼
   dish_fuzzy          restaurant_fuzzy
   (name + desc          (name + cuisine
    + tags)               + tags)

Phase 2 — Assistant
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
pip install rapidfuzz
```

---

# Phase 1 — Ingestion

The ingestion phase has two distinct data sources. Restaurant records already carry all their operational fields — rating, delivery time, minimum order — generated here with `random` to simulate a real database. Dish records, on the other hand, start with only an informal name and a restaurant reference. A `DishEnricher` agent fills in everything else: description, category, price, flavor tags, and dietary restrictions.

---

## Step 1 — Models

```python
import msgflux as mf

mf.load_dotenv()
chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
```

---

## Step 2 — Raw Data

Raw restaurants carry only identity fields. Faker generates the operational metadata — rating, delivery time, minimum order. Raw dishes carry only an informal name and a reference to the restaurant.

```python
RAW_RESTAURANTS = [
    {"id": "REST001", "name": "Napoli's Pizza",   "cuisine": "pizza"},
    {"id": "REST002", "name": "Tokyo Garden",     "cuisine": "japanese"},
    {"id": "REST003", "name": "Burger Bros",      "cuisine": "burger"},
    {"id": "REST004", "name": "Southern Table",   "cuisine": "brazilian"},
    {"id": "REST005", "name": "Shawarma Palace",  "cuisine": "arabic"},
    {"id": "REST006", "name": "Green & Good",     "cuisine": "vegan"},
    {"id": "REST007", "name": "Wok House",        "cuisine": "chinese"},
    {"id": "REST008", "name": "Tacos & Co",       "cuisine": "mexican"},
    {"id": "REST009", "name": "Chicken House",    "cuisine": "brazilian"},
    {"id": "REST010", "name": "Pasta Mia",        "cuisine": "italian"},
]

RAW_DISHES = [
    # Napoli's Pizza
    {"restaurant_id": "REST001", "raw_name": "margherita"},
    {"restaurant_id": "REST001", "raw_name": "sausage pizza"},
    {"restaurant_id": "REST001", "raw_name": "four cheese pizza"},
    {"restaurant_id": "REST001", "raw_name": "chicken cream cheese pizza"},
    {"restaurant_id": "REST001", "raw_name": "ham calzone"},
    {"restaurant_id": "REST001", "raw_name": "garlic bread"},
    # Tokyo Garden
    {"restaurant_id": "REST002", "raw_name": "20-piece sushi combo"},
    {"restaurant_id": "REST002", "raw_name": "salmon hand roll"},
    {"restaurant_id": "REST002", "raw_name": "philadelphia roll"},
    {"restaurant_id": "REST002", "raw_name": "salmon sashimi"},
    {"restaurant_id": "REST002", "raw_name": "chicken yakisoba"},
    {"restaurant_id": "REST002", "raw_name": "miso soup"},
    # Burger Bros
    {"restaurant_id": "REST003", "raw_name": "classic burger"},
    {"restaurant_id": "REST003", "raw_name": "double smash burger"},
    {"restaurant_id": "REST003", "raw_name": "crispy chicken sandwich"},
    {"restaurant_id": "REST003", "raw_name": "veggie burger"},
    {"restaurant_id": "REST003", "raw_name": "french fries"},
    {"restaurant_id": "REST003", "raw_name": "chocolate milkshake"},
    # Southern Table
    {"restaurant_id": "REST004", "raw_name": "chicken with okra"},
    {"restaurant_id": "REST004", "raw_name": "bean stew"},
    {"restaurant_id": "REST004", "raw_name": "pork ribs with cassava"},
    {"restaurant_id": "REST004", "raw_name": "tropeiro beans"},
    {"restaurant_id": "REST004", "raw_name": "pequi rice"},
    # Shawarma Palace
    {"restaurant_id": "REST005", "raw_name": "chicken shawarma"},
    {"restaurant_id": "REST005", "raw_name": "beef shawarma"},
    {"restaurant_id": "REST005", "raw_name": "falafel wrap"},
    {"restaurant_id": "REST005", "raw_name": "grilled kafta"},
    {"restaurant_id": "REST005", "raw_name": "arabic platter"},
    # Green & Good
    {"restaurant_id": "REST006", "raw_name": "protein bowl"},
    {"restaurant_id": "REST006", "raw_name": "chickpea burger"},
    {"restaurant_id": "REST006", "raw_name": "açaí bowl"},
    {"restaurant_id": "REST006", "raw_name": "vegan wrap"},
    {"restaurant_id": "REST006", "raw_name": "vegan caesar salad"},
    # Wok House
    {"restaurant_id": "REST007", "raw_name": "sweet and sour chicken"},
    {"restaurant_id": "REST007", "raw_name": "mixed yakisoba"},
    {"restaurant_id": "REST007", "raw_name": "chop suey rice"},
    {"restaurant_id": "REST007", "raw_name": "garlic shrimp"},
    # Tacos & Co
    {"restaurant_id": "REST008", "raw_name": "beef tacos"},
    {"restaurant_id": "REST008", "raw_name": "chicken tacos"},
    {"restaurant_id": "REST008", "raw_name": "beef burrito"},
    {"restaurant_id": "REST008", "raw_name": "cheese quesadilla"},
    {"restaurant_id": "REST008", "raw_name": "nachos with guacamole"},
    # Chicken House
    {"restaurant_id": "REST009", "raw_name": "half grilled chicken"},
    {"restaurant_id": "REST009", "raw_name": "fried chicken platter"},
    {"restaurant_id": "REST009", "raw_name": "chicken bites"},
    {"restaurant_id": "REST009", "raw_name": "chicken sandwich"},
    {"restaurant_id": "REST009", "raw_name": "fried cassava"},
    # Pasta Mia
    {"restaurant_id": "REST010", "raw_name": "spaghetti bolognese"},
    {"restaurant_id": "REST010", "raw_name": "fettuccine alfredo"},
    {"restaurant_id": "REST010", "raw_name": "penne all'arrabbiata"},
    {"restaurant_id": "REST010", "raw_name": "meat lasagna"},
    {"restaurant_id": "REST010", "raw_name": "mushroom risotto"},
    {"restaurant_id": "REST010", "raw_name": "tiramisu"},
]
```

Restaurant operational metadata is generated with `random`. Delivery times and minimum orders follow realistic ranges per cuisine.

```python
import random

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

print(RESTAURANTS[0])
# {'id': 'REST001',
#  'name': "Napoli's Pizza",
#  'cuisine': 'pizza',
#  'rating': 4.1,
#  'delivery_min': 36,
#  'min_order': 40.0,
#  'tags': []}             # ← filled in Step 4 after dish enrichment
```

---

## Step 3 — DishEnricher

`DishEnricher` receives an informal dish name and the restaurant's cuisine type and returns a fully populated catalog entry. `DishEntry` is a `msgspec.Struct` that pins the output shape — every field the search layer needs (price, category, tags, dietary restrictions) is inferred from the minimal input.

```python
import msgflux.nn as nn
from msgspec import Struct

class DishEntry(Struct):
    name:        str
    description: str
    category:    Annotated[
                     Literal["main course", "starter", "dessert", "drink", "side dish"],
                     Meta(description="Dish category")
                 ]
    price:       Annotated[float,      Meta(description="Price in US$ — delivery app range: side $3-7, starter $5-10, main $9-18, dessert $4-8, drink $2-5")]
    tags:        Annotated[List[str],  Meta(description="Cuisine and flavor tags")]
    dietary:     Annotated[List[str],  Meta(description="Dietary tags: vegetarian, vegan, gluten-free, spicy, etc.")]


class DishEnricher(nn.Agent):
    """Expands a raw dish name into a full catalog entry."""
    model             = chat_model
    system_message    = """
    You are a food catalog specialist for a food delivery platform.
    """
    instructions      = """
    Generate realistic, appetizing catalog entries in English.
    Use typical US food delivery prices: sides $3-7, starters $5-10,
    mains $9-18, desserts $4-8, drinks $2-5.
    """
    generation_schema = DishEntry
    templates         = {"task": "Dish: {{ raw_name }}\nCuisine: {{ cuisine }}"}


enricher = DishEnricher()
```

For a single call, the input is just two strings and the output is a fully populated `DishEntry`:

```python
result = enricher(raw_name="margherita", cuisine="pizza")

print(result)
# {'name': 'Margherita Pizza',
#  'description': 'Classic Neapolitan pizza with San Marzano tomato sauce, '
#                 'fresh mozzarella, and fragrant basil leaves on a thin, '
#                 'hand-tossed crust.',
#  'category': 'main course',
#  'price': 13.99,
#  'tags': ['pizza', 'italian', 'classic', 'tomato', 'cheese'],
#  'dietary': ['vegetarian']}
```

The `dietary` field carries restriction tags like `vegetarian`, `vegan`, `gluten-free`, `spicy` — the search layer uses them to filter results later.

---

## Step 4 — Parallel Enrichment

`F.map_gather` runs `DishEnricher` over all raw dishes concurrently. Each call is independent, so the entire catalog is enriched in a single parallel batch instead of sequentially.

```python
enriched = F.map_gather(
    enricher,
    args_list=[() for _ in RAW_DISHES],
    kwargs_list=[
        {"raw_name": d["raw_name"], "cuisine": _rest_by_id[d["restaurant_id"]]["cuisine"]}
        for d in RAW_DISHES
    ],
)
```

Assemble the final catalog and assign sequential IDs. Each `result` returned by `enricher` is a `dotdict` — msgflux converts the `DishEntry` Struct response into a plain dict-like object — so `**result` spreads the fields (`name`, `description`, `category`, `price`, `tags`, `dietary`) directly into the dish entry.

```python
DISHES = []
for i, (raw, result) in enumerate(zip(RAW_DISHES, enriched), start=1):
    DISHES.append({
        "id":            f"D{i:03d}",
        "restaurant_id": raw["restaurant_id"],
        **result,
    })

_dish_by_id = {d["id"]: d for d in DISHES}

print(DISHES[0])
# {'id': 'D001',
#  'restaurant_id': 'REST001',
#  'name': 'Margherita Pizza',
#  'description': 'Classic Neapolitan pizza with San Marzano tomato sauce, '
#                 'fresh mozzarella, and fragrant basil leaves on a thin, '
#                 'hand-tossed crust.',
#  'category': 'main course',
#  'price': 13.99,
#  'tags': ['pizza', 'italian', 'classic', 'tomato', 'cheese'],
#  'dietary': ['vegetarian']}
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

print(RESTAURANTS[0])
# {'id': 'REST001',
#  'name': "Napoli's Pizza",
#  'cuisine': 'pizza',
#  'rating': 4.1,
#  'delivery_min': 36,
#  'min_order': 40.0,
#  'tags': ['pizza', 'italian', 'classic', 'tomato', 'cheese', 'creamy', 'savory', 'hearty']}
```

---

## Step 5 — Building the Indexes

Two fuzzy indexes. The dish corpus packs name, description, price, tags, and the restaurant ID. The restaurant ID is embedded directly in the string so `place_order` can receive it from the tool output without a separate lookup. The restaurant corpus packs name, cuisine, rating, delivery time, and tags.

```python
def _dish_corpus(dishes: list[dict]) -> list[str]:
    entries = []
    for d in dishes:
        rest = _rest_by_id.get(d["restaurant_id"], {})
        tags = " ".join(d.get("tags", []) + d.get("dietary", []))
        entries.append(
            f"{d['id']} | {d['name']} | {rest.get('name', '')} "
            f"(id: {d['restaurant_id']}) | {d.get('description', '')} "
            f"| US${d['price']:.2f} | {tags}"
        )
    return entries


def _restaurant_corpus(restaurants: list[dict]) -> list[str]:
    return [
        f"{r['id']} | {r['name']} | {r['cuisine']} | "
        f"rating: {r['rating']} | {r['delivery_min']}min | "
        f"mín: US${r['min_order']:.0f} | {' '.join(r['tags'])}"
        for r in restaurants
    ]


dish_fuzzy = mf.Retriever.fuzzy("rapidfuzz")
dish_fuzzy.add(_dish_corpus(DISHES))

restaurant_fuzzy = mf.Retriever.fuzzy("rapidfuzz")
restaurant_fuzzy.add(_restaurant_corpus(RESTAURANTS))
```

The ingestion phase is complete. `DISHES`, `RESTAURANTS`, `dish_fuzzy`, and `restaurant_fuzzy` are ready for Phase 2.

---

# Phase 2 — Conversational Assistant

---

## Step 6 — Searchers and Tools

Each searcher is a self-contained `nn.Searcher` that plugs directly into the agent as a tool — no wrapper function needed. The docstring becomes the tool description, `name` overrides the default class-name derived tool name, and `templates={"response": ...}` formats the retriever output into a plain string before it reaches the model.

The `(id: REST001)` token embedded in each dish result is what makes `place_order` work: the model never needs to guess or look up the restaurant ID — it reads it directly from the search output.

```python
class DishSearcher(nn.Searcher):
    """
    Search for dishes by name, description, ingredients, cuisine, or dietary tag.
    Include price constraints and dietary restrictions directly in the query
    (e.g. "vegan under US$35", "gluten-free japanese").
    """
    name      = "search_dishes"
    retriever = dish_fuzzy
    config    = {"top_k": 5}
    templates = {"response": "{% for r in results %}{{ r.data }}\n{% endfor %}"}


class RestaurantSearcher(nn.Searcher):
    """
    Search for restaurants by name, cuisine type, or tags.
    Include delivery time constraints directly in the query
    (e.g. "japanese fast delivery", "pizza 30 minutes").
    """
    name      = "search_restaurants"
    retriever = restaurant_fuzzy
    config    = {"top_k": 5}
    templates = {"response": "{% for r in results %}{{ r.data }}\n{% endfor %}"}
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
        f"rating: {rest['rating']} | {rest['delivery_min']}min | mín: US${rest['min_order']:.0f}",
        "",
    ]
    for d in dishes:
        dietary = f" [{', '.join(d['dietary'])}]" if d.get("dietary") else ""
        lines.append(f"{d['id']} | {d['name']} — US${d['price']:.2f}{dietary}")
        lines.append(f"     {d.get('description', '')}")
    return "\n".join(lines)
```

### place_order

```python
import uuid

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
        lines.append(f"  {qty}x {name} — US${price * qty:.2f}")

    lines += ["", f"Total: US${total:.2f}", f"Estimated delivery: {rest['delivery_min']}min"]
    return "\n".join(lines)
```

---

## Step 7 — FoodAssistant

The four tools are wired into a single `nn.Agent`. `system_message` establishes the role, `instructions` describes the conversation flow, and `expected_output` pins the behavioral guidelines.

```python
class FoodAssistant(nn.Agent):
    """Food delivery assistant with restaurant and dish search."""
    model           = chat_model
    system_message  = """
    You are a food delivery assistant, similar to iFood or UberEats.
    """
    instructions    = """
    Help the user find and order food through a natural conversation.

    Available tools:
    - search_dishes: search by name, ingredient, cuisine, or dietary tag
    - search_restaurants: search by name or cuisine type
    - get_menu: get the full menu of a specific restaurant
    - place_order: submit the order after user confirmation
    """
    expected_output = """
    - When the request is vague, search both dishes and restaurants.
    - Always show dish ID, name, restaurant, price, and dietary tags.
    - Ask clarifying questions when the user has dietary restrictions.
    - Before calling place_order, confirm the exact items and quantities.
    - If nothing matches, suggest the closest alternatives.
    - When calling place_order, use the dish_id (e.g. "D026") and
      restaurant_id (e.g. "REST005") exactly as shown in the search results.
      Never invent or paraphrase these IDs.
    """
    tools  = [DishSearcher, RestaurantSearcher, get_menu, place_order]
    config = {"verbose": True}


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
        user_msg_1 = "I want something Japanese today"
        response = assistant(user_msg_1, messages=history)
        history.append(mf.ChatBlock.assist(response))
        print("Assistant:", response)

        # Turn 2 — refinement
        user_msg_2 = "Anything gluten-free under US$15?"
        response = assistant(user_msg_2, messages=history)
        history.append(mf.ChatBlock.assist(response))
        print("Assistant:", response)

        # Turn 3 — confirm
        user_msg_3 = "I'll take the salmon sashimi. Place the order."
        response = assistant(user_msg_3, messages=history)
        print("Assistant:", response)
        ```

        ```
        # Turn 1
        [FoodAssistant][tool_call] search_dishes: {'query': 'Japanese'}
        [FoodAssistant][tool_call] search_restaurants: {'query': 'Japanese'}

        Assistant: Here are some Japanese options from Tokyo Garden:
        - D007 | 20-Piece Sushi Combo — US$14.99 [contains fish]
        - D008 | Salmon Hand Roll — US$7.99
        - D010 | Salmon Sashimi — US$12.99 [gluten-free]
        - D011 | Chicken Yakisoba — US$11.99
        Would you like to filter by dietary restriction or price?

        # Turn 2
        [FoodAssistant][tool_call] search_dishes: {'query': 'Japanese gluten-free under US$15'}

        Assistant: Gluten-free Japanese dishes under US$15:
        - D010 | Salmon Sashimi | Tokyo Garden (id: REST002) — US$12.99 [gluten-free]
        - D012 | Miso Soup | Tokyo Garden (id: REST002) — US$4.99 [gluten-free, vegan]
        Want to order the Salmon Sashimi?

        # Turn 3
        [FoodAssistant][tool_call] place_order: {'restaurant_id': 'REST002', 'dish_ids': ['D010'], 'names': ['Salmon Sashimi'], 'quantities': [1]}

        Assistant: Order A3F1B2C4 confirmed at Tokyo Garden
          1x Salmon Sashimi — US$12.99
        Total: US$12.99
        Estimated delivery: 42min
        ```

    === "Dietary restriction"

        ```python
        assistant = FoodAssistant()
        history   = []

        user_msg_1 = "I want something vegan, under US$15, fast delivery"

        response = assistant(user_msg_1, messages=history)
        history.append(mf.ChatBlock.assist(response))
        print("Assistant:", response)

        user_msg_2 = "I'll take the protein bowl. Go ahead and order."

        response = assistant(user_msg_2, messages=history)
        print("Assistant:", response)
        ```

        ```
        # Turn 1
        [FoodAssistant][tool_call] search_dishes: {'query': 'vegan under US$15 fast delivery'}
        [FoodAssistant][tool_call] search_restaurants: {'query': 'vegan fast delivery'}

        Assistant: Vegan options under US$15 from fast-delivery restaurants:
        - D029 | Vegan Protein Bowl | Green & Good (id: REST006) — US$14.00 [vegan, gluten-free]
        - D030 | Chickpea Burger | Green & Good (id: REST006) — US$12.00 [vegan]
        - D031 | Açaí Bowl | Green & Good (id: REST006) — US$9.99 [vegan, gluten-free]
        Green & Good delivers in ~32 min. Want to place an order?

        # Turn 2
        [FoodAssistant][tool_call] place_order: {'restaurant_id': 'REST006', 'dish_ids': ['D029'], 'names': ['Vegan Protein Bowl'], 'quantities': [1]}

        Assistant: Order 7D9E1A2B confirmed at Green & Good
          1x Vegan Protein Bowl — US$14.00
        Total: US$14.00
        Estimated delivery: 32min
        ```

    === "Browse restaurant"

        ```python
        assistant = FoodAssistant()
        history   = []

        user_msg_1 = "Show me the menu at Napoli's Pizza"

        response = assistant(user_msg_1, messages=history)
        history.append(mf.ChatBlock.assist(response))
        print("Assistant:", response)

        user_msg_2 = "I'd like a Margherita and a Garlic Bread"

        response = assistant(user_msg_2, messages=history)
        history.append(mf.ChatBlock.assist(response))
        print("Assistant:", response)

        response = assistant("Confirm.", messages=history)
        print("Assistant:", response)
        ```

        ```
        # Turn 1
        [FoodAssistant][tool_call] get_menu: {'restaurant_id': 'REST001'}

        Assistant: Napoli's Pizza (pizza) — rating: 4.3 | 36min | min: US$20
        - D001 | Margherita Pizza — US$13.99 [vegetarian]
        - D002 | Sausage Pizza — US$15.99
        - D003 | Four Cheese Pizza — US$16.99 [vegetarian]
        - D004 | Chicken Cream Cheese Pizza — US$16.99
        - D005 | Ham Calzone — US$14.99
        - D006 | Garlic Bread — US$5.99 [vegetarian]
        What would you like to order?

        # Turn 2
        Assistant: To confirm: 1x Margherita Pizza (US$13.99) and 1x Garlic Bread (US$5.99)
        from Napoli's Pizza. Total: US$19.98. Shall I place the order?

        # Turn 3
        [FoodAssistant][tool_call] place_order: {'restaurant_id': 'REST001', 'dish_ids': ['D001', 'D006'], 'names': ['Margherita Pizza', 'Garlic Bread'], 'quantities': [1, 1]}

        Assistant: Order 2C4E6F8A confirmed at Napoli's Pizza
          1x Margherita Pizza — US$13.99
          1x Garlic Bread — US$5.99
        Total: US$19.98
        Estimated delivery: 36min
        ```
        ```

---

## Extending

### Structured Recommender

The `FoodAssistant` built above is designed for interactive chat — the user refines the request across turns. A different scenario is a feed or push recommendation where a single message must immediately produce a ranked list. The `StructuredRecommender` handles that: it extracts structured preferences from the message, searches dishes and restaurants in parallel, and returns a typed top-3 ranking.

---

#### Schemas

Three `msgspec.Struct` types define the data contract. `UserPreferences` captures everything that can be inferred from a natural language message — cuisine, price ceiling, delivery constraint, dietary restrictions, and free-form keywords. `Recommendation` is a single ranked item with a one-sentence justification. `RankerOutput` wraps the top-3 list together with an overall reasoning string.

```python
class UserPreferences(Struct):
    restrictions:     Annotated[List[str],       Meta(description="Dietary restrictions (vegan, gluten-free, etc.)")]
    keywords:         Annotated[List[str],       Meta(description="Food keywords extracted from the message")]
    cuisine:          Annotated[Optional[str],   Meta(description="Preferred cuisine, null if not specified")] = None
    max_price:        Annotated[Optional[float], Meta(description="Maximum price in US$, null if not specified")] = None
    max_delivery_min: Annotated[Optional[int],   Meta(description="Maximum delivery time in minutes, null if not specified")] = None


class Recommendation(Struct):
    dish_id:    str
    name:       str
    restaurant: str
    price:      Annotated[float, Meta(description="Price in US$")]
    reason:     Annotated[str,   Meta(description="One sentence explaining why this matches the preferences")]


class RankerOutput(Struct):
    recommendations: Annotated[List[Recommendation], Meta(description="Top 3 dish recommendations")]
    reasoning:       Annotated[str,                  Meta(description="Overall reasoning for the selections")]
```

---

#### PreferenceExtractor

`PreferenceExtractor` parses the user's free-form message into a `UserPreferences` struct. `generation_schema` pins the output shape so the model cannot return anything outside the contract.

```python
class PreferenceExtractor(nn.Agent):
    """Extracts structured food preferences from a natural language message."""
    model             = chat_model
    generation_schema = UserPreferences
```

For the message `"I want something vegan, light, under US$15"`, the output looks like:

```python
# {'restrictions': ['vegan'],
#  'keywords': ['vegan', 'light'],
#  'cuisine': None,
#  'max_price': 15.0,
#  'max_delivery_min': None}
```

---

#### Ranker

`Ranker` receives the extracted preferences alongside the raw search results and selects the top 3 matches. The task template inlines all three inputs so the model sees preferences and candidates in a single prompt.

```python
class Ranker(nn.Agent):
    """Ranks search results and produces the top 3 recommendations with reasoning."""
    model             = chat_model
    instructions      = """
    You are a food recommendation engine. Be concise and opinionated.
    Given user preferences and search results, select the 3 best options.
    For each one explain in one sentence why it matches.
    """
    generation_schema = RankerOutput
    templates         = {"task": "Preferences: {{ preferences }}\nDishes: {{ dish_results }}\nRestaurants: {{ restaurant_results }}"}
```

---

#### StructuredRecommender

`StructuredRecommender` composes the three steps. `DishSearcher` and `RestaurantSearcher` are registered as submodules so they participate in the module tree. `F.bcast_gather` fans the same query out to both searchers concurrently and collects the results.

```python
class StructuredRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor   = PreferenceExtractor()
        self.ranker      = Ranker()
        self.dish_search = DishSearcher()
        self.rest_search = RestaurantSearcher()

    def forward(self, message: str) -> dict:
        preferences = self.extractor(message)

        keywords = preferences.get("keywords") or []
        cuisine  = preferences.get("cuisine") or ""
        query    = " ".join(keywords + ([cuisine] if cuisine else []))

        dish_results, restaurant_results = F.bcast_gather(
            [self.dish_search, self.rest_search],
            query,
        )

        return self.ranker(
            preferences=preferences,
            dish_results=dish_results,
            restaurant_results=restaurant_results,
        )
```

---

#### Usage

```python
recommender = StructuredRecommender()
result      = recommender("I want something vegan, light, under US$15")

print(result["recommendations"])
# [Recommendation(dish_id='D029', name='Vegan Protein Bowl', restaurant='Green & Good',
#                 price=14.0, reason='Matches all restrictions and stays well under the budget.'),
#  ...]
print(result["reasoning"])
# 'Green & Good dominates the top picks given the vegan restriction and $15 ceiling...'
```


---

## Complete Script

??? example "Expand full script"
    ```python
    # /// script
    # dependencies = [
    #   "rapidfuzz",
    #   "typing-extensions",
    # ]
    # ///

    import uuid
    import random
    from typing import List, Literal

    from msgspec import Meta, Struct
    from typing_extensions import Annotated

    import msgflux as mf
    import msgflux.nn as nn
    import msgflux.nn.functional as F

    mf.load_dotenv()
    chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    RAW_RESTAURANTS = [
        {"id": "REST001", "name": "Napoli's Pizza",   "cuisine": "pizza"},
        {"id": "REST002", "name": "Tokyo Garden",     "cuisine": "japanese"},
        {"id": "REST003", "name": "Burger Bros",      "cuisine": "burger"},
        {"id": "REST004", "name": "Southern Table",   "cuisine": "brazilian"},
        {"id": "REST005", "name": "Shawarma Palace",  "cuisine": "arabic"},
        {"id": "REST006", "name": "Green & Good",     "cuisine": "vegan"},
        {"id": "REST007", "name": "Wok House",        "cuisine": "chinese"},
        {"id": "REST008", "name": "Tacos & Co",       "cuisine": "mexican"},
        {"id": "REST009", "name": "Chicken House",    "cuisine": "brazilian"},
        {"id": "REST010", "name": "Pasta Mia",        "cuisine": "italian"},
    ]

    RAW_DISHES = [
        # Napoli's Pizza
        {"restaurant_id": "REST001", "raw_name": "margherita"},
        {"restaurant_id": "REST001", "raw_name": "sausage pizza"},
        {"restaurant_id": "REST001", "raw_name": "four cheese pizza"},
        {"restaurant_id": "REST001", "raw_name": "chicken cream cheese pizza"},
        {"restaurant_id": "REST001", "raw_name": "ham calzone"},
        {"restaurant_id": "REST001", "raw_name": "garlic bread"},
        # Tokyo Garden
        {"restaurant_id": "REST002", "raw_name": "20-piece sushi combo"},
        {"restaurant_id": "REST002", "raw_name": "salmon hand roll"},
        {"restaurant_id": "REST002", "raw_name": "philadelphia roll"},
        {"restaurant_id": "REST002", "raw_name": "salmon sashimi"},
        {"restaurant_id": "REST002", "raw_name": "chicken yakisoba"},
        {"restaurant_id": "REST002", "raw_name": "miso soup"},
        # Burger Bros
        {"restaurant_id": "REST003", "raw_name": "classic burger"},
        {"restaurant_id": "REST003", "raw_name": "double smash burger"},
        {"restaurant_id": "REST003", "raw_name": "crispy chicken sandwich"},
        {"restaurant_id": "REST003", "raw_name": "veggie burger"},
        {"restaurant_id": "REST003", "raw_name": "french fries"},
        {"restaurant_id": "REST003", "raw_name": "chocolate milkshake"},
        # Southern Table
        {"restaurant_id": "REST004", "raw_name": "chicken with okra"},
        {"restaurant_id": "REST004", "raw_name": "bean stew"},
        {"restaurant_id": "REST004", "raw_name": "pork ribs with cassava"},
        {"restaurant_id": "REST004", "raw_name": "tropeiro beans"},
        {"restaurant_id": "REST004", "raw_name": "pequi rice"},
        # Shawarma Palace
        {"restaurant_id": "REST005", "raw_name": "chicken shawarma"},
        {"restaurant_id": "REST005", "raw_name": "beef shawarma"},
        {"restaurant_id": "REST005", "raw_name": "falafel wrap"},
        {"restaurant_id": "REST005", "raw_name": "grilled kafta"},
        {"restaurant_id": "REST005", "raw_name": "arabic platter"},
        # Green & Good
        {"restaurant_id": "REST006", "raw_name": "protein bowl"},
        {"restaurant_id": "REST006", "raw_name": "chickpea burger"},
        {"restaurant_id": "REST006", "raw_name": "açaí bowl"},
        {"restaurant_id": "REST006", "raw_name": "vegan wrap"},
        {"restaurant_id": "REST006", "raw_name": "vegan caesar salad"},
        # Wok House
        {"restaurant_id": "REST007", "raw_name": "sweet and sour chicken"},
        {"restaurant_id": "REST007", "raw_name": "mixed yakisoba"},
        {"restaurant_id": "REST007", "raw_name": "chop suey rice"},
        {"restaurant_id": "REST007", "raw_name": "garlic shrimp"},
        # Tacos & Co
        {"restaurant_id": "REST008", "raw_name": "beef tacos"},
        {"restaurant_id": "REST008", "raw_name": "chicken tacos"},
        {"restaurant_id": "REST008", "raw_name": "beef burrito"},
        {"restaurant_id": "REST008", "raw_name": "cheese quesadilla"},
        {"restaurant_id": "REST008", "raw_name": "nachos with guacamole"},
        # Chicken House
        {"restaurant_id": "REST009", "raw_name": "half grilled chicken"},
        {"restaurant_id": "REST009", "raw_name": "fried chicken platter"},
        {"restaurant_id": "REST009", "raw_name": "chicken bites"},
        {"restaurant_id": "REST009", "raw_name": "chicken sandwich"},
        {"restaurant_id": "REST009", "raw_name": "fried cassava"},
        # Pasta Mia
        {"restaurant_id": "REST010", "raw_name": "spaghetti bolognese"},
        {"restaurant_id": "REST010", "raw_name": "fettuccine alfredo"},
        {"restaurant_id": "REST010", "raw_name": "penne all'arrabbiata"},
        {"restaurant_id": "REST010", "raw_name": "meat lasagna"},
        {"restaurant_id": "REST010", "raw_name": "mushroom risotto"},
        {"restaurant_id": "REST010", "raw_name": "tiramisu"},
    ]

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

    class DishEntry(Struct):
        name:        str
        description: str
        category:    Annotated[
                         Literal["main course", "starter", "dessert", "drink", "side dish"],
                         Meta(description="Dish category")
                     ]
        price:       Annotated[float,      Meta(description="Price in US$ — delivery app range: side $3-7, starter $5-10, main $9-18, dessert $4-8, drink $2-5")]
        tags:        Annotated[List[str],  Meta(description="Cuisine and flavor tags")]
        dietary:     Annotated[List[str],  Meta(description="Dietary tags: vegetarian, vegan, gluten-free, spicy, etc.")]


    class DishEnricher(nn.Agent):
        """Expands a raw dish name into a full catalog entry."""
        model             = chat_model
        system_message    = """
        You are a food catalog specialist for a food delivery platform.
        """
        instructions      = """
        Generate realistic, appetizing catalog entries in English.
        Use typical US food delivery prices: sides $3-7, starters $5-10,
        mains $9-18, desserts $4-8, drinks $2-5.
        """
        generation_schema = DishEntry
        templates         = {"task": "Dish: {{ raw_name }}\nCuisine: {{ cuisine }}"}


    enricher = DishEnricher()

    enriched = F.map_gather(
        enricher,
        args_list=[() for _ in RAW_DISHES],
        kwargs_list=[
            {"raw_name": d["raw_name"], "cuisine": _rest_by_id[d["restaurant_id"]]["cuisine"]}
            for d in RAW_DISHES
        ],
    )

    DISHES = []
    for i, (raw, result) in enumerate(zip(RAW_DISHES, enriched), start=1):
        DISHES.append({
            "id":            f"D{i:03d}",
            "restaurant_id": raw["restaurant_id"],
            **result,
        })

    _dish_by_id = {d["id"]: d for d in DISHES}

    for rest in RESTAURANTS:
        dish_tags = [
            tag
            for d in DISHES
            if d["restaurant_id"] == rest["id"]
            for tag in d.get("tags", [])
        ]
        rest["tags"] = list(dict.fromkeys(dish_tags))[:8]  # deduplicated, top 8

    def _dish_corpus(dishes: list[dict]) -> list[str]:
        entries = []
        for d in dishes:
            rest = _rest_by_id.get(d["restaurant_id"], {})
            tags = " ".join(d.get("tags", []) + d.get("dietary", []))
            entries.append(
                f"{d['id']} | {d['name']} | {rest.get('name', '')} "
                f"(id: {d['restaurant_id']}) | {d.get('description', '')} "
                f"| US${d['price']:.2f} | {tags}"
            )
        return entries


    def _restaurant_corpus(restaurants: list[dict]) -> list[str]:
        return [
            f"{r['id']} | {r['name']} | {r['cuisine']} | "
            f"rating: {r['rating']} | {r['delivery_min']}min | "
            f"mín: US${r['min_order']:.0f} | {' '.join(r['tags'])}"
            for r in restaurants
        ]


    dish_fuzzy = mf.Retriever.fuzzy("rapidfuzz")
    dish_fuzzy.add(_dish_corpus(DISHES))

    restaurant_fuzzy = mf.Retriever.fuzzy("rapidfuzz")
    restaurant_fuzzy.add(_restaurant_corpus(RESTAURANTS))

    class DishSearcher(nn.Searcher):
        """
        Search for dishes by name, description, ingredients, cuisine, or dietary tag.
        Include price constraints and dietary restrictions directly in the query
        (e.g. "vegan under US$35", "gluten-free japanese").
        """
        name      = "search_dishes"
        retriever = dish_fuzzy
        config    = {"top_k": 5}
        templates = {"response": "{% for r in results %}{{ r.data }}\n{% endfor %}"}


    class RestaurantSearcher(nn.Searcher):
        """
        Search for restaurants by name, cuisine type, or tags.
        Include delivery time constraints directly in the query
        (e.g. "japanese fast delivery", "pizza 30 minutes").
        """
        name      = "search_restaurants"
        retriever = restaurant_fuzzy
        config    = {"top_k": 5}
        templates = {"response": "{% for r in results %}{{ r.data }}\n{% endfor %}"}

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
            f"rating: {rest['rating']} | {rest['delivery_min']}min | mín: US${rest['min_order']:.0f}",
            "",
        ]
        for d in dishes:
            dietary = f" [{', '.join(d['dietary'])}]" if d.get("dietary") else ""
            lines.append(f"{d['id']} | {d['name']} — US${d['price']:.2f}{dietary}")
            lines.append(f"     {d.get('description', '')}")
        return "\n".join(lines)    

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
            lines.append(f"  {qty}x {name} — US${price * qty:.2f}")

        lines += ["", f"Total: US${total:.2f}", f"Estimated delivery: {rest['delivery_min']}min"]
        return "\n".join(lines)


    class FoodAssistant(nn.Agent):
        """Food delivery assistant with restaurant and dish search."""
        model           = chat_model
        system_message  = """
        You are a food delivery assistant, similar to iFood or UberEats.
        """
        instructions    = """
        Help the user find and order food through a natural conversation.

        Available tools:
        - search_dishes: search by name, ingredient, cuisine, or dietary tag
        - search_restaurants: search by name or cuisine type
        - get_menu: get the full menu of a specific restaurant
        - place_order: submit the order after user confirmation
        """
        expected_output = """
        - When the request is vague, search both dishes and restaurants.
        - Always show dish ID, name, restaurant, price, and dietary tags.
        - Ask clarifying questions when the user has dietary restrictions.
        - Before calling place_order, confirm the exact items and quantities.
        - If nothing matches, suggest the closest alternatives.
        - When calling place_order, use the dish_id (e.g. "D026") and
          restaurant_id (e.g. "REST005") exactly as shown in the search results.
          Never invent or paraphrase these IDs.
        """
        tools  = [DishSearcher, RestaurantSearcher, get_menu, place_order]
        config = {"verbose": True}


    if __name__ == "__main__":
        assistant = FoodAssistant()
        history   = []

        # Turn 1 — vague request
        user_msg = "I want something vegan under US$15"
        r1 = assistant(user_msg, messages=history)
        history.append(mf.ChatBlock.assist(r1))
        print("Assistant:", r1)

        print()

        # Turn 2 — refinement + confirmation
        user_msg2 = "I'll take the vegan wrap. Place the order."
        r2 = assistant(user_msg2, messages=history)
        history.append(mf.ChatBlock.assist(r2))
        print("Assistant:", r2)

        print()

        # Turn 3 — confirm quantity
        r3 = assistant("1 item.", messages=history)
        print("Assistant:", r3)
    ```

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — tools, multi-turn, system messages
- [nn.Searcher](../learn/nn/searcher.md) — BM25 and semantic retrieval
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Functional API](../learn/nn/functional.md) — `map_gather` and `bcast_gather`
