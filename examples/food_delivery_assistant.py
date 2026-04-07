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
    r1 = assistant("I want something vegan under US$15", messages=history)
    history += [mf.ChatBlock.user("I want something vegan under US$15"), mf.ChatBlock.assist(str(r1))]
    print("Assistant:", r1)

    print()

    # Turn 2 — refinement + confirmation
    r2 = assistant("I'll take the vegan wrap. Place the order.", messages=history)
    history += [mf.ChatBlock.user("I'll take the vegan wrap. Place the order."), mf.ChatBlock.assist(str(r2))]
    print("Assistant:", r2)

    print()

    # Turn 3 — confirm quantity
    r3 = assistant("1 item.", messages=history)
    print("Assistant:", r3)
