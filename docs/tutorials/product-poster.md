# Product Poster Generator

Build a pipeline that creates professional marketing posters automatically: scrape a product page, analyze it with a vision model, and generate a polished poster image.

## What You'll Build

```
Product URL
    │
    ▼
HTML Parser ──────────────► product text + image URL
                                        │
                            ┌───────────┴───────────┐
                            ▼                       ▼
                       product_text          image download
                            │                       │
                            └──────────┬────────────┘
                                       ▼
                                  Vision Agent
                                 (text + image)
                                       │
                                       ▼
                                 poster_prompt
                                       │
                                       ▼
                                   MediaMaker
                                       │
                                       ▼
                                   poster.png
```

---

## Setup

```bash
pip install msgflux[openai] beautifulsoup4 httpx
```

```bash
export OPENAI_API_KEY="sk-..."
export IMAGEROUTER_API_KEY="..."
```

---

## Step 1: Parse the Product Page

`Parser.html()` fetches the page and returns clean text plus all image URLs:

```python
import msgflux as mf

parser = mf.Parser.html("beautifulsoup", extract_images=True)

# Pass a URL directly — the parser fetches and converts to Markdown
response = parser("https://your-shop.com/products/some-sku")

product_text = response.data["text"]
product_images = response.data["images"]  # [{"alt": "...", "url": "..."}, ...]

print(f"Extracted {len(product_text)} characters of text")
print(f"Found {len(product_images)} images")
```

!!! tip
    For sites that require custom headers (e.g., a `User-Agent`), fetch the HTML
    yourself with `httpx` and pass the raw HTML string to the parser instead of
    the URL — the parser detects `<!DOCTYPE` and `<html` automatically.

---

## Step 2: Download the Product Image

The parser gives you image URLs; download the first one as bytes to feed into the vision model:

```python
import httpx

def fetch_image(url: str) -> bytes:
    """Download an image from a URL."""
    r = httpx.get(url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.content

product_image = fetch_image(product_images[0]["url"])
```

---

## Step 3: Generate a Poster Prompt with a Vision Agent

An `Agent` backed by a vision model (`gpt-4o`) reads the product text and inspects the product image to produce a detailed poster-generation prompt:

```python
import msgflux.nn as nn

class PosterPromptAgent(nn.Agent):
    """Analyzes a product and crafts a poster generation prompt."""
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    instructions = """
    You are an expert creative director specializing in product advertising.

    Given a product description and its image, create a highly detailed prompt
    for generating a professional marketing poster. Include:
    - Product name and key selling points
    - Visual style (e.g., minimalist, bold, luxury, playful)
    - Lighting and mood
    - Background and composition
    - Color palette
    - Typography hints (tagline, price, brand name placement)

    Return only the image generation prompt — nothing else.
    """
    message_fields = {
        "task_inputs": "product_text",
        "task_multimodal_inputs": {"image": "product_image"},
    }
    response_mode = "poster_prompt"
```

---

## Step 4: Generate the Poster

A `MediaMaker` takes the prompt and calls an image model to produce the poster bytes:

```python
class PosterMaker(nn.MediaMaker):
    """Generates a marketing poster from a descriptive prompt."""
    model = mf.Model.text_to_image("openai/gpt-image-1.5")
    message_fields = {"task_inputs": "poster_prompt"}
    response_mode = "poster"
    negative_prompt = "blurry, distorted, low quality, pixelated, watermark, text artifacts"
```

!!! note
    You can swap the image model for any `TextToImageModel` — `openai/dall-e-3`,
    `openai/gpt-image-1.5`, or any model available through ImageRouter.

---

## Step 5: Compose the Pipeline

Wire the two modules with `Inline` so they share a single `Message` object:

```python
from msgflux import Message, Inline

pipeline = Inline(
    "prompt_agent -> poster_maker",
    {
        "prompt_agent": PosterPromptAgent(),
        "poster_maker": PosterMaker(),
    },
)

msg = Message()
msg.product_text = product_text
msg.product_image = product_image

pipeline(msg)

print("Poster prompt:\n", msg.poster_prompt)

with open("poster.png", "wb") as f:
    f.write(msg.poster)

print("Saved to poster.png")
```

---

## Complete Example

```python
import httpx
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Inline


# ── Helpers ────────────────────────────────────────────────────────────────

def fetch_product(url: str) -> tuple[str, bytes]:
    """Scrape a product page and download its main image."""
    parser = mf.Parser.html("beautifulsoup", extract_images=True)
    parsed = parser(url)

    text = parsed.data["text"]
    images = parsed.data["images"]

    if not images:
        raise ValueError("No images found on the product page")

    r = httpx.get(images[0]["url"], follow_redirects=True, timeout=30)
    r.raise_for_status()

    return text, r.content


# ── Modules ─────────────────────────────────────────────────────────────────

class PosterPromptAgent(nn.Agent):
    """Analyzes a product and crafts a poster generation prompt."""
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    instructions = """
    You are an expert creative director specializing in product advertising.
    Given a product description and its image, create a highly detailed prompt
    for generating a professional marketing poster.
    Include: visual style, lighting, mood, composition, color palette, and
    typography hints. Return only the image generation prompt.
    """
    message_fields = {
        "task_inputs": "product_text",
        "task_multimodal_inputs": {"image": "product_image"},
    }
    response_mode = "poster_prompt"


class PosterMaker(nn.MediaMaker):
    """Generates a marketing poster from a descriptive prompt."""
    model = mf.Model.text_to_image("openai/gpt-image-1.5")
    message_fields = {"task_inputs": "poster_prompt"}
    response_mode = "poster"
    negative_prompt = "blurry, distorted, low quality, pixelated, watermark"


# ── Run ──────────────────────────────────────────────────────────────────────

product_url = "https://your-shop.com/products/some-sku"

product_text, product_image = fetch_product(product_url)

pipeline = Inline(
    "prompt_agent -> poster_maker",
    {
        "prompt_agent": PosterPromptAgent(),
        "poster_maker": PosterMaker(),
    },
)

msg = Message()
msg.product_text = product_text
msg.product_image = product_image

pipeline(msg)

print("Prompt used:\n", msg.poster_prompt)

with open("poster.png", "wb") as f:
    f.write(msg.poster)

print("Poster saved to poster.png")
```

---

## Async Version

```python
import asyncio
import httpx
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Inline


async def main():
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        parser = mf.Parser.html("beautifulsoup", extract_images=True)
        parsed = parser("https://your-shop.com/products/some-sku")

        img_url = parsed.data["images"][0]["url"]
        r = await client.get(img_url)
        r.raise_for_status()

        msg = Message()
        msg.product_text = parsed.data["text"]
        msg.product_image = r.content

    pipeline = Inline(
        "prompt_agent -> poster_maker",
        {
            "prompt_agent": PosterPromptAgent(),
            "poster_maker": PosterMaker(),
        },
    )
    await pipeline.acall(msg)

    with open("poster.png", "wb") as f:
        f.write(msg.poster)

    print("Poster saved to poster.png")


asyncio.run(main())
```
