# Product Poster Generator

Build a pipeline that creates professional marketing posters automatically: scrape a product page, analyze it with a vision model, and generate a polished poster image.

## What You'll Build

```
Product URL
    │
    ▼
ProductScraper ────────────► product text + image bytes
                                        │
                                        ▼
                                  PosterPromptAgent
                                 (text + image → prompt)
                                        │
                                        ▼
                                    PosterMaker
                                  (prompt → image)
                                        │
                                        ▼
                                    poster.png
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1: Scrape the Product Page

Wrap the scraping logic in a module with `__call__` and `acall` so it fits naturally into an `Inline` pipeline:

```python
import base64
from urllib.parse import urljoin

import httpx
import msgflux as mf
import msgflux.nn as nn


class ProductScraper:
    """Fetches a product page and downloads its main image."""

    def _fetch(self, url: str) -> tuple[str, bytes]:
        r = httpx.get(url, follow_redirects=True, timeout=30)
        r.raise_for_status()

        parser = mf.Parser.html("beautifulsoup", extract_images=True)
        parsed = parser(r.content)

        text = parsed.data["text"]
        images = parsed.data["images"]

        if not images:
            raise ValueError("No images found on the product page")

        image_url = urljoin(url, images[0]["url"])
        img = httpx.get(image_url, follow_redirects=True, timeout=30)
        img.raise_for_status()

        return text, img.content

    async def _afetch(self, url: str) -> tuple[str, bytes]:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()

            parser = mf.Parser.html("beautifulsoup", extract_images=True)
            parsed = parser(r.content)

            text = parsed.data["text"]
            images = parsed.data["images"]

            if not images:
                raise ValueError("No images found on the product page")

            image_url = urljoin(url, images[0]["url"])
            img = await client.get(image_url)
            img.raise_for_status()

        return text, img.content

    def __call__(self, msg: mf.Message) -> mf.Message:
        msg.product_text, msg.product_image = self._fetch(msg.product_url)
        return msg

    async def acall(self, msg: mf.Message) -> mf.Message:
        msg.product_text, msg.product_image = await self._afetch(msg.product_url)
        return msg
```

!!! tip
    For sites that require custom headers (e.g., a `User-Agent`), add them to the
    `httpx.get` calls inside `_fetch`. The parser accepts raw `bytes` directly —
    pass `r.content` instead of `r.text` to avoid extension-validation issues.

---

## Step 2: Generate a Poster Prompt with a Vision Agent

An `Agent` backed by a vision model reads the product text and inspects the product image to produce a detailed poster-generation prompt:

```python
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
        "task": "product_text",
        "task_multimodal": {"image": "product_image"},
    }
    response_mode = "poster_prompt"
```

---

## Step 3: Generate the Poster

A `MediaMaker` takes the prompt and calls an image model to produce the poster bytes:

```python
class PosterMaker(nn.MediaMaker):
    """Generates a marketing poster from a descriptive prompt."""
    model = mf.Model.text_to_image("openai/gpt-image-1.5")
    message_fields = {"task": "poster_prompt"}
    response_mode = "poster"
```

!!! note
    You can swap the image model for any `TextToImageModel` — `openai/dall-e-3`,
    `openai/gpt-image-1.5`, or any model available through ImageRouter.

---

## Step 4: Compose the Pipeline

Wire all three steps with `Inline` so they share a single `Message` object:

```python
flux = mf.Inline(
    "scraper -> prompt_agent -> poster_maker",
    {
        "scraper":      ProductScraper(),
        "prompt_agent": PosterPromptAgent(),
        "poster_maker": PosterMaker(),
    },
)

msg = mf.Message()
msg.product_url = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"

flux(msg)

print("Poster prompt:\n", msg.poster_prompt)

poster = msg.poster
if isinstance(poster, str):
    poster = base64.b64decode(poster)

with open("poster.png", "wb") as f:
    f.write(poster)

print("Saved to poster.png")
```

---

## Complete Example

```python
import asyncio
import base64
from urllib.parse import urljoin

import httpx
import msgflux as mf
import msgflux.nn as nn


PRODUCT_URL = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"


class ProductScraper:
    """Fetches a product page and downloads its main image."""

    def _fetch(self, url: str) -> tuple[str, bytes]:
        r = httpx.get(url, follow_redirects=True, timeout=30)
        r.raise_for_status()

        parser = mf.Parser.html("beautifulsoup", extract_images=True)
        parsed = parser(r.content)

        text = parsed.data["text"]
        images = parsed.data["images"]

        if not images:
            raise ValueError("No images found on the product page")

        image_url = urljoin(url, images[0]["url"])
        img = httpx.get(image_url, follow_redirects=True, timeout=30)
        img.raise_for_status()

        return text, img.content

    async def _afetch(self, url: str) -> tuple[str, bytes]:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()

            parser = mf.Parser.html("beautifulsoup", extract_images=True)
            parsed = parser(r.content)

            text = parsed.data["text"]
            images = parsed.data["images"]

            if not images:
                raise ValueError("No images found on the product page")

            image_url = urljoin(url, images[0]["url"])
            img = await client.get(image_url)
            img.raise_for_status()

        return text, img.content

    def __call__(self, msg: mf.Message) -> mf.Message:
        msg.product_text, msg.product_image = self._fetch(msg.product_url)
        return msg

    async def acall(self, msg: mf.Message) -> mf.Message:
        msg.product_text, msg.product_image = await self._afetch(msg.product_url)
        return msg


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
        "task": "product_text",
        "task_multimodal": {"image": "product_image"},
    }
    response_mode = "poster_prompt"


class PosterMaker(nn.MediaMaker):
    """Generates a marketing poster from a descriptive prompt."""
    model = mf.Model.text_to_image("openai/gpt-image-1.5")
    message_fields = {"task": "poster_prompt"}
    response_mode = "poster"


def save_poster(data: str | bytes, path: str = "poster.png") -> None:
    if isinstance(data, str):
        data = base64.b64decode(data)
    with open(path, "wb") as f:
        f.write(data)


def main():
    flux = mf.Inline(
        "scraper -> prompt_agent -> poster_maker",
        {
            "scraper":      ProductScraper(),
            "prompt_agent": PosterPromptAgent(),
            "poster_maker": PosterMaker(),
        },
    )

    msg = mf.Message()
    msg.product_url = PRODUCT_URL

    flux(msg)

    print("Prompt used:\n", msg.poster_prompt)
    save_poster(msg.poster)
    print("Poster saved to poster.png")


async def amain():
    flux = mf.Inline(
        "scraper -> prompt_agent -> poster_maker",
        {
            "scraper":      ProductScraper(),
            "prompt_agent": PosterPromptAgent(),
            "poster_maker": PosterMaker(),
        },
    )

    msg = mf.Message()
    msg.product_url = PRODUCT_URL

    await flux.acall(msg)

    print("Prompt used:\n", msg.poster_prompt)
    save_poster(msg.poster)
    print("Poster saved to poster.png")


main()
# asyncio.run(amain())
```

---

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — vision inputs, instructions, and message fields
- [nn.MediaMaker](../learn/nn/mediamaker.md) — image and video generation modules
- [Inline](../learn/inline.md) — composing multi-step pipelines with a shared Message
