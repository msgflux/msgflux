# Product Poster Generator

<span class="tag tag-purple">Intermediate</span><span class="tag tag-gray">Multimodal</span><span class="tag tag-gray">MediaMaker</span>

Build a pipeline that creates professional marketing posters automatically: scrape a product page, analyze it with a vision model, and generate a polished poster image.

---

## The Problem

Product marketing at scale is a bottleneck. A team managing hundreds of SKUs needs a poster for each one — every poster requiring a designer to research the product, extract the selling points, choose a visual direction, and write a brief before a single pixel is generated.

The catalog already contains everything needed: product pages with text descriptions and images. The only missing step is converting that existing data into a prompt that an image model can turn into a poster.

---

## The Plan

We will build a three-step pipeline. A scraper fetches the product page and downloads its main image. A vision agent reads both — the product text and the image — and produces a detailed image generation prompt tailored to the product's category and aesthetic. A media maker feeds that prompt to an image model and returns the finished poster.

All three steps share a single `Message` via `Inline`, so no intermediate state needs to be passed manually between them.

---

## Architecture

```
Product URL
      │
      ▼
 ProductScraper
      │  product_text + product_image
      ▼
 PosterPromptAgent  (vision model)
      │  poster_prompt
      ▼
 PosterMaker  (image model)
      │
      ▼
 poster.png
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

```bash
pip install httpx beautifulsoup4
```

---

## Step 1 — Models

```python
import base64
from urllib.parse import urljoin

import httpx
import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()

vision_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
image_model  = mf.Model.text_to_image("openai/gpt-image-1.5")
```

`vision_model` must be a vision-capable chat model — it receives both the product text and the product image. `image_model` can be swapped for any `TextToImageModel` supported by ImageRouter.

---

## Step 2 — Scraper

`ProductScraper` fetches the page, parses the HTML, and downloads the first product image. It exposes `__call__` and `acall` so it plugs directly into an `Inline` pipeline alongside `nn.Module` components.

```python
class ProductScraper:
    """Fetches a product page and downloads its main image."""

    def _fetch(self, url: str) -> tuple[str, bytes]:
        r = httpx.get(url, follow_redirects=True, timeout=30)
        r.raise_for_status()

        parser = mf.Parser.html("beautifulsoup", extract_images=True)
        parsed = parser(r.content)

        text   = parsed.data["text"]
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

            text   = parsed.data["text"]
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
    For sites that require a custom `User-Agent` or session headers, add them to the `httpx.get` calls inside `_fetch`. The parser accepts raw `bytes` — pass `r.content`, not `r.text`.

---

## Step 3 — Prompt Agent

`PosterPromptAgent` receives the product text as its task and the product image as a multimodal attachment. Its output — a detailed image generation prompt — is written directly to `msg.poster_prompt`.

```python
class PosterPromptAgent(nn.Agent):
    """Analyzes a product and crafts a poster generation prompt."""
    model        = vision_model
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
        "task":           "product_text",
        "task_multimodal": {"image": "product_image"},
    }
    response_mode = "poster_prompt"
```

---

## Step 4 — Poster Maker

`PosterMaker` reads `msg.poster_prompt` and calls the image model. The result is stored on `msg.poster` as base64-encoded bytes.

```python
class PosterMaker(nn.MediaMaker):
    """Generates a marketing poster from a descriptive prompt."""
    model          = image_model
    message_fields = {"task": "poster_prompt"}
    response_mode  = "poster"
```

---

## Step 5 — Pipeline

Wire the three steps with `Inline`. All components share the same `Message` object — no manual wiring between steps.

```python
flux = mf.Inline(
    "scraper -> prompt_agent -> poster_maker",
    {
        "scraper":      ProductScraper(),
        "prompt_agent": PosterPromptAgent(),
        "poster_maker": PosterMaker(),
    },
)
```

---

## Examples

???+ example

    === "Sync"

        ```python
        msg = mf.Message()
        msg.product_url = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"

        flux(msg)

        print("Prompt used:\n", msg.poster_prompt)

        poster = msg.poster
        if isinstance(poster, str):
            poster = base64.b64decode(poster)

        with open("poster.png", "wb") as f:
            f.write(poster)

        print("Saved to poster.png")
        ```

    === "Async"

        ```python
        import asyncio

        async def main():
            msg = mf.Message()
            msg.product_url = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"

            await flux.acall(msg)

            poster = msg.poster
            if isinstance(poster, str):
                poster = base64.b64decode(poster)

            with open("poster.png", "wb") as f:
                f.write(poster)

        asyncio.run(main())
        ```

    === "Batch — multiple products"

        ```python
        URLS = [
            "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
            "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
        ]

        for i, url in enumerate(URLS, 1):
            msg = mf.Message()
            msg.product_url = url
            flux(msg)

            poster = msg.poster
            if isinstance(poster, str):
                poster = base64.b64decode(poster)

            path = f"poster_{i:02d}.png"
            with open(path, "wb") as f:
                f.write(poster)
            print(f"Saved {path}")
        ```

---

## Extending

### Applying a style reference image

Pass a second image to `PosterPromptAgent` to guide the visual style. Add a `style_image` field to the message before calling the pipeline:

```python
msg.style_image = open("brand_style.jpg", "rb").read()
```

Then extend `message_fields` on `PosterPromptAgent`:

```python
message_fields = {
    "task":            "product_text",
    "task_multimodal": {"image": "product_image", "style": "style_image"},
}
```

Update the instructions to mention the style reference and the model will incorporate it.

### Saving with a custom filename

Wrap the save step in a helper and derive the filename from the URL:

```python
from pathlib import Path
from urllib.parse import urlparse

def save_poster(msg: mf.Message, out_dir: str = ".") -> str:
    slug = Path(urlparse(msg.product_url).path).parent.name
    path = f"{out_dir}/{slug}.png"
    data = msg.poster
    if isinstance(data, str):
        data = base64.b64decode(data)
    with open(path, "wb") as f:
        f.write(data)
    return path
```

---

## Complete Script

```python
import asyncio
import base64
from urllib.parse import urljoin

import httpx
import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()

PRODUCT_URL = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"


# Models
vision_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
image_model  = mf.Model.text_to_image("openai/gpt-image-1.5")


# Scraper
class ProductScraper:
    """Fetches a product page and downloads its main image."""

    def _fetch(self, url: str) -> tuple[str, bytes]:
        r = httpx.get(url, follow_redirects=True, timeout=30)
        r.raise_for_status()
        parser = mf.Parser.html("beautifulsoup", extract_images=True)
        parsed = parser(r.content)
        text   = parsed.data["text"]
        images = parsed.data["images"]
        if not images:
            raise ValueError("No images found on the product page")
        img = httpx.get(urljoin(url, images[0]["url"]), follow_redirects=True, timeout=30)
        img.raise_for_status()
        return text, img.content

    async def _afetch(self, url: str) -> tuple[str, bytes]:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()
            parser = mf.Parser.html("beautifulsoup", extract_images=True)
            parsed = parser(r.content)
            text   = parsed.data["text"]
            images = parsed.data["images"]
            if not images:
                raise ValueError("No images found on the product page")
            img = await client.get(urljoin(url, images[0]["url"]))
            img.raise_for_status()
        return text, img.content

    def __call__(self, msg: mf.Message) -> mf.Message:
        msg.product_text, msg.product_image = self._fetch(msg.product_url)
        return msg

    async def acall(self, msg: mf.Message) -> mf.Message:
        msg.product_text, msg.product_image = await self._afetch(msg.product_url)
        return msg


# Agents
class PosterPromptAgent(nn.Agent):
    """Analyzes a product and crafts a poster generation prompt."""
    model        = vision_model
    instructions = """
    You are an expert creative director specializing in product advertising.
    Given a product description and its image, create a highly detailed prompt
    for generating a professional marketing poster.
    Include: visual style, lighting, mood, composition, color palette, and
    typography hints. Return only the image generation prompt.
    """
    message_fields = {
        "task":            "product_text",
        "task_multimodal": {"image": "product_image"},
    }
    response_mode = "poster_prompt"


class PosterMaker(nn.MediaMaker):
    """Generates a marketing poster from a descriptive prompt."""
    model          = image_model
    message_fields = {"task": "poster_prompt"}
    response_mode  = "poster"


# Pipeline
flux = mf.Inline(
    "scraper -> prompt_agent -> poster_maker",
    {
        "scraper":      ProductScraper(),
        "prompt_agent": PosterPromptAgent(),
        "poster_maker": PosterMaker(),
    },
)


# Run
def save_poster(data: str | bytes, path: str = "poster.png") -> None:
    if isinstance(data, str):
        data = base64.b64decode(data)
    with open(path, "wb") as f:
        f.write(data)


msg = mf.Message()
msg.product_url = PRODUCT_URL

flux(msg)

print("Prompt used:\n", msg.poster_prompt)
save_poster(msg.poster)
print("Poster saved to poster.png")
```

---

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — vision inputs, instructions, and message fields
- [nn.MediaMaker](../learn/nn/mediamaker.md) — image and video generation modules
- [Inline](../learn/inline.md) — composing multi-step pipelines with a shared Message
