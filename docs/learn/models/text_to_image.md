# Text to Image

The `text_to_image` model generates images from text descriptions. These models can create photorealistic images, artwork, illustrations, and more from natural language prompts.

## ✦₊⁺ Overview

Text-to-image models transform textual descriptions into visual content. They enable:

- **Image Generation**: Create images from scratch using text prompts
- **Style Control**: Generate images in specific artistic styles
- **Variations**: Create multiple variations of the same concept
- **Quality Control**: Adjust output quality and resolution

### Common Use Cases

- **Content Creation**: Marketing materials, social media posts
- **Prototyping**: Visual concepts for design projects
- **Art Generation**: Digital artwork and illustrations
- **Product Visualization**: Product mockups and concepts
- **Education**: Visual aids and explanations

## 1. **Quick Start**

### Basic Usage

???+ example

    ```python
    import msgflux as mf

    # Create image generator
    model = mf.Model.text_to_image("openai/gpt-image-1")

    # Generate image
    response = model(prompt="A serene lake at sunset with mountains")

    # Get image URL
    image_url = response.consume()
    print(image_url)  # https://...
    ```

### With Custom Parameters

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    response = model(
        prompt="A futuristic city with flying cars",
        size="1536x1024",  # Landscape
        quality="high",    # High quality
        n=1                # Number of images
    )

    image_url = response.consume()
    ```

## 2. **Supported Providers**

!!! info "Dependencies"
    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

### OpenAI

???+ example

    ```python
    import msgflux as mf

    # gpt-image-1 — best quality/cost balance (recommended)
    model = mf.Model.text_to_image("openai/gpt-image-1")

    # gpt-image-1.5 — latest, superior detail and prompt adherence
    model = mf.Model.text_to_image("openai/gpt-image-1.5")
    ```

### Replicate

???+ example

    ```python
    import msgflux as mf

    # FLUX 2 Flex — high-quality open model
    model = mf.Model.text_to_image("replicate/black-forest-labs/flux-2-flex")
    ```

### ImageRouter

???+ example

    ```python
    import msgflux as mf

    # Router to multiple image generation providers
    model = mf.Model.text_to_image("imagerouter/default")
    ```

## 3. **Image Sizes**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    # Square (default)
    response = model(prompt="A cat wearing sunglasses", size="1024x1024")

    # Landscape
    response = model(prompt="A panoramic mountain view", size="1536x1024")

    # Portrait
    response = model(prompt="A full-length portrait", size="1024x1536")

    # Auto (model decides based on prompt)
    response = model(prompt="A wide forest path", size="auto")
    ```

## 4. **Quality Settings**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    # High quality — more detail, higher cost
    response = model(prompt="A detailed landscape", quality="high")

    # Medium quality — balanced
    response = model(prompt="A detailed landscape", quality="medium")

    # Low quality — faster, cheaper, good for drafts
    response = model(prompt="A detailed landscape", quality="low")
    ```

## 5. **Response Formats**

### URL Response (Default)

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    response = model(
        prompt="A sunset over the ocean",
        response_format="url"  # Default
    )

    # Get URL
    image_url = response.consume()
    print(image_url)  # https://...

    # Download image
    import requests
    img_data = requests.get(image_url).content
    with open("image.png", "wb") as f:
        f.write(img_data)
    ```

### Base64 Response

???+ example

    ```python
    import msgflux as mf
    import base64

    model = mf.Model.text_to_image("openai/gpt-image-1")

    response = model(
        prompt="A colorful abstract painting",
        response_format="base64"
    )

    # Decode and save
    img_data = base64.b64decode(response.consume())
    with open("image.png", "wb") as f:
        f.write(img_data)
    ```

## 6. **Multiple Images**

Generate multiple variations in a single call:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    response = model(
        prompt="A cute robot",
        n=4,  # Generate 4 variations
        size="1024x1024"
    )

    # Get all image URLs
    images = response.consume()
    print(f"Generated {len(images)} images")

    for i, url in enumerate(images):
        print(f"Image {i+1}: {url}")
    ```

## 7. **Background Control**

Control background transparency:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    # Transparent background (useful for product shots)
    response = model(prompt="A red apple", background="transparent")

    # Opaque background
    response = model(prompt="A red apple", background="opaque")

    # Auto — model decides (default)
    response = model(prompt="A red apple", background="auto")
    ```

## 8. **Content Moderation**

Control content filtering:

???+ example

    ```python
    import msgflux as mf

    # Auto moderation (default)
    model = mf.Model.text_to_image("openai/gpt-image-1", moderation="auto")

    # Low moderation — more permissive
    model = mf.Model.text_to_image("openai/gpt-image-1", moderation="low")

    response = model(prompt="Your prompt here")
    ```

## 9. **Async Support**

Generate images concurrently with `F.map_gather`:

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    model = mf.Model.text_to_image("openai/gpt-image-1")

    prompts = [
        "A serene lake",
        "A bustling city",
        "A quiet forest"
    ]

    results = F.map_gather(
        model,
        args_list=[(p,) for p in prompts]
    )

    for prompt, result in zip(prompts, results):
        print(f"{prompt}: {result.consume()}")
    ```

## 10. **Response Metadata**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    response = model(
        prompt="A beautiful sunset",
        size="1024x1024",
        quality="high"
    )

    # Access metadata
    print(response.metadata)
    # {'created': 1234567890, 'content_filter_results': {...}}

    # Access the model profile
    print(model.profile.cost.input_per_million)
    ```

## 11. **Prompt Engineering**

### Effective Prompts

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    # Good - Specific and detailed
    response = model(
        prompt="""A professional photograph of a modern minimalist living room with:
        - Large floor-to-ceiling windows
        - Natural light streaming in
        - Scandinavian furniture
        - Indoor plants
        - Neutral color palette
        - Shot with a wide-angle lens"""
    )

    # Less effective - Too vague
    response = model(prompt="A nice room")
    ```

### Style Specifications

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    # Different art styles
    styles = {
        "photorealistic": "A photorealistic portrait of a woman, studio lighting, 85mm lens",
        "oil painting": "An oil painting of a countryside landscape in the style of Van Gogh",
        "digital art": "A digital art illustration of a fantasy castle, vibrant colors",
        "3d render": "A 3D render of a futuristic car, octane render, high detail",
        "sketch": "A pencil sketch of a cat, detailed crosshatching",
    }

    for style_name, prompt in styles.items():
        response = model(prompt=prompt)
        print(f"{style_name}: {response.consume()}")
    ```

### Composition Control

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    prompts = [
        # Rule of thirds
        "A lone tree positioned on the right third of the image, sunset on the left",

        # Centered composition
        "A symmetrical view of a building, centered composition, front view",

        # Leading lines
        "A road leading into the distance toward mountains, vanishing point",

        # Foreground/background
        "A flower in sharp focus in the foreground, blurred forest in background"
    ]

    for prompt in prompts:
        response = model(prompt=prompt)
        print(response.consume())
    ```

## 12. **Error Handling**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_image("openai/gpt-image-1")

    try:
        response = model(prompt="A landscape")
        image_url = response.consume()
    except ImportError:
        print("Provider not installed")
    except ValueError as e:
        print(f"Invalid parameters: {e}")
    except Exception as e:
        print(f"Generation failed: {e}")
        # Common errors:
        # - Content policy violation
        # - Rate limits
        # - Network issues
    ```
