# Image Editing

The `image_text_to_image` model edits existing images using text descriptions. This enables modifications, inpainting, object removal, style changes, and creative transformations of existing visual content.

## ✦₊⁺ Overview

Image editing models take an existing image and modify it based on text prompts. They enable:

- **Image Modification**: Edit specific parts of an image
- **Inpainting**: Fill in or replace masked regions
- **Object Removal**: Remove unwanted elements
- **Style Transfer**: Change artistic style while preserving content
- **Creative Editing**: Add, modify, or transform elements

### Common Use Cases

- **Photo Editing**: Remove objects, change backgrounds
- **Product Photography**: Modify product colors, backgrounds, settings
- **Content Creation**: Transform existing images for marketing
- **Image Restoration**: Fill in missing or damaged areas
- **Creative Variations**: Generate alternative versions of images

## 1. **Quick Start**

### Basic Usage

???+ example

    ```python
    import msgflux as mf

    # Create image editor
    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Edit image
    response = model(
        prompt="Add a sunset sky",
        image="path/to/image.png"
    )

    # Get edited image URL
    edited_url = response.consume()
    print(edited_url)  # https://...
    ```

### With Mask (Inpainting)

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Edit only masked area
    response = model(
        prompt="A wooden table",
        image="room.png",
        mask="table_mask.png"  # Transparent areas will be edited
    )

    edited_url = response.consume()
    ```

## 2. **Supported Providers**

!!! info "Dependencies"
    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

### OpenAI

???+ example

    ```python
    import msgflux as mf

    # Latest model — precise editing with logo and face preservation
    model = mf.Model.image_text_to_image("openai/gpt-image-1.5")

    # Stable release
    model = mf.Model.image_text_to_image("openai/gpt-image-1")
    ```

### Replicate

???+ example

    ```python
    import msgflux as mf

    # Various Replicate models support image editing
    model = mf.Model.image_text_to_image("replicate/stability-ai/stable-diffusion-inpainting")
    ```

## 3. **How Image Editing Works**

### Without Mask

When no mask is provided, the model edits the entire image:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Edit entire image
    response = model(
        prompt="Make it look like a watercolor painting",
        image="photo.jpg"
    )
    ```

### With Mask (Inpainting)

Masks define which areas to edit:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Only edit masked areas
    response = model(
        prompt="A blue sky with clouds",
        image="landscape.png",
        mask="sky_mask.png"  # Transparent PNG showing sky area
    )
    ```

The mask is a hint to the model alongside the prompt — masking with `gpt-image-1` is entirely prompt-based. The image must be a PNG and the mask (if provided) the same dimensions as the input.

## 4. **Image Input Formats**

The `image` parameter accepts multiple formats:

???+ example

    === "File Path"

        ```python
        import msgflux as mf

        model = mf.Model.image_text_to_image("openai/gpt-image-1")

        response = model(
            prompt="Change background to beach",
            image="/path/to/photo.png"
        )
        ```

    === "URL"

        ```python
        import msgflux as mf

        model = mf.Model.image_text_to_image("openai/gpt-image-1")

        response = model(
            prompt="Add winter atmosphere",
            image="https://example.com/photo.jpg"
        )
        ```

    === "Base64"

        ```python
        import msgflux as mf
        import base64

        with open("photo.png", "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        model = mf.Model.image_text_to_image("openai/gpt-image-1")

        response = model(
            prompt="Make it artistic",
            image=f"data:image/png;base64,{img_data}"
        )
        ```

## 5. **Response Formats**

### URL Response (Default)

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    response = model(
        prompt="Add flowers in foreground",
        image="garden.png",
        response_format="url"  # Default
    )

    # Get URL
    url = response.consume()
    print(url)  # https://...

    # Download
    import requests
    img_data = requests.get(url).content
    with open("edited.png", "wb") as f:
        f.write(img_data)
    ```

### Base64 Response

???+ example

    ```python
    import msgflux as mf
    import base64

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    response = model(
        prompt="Change to evening lighting",
        image="scene.png",
        response_format="base64"
    )

    # Get base64 data
    b64_data = response.consume()

    # Decode and save
    img_data = base64.b64decode(b64_data)
    with open("edited.png", "wb") as f:
        f.write(img_data)
    ```

## 6. **Multiple Variations**

Generate multiple edited versions:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Generate 4 variations
    response = model(
        prompt="Add dramatic storm clouds",
        image="landscape.jpg",
        n=4  # Number of variations
    )

    # Get all URLs
    edited_images = response.consume()
    print(f"Generated {len(edited_images)} variations")

    for i, url in enumerate(edited_images):
        print(f"Variation {i+1}: {url}")
    ```

## 7. **Image Size Requirements**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Image requirements:
    # - PNG, WEBP, or JPEG format
    # - Less than 25MB
    # - If using mask, same dimensions as the image
    # Supported output sizes: 1024x1024, 1536x1024, 1024x1536, auto

    response = model(
        prompt="Edit the background",
        image="photo.png",
        size="1024x1024"
    )
    ```

## 8. **Creating Masks**

With `gpt-image-1`, masking is **prompt-based**: the mask tells the model *where* to focus, and the prompt tells it *what* to do.

### Programmatic Mask Creation

???+ example

    ```python
    from PIL import Image, ImageDraw

    # Create a mask the same size as the input image
    # White = keep, Black = edit region (for gpt-image-1, this is a hint)
    img = Image.open("photo.png")
    mask = Image.new("RGBA", img.size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)

    # Mark the region to edit as transparent
    draw.rectangle([200, 100, 800, 500], fill=(0, 0, 0, 0))

    mask.save("edit_mask.png")
    ```

### Prompt-Only Editing (no mask)

With `gpt-image-1` you can edit without a mask — the model infers the region from the prompt:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # The model understands "the background" from the prompt alone
    response = model(
        prompt="Replace the background with a sunset beach",
        image="portrait.png"
    )
    ```

## 9. **Common Editing Tasks**

???+ example

    === "Background Replacement"

        ```python
        import msgflux as mf

        model = mf.Model.image_text_to_image("openai/gpt-image-1")

        response = model(
            prompt="Professional studio background with soft lighting",
            image="portrait.png",
            mask="background_mask.png"  # Background area transparent
        )
        ```

    === "Object Removal"

        ```python
        import msgflux as mf

        model = mf.Model.image_text_to_image("openai/gpt-image-1")

        response = model(
            prompt="Natural grass lawn",
            image="yard.png",
            mask="object_mask.png"  # Object area transparent
        )
        ```

    === "Style Transfer"

        ```python
        import msgflux as mf

        model = mf.Model.image_text_to_image("openai/gpt-image-1")

        response = model(
            prompt="Oil painting in impressionist style, vibrant colors",
            image="photo.jpg"
            # No mask = edit entire image
        )
        ```

    === "Adding Elements"

        ```python
        import msgflux as mf

        model = mf.Model.image_text_to_image("openai/gpt-image-1")

        response = model(
            prompt="A red sports car parked",
            image="driveway.png",
            mask="parking_spot_mask.png"
        )
        ```

    === "Lighting Adjustments"

        ```python
        import msgflux as mf

        model = mf.Model.image_text_to_image("openai/gpt-image-1")

        response = model(
            prompt="Warm sunset lighting, golden hour atmosphere",
            image="scene.jpg"
        )
        ```

## 10. **Async Support**

Edit images asynchronously:

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    edits = [
        ("photo1.png", "Add sunset sky"),
        ("photo2.png", "Change to winter scene"),
        ("photo3.png", "Make it vintage style")
    ]

    results = F.map_gather(
        model,
        args_list=[(prompt, img) for img, prompt in edits]
    )

    for (img, prompt), result in zip(edits, results):
        print(f"{img} ({prompt}): {result.consume()}")
    ```

## 11. **Batch Processing**

Edit multiple images:

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    images = ["photo1.png", "photo2.png", "photo3.png"]
    prompt = "Professional studio background"

    # Process in parallel
    results = F.map_gather(
        model,
        args_list=[
            (prompt, img) for img in images
        ]
    )

    # Get all edited URLs
    edited_urls = [r.consume() for r in results]

    for original, edited in zip(images, edited_urls):
        print(f"{original} -> {edited}")
    ```

## 12. **Error Handling**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    try:
        response = model(
            prompt="Edit this image",
            image="photo.png",
            mask="mask.png"
        )
        url = response.consume()
    except ImportError:
        print("Provider not installed")
    except ValueError as e:
        print(f"Invalid parameters: {e}")
        # Common issues:
        # - Image too large (>25MB)
        # - Mask doesn't match image dimensions
        # - Invalid image format
    except Exception as e:
        print(f"Edit failed: {e}")
        # Common errors:
        # - Content policy violation
        # - Rate limits
        # - Network issues
    ```

## 13. **Limitations**

- **Format**: PNG, WEBP, or JPEG for input; PNG for masks
- **File Size**: Up to 25MB per image
- **Output sizes**: 1024x1024, 1536x1024, 1024x1536, or `auto`
- **Masking**: Prompt-based — complex pixel-perfect edits may need multiple iterations
