# nn.MediaMaker

## ✦₊⁺ Overview

`nn.MediaMaker` is a Module for **generating visual and audio content** (images, videos, 3D models, audio) from prompts.

---

## 1. **Quick Start**

!!! info "Initialization styles"

    === "Declarative (recommended)"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class ImageGenerator(nn.MediaMaker):
            model = mf.Model.text_to_image("openai/gpt-image-1")
            config = {"quality": "high"}

        generator = ImageGenerator()
        image_b64 = generator("A futuristic city at sunset, cyberpunk style")
        ```

    === "Direct"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        maker = nn.MediaMaker(model=mf.Model.text_to_image("openai/gpt-image-1"))
        image_b64 = maker("A futuristic city at sunset, cyberpunk style")
        ```

---

## 2. **Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `MEDIA_MODEL_TYPES` | Media generation model |
| `message_fields` | `dict` | Map `Message` field names to inputs. Valid keys: `task_inputs`, `task_multimodal_inputs` |
| `response_mode` | `str \| None` | Field path on the `Message` where the result is written. `None` returns the result directly |
| `response_format` | `"base64" \| "url" \| None` | Output format for the generated media |
| `negative_prompt` | `str \| None` | What to avoid in generation |
| `config` | `dict \| None` | Extra parameters passed to the model. Common keys: `fps`, `duration_seconds`, `aspect_ratio`, `n` |
| `hooks` | `list[Hook] \| None` | Hook instances registered on the module |
| `name` | `str \| None` | Module name in snake_case |

---

## 3. **Generation Modes**

!!! info "Examples by modality"

    === "Text to Image"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class PhotoMaker(nn.MediaMaker):
            model = mf.Model.text_to_image("openai/gpt-image-1")
            config = {
                "size": "1536x1024",   # landscape
                "quality": "high",
                "background": "transparent",
            }

        maker = PhotoMaker()
        image_b64 = maker("A serene Japanese garden with cherry blossoms")
        ```

        !!! note
            `size`, `quality`, `background`, and `n` are not call kwargs — pass them via `config`.

    === "Multiple Images"

        ```python
        class Maker(nn.MediaMaker):
            model = mf.Model.text_to_image("openai/gpt-image-1")
            config = {"n": 4, "size": "1024x1024", "quality": "low"}

        maker = Maker()
        images = maker("A cute robot")  # returns list of base64 strings when n > 1
        ```

    === "Image Editing"

        ```python
        class Editor(nn.MediaMaker):
            model = mf.Model.image_text_to_image("openai/gpt-image-1")

        editor = Editor()

        # Edit with reference image
        edited = editor(
            "Make it look like sunset",
            task_multimodal_inputs={"image": "/path/to/photo.jpg"}
        )

        # Edit with mask
        edited = editor(
            "Add a flamingo in the pool",
            task_multimodal_inputs={
                "image": "/path/to/pool.jpg",
                "mask": "/path/to/mask.png"
            }
        )
        ```

    === "Video"

        ```python
        class VideoMaker(nn.MediaMaker):
            model = mf.Model.text_to_video("sora/text-to-video")
            config = {
                "duration_seconds": 5,
                "aspect_ratio": "16:9",
                "fps": 24,
            }

        maker = VideoMaker()
        video = maker("A timelapse of a blooming flower")
        ```

    === "With Message"

        Bind inputs and outputs to fields on a shared `Message` for pipeline composition:

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class Maker(nn.MediaMaker):
            model = mf.Model.text_to_image("openai/gpt-image-1")
            message_fields = {"task_inputs": "prompt"}
            response_mode = "generated_image"

        maker = Maker()

        msg = mf.dotdict(prompt="A cozy cabin in the mountains during winter")
        maker(msg)
        # msg.generated_image contains the base64 string
        ```

---

## 4. **Configuration**

!!! info "Controlling generation"

    === "Negative Prompt"

        Specify what to avoid in the output:

        ```python
        class Designer(nn.MediaMaker):
            model = mf.Model.text_to_image("openai/gpt-image-1")
            negative_prompt = "blurry, low quality, distorted, watermark"

        maker = Designer()
        image_b64 = maker("A professional portrait photo")
        ```

    === "Providers"

        ```python
        # OpenAI — superior detail and prompt adherence
        model = mf.Model.text_to_image("openai/gpt-image-1")

        # Replicate — FLUX 2 Flex open model
        model = mf.Model.text_to_image("replicate/black-forest-labs/flux-2-flex")
        ```

---

## 5. **Guardrails**

Use `Guard` hooks to validate prompts before generation — useful for blocking unsafe content or enforcing prompt policies.

!!! info "Guard patterns"

    === "Short-circuit (with message)"

        When `message` is provided, the guard returns it directly — the model is never called:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard

        BLOCKED = {"violence", "explicit", "nsfw"}

        def content_policy(data):
            text = str(data).lower()
            return {"safe": not any(w in text for w in BLOCKED)}

        class SafeImageMaker(nn.MediaMaker):
            model = mf.Model.text_to_image("openai/gpt-image-1")
            hooks = [
                Guard(
                    validator=content_policy,
                    on="pre",
                    message="Prompt violates content policy.",
                )
            ]

        maker = SafeImageMaker()
        result = maker("explicit content")  # → "Prompt violates content policy."
        ```

    === "Raises exception"

        Without `message`, a `UnsafeUserInputError` is raised instead:

        ```python
        from msgflux.exceptions import UnsafeUserInputError

        class StrictMaker(nn.MediaMaker):
            model = mf.Model.text_to_image("openai/gpt-image-1")
            hooks = [Guard(validator=content_policy, on="pre")]

        maker = StrictMaker()

        try:
            maker("violence scene")
        except UnsafeUserInputError:
            print("Prompt blocked.")
        ```

---

## 6. **Creative Pipeline**

Chain an `Agent` (prompt engineer) and a `MediaMaker` (image generator) using `Inline`:

!!! info "Story → Image pipeline"

    ```python
    import msgflux as mf
    import msgflux.nn as nn

    class StoryToPrompt(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        instructions = """
        Create a detailed image generation prompt from the story scene.
        Focus on visual elements, lighting, style, and composition.
        """
        message_fields = {"task_inputs": "scene"}
        response_mode = "prompt"

    class ImageGenerator(nn.MediaMaker):
        model = mf.Model.text_to_image("openai/gpt-image-1")
        message_fields = {"task_inputs": "prompt"}
        response_mode = "illustration"

    prompter = StoryToPrompt()
    generator = ImageGenerator()

    pipeline = mf.Inline(
        "prompter -> generator",
        {"prompter": prompter, "generator": generator},
    )

    msg = mf.dotdict(scene="""
    Chapter 3: The hero stood at the edge of the cliff,
    watching the dragon descend from the storm clouds.
    Lightning illuminated the ancient castle behind them.
    """)

    pipeline(msg)
    print(msg.prompt)       # Detailed image prompt
    # msg.illustration contains the base64 string
    ```

---

## 7. **Async**

```python
image = await maker.acall("A colorful abstract painting")
```

---

## 8. **Debugging**

```python
params = maker.inspect_model_execution_params("test prompt")
print(params)
```
