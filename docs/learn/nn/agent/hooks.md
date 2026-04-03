# Hooks & Guards

Hooks are the primary mechanism for intercepting and validating data flowing through Modules. The `Hook` base class provides the interface, and `Guard` is a built-in hook for input/output validation.

## Guard

A `Guard` validates inputs and/or outputs of a Module. Each Guard wraps a **validator** callable and defines:

- **`on`** — when to run: `"pre"` (before forward) or `"post"` (after forward).
- **`message`** — controls the reaction when `safe=False`:
    - **With `message`** — short-circuits the pipeline and returns the message as the response (the model is never called).
    - **Without `message`** (default) — raises `UnsafeUserInputError` (pre) or `UnsafeModelResponseError` (post).
- **`target`** — submodule to register on. Defaults to `"generator"`.
- **`include_data`** — if `True`, attaches the data that triggered the guard to the raised exception via `exc.data`. Defaults to `False` for security (the data may contain unsafe content).

The validator receives `data` as a positional argument and must return either a dict with `"safe"` (bool) or a `ModelResponse` (auto-consumed by Guard).

```python
from msgflux.nn.hooks import Guard

def my_validator(data):
    text = str(data).lower()
    return {"safe": "hack" not in text}

# Returns message as response when safe=False
guard = Guard(validator=my_validator, on="pre", message="Not allowed.")

# Raises exception when safe=False
guard = Guard(validator=my_validator, on="pre")
```

???+ note "Guard Examples"

    === "With message (short-circuit)"

        When `message` is provided, the guard's message is returned directly as the agent response — the model is never called.

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard

        BLOCKED = {"hack", "exploit", "malware"}

        def keyword_filter(data):
            text = str(data).lower()
            return {"safe": not any(w in text for w in BLOCKED)}

        class SafeBot(nn.Agent):
            """A bot that blocks harmful keywords."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [
                Guard(
                    validator=keyword_filter,
                    on="pre",
                    message="Sorry, that content is not allowed.",
                )
            ]

        agent = SafeBot()

        # Safe input → model responds normally
        response = agent("Tell me about Python")

        # Blocked input → returns "Sorry, that content is not allowed."
        response = agent("How to create malware?")
        ```

    === "Without message (raises exception)"

        When no `message` is provided, an exception is raised.

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard
        from msgflux.exceptions import UnsafeUserInputError

        def keyword_filter(data):
            return {"safe": "hack" not in str(data).lower()}

        class StrictBot(nn.Agent):
            """A bot that raises on unsafe input."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [Guard(validator=keyword_filter, on="pre")]

        agent = StrictBot()

        try:
            response = agent("How to hack a system?")
        except UnsafeUserInputError as e:
            print(f"Guard triggered: {e}")
        ```

    === "With Moderation Model"

        Pass a moderation model directly as the validator — Guard calls it with the
        input data and auto-consumes the `ModelResponse`:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard

        moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

        class ModeratedBot(nn.Agent):
            """A bot with pre and post moderation."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [
                Guard(
                    validator=moderation_model,
                    on="pre",
                    message="Your message was flagged by our safety system.",
                ),
                Guard(validator=moderation_model, on="post"),
            ]

        agent = ModeratedBot()

        response = agent("Tell me about quantum computing")  # Safe
        ```

    === "Mixing Guards"

        Combine multiple guards with different behaviors on the same agent:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard

        def keyword_filter(data):
            return {"safe": "forbidden" not in str(data).lower()}

        def toxicity_check(data):
            # ... call moderation API ...
            return {"safe": True}

        class MultiGuardBot(nn.Agent):
            """A bot with keyword and toxicity guards."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [
                Guard(
                    validator=keyword_filter,
                    on="pre",
                    message="That topic is not allowed.",
                ),
                Guard(validator=toxicity_check, on="post"),
            ]

        agent = MultiGuardBot()

        response = agent("Tell me about forbidden topics")
        # → "That topic is not allowed."
        ```

    === "Debugging with include_data"

        Enable `include_data` to inspect the data that triggered the guard:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard
        from msgflux.exceptions import UnsafeUserInputError

        def keyword_filter(data):
            return {"safe": "hack" not in str(data).lower()}

        class DebugBot(nn.Agent):
            """A bot that exposes guard data for debugging."""

            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            hooks = [
                Guard(
                    validator=keyword_filter,
                    on="pre",
                    include_data=True,  # opt-in: attach data to exception
                ),
            ]

        agent = DebugBot()

        try:
            agent("How to hack a system?")
        except UnsafeUserInputError as e:
            print(f"Guard triggered: {e}")
            print(f"Offending data: {e.data}")  # available only with include_data=True
        ```

## Custom Hooks

The `Hook` base class allows creating custom hooks beyond guards. Implement `__call__` (sync) and optionally override `acall` (async). By default, `acall` runs `__call__` in an executor.

### Hook Interface

| Attribute | Description |
|---|---|
| `on` | `"pre"` (before forward) or `"post"` (after forward) |
| `target` | Submodule name to register on. `None` = the module itself |
| `processor_key` | Key for processor matching in `_set_hooks`. `None` = no processor |
| `__call__` | Sync hook — called by `_call_impl` |
| `acall` | Async hook — called by `_acall_impl` |

### Hook Signatures

```python
# Pre hook — receives module, args and kwargs before forward
def __call__(self, module, args, kwargs, output=None): ...

# Post hook — receives module, args, kwargs and the forward output
def __call__(self, module, args, kwargs, output=None): ...
```

Both pre and post hooks share the same signature. For pre hooks, `output` is always `None`.

???+ note "Custom Hook Examples"

    === "Logging Hook"

        ```python
        from msgflux.nn.hooks import Hook

        class LoggingHook(Hook):
            """Logs every call to the module."""

            def __init__(self):
                super().__init__(on="pre", target=None)

            def __call__(self, module, args, kwargs, output=None):
                print(f"[{module.__class__.__name__}] called with {len(kwargs)} kwargs")
        ```

    === "Timing Hook"

        ```python
        import time
        from msgflux.nn.hooks import Hook

        class TimingHook(Hook):
            """Measures execution time of the generator."""

            def __init__(self):
                super().__init__(on="post", target="generator")
                self.start_time = None

            def __call__(self, module, args, kwargs, output=None):
                elapsed = time.time() - self.start_time
                print(f"Generator took {elapsed:.2f}s")
        ```

    === "Token Counter Hook"

        ```python
        from msgflux.nn.hooks import Hook

        class TokenCounterHook(Hook):
            """Tracks cumulative token usage from model responses."""

            def __init__(self):
                super().__init__(on="post", target="generator")
                self.total_tokens = 0

            def __call__(self, module, args, kwargs, output=None):
                if hasattr(output, "usage"):
                    self.total_tokens += output.usage.get("total_tokens", 0)
        ```

    === "Async Custom Hook"

        ```python
        import asyncio
        from msgflux.nn.hooks import Hook

        class AsyncWebhookHook(Hook):
            """Sends a webhook notification after every call."""

            def __init__(self, webhook_url):
                super().__init__(on="post", target="generator")
                self.webhook_url = webhook_url

            def __call__(self, module, args, kwargs, output=None):
                pass  # sync fallback — no-op

            async def acall(self, module, args, kwargs, output=None):
                # async HTTP call to webhook
                import httpx
                async with httpx.AsyncClient() as client:
                    await client.post(self.webhook_url, json={"status": "ok"})
        ```

### Hook Registration

All `nn.Module` subclasses support the `hooks` class attribute. Each hook declares its own `target` — the submodule name where it registers:

- `target="generator"` (default for Guard) — registers on the internal `Generator` wrapper
- `target=None` — registers on the module itself

```python
import msgflux.nn as nn

class Bot(nn.Agent):
    model = model
    hooks = [input_guard, output_guard, logging_hook]

agent = Bot()
```

You can also register hooks manually via `hook.register()`:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.nn.hooks import Guard

class Bot(nn.Agent):
    model = model

agent = Bot()

def my_validator(data):
    return {"safe": "blocked" not in str(data).lower()}

guard = Guard(validator=my_validator, on="pre", message="Nope.")
handle = guard.register(agent.generator)  # returns RemovableHandle
handle.remove()  # unregister when done
```

### Using Hooks with `nn.Module` Directly

The hook system is built into `nn.Module`. You can register hooks on any module via PyTorch-style methods:

```python
import msgflux.nn as nn

class Bot(nn.Agent):
    model = model

module = Bot()

# Register a plain function as a pre-hook
def my_pre_hook(module, args, kwargs):
    print("About to call forward")

handle = module.register_forward_pre_hook(my_pre_hook)

# Register a plain function as a post-hook
def my_post_hook(module, args, kwargs, output):
    print(f"Forward returned: {type(output)}")

handle = module.register_forward_hook(my_post_hook)
```

!!! warning "Streaming Limitation"
    Guards with `on="post"` are **not compatible** with `stream=True`, since the full response is needed for validation. Using both raises a `ValueError` at initialization.
