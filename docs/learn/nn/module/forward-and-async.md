# Forward and Async

## ✦₊⁺ Overview

All module logic is defined in `forward()`. The async counterpart `aforward()` enables non-blocking execution, and hooks let you enrich inputs or inspect outputs without modifying the module itself.

## 1. **`forward()`**

The main execution method. All module logic is defined in `forward`:


Make synchronous HTTP requests in the main execution method:

```python
import httpx
import msgflux.nn as nn

class APIConsumer(nn.Module):
    def __init__(self, api_url: str):
        super().__init__()
        self.api_url = api_url

    def forward(self, query: str, **kwargs):
        # Synchronous HTTP request
        with httpx.Client() as client:
            response = client.get(
                f"{self.api_url}/search",
                params={"q": query}
            )
            data = response.json()
        return f"Query: {query}, Result: {data['result']}"

consumer = APIConsumer("https://api.example.com")
result = consumer("what is AI?")
print(result)
```

## 2. **`aforward()`**

The async execution method enables non-blocking HTTP calls. Called via `.acall()`:

```python
import httpx
import msgflux.nn as nn

class AsyncAPIConsumer(APIConsumer):
    async def aforward(self, query: str, **kwargs):
        # Asynchronous HTTP request
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/search",
                params={"q": query}
            )
            data = response.json()
        return f"Query: {query}, Result: {data['result']}"

processor = AsyncProcessor("https://api.example.com")
result = await processor.acall("what is AI?")  # Calls aforward()
print(result)
```

!!! note
    If `aforward` is not implemented, `.acall()` will fall back to `forward()`.

## 3. **Hooks**

Hooks allow you to execute code before or after the forward pass without modifying the module itself. This is useful for context enrichment, logging, inspection, and monitoring.

### Pre-Forward Hook

Called **before** `forward()`. Can modify the input arguments:

```python
def retrieve_user_name(user_id: str):
    if user_id == "123":
        return "Clark"
    return None

def pre_hook(module, args, kwargs):
    """Enrich context before forward."""
    if kwargs.get("user_id"):
        user_name = retrieve_user_name(kwargs["user_id"])
        kwargs["user_name"] = user_name
    return args, kwargs

model = Model()

# Register hook - returns a handle object
pre_hook_handle = model.register_forward_pre_hook(pre_hook)
```

### Post-Forward Hook

Called **after** `forward()`. Can inspect or modify the output:

```python
def post_hook(module, args, kwargs, output):
    """Inspect output after forward."""
    print(f"inspect output: {output}")
    return output

# Register hook
post_hook_handle = model.register_forward_hook(post_hook)
```

### Method Hooks

When the extension point is not `forward`, use method hooks. They apply the same pre/post contract to an arbitrary method on the module:

```python
class Model(nn.Module):
    def forward(self, x):
        return self._normalize(x)

    def _normalize(self, x):
        return x.strip().lower()

model = Model()

def normalize_pre_hook(module, args, kwargs):
    return (args[0] + "  ",), kwargs

def normalize_post_hook(module, args, kwargs, output):
    return f"[{output}]"

pre_handle = model.register_method_pre_hook("_normalize", normalize_pre_hook)
post_handle = model.register_method_hook("_normalize", normalize_post_hook)

result = model(" Hello ")
print(result)  # "[hello]"
```

### Complete Hook Example

```python
import msgflux.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Yes I did.")

    def forward(self, x, **kwargs):
        user_name = kwargs.get("user_name", None)
        if user_name:
            return x + " Hi " + user_name + " " + self.response
        return x + self.response

model = Model()

# Register hooks
pre_hook_handle = model.register_forward_pre_hook(pre_hook)
post_hook_handle = model.register_forward_hook(post_hook)

# Execute - pre_hook enriches kwargs, post_hook inspects output
result = model("You did the work?", user_id="123")
print(f"Output: {result}")
# inspect output: You did the work? Hi Clark Yes I did.
# Output: You did the work? Hi Clark Yes I did.
```

### Removing Hooks

```python
# Remove hooks when no longer needed
pre_hook_handle.remove()
post_hook_handle.remove()

# Now forward runs without hooks
result = model("You did the work?", user_id="123")
print(result)  # "You did the work?Yes I did." (no user_name enrichment)
```

### Inspecting Registered Hooks

```python
# View registered hooks
print(model._forward_pre_hooks)   # Pre-forward hooks
print(model._forward_hooks)       # Post-forward hooks
print(model._method_pre_hooks)    # Pre-method hooks by method name
print(model._method_hooks)        # Post-method hooks by method name
```
