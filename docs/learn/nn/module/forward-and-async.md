# Forward and Async

## ✦₊⁺ Overview

All module logic is defined in `forward()`. The async counterpart `aforward()` enables non-blocking execution, and hooks let you enrich inputs or inspect outputs without modifying the module itself.

## 1. **`forward()`**

The main execution method. All module logic is defined in `forward`:

```python
import msgflux.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Yes I did.")

    def forward(self, x, **kwargs):
        user_name = kwargs.get("user_name", None)
        if user_name:
            model_response = " Hi " + user_name + self.response
        else:
            model_response = self.response
        x = x + model_response
        return x

model = Model()
result = model("You did the work?")  # Calls forward()
print(result)  # "You did the work?Yes I did."
```

## 2. **`aforward()`**

The async execution method. Called via `.acall()`:

```python
class AsyncProcessor(nn.Module):
    async def aforward(self, data, **kwargs):
        result = await some_async_operation(data)
        return result

processor = AsyncProcessor()
result = await processor.acall(data)  # Calls aforward()
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
```
