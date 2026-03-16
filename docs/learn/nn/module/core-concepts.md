# Core Concepts

## ✦₊⁺ Overview

Modules store state through **Parameters** and **Buffers**, both of which are captured in the state dict. The state dict is serializable, enabling saving, loading, and hot-swapping values at runtime.

## 1. **Parameters and Buffers**

Modules store their state through **Parameters** and **Buffers**, both are registered in the state dict.

**Parameters** are values that can be optimized (e.g., prompts, instructions):

```python
import msgflux.nn as nn

class Workflow(nn.Module):
    def __init__(self):
        super().__init__()
        self.instructions = nn.Parameter(
            "Be helpful and concise",  # value
            "system_prompt"             # spec/category
        )

print(Workflow().state_dict())
# {'instructions': 'Be helpful and concise'}
```

**Buffers** are constant values that should be serialized but not optimized:

```python
import msgflux.nn as nn

class Workflow(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("expected_output", "JSON format")
        self.register_buffer("flux", "preprocess -> analyze")

print(Workflow().state_dict())
# {'expected_output': 'JSON format', 'flux': 'preprocess -> analyze'}
```

## 2. **State Dict**

Every module can export its complete state as a dictionary. This enables saving, loading, and updating module configurations without reloading the module.

### Export and Inspect

```python
import msgflux as mf
import msgflux.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Yes I did.")

    def forward(self, x, **kwargs):
        user_name = kwargs.get("user_name", None)
        if user_name:
            return x + " Hi " + user_name + self.response
        return x + self.response

model = Model()
print(model.state_dict())  # {"response": "Yes I did."}
```

### Save and Load

```python
# Save to file (supports toml and json)
mf.save(model.state_dict(), "state_dict.toml")

# Load from file
state_dict = mf.load("state_dict.toml")
print(state_dict)  # {"response": "Yes I did."}
```

### Update at Runtime

Update parameters without reloading the module:

```python
# Modify a value
state_dict["response"] = "No, I didn't."

# Apply changes
model.load_state_dict(state_dict)

# Module now uses updated values
result = model("You did the work?", user_id="123")
print(result)  # "You did the work? Hi Clark No, I didn't."
```

### Nested State Dict

Sub-modules are automatically tracked in the state dict with dot-separated keys:

```python
class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessor = Preprocessor()
        self.analyzer = Analyzer()

pipeline = Pipeline()
print(pipeline.state_dict())
# {'preprocessor.buffer_name': '...', 'analyzer.buffer_name': '...'}
```
