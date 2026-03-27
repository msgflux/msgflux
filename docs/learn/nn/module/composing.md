# Composing Modules

## ✦₊⁺ Overview

Modules can contain other modules as sub-modules. Sub-modules are **automatically tracked** when assigned to attributes — no extra registration needed. They appear in the state dict with dot-separated keys and are visible to all `Module` methods like `named_modules()`, `state_dict()`, and `parameters()`.

## 1. **Sub-Modules**

Any `Module` assigned to `self.<attr>` is automatically registered as a sub-module:

```python
import msgflux.nn as nn

class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("prefix", "[normalized]")

    def forward(self, text: str) -> str:
        return f"{self.prefix} {text.strip().lower()}"


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("label", "spam")

    def forward(self, text: str) -> str:
        return f"{text} → label={self.label}"


class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = Normalizer()   # auto-registered
        self.classifier = Classifier()   # auto-registered

    def forward(self, text: str) -> str:
        text = self.normalizer(text)
        return self.classifier(text)


pipeline = Pipeline()
result = pipeline("  HELLO WORLD  ")
print(result)
# [normalized] hello world → label=spam
```

## 2. **State Dict with Nested Keys**

Sub-module buffers and parameters appear in the state dict with **dot-separated paths**:

```python
import msgflux.nn as nn

class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("prefix", "[normalized]")

    def forward(self, text: str) -> str:
        return f"{self.prefix} {text.strip().lower()}"


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("label", "spam")

    def forward(self, text: str) -> str:
        return f"{text} → label={self.label}"


class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = Normalizer()
        self.classifier = Classifier()

    def forward(self, text: str) -> str:
        text = self.normalizer(text)
        return self.classifier(text)


pipeline = Pipeline()
print(pipeline.state_dict())
# {
#   "normalizer.prefix": "[normalized]",
#   "classifier.label": "spam"
# }
```

## 3. **Inspecting the Module Tree**

Use `named_modules()` to traverse the entire hierarchy:

```python
import msgflux.nn as nn

class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("prefix", "[normalized]")

    def forward(self, text: str) -> str:
        return f"{self.prefix} {text.strip().lower()}"


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("label", "spam")

    def forward(self, text: str) -> str:
        return f"{text} → label={self.label}"


class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = Normalizer()
        self.classifier = Classifier()

    def forward(self, text: str) -> str:
        text = self.normalizer(text)
        return self.classifier(text)


pipeline = Pipeline()

for name, module in pipeline.named_modules():
    print(f"{name!r:20} → {type(module).__name__}")
# ''                   → Pipeline
# 'normalizer'         → Normalizer
# 'classifier'         → Classifier
```

Use `named_children()` for only **direct** children (one level deep):

```python
for name, module in pipeline.named_children():
    print(f"{name}: {type(module).__name__}")
# normalizer: Normalizer
# classifier: Classifier
```

## 4. **Deep Nesting**

Sub-modules can be nested to any depth. State dict keys reflect the full path:

```python
import msgflux.nn as nn

class Formatter(nn.Module):
    def __init__(self, tag: str):
        super().__init__()
        self.register_buffer("tag", tag)

    def forward(self, text: str) -> str:
        return f"<{self.tag}>{text}</{self.tag}>"


class InnerPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.bold = Formatter("b")
        self.italic = Formatter("i")

    def forward(self, text: str) -> str:
        return self.italic(self.bold(text))


class OuterPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = InnerPipeline()
        self.wrapper = Formatter("div")

    def forward(self, text: str) -> str:
        return self.wrapper(self.inner(text))


outer = OuterPipeline()
print(outer("hello"))
# <div><i><b>hello</b></i></div>

print(outer.state_dict())
# {
#   "inner.bold.tag": "b",
#   "inner.italic.tag": "i",
#   "wrapper.tag": "div"
# }
```

## 5. **Loading State**

Use `load_state_dict()` to restore a saved configuration across the whole tree:

```python
import msgflux.nn as nn

class Formatter(nn.Module):
    def __init__(self, tag: str):
        super().__init__()
        self.register_buffer("tag", tag)

    def forward(self, text: str) -> str:
        return f"<{self.tag}>{text}</{self.tag}>"


class InnerPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.bold = Formatter("b")
        self.italic = Formatter("i")

    def forward(self, text: str) -> str:
        return self.italic(self.bold(text))


class OuterPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = InnerPipeline()
        self.wrapper = Formatter("div")

    def forward(self, text: str) -> str:
        return self.wrapper(self.inner(text))


outer = OuterPipeline()

# Save state
saved = outer.state_dict()

# Mutate
outer.inner.bold.tag = "strong"
print(outer("hello"))
# <div><i><strong>hello</strong></i></div>

# Restore
outer.load_state_dict(saved)
print(outer("hello"))
# <div><i><b>hello</b></i></div>
```

## 6. **Container Modules**

For common patterns, msgFlux provides dedicated containers — see their individual pages:

| Container | When to use |
|-----------|-------------|
| [`nn.Sequential`](sequential.md) | Fixed chain: output of step N feeds step N+1 |
| [`nn.ModuleList`](module-list.md) | Ordered collection iterated manually in `forward` |
| [`nn.ModuleDict`](module-dict.md) | Named collection, selected by key at runtime |
