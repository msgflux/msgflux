# Composing Modules

## ✦₊⁺ Overview

Modules can contain other modules as sub-modules. Sub-modules are automatically tracked in the state dict with dot-separated keys and visible to all Module methods.

## 1. **Sub-Modules**

Modules can contain other modules:

```python
class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessor = Preprocessor()
        self.analyzer = Analyzer()

    def forward(self, data):
        data = self.preprocessor(data)
        return self.analyzer(data)
```

Sub-modules are automatically tracked in the state dict.

See also: [ModuleDict](module-dict.md), [ModuleList](module-list.md), [Sequential](sequential.md)
