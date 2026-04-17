# ModuleList

## ✦₊⁺ Overview

Holds submodules in a list.

`nn.ModuleList` can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all `nn.Module` methods.

## 1. **Example**

```python
import msgflux.nn as nn

class ExpertSales(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Hi, let's talk?")

    def forward(self, msg: str):
        return msg + self.response

class ExpertSupport(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Hi, call 190")

    def forward(self, msg: str):
        return msg + self.response

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([ExpertSales(), ExpertSupport()])

    def forward(self, msg: str) -> str:
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.experts):
            msg = self.experts[i](msg)
        return msg

expert = Expert()
expert("I need help with my tv.")
```
