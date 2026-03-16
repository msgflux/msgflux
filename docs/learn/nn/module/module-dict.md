# ModuleDict

## ✦₊⁺ Overview

Holds submodules in a dictionary.

`nn.ModuleDict` can be indexed like a regular Python dictionary, but modules it contains are properly registered, and will be visible by all Module methods.

`nn.ModuleDict` is an **ordered** dictionary that respects:

- the order of insertion, and
- in `update()`, the order of the merged `OrderedDict`, `dict` (started from Python 3.6) or another `nn.ModuleDict` (the argument to `update()`).

## 1. **Example**

```python
import random
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

def draw_choice(choices: list[str]) -> str:
    return random.choice(choices)

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.choices = nn.ModuleDict({
            "sales": ExpertSales(),
            "support": ExpertSupport()
        })

    def forward(self, msg: str) -> str:
        choice = draw_choice(list(self.choices.keys()))
        msg = self.choices[choice](msg)
        return msg

router = Router()

# Sub-modules are tracked in state_dict
print(router.state_dict())
# {'choices.sales.response': "Hi, let's talk?", 'choices.support.response': 'Hi, call 190'}

router("I need help with my tv.")
```
