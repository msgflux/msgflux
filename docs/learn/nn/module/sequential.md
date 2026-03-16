# Sequential

## ✦₊⁺ Overview

A sequential container. Modules are added in constructor order; `forward()` chains their outputs automatically, passing the result of each module as input to the next.

The value a `Sequential` provides over manually calling a sequence of modules is that it allows treating the whole container as a single module, such that performing a transformation on the `Sequential` applies to each of the modules it stores (which are each a registered submodule of the `Sequential`).

!!! info "Sequential vs ModuleList"
    A [ModuleList](module-list.md) is exactly what it sounds like -- a list for storing `Module`s! On the other hand, the layers in a `Sequential` are connected in a cascading way.

## 1. **Basic Usage**

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

# Using Sequential to create a small workflow. When **experts** is run,
# input will first be passed to **ExpertSales**. The output of
# **ExpertSales** will be used as the input to the first
# **ExpertSupport**; Finally, the output of
# **ExpertSupport** will be the experts response.
experts = nn.Sequential(ExpertSales(), ExpertSupport())
experts("I need help with my tv.")
```

## 2. **Using OrderedDict**

You can pass an `OrderedDict` to name each step in the sequence:

```python
from collections import OrderedDict
import msgflux.nn as nn

experts_dict = nn.Sequential(OrderedDict([
    ("expert_sales", ExpertSales()),
    ("expert_support", ExpertSupport())
]))

experts_dict("I need help with my tv.")
```

## 3. **Async Support**

`Sequential` supports async execution via `.acall()`. It checks each module for an `acall` method first, then falls back to sync `forward()`:

```python
experts = nn.Sequential(ExpertSales(), ExpertSupport())
result = await experts.acall("I need help with my tv.")
```
