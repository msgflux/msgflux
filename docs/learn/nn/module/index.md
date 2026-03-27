# nn.Module

## ✦₊⁺ Overview

The `nn.Module` class is the foundation for all AI components in msgFlux, inspired by `torch.nn.Module`**.

It provides a structured way to build, compose, and manage AI workflows with features like parameter serialization, hooks, and async support.

## 1. **Quick Start**

```python
import msgflux.nn as nn

class MyWorkflow(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("greeting", "Hello!")

    def forward(self, name: str) -> str:
        return f"{self.greeting} {name}"

workflow = MyWorkflow()
result = workflow("World")  # "Hello! World"
print(result)
print(workflow.state_dict())
```

## 2. **Built-in Modules**

msgFlux provides ready-to-use modules:

| Module | Description |
|--------|-------------|
| `nn.Sequential` | Chain modules in sequence |
| `nn.ModuleDict` | Dictionary of named modules |
| `nn.ModuleList` | Ordered list of modules |
| `nn.Agent` | LM with tools and reasoning |
| `nn.Transcriber` | Speech-to-text |
| `nn.Speaker` | Text-to-speech |
| `nn.Searcher` | Data retrieval |
| `nn.Embedder` | Text embeddings |
| `nn.MediaMaker` | Image/video generation |
| `nn.Predictor` | ML model wrapper (sklearn, etc.) |

## 4. **Contents**

| Topic | Description |
|-------|-------------|
| [Core Concepts](core-concepts.md) | Parameters, buffers, and state dict |
| [Forward and Async](forward-and-async.md) | Execution methods and hooks |
| [Composing Modules](composing.md) | Sub-modules and nested composition |
| [ModuleDict](module-dict.md) | Dictionary of named modules |
| [ModuleList](module-list.md) | Ordered list of modules |
| [Sequential](sequential.md) | Chain of modules |
| [Visualization](visualization.md) | Flow diagrams and complete example |
