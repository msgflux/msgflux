# Visualization

## ✦₊⁺ Overview

Generate a Mermaid diagram of any module with `module.plot()`. Useful for inspecting the structure of composed pipelines.

```python
module.plot()  # Displays a flow diagram
```

!!! warning
    Flow visualization is **experimental** and may be incomplete for complex conditionals.

## 1. **Example: Complete Module**

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.dsl.inline import Inline

class QAWorkflow(nn.Module):
    """A question-answering workflow with retrieval."""

    def __init__(self):
        super().__init__()

        # Models
        chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # Sub-modules
        self.retriever = nn.Searcher("wikipedia")
        self.agent = nn.Agent(
            "qa-agent",
            chat_model,
            instructions="Answer questions using the provided context."
        )

        # Components for inline DSL
        self.components = nn.ModuleDict({
            "retrieve": self.retriever,
            "answer": self.agent
        })

        # Workflow definition
        self.register_buffer("flux", "retrieve -> answer")

    def forward(self, question: str) -> str:
        msg = mf.dotdict(query=question)
        msg = Inline(self.flux, self.components)(msg)
        return msg.answer

    async def aforward(self, question: str) -> str:
        msg = mf.dotdict(query=question)
        msg = await Inline(self.flux, self.components).acall(msg)
        return msg.answer

# Use it
qa = QAWorkflow()
answer = qa("What is quantum entanglement?")
print(answer)

# Save state
mf.save(qa.state_dict(), "qa_workflow.toml")
```
