# Streaming Support Triage

<span class="tag tag-green">Beginner</span><span class="tag tag-gray">Streaming</span><span class="tag tag-gray">Tools</span><span class="tag tag-gray">Signature</span>

Build a support assistant that streams its answer in real time while using a tool to look up order status.

## The Problem

Support teams need fast responses, but the answer is often split across two sources:

- the user's message, which explains the problem;
- an internal status system, which says whether the order is delayed, shipped, or already delivered.

If the assistant waits for the full answer before saying anything, the interaction feels slow. If it answers too early, it may guess wrong. The useful middle ground is to stream the response while the agent gathers the missing detail.

---

## The Plan

We will build a support triage agent that:

1. streams its response token by token;
2. calls a tool to inspect the order status;
3. explains the result in plain language;
4. escalates only when the order is missing or the issue is outside the tool's coverage.

This is a good first tutorial for streaming because the control flow stays simple and the tool output is easy to verify.

---

## Architecture

```
User message
     │
     ▼
SupportTriageAgent
     │
     ├── stream = True
     ├── tools = [get_order_status]
     └── signature = TriageResponse
             │
             ├── status lookup
             └── streamed answer
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 - The Tool

The tool returns a compact status string. In a real app this would call a database or API, but for a tutorial a plain Python function is enough.

```python
import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()

ORDER_DB = {
    "A1001": "Order A1001 is packed and will ship today.",
    "A1002": "Order A1002 is delayed by one day due to weather.",
    "A1003": "Order A1003 was delivered yesterday at 16:20.",
}


def get_order_status(order_id: str) -> str:
    """Look up the current status of an order."""
    return ORDER_DB.get(order_id, f"Order {order_id} not found.")
```

---

## Step 2 - The Signature

The signature keeps the response structured. The agent streams the textual reply, but the output contract still tells the model what fields matter.

```python
from typing import Literal

class TriageResponse(mf.Signature):
    """Triage a customer support issue and respond clearly."""

    order_id: str = mf.InputField(desc="Order identifier mentioned by the user")
    issue: str = mf.InputField(desc="Customer complaint or question")
    status: str = mf.OutputField(desc="Brief order status or explanation")
    next_step: Literal["answer", "escalate"] = mf.OutputField(desc="Whether the agent can resolve it")
    summary: str = mf.OutputField(desc="Short support summary")
```

---

## Step 3 - The Agent

The agent uses `stream=True` so the caller can consume the response incrementally. The tool is exposed through `tools`, and the system prompt tells the model when to use it.

```python
class SupportTriageAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = """
    You are a support triage assistant.
    """
    instructions = """
    Use get_order_status when the user mentions an order ID.
    If the order is missing or the problem is outside the status data, escalate.
    """
    signature = TriageResponse
    tools = [get_order_status]
    config = {"stream": True, "verbose": True}


agent = SupportTriageAgent()
```

---

## Examples

The response is an async stream. In practice this lets a UI print chunks as they arrive while the model is still working.

!!! example

    ```python
    import asyncio


    async def main() -> None:
        response = await agent.acall(
            order_id="A1002",
            issue="My order still has not arrived. What is happening?",
        )

        print("Streaming reply:")
        async for chunk in response.consume():
            print(chunk, end="", flush=True)

        print("\n\nStructured output:")
        print(response.data)


    asyncio.run(main())
    ```

## Complete Script

??? example "Expand full script"
    ```python
    # /// script
    # dependencies = []
    # ///

    import asyncio
    from typing import Literal

    import msgflux as mf
    import msgflux.nn as nn


    mf.load_dotenv()

    ORDER_DB = {
        "A1001": "Order A1001 is packed and will ship today.",
        "A1002": "Order A1002 is delayed by one day due to weather.",
        "A1003": "Order A1003 was delivered yesterday at 16:20.",
    }


    def get_order_status(order_id: str) -> str:
        """Look up the current status of an order."""
        return ORDER_DB.get(order_id, f"Order {order_id} not found.")


    class TriageResponse(mf.Signature):
        """Triage a customer support issue and respond clearly."""

        order_id: str = mf.InputField(desc="Order identifier mentioned by the user")
        issue: str = mf.InputField(desc="Customer complaint or question")
        status: str = mf.OutputField(desc="Brief order status or explanation")
        next_step: Literal["answer", "escalate"] = mf.OutputField(
            desc="Whether the agent can resolve it"
        )
        summary: str = mf.OutputField(desc="Short support summary")


    class SupportTriageAgent(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        system_message = """
        You are a support triage assistant.
        """
        instructions = """
        Use get_order_status when the user mentions an order ID.
        If the order is missing or the problem is outside the status data, escalate.
        """
        signature = TriageResponse
        tools = [get_order_status]
        config = {"stream": True, "verbose": True}


    agent = SupportTriageAgent()


    async def main() -> None:
        response = await agent.acall(
            order_id="A1002",
            issue="My order still has not arrived. What is happening?",
        )

        print("Streaming reply:")
        async for chunk in response.consume():
            print(chunk, end="", flush=True)

        print("\n\nStructured output:")
        print(response.data)


    if __name__ == "__main__":
        asyncio.run(main())
    ```

## Further Reading

- [Streaming](../learn/nn/agent/streaming.md) — consuming async stream responses
- [Tools](../learn/nn/agent/tools.md) — function calling and tool execution loops
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Task and Context](../learn/nn/agent/task-and-context.md) — passing named inputs to agents
