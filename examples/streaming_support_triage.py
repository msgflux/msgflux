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
