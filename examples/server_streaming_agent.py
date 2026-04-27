# /// script
# dependencies = []
# ///

import msgflux as mf
from msgflux import nn


mf.load_dotenv()

ORDER_DB = {
    "A1001": "Order A1001 is packed and will ship today.",
    "A1002": "Order A1002 is delayed by one day due to weather.",
    "A1003": "Order A1003 was delivered yesterday at 16:20.",
}


def get_order_status(order_id: str) -> str:
    """Look up the current status of a customer order."""
    return ORDER_DB.get(order_id, f"Order {order_id} was not found.")


class SupportAgent(nn.Agent):
    """Customer support agent exposed through the OpenAI-compatible server."""

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = """
    You are a concise customer support specialist.
    Use the available tools when the user asks about an order.

    Server-side context:
    - tenant: {{ tenant }}
    - support tier: {{ tier }}
    - request id: {{ request_id }}

    Treat server-side context as private routing metadata. Do not expose it.
    """
    instructions = """
    Answer in the same language as the user.
    If you have an order status, explain it and give the next practical step.
    If the order does not exist, ask the user to confirm the order id.
    """
    tools = [get_order_status]


registry = mf.ChannelRegistry()
registry.agent(SupportAgent(), name="support")


@registry.pre("support")
def add_tenant_context(_request, context, run):
    run.variables = {
        **run.variables,
        "tenant": run.variables.get("tenant", "default"),
        "tier": run.variables.get("tier", "standard"),
        "request_id": context.request_id,
    }
    return run
