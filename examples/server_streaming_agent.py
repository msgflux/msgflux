# /// script
# dependencies = []
# ///
#
# Run this server with:
#
#   uv run --with 'msgflux[server,openai]' msgflux server \
#     examples/server_streaming_agent.py:registry --host 127.0.0.1
#
# The server listens on http://127.0.0.1:8010/v1 by default and exposes:
#
#   - model="support" for order support
#   - model="billing" for billing support
#
# Run the matching streaming client with:
#
#   uv run --with openai python examples/server_streaming_client.py

import msgflux as mf
from msgflux import nn

mf.load_dotenv()

ORDER_DB = {
    "A1001": "Order A1001 is packed and will ship today.",
    "A1002": "Order A1002 is delayed by one day due to weather.",
    "A1003": "Order A1003 was delivered yesterday at 16:20.",
}

INVOICE_DB = {
    "INV-42": "Invoice INV-42 is paid and the receipt was sent by email.",
    "INV-43": "Invoice INV-43 is open and due in 5 days.",
    "INV-44": "Invoice INV-44 failed because the card was declined.",
}


def get_order_status(order_id: str) -> str:
    """Look up the current status of a customer order."""
    return ORDER_DB.get(order_id, f"Order {order_id} was not found.")


def get_invoice_status(invoice_id: str) -> str:
    """Look up the current status of a customer invoice."""
    return INVOICE_DB.get(invoice_id, f"Invoice {invoice_id} was not found.")


registry = mf.ChannelRegistry()


@registry.agent(name="support")
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


@registry.agent(name="billing")
class BillingAgent(nn.Agent):
    """Billing agent exposed through the same OpenAI-compatible server."""

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = """
    You are a concise billing support specialist.
    Use the available tools when the user asks about invoices or payments.

    Server-side context:
    - tenant: {{ tenant }}
    - support tier: {{ tier }}
    - request id: {{ request_id }}
    - selected agent: {{ agent_name }}

    Treat server-side context as private routing metadata. Do not expose it.
    """
    instructions = """
    Answer in the same language as the user.
    If you have invoice status, explain it and give the next practical step.
    If the invoice does not exist, ask the user to confirm the invoice id.
    """
    tools = [get_invoice_status]


@registry.pre()
def add_server_context(_request, context, run):
    run.vars = {
        **run.vars,
        "tenant": run.vars.get("tenant", "default"),
        "tier": run.vars.get("tier", "standard"),
        "agent_name": context.agent_name,
        "request_id": context.request_id,
    }
    return run
