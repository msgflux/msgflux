# /// script
# dependencies = []
# ///

from typing import Literal

import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought


mf.load_dotenv()
model = mf.Model.chat_completion("openai/gpt-4.1-mini")

HANDBOOK = """
Pricing:
- Starter costs US$29/month.
- Pro costs US$99/month and includes API access plus webhooks.
- Team costs US$249/month and adds SAML SSO plus audit logs.

Refunds:
- First-time purchases are refundable within 30 days.
- Renewals are refundable only within 7 days.

Security:
- Data is encrypted in transit and at rest.
- SAML SSO is available only on the Team plan.

Support:
- Starter receives email support.
- Pro and Team receive priority email support.
"""


class AdvisorQuestion(mf.Signature):
    """Answer handbook questions using only the provided internal documentation."""

    question: str = mf.InputField(desc="Question delegated by the root assistant")
    answer: str = mf.OutputField(desc="Short factual answer grounded in the handbook")
    confidence: Literal["high", "medium", "low"] = mf.OutputField(
        desc="Confidence in the answer based on how directly the handbook supports it"
    )
    source_section: Literal[
        "pricing", "refunds", "security", "support", "unknown"
    ] = mf.OutputField(desc="Most relevant handbook section")


@mf.tool_config(name_override="advisor", inject_messages=True)
class AdvisorTool(nn.Agent):
    """Specialist that answers product and policy questions from the handbook."""

    model = model
    system_message = """
    You are the Advisor specialist.
    """
    instructions = """
    Answer using only the handbook and the shared conversation context.
    If the handbook or the conversation context is insufficient, say so and lower confidence.
    """
    generation_schema = ChainOfThought
    signature = AdvisorQuestion
    templates = {
        "response": (
            "Advisor answer "
            "(section={{ final_answer.source_section }}, "
            "confidence={{ final_answer.confidence }}): "
            "{{ final_answer.answer }}"
        )
    }
    context_cache = HANDBOOK
    config = {"verbose": True}


class RootAssistant(nn.Agent):
    model = model
    system_message = """
    You are the root assistant for AcmeCloud.
    """
    instructions = """
    Use the advisor tool for product, pricing, refund, security, and support-policy
    questions. For greetings or general conversational help, answer directly.

    If advisor returns low confidence, say that the answer needs human follow-up.
    """
    tools = [AdvisorTool]
    config = {"verbose": True}


assistant = RootAssistant()

print(assistant("Does the Pro plan include SAML SSO?"))
print()
print(assistant("Can a customer get a refund 45 days after purchase?"))
