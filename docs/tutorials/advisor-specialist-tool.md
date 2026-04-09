# Advisor Specialist Tool

<span class="tag tag-green">Beginner</span><span class="tag tag-gray">Tools</span><span class="tag tag-gray">ChainOfThought</span><span class="tag tag-gray">Templates</span>

Some questions should not be answered by the root assistant directly. Pricing, refunds, and product policy change over time, and they are easier to maintain in one specialist than inside the root prompt.

In this tutorial, the root assistant delegates those questions to an `advisor` tool.

## The Problem

The root assistant has two competing jobs:

- keep the conversation natural and helpful;
- answer product and policy questions precisely.

Trying to do both in one prompt usually degrades over time. The prompt gets bloated with handbook details, the assistant starts mixing policy with general conversational behavior, and a small pricing change forces you to edit the root prompt instead of the specialist logic that actually owns that knowledge.

The cleaner design is delegation. Let the root assistant manage the conversation and let a specialist answer handbook questions. In msgFlux, that specialist can be another agent exposed as a tool.

---

## The Plan

We will build a small two-agent setup.

The **root assistant** handles the conversation. When the user asks about pricing, refunds, security, or support policy, it calls the **advisor specialist** instead of answering from memory.

The Advisor reads a small internal handbook and returns a clean tool response that the root assistant can use in its final answer.

---

## Architecture

```
User question
     │
     ▼
RootAssistant
     │
     ├── direct answer for simple conversation
     └── advisor(question=...)
             │
             ▼
      AdvisorTool (Agent-as-Tool)
             │
             ├── Signature
             ├── ChainOfThought
             ├── context_cache = internal handbook
             └── response template → formatted tool output
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 - Internal Handbook

The handbook is deliberately small. It stands in for your pricing table, refund policy, security policy, or product documentation.

```python
import msgflux as mf
import msgflux.nn as nn

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
```

---

## Step 2 - Advisor Signature

The Advisor should answer with a compact, typed payload. The root assistant does not need raw chain-of-thought text; it needs a reliable answer plus enough metadata to decide how much to trust it.

```python
from typing import Literal


class AdvisorQuestion(mf.Signature):
    """Answer handbook questions using only the provided internal documentation."""

    question: str = mf.InputField(desc="Question delegated by the root assistant")
    answer: str = mf.OutputField(desc="Short factual answer grounded in the handbook")
    confidence: Literal["high", "medium", "low"] = mf.OutputField(
        desc="Confidence in the answer based on how directly the handbook supports it"
    )
    source_section: Literal["pricing", "refunds", "security", "support", "unknown"] = mf.OutputField(
        desc="Most relevant handbook section"
    )
```

---

## Step 3 - Advisor as a Tool

This is the key pattern: the tool is itself an `nn.Agent`.

`ChainOfThought` helps the Advisor think before answering. The response template keeps the tool output clean, so the root assistant receives a short, readable result instead of the raw structured payload.

```python
from msgflux.generation.reasoning import ChainOfThought


@mf.tool_config(name_override="advisor")
class AdvisorTool(nn.Agent):
    """Specialist that answers product and policy questions from the handbook."""

    model = model
    system_message = """
    You are the Advisor specialist.
    """
    instructions = """
    Answer using only the handbook.
    If the handbook is insufficient, say so and lower confidence.
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
```

## Extending

If you want the Advisor to see the same conversation context as the root assistant, add `inject_messages=True` to the tool config. The tool still receives the delegated task from the root, but it also gets the root message history as `messages`, which is useful when the answer depends on earlier turns.

```python
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
```

## Step 4 - Root Assistant

The root assistant owns the conversation and decides when to call `advisor`.

```python
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
```

---

## Examples

The user asks the root assistant a product question. The root decides to call `advisor`, receives the formatted tool output, and then answers the user.

!!! example

    ```python
    response = assistant("Does the Pro plan include SAML SSO?")
    print(response)
    ```

    Expected behavior:

    - the root agent calls `advisor(question="Does the Pro plan include SAML SSO?")`;
    - the Advisor reasons with `ChainOfThought`;
    - the response template turns the structured output into a clean string;
    - the root assistant uses that tool result in its final answer.

---

## Complete Script

??? example "Expand full script"
    ```python
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
    ```

## Further Reading

- [Tools](../learn/nn/agent/tools.md) — agent-as-tool patterns and tool configuration
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) — `ChainOfThought` and structured reasoning
- [Task and Context](../learn/nn/agent/task-and-context.md) — response templates and context caches
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts for specialists
