# Plan Tool with Root Context

<span class="tag tag-green">Beginner</span><span class="tag tag-gray">Tools</span><span class="tag tag-gray">ChainOfThought</span><span class="tag tag-gray">Templates</span>

Some planning tasks depend on details spread across earlier turns. The root assistant should not have to rewrite that context every time it wants a plan.

In this tutorial, the root assistant delegates planning to a `plan` tool that receives both the current task and the full conversation history.

## The Problem

Users rarely ask for a plan in a single clean message. The real constraints usually appear across the conversation:

- the target plan or migration;
- the number of users or systems involved;
- the downtime limit;
- the approvals that must happen before launch.

If the root assistant has to restate all of that every time it calls a planning tool, the system becomes repetitive and fragile. The better pattern is to let the tool see the root conversation directly.

---

## The Plan

We will build a small two-agent setup.

The **root assistant** handles the conversation. When the user asks for a rollout plan, checklist, roadmap, or next steps, it calls the **plan tool**.

The plan tool receives:

- the delegated `task` from the root assistant;
- the full root conversation history through `inject_messages=True`.

That gives the specialist enough context to write a useful plan without asking the root to repeat earlier details.

---

## Architecture

```
Conversation history
     + current request
            │
            ▼
      RootAssistant
            │
            └── plan(task=...)
                    │
                    ▼
                PlanTool
         inject_messages = True
                    │
                    ├── sees root history
                    ├── reasons with ChainOfThought
                    └── returns clean plan text
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 - The Plan Tool

The plan tool is itself an `nn.Agent`. It has no explicit `Signature`, so the default public annotation is `task: str`. That is exactly what the root assistant will pass.

`inject_messages=True` is the important part: the tool receives the root conversation history automatically.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought

mf.load_dotenv()
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


@mf.tool_config(inject_messages=True)
class PlanTool(nn.Agent):
    """Create an action plan using the delegated task and the full root conversation."""

    name = "plan"
    model = model
    system_message = """
    You are a planning specialist.
    """
    instructions = """
    Build plans using both the delegated task and the full conversation history.
    Use the history to extract constraints, goals, deadlines, stakeholders, and risks.
    Do not ignore details mentioned earlier in the conversation.
    """
    expected_output = """
    Return a concise action plan with:
    - a one-line objective
    - the key constraints
    - 3 to 5 ordered steps
    - the immediate next action
    """
    generation_schema = ChainOfThought
    templates = {"response": "{{ final_answer }}"}
    config = {"verbose": True}


planner = PlanTool()
print(planner.annotations)
```

If you inspect `planner.annotations`, you will see the default public tool schema:

```python
{'task': str, 'return': str}
```

That means the root assistant can call the tool naturally with `plan(task=...)` without defining a separate signature.

The internal flow is:

1. the root assistant passes a `task` string to the tool;
2. `inject_messages=True` also gives the tool the full root conversation history;
3. `ChainOfThought` makes the model produce both `reasoning` and `final_answer`;
4. `templates={"response": "{{ final_answer }}"}` strips the reasoning from the tool response and returns only the final plan text to the root assistant.

Without that template, the root assistant would receive the raw reasoning payload instead of a clean string.

---

## Step 2 - The Root Assistant

The root assistant owns the conversation and decides when to delegate planning.

```python
class RootAssistant(nn.Agent):
    model = model
    system_message = """
    You are the root assistant for AcmeCloud.
    """
    instructions = """
    Use the plan tool whenever the user asks for a plan, rollout, checklist, roadmap,
    or next steps.

    Pass a clear task to the tool. The tool already receives the full conversation
    history, so you do not need to restate all prior details in the task.
    """
    tools = [planner]
    config = {"verbose": True}


assistant = RootAssistant()
```

---

## Examples

The conversation history is built with `ChatBlock`, not manual role dictionaries. The root assistant receives that history, and the plan tool sees the same history when it is called.

!!! example

    ```python
    history = [
        mf.ChatBlock.user("We are moving from the Pro plan to Team next month for 40 users."),
        mf.ChatBlock.assist(
            "Understood. You need SAML SSO, audit logs, and a controlled migration."
        ),
        mf.ChatBlock.user(
            "We have two environments, one hour of downtime, and security needs sign-off before launch."
        ),
    ]

    response = assistant(
        "Create a rollout plan for this migration.",
        messages=history,
    )
    print(response)
    ```

    Expected behavior:

    - the root assistant calls `plan(task=...)`;
    - the tool reads both the delegated task and the full root history;
    - the plan includes constraints from earlier turns, like `40 users`, `two environments`, `one hour of downtime`, and `security sign-off`.

---

## Complete Script

??? example "Expand full script"
    ```python
    # /// script
    # dependencies = []
    # ///

    import msgflux as mf
    import msgflux.nn as nn
    from msgflux.generation.reasoning import ChainOfThought


    mf.load_dotenv()
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")


    @mf.tool_config(inject_messages=True)
    class PlanTool(nn.Agent):
        """Create an action plan using the delegated task and the full root conversation."""

        name = "plan"
        model = model
        system_message = """
        You are a planning specialist.
        """
        instructions = """
        Build plans using both the delegated task and the full conversation history.
        Use the history to extract constraints, goals, deadlines, stakeholders, and risks.
        Do not ignore details mentioned earlier in the conversation.
        """
        expected_output = """
        Return a concise action plan with:
        - a one-line objective
        - the key constraints
        - 3 to 5 ordered steps
        - the immediate next action
        """
        generation_schema = ChainOfThought
        templates = {"response": "{{ final_answer }}"}
        config = {"verbose": True}


    planner = PlanTool()


    class RootAssistant(nn.Agent):
        model = model
        system_message = """
        You are the root assistant for AcmeCloud.
        """
        instructions = """
        Use the plan tool whenever the user asks for a plan, rollout, checklist, roadmap,
        or next steps.

        Pass a clear task to the tool. The tool already receives the full conversation
        history, so you do not need to restate all prior details in the task.
        """
        tools = [planner]
        config = {"verbose": True}


    assistant = RootAssistant()

    history = [
        mf.ChatBlock.user("We are moving from the Pro plan to Team next month for 40 users."),
        mf.ChatBlock.assist(
            "Understood. You need SAML SSO, audit logs, and a controlled migration."
        ),
        mf.ChatBlock.user(
            "We have two environments, one hour of downtime, and security needs sign-off before launch."
        ),
    ]

    print("Plan tool annotations:", planner.annotations)
    print()

    response = assistant(
        "Create a rollout plan for this migration.",
        messages=history,
    )
    print(response)
    ```

## Further Reading

- [Tools](../learn/nn/agent/tools/index.md) — `inject_messages=True` and agent-as-tool patterns
- [Task and Context](../learn/nn/agent/task-and-context.md) — conversation history with `messages`
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) — `ChainOfThought` reasoning
- [Chat Completion](../learn/models/chat_completion.md) — `ChatBlock` helpers and message formats
