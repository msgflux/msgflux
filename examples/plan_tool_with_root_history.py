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
