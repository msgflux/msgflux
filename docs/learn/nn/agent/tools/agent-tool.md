# Agent Tool

Agents can be used as tools for other agents, enabling hierarchical task delegation, also known as **SubAgents**.

The Coordinator calls the Specialist as any other tool. The result returns to the Coordinator's model, which synthesizes it into the final response.

There are two common patterns:

- Register agents directly as tools, where each agent appears as its own tool.
- Register one `AgentTool`, where the model calls a single `agent(name, message)`
  tool and chooses the target agent by name.

## AgentTool

`AgentTool` exposes a group of agents through one tool named `agent`.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import AgentTool

model = mf.Model.chat_completion("openai/gpt-4.1-mini")

class Researcher(nn.Agent):
    model = model
    instructions = "Research the user's question and return concise findings."

class Reviewer(nn.Agent):
    model = model
    instructions = "Review the answer for correctness and missing details."

agent_tool = AgentTool(agents=[Researcher(), Reviewer()])

coordinator = nn.Agent(
    name="coordinator",
    model=model,
    instructions="Delegate specialized work to the agent tool when useful.",
    tools=[agent_tool],
)
```

The model sees one callable shape:

```python
agent(name: str, message: str) -> str
```

`AgentTool` also accepts runtime-injected `messages` and `vars`; those are
provided by msgFlux and are not exposed as normal model parameters.

## Tool Bucket Capture

Internally, `AgentTool` is a `ToolBucket`. A bucket is a tool that absorbs other
tools of a specific kind and exposes them through one public tool. `AgentTool`
uses:

```python
tool_kind = "bucket"
capture_kind = "agent"
```

Agents are registered with `tool_kind="agent"`. When a `ToolLibrary` contains
an `AgentTool`, adding an agent tool causes the library to route that agent into
the bucket instead of exposing it as a separate top-level tool:

```python
from msgflux.tools.builtin import AgentTool

library = nn.ToolLibrary(name="coordinator", tools=[AgentTool()])
library.add(Researcher())
library.add(Reviewer())

print(library.get_tool_names())
# ["agent"]
```

The bucket updates its own description and usage guidance when agents are
captured, so the single `agent` tool still tells the model which agents are
available and when each one should be used.

This is useful with on-demand tools as well: an agent can be selected later,
loaded into the library, and captured by the existing `AgentTool` bucket.

```
              Input
                │
                ▼
  ┌──────────────────────────────┐
  │         Coordinator          │
  │                              │
  │   ┌──────────┐               │
  │   │  Model   │──▶ "call      │
  │   └──────────┘    Specialist │
  └─────────────┬────────────────┘
                │  call(task)
                ▼
  ┌──────────────────────────────┐
  │         Specialist           │
  │        (SubAgent)            │
  │                              │
  │  processes task independently│
  │  may call its own tools      │
  └─────────────┬────────────────┘
                │  result
                ▼
  ┌──────────────────────────────┐
  │         Coordinator          │
  │                              │
  │   ┌──────────┐               │
  │   │  Model   │──▶ synthesized│
  │   └──────────┘    response   │
  └─────────────┬────────────────┘
                │
                ▼
                Output
```

???+ note "Agent-as-Tool Examples"

    === "Health Team"

        A coordinator agent delegates to specialist agents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Nutritionist(nn.Agent):
            """Specialist in nutrition, diet planning, and healthy eating habits.
            Consult for meal plans, dietary recommendations, and nutritional advice."""

            model = model
            system_message = "You are a certified nutritionist."
            instructions = """Create clear and practical meal plans tailored to the user's goals.
            Be objective, technical, and structured."""

        class FitnessTrainer(nn.Agent):
            """Specialist in fitness, exercise routines, and physical training.
            Consult for workout plans, training schedules, and exercise guidance."""

            model = model
            system_message = "You are a certified personal trainer."
            instructions = """Design workout routines based on the user's fitness level and goals.
            Focus on safety, progression, and sustainability."""

        class HealthCoordinator(nn.Agent):
            """Coordinates health specialists to provide comprehensive wellness advice."""

            model = model
            system_message = "You coordinate a team of health specialists."
            instructions = "Delegate user requests to the appropriate specialist."
            tools = [Nutritionist, FitnessTrainer]
            config = {"verbose": True}

        coordinator = HealthCoordinator()

        response = coordinator("I want to lose 10kg and build muscle")
        ```

    === "Research Team"

        Multiple research specialists with a coordinator:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class AcademicResearcher(nn.Agent):
            """Expert in academic research with peer-reviewed sources.
            Use for scholarly inquiries and scientific topics."""

            model = model
            system_message = "You are an academic researcher."
            expected_output = "Provide academic-level analysis with citations."

        class MarketResearcher(nn.Agent):
            """Expert in market research and competitive analysis.
            Use for business intelligence and market sizing."""

            model = model
            system_message = "You are a market research analyst."
            expected_output = "Provide actionable business insights."

        class TechnicalResearcher(nn.Agent):
            """Expert in technical documentation and APIs.
            Use for programming questions and library comparisons."""

            model = model
            system_message = "You are a technical researcher."
            expected_output = "Provide technical details with code examples."

        class ResearchCoordinator(nn.Agent):
            model = model
            system_message = "You coordinate research specialists."
            instructions = "Delegate to the appropriate researcher based on the query type."
            tools = [
                AcademicResearcher,
                MarketResearcher,
                TechnicalResearcher
            ]
            config = {"verbose": True}

        coordinator = ResearchCoordinator()

        response = coordinator("Compare FastAPI vs Flask for building REST APIs")
        ```

    === "Agent Router"

        Route requests directly to specialists using `return_direct`:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        @mf.tool_config(return_direct=True)
        class PythonExpert(nn.Agent):
            """Expert in Python performance optimization."""

            model = model
            system_message = "You specialize in Python performance."

        @mf.tool_config(return_direct=True)
        class JavaScriptExpert(nn.Agent):
            """Expert in JavaScript and Node.js."""

            model = model
            system_message = "You specialize in JavaScript."

        class Router(nn.Agent):
            model = model
            system_message = "Route programming questions to the right expert."
            tools = [PythonExpert, JavaScriptExpert]
            config = {"verbose": True}

        router = Router()

        # Response comes directly from the specialist
        response = router("How do I optimize a Python loop?")
        ```

    === "Handoff Pattern"

        Seamless conversation handoff between agents:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # Enable handoff - transfers conversation history
        @mf.tool_config(handoff=True)
        class StartupSpecialist(nn.Agent):
            """Specialist in scaling digital startups.
            Use for growth strategies, metrics, and funding."""

            model = model
            system_message = "You are a startup scaling expert."

        class BusinessConsultant(nn.Agent):
            model = model
            system_message = """You are a business consultant.
            If the context is a startup, transfer to the specialist."""
            tools = [StartupSpecialist]
            config = {"verbose": True}

        consultant = BusinessConsultant()

        # Conversation is handed off to specialist
        response = consultant(
            "My SaaS has a CAC of $120 and LTV of $600. How do I scale?"
        )
        ```
