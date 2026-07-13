# AgentTool

`AgentTool` is a built-in tool for exposing a group of agents through one
callable model tool:

```python
agent(name: str, message: str) -> str
```

Use it when a coordinator should delegate to specialized agents, but the model
should not see every specialist as a separate top-level tool. The coordinator
calls `agent(...)`, `AgentTool` resolves the requested agent by `name`, and the
selected agent processes the delegated `message`. The result then returns to the
coordinator's model, which can synthesize the final response.

If you want to expose each agent as its own tool instead, see
[Agents as Tools](index.md#agents-as-tools).

## Basic Usage

Create an `AgentTool` with the agents that should be available behind the
single `agent` tool, then add that `AgentTool` to the coordinator.

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

`AgentTool` also accepts runtime-injected `messages` and `vars`; those are
provided by msgFlux and are not exposed as normal model parameters. Before
dispatching, it only forwards those runtime values to the selected agent when
that agent's own `tool_config` requests them.

## Tool Bucket Capture

Internally, `AgentTool` is a `ToolBucket`. A bucket is a tool that absorbs other
tools of a specific kind and exposes them through one public tool. `AgentTool`
uses:

```python
class AgentTool(ToolBucket):
    capture = {"tool_kind": "agent", "on_demand": False}
```

`ToolBucket` supplies `tool_kind="bucket"`; the library stores that value in
the bucket's `tool_config`. Agents are registered with
`tool_config["tool_kind"]="agent"`. When a `ToolLibrary` contains an
`AgentTool`, adding an agent tool causes the library to route that agent into
the bucket instead of exposing it as a separate top-level tool:

```python
from msgflux.tools.builtin import AgentTool

library = nn.ToolLibrary(name="coordinator", tools=[AgentTool()])
library.add(Researcher())
library.add(Reviewer())

print(library.get_tool_names())
# ["agent"]
```

The bucket description is a compact list of the available agents. Explicit
`usage_guidance` on individual agents is aggregated separately, so the model
gets delegation guidance without inflating the tool description.

If no captured agent defines `usage_guidance`, the bucket guidance is `None` and
no guidance block is added to the system prompt.

`capture` can match any tool configuration field. `capture["tool_kind"]` can
also group multiple kinds with `|`, for example
`capture = {"tool_kind": "research|review", "on_demand": False}`. Overlapping
captures are rejected, and the base `add()` method rejects duplicate captured
names and calls the bucket's `refresh()` hook so `AgentTool` can update its
description and usage guidance.

[Tool Search](tool-search.md) is compatible with `ToolBucket` capture.
On-demand agents can stay outside the active tool set until `tool_search` loads
them by exact name. When the agent is promoted into the `ToolLibrary`, the
existing `AgentTool` bucket captures it and makes it available as another
`agent(name, message)` target.

`AgentTool` also works with [Background Tasks](background-tasks.md). When
configured with background support, the selected subagent can run through the
same task dispatch, status, wait, and task-message flow as other background
tools.

## Dynamic Agent Registration

A tool can request `tools.register` access to register agents while the
coordinator is running. The model writes the subagent specification through
normal tool arguments.

When the library already contains `AgentTool()`, `handle.tools.register(agent)` follows the
same `ToolLibrary.add(...)` path as static registration. Because `nn.Agent`
uses `tool_kind="agent"`, the existing `AgentTool` bucket captures the new
agent instead of exposing it as another top-level tool.

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import AgentTool

# mf.set_envs(OPENAI_API_KEY="...")

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


@mf.tool_config(handle={"tools": ["register"]})
def create_specialist(
    agent_name: str,
    description: str,
    instructions: str,
    handle: mf.Hidden,
) -> str:
    """Create a specialist agent from a model-written specification."""
    specialist = nn.Agent(
        name=agent_name,
        model=model,
        description=description,
        instructions=instructions,
    )

    registered_name = handle.tools.register(specialist)
    return (
        f"Registered `{registered_name}`. Delegate with "
        f"agent(name='{registered_name}', message='...')."
    )


coordinator = nn.Agent(
    name="coordinator",
    model=model,
    instructions=(
        "When a missing specialist would help, call create_specialist with a "
        "clear name, description, and instructions. Then delegate through the "
        "agent tool."
    ),
    tools=[
        AgentTool(),
        create_specialist,
    ],
)
```

The model-facing schema for `create_specialist` contains `agent_name`,
`description`, and `instructions`. It does not contain `handle`; msgFlux injects
that runtime value because the parameter is annotated with `mf.Hidden`.

After `create_specialist` runs, the public tool list remains small:
`create_specialist(...)` creates specialists, and `agent(name, message)`
delegates to any captured specialist in the bucket.

```text
User input
    |
    v
+------------------------------+
| Coordinator Agent            |
|                              |
| model sees one public tool:  |
| agent(name, message)         |
+--------------+---------------+
               |
               | calls agent(
               |   name="reviewer",
               |   message="Check this answer"
               | )
               v
+------------------------------+
| ToolLibrary                  |
| exposes the AgentTool bucket |
+--------------+---------------+
               |
               v
+------------------------------+
| AgentTool                    |
| - resolves name              |
| - injects messages and vars  |
| - dispatches to one agent    |
+--------------+---------------+
               |
               | selected agent: reviewer
               v
+------------------------------+
| reviewer subagent            |
| processes delegated message  |
+--------------+---------------+
               |
               | result
               v
+------------------------------+
| Coordinator model            |
| synthesizes final response   |
+--------------+---------------+
               |
               v
Output
```

## AgentTool Examples

These examples use `AgentTool`, so the coordinator exposes one public
`agent(name, message)` tool instead of exposing each specialist as a separate
top-level tool.

???+ example "AgentTool"

    === "Health Team"

        A coordinator exposes one `agent` tool backed by multiple specialists:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import AgentTool

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        nutritionist = nn.Agent(
            name="nutritionist",
            model=model,
            description="""Specialist in nutrition, diet planning, and healthy
            eating habits. Use for meal plans, dietary recommendations, and
            nutritional advice.""",
            system_message="You are a certified nutritionist.",
            instructions="""Create clear and practical meal plans tailored to
            the user's goals. Be objective, technical, and structured.""",
        )

        fitness_trainer = nn.Agent(
            name="fitness_trainer",
            model=model,
            description="""Specialist in fitness, exercise routines, and
            physical training. Use for workout plans, training schedules, and
            exercise guidance.""",
            system_message="You are a certified personal trainer.",
            instructions="""Design workout routines based on the user's fitness
            level and goals. Focus on safety and sustainable progression.""",
        )

        agent_tool = AgentTool([nutritionist, fitness_trainer])

        coordinator = nn.Agent(
            name="health_coordinator",
            model=model,
            system_message="You coordinate a team of health specialists.",
            instructions="""Use the agent tool when a specialist should handle
            the request. Choose `nutritionist` for diet questions and
            `fitness_trainer` for workout questions.""",
            tools=[agent_tool],
        )

        response = coordinator("I want to lose 10kg and build muscle")
        ```

        The model sees one tool named `agent`. To delegate, it calls that tool
        with a target name such as `nutritionist` or `fitness_trainer` and a
        message for the selected specialist.

    === "Bucket Capture"

        `AgentTool` can start empty and capture agent tools when they are added
        to the same tool library:

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import AgentTool

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        reviewer = nn.Agent(
            name="reviewer",
            model=model,
            description="Review answers for correctness and missing details.",
            instructions="Return a concise review with concrete corrections.",
        )

        planner = nn.Agent(
            name="planner",
            model=model,
            description="Break complex requests into ordered implementation steps.",
            instructions="Return a practical plan with risks and validation steps.",
        )

        coordinator = nn.Agent(
            name="coordinator",
            model=model,
            instructions="Delegate planning and review work through the agent tool.",
            tools=[AgentTool(), reviewer, planner],
        )

        response = coordinator("Plan and review a migration to a new provider")
        ```

        Because `AgentTool` is present, `reviewer` and `planner` are captured by
        the bucket. The model still sees only the single public `agent` tool.
