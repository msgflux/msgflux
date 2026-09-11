# AgentTool

`AgentTool` is a built-in tool for exposing a group of agents through one
callable model tool:

```text
agent(name: str, message: str, model: str | None = None) -> str
```

Use it when a coordinator should delegate to specialized agents, but the model
should not see every specialist as a separate top-level tool. The coordinator
calls `agent(...)`, `AgentTool` resolves the requested agent by `name`, and the
selected agent processes the delegated `message`. The result then returns to the
coordinator's model, which can synthesize the final response.

If you want to expose each agent as its own tool instead, see
[Agents as Tools](index.md#agents-as-tools).

## Basic Usage

Add an empty `AgentTool()` and the specialist agents directly to the
coordinator's `tools`. Every `nn.Agent` is registered with
`tool_kind="agent"`, so the coordinator's `ToolLibrary` routes it into the
bucket automatically. The model sees one `agent(name, message)` tool rather
than one schema per specialist.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import AgentTool

model = mf.Model.chat_completion("openai/gpt-5.6-luna")

class Researcher(nn.Agent):
    model = model
    system_prompt = "Research the user's question and return concise findings."

class Reviewer(nn.Agent):
    model = model
    system_prompt = "Review the answer for correctness and missing details."

coordinator = nn.Agent(
    name="coordinator",
    model=model,
    system_prompt="Delegate specialized work to the agent tool when useful.",
    tools=[AgentTool(), Researcher(), Reviewer()],
)
```

Registration order does not matter. If the agents are added before the bucket,
registering `AgentTool()` discovers and captures the matching tools already in
the library. If the bucket is present first, each later agent is routed into it.

## Selecting A Subagent Model

Give a specialist a `ModelGateway` when the coordinator should be able to
choose the model for a delegated task. Each deployment needs a concise
`description`: `AgentTool` adds the aliases and descriptions to its public tool
description so the coordinator can make an informed choice.

```python
# pip install msgflux
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import AgentTool

# mf.set_envs(OPENAI_API_KEY="...")

review_models = mf.ModelGateway(
    [
        {
            "model_name": "fast",
            "model": "openai/gpt-5.6-luna",
            "description": "Fast for clear, repeatable review tasks.",
        },
        {
            "model_name": "deep",
            "model": "openai/gpt-5.6-sol",
            "description": (
                "Stronger reasoning for ambiguous changes and architecture."
            ),
        },
    ]
)

reviewer = nn.Agent(
    name="reviewer",
    model=review_models,
    description="Reviews code changes for correctness and maintainability.",
    system_prompt="Review the delegated change and return concise findings.",
)

coordinator = nn.Agent(
    name="coordinator",
    model="openai/gpt-5.6-sol",
    system_prompt=(
        "Delegate code reviews to reviewer. Choose fast for mechanical changes "
        "and deep for ambiguous or architectural changes."
    ),
    tools=[AgentTool(), reviewer],
)

result = coordinator(
    "Ask reviewer to analyze whether replacing a shared mutable model during "
    "concurrent requests is safe. Use the deep model."
)
print(result)
```

The model selector appears only while the bucket contains at least one Agent
whose model is a `ModelGateway`. Agents backed by a single model remain valid
targets. If the coordinator supplies `model` for a single-model Agent, the
preference is ignored and that Agent uses its configured model. If the selected
Agent has a Gateway, an unknown alias raises instead of silently falling back.

`model` is forwarded as the selected Agent's `model_preference`; the tool never
mutates `agent.model`. Concurrent delegations can therefore choose different
aliases safely. The Gateway keeps its normal fallback behavior: it tries the
selected deployment first and then another available deployment if necessary.
Set `fallback=False` on the Gateway when selection must be strict.

The selected alias is retained when a background Agent task is resumed. Model
descriptions are part of the Gateway's serialized state, so restored Agents
produce the same selection guidance.

!!! warning "Register agents through ToolLibrary"

    `AgentTool` does not accept agents in its constructor. Pass each agent in
    `tools=[AgentTool(), researcher, reviewer]` so registration, duplicate
    checks, bucket capture, and deferred-tool routing all use `ToolLibrary`.

`AgentTool` dispatches through its scoped bucket handle instead of calling an
agent implementation directly. `ToolLibrary` therefore remains responsible for
the selected agent's runtime-injected `messages` and `vars`, retries,
telemetry, and errors; none of those values appear as model parameters.
The default child scope uses the agent's canonical module name. A
`name_override` may change the public selector shown to the model without
changing the agent's checkpoint namespace.

## Tool Bucket Capture

Internally, `AgentTool` is a [ToolBucket](tool-bucket.md). A bucket is a tool that absorbs other
tools of a specific kind and exposes them through one public tool. `AgentTool`
uses:

```python
class AgentTool(ToolBucket):
    capture = {"tool_kind": "agent", "defer_loading": False}
```

`ToolBucket` supplies `tool_kind="bucket"`; the library stores that value in
the bucket's `tool_config`. Every `nn.Agent` supplies
`tool_config["tool_kind"]="agent"`. Therefore `AgentTool` captures any agent
registered in the same library whose configuration still matches
`defer_loading=False`; it does not need to know that agent's concrete class in
advance:

```python
from msgflux.tools.builtin import AgentTool

library = nn.ToolLibrary(name="coordinator", tools=[AgentTool()])
library.add(Researcher())
library.add(Reviewer())

print(library.get_tool_names())
# ["agent"]
```

The bucket stores only stable references to those agents. The library projects
their names, descriptions, usage guidance, and selectable gateway model aliases
into execution-free entries whenever membership changes. The bucket description
is therefore a compact list of the available agents without retaining or
calling agent implementations. Explicit
`usage_guidance` on individual agents is aggregated separately, so the model
gets delegation guidance without inflating the tool description.

The generic guidance for `agent` is opt-in through
`apply_tool_guidance([AgentTool()])`, like other builtin guidance entries.

`capture` can match any tool configuration field. `capture["tool_kind"]` can
also group multiple kinds with `|`, for example
`capture = {"tool_kind": "research|review", "defer_loading": False}`. Overlapping
captures are rejected, and the base `add()` method rejects duplicate captured
names and calls the bucket's `refresh()` hook so `AgentTool` can update its
description and usage guidance.

[Tool Search](tool-search.md) and `AgentTool` can coexist, but they provide
different public shapes. `AgentTool` deliberately captures only
`defer_loading=False` agents. An agent configured with `defer_loading=True` is
owned by the tool-search bucket and, after activation, is exposed under its own
tool schema rather than becoming an `agent(name, message)` target. Keep an
agent non-deferred when it must remain behind the single `AgentTool` entry
point. Loading a deferred agent therefore does not rewrite the `AgentTool`
description or schema, preserving a stable tool prefix for provider caching.

`AgentTool` also works with [Background Tasks](background-tasks.md). When
configured with background support, the selected subagent can run through the
same task dispatch, status, wait, and task-message flow as other background
tools.

Bucket membership belongs to the shared `ToolLibrary`. Adding an agent with
`handle.add(...)` makes it available to every concurrent call using that
library. Deferred activation is different: an agent marked
`defer_loading=True` stays in the tool-search catalog, and loading it is
recorded in that thread's `ChatMessages`. Because that agent does not match
`AgentTool.capture`, it appears as its own callable after loading.

## Dynamic Agent Registration

A tool can request `runtime_inputs=["handle"]` to register agents while the coordinator is
running. The model writes the subagent specification through normal tool
arguments, and the injected `handle` gives the Python tool access to the current
`ToolLibrary`.

When the library already contains `AgentTool()`, `handle.add(agent)` follows the
same `ToolLibrary.add(...)` path as static registration. Because `nn.Agent`
uses `tool_kind="agent"`, the existing `AgentTool` bucket captures the new
agent instead of exposing it as another top-level tool.

```python
# pip install msgflux
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import AgentTool

# mf.set_envs(OPENAI_API_KEY="...")

model = mf.Model.chat_completion("openai/gpt-5.6-luna")


@mf.tool_config(runtime_inputs=["handle"])
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
        system_prompt=instructions,
    )

    registered_name = handle.add(specialist)
    return (
        f"Registered `{registered_name}`. Delegate with "
        f"agent(name='{registered_name}', message='...')."
    )


coordinator = nn.Agent(
    name="coordinator",
    model=model,
    system_prompt=(
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
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import AgentTool

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-5.6-luna")

        nutritionist = nn.Agent(
            name="nutritionist",
            model=model,
            description="""Specialist in nutrition, diet planning, and healthy
            eating habits. Use for meal plans, dietary recommendations, and
            nutritional advice.""",
            system_prompt="""You are a certified nutritionist.
            Create clear and practical meal plans tailored to
            the user's goals. Be objective, technical, and structured.""",
        )

        fitness_trainer = nn.Agent(
            name="fitness_trainer",
            model=model,
            description="""Specialist in fitness, exercise routines, and
            physical training. Use for workout plans, training schedules, and
            exercise guidance.""",
            system_prompt="""You are a certified personal trainer.
            Design workout routines based on the user's fitness
            level and goals. Focus on safety and sustainable progression.""",
        )

        coordinator = nn.Agent(
            name="health_coordinator",
            model=model,
            system_prompt="""You coordinate a team of health specialists.
            Use the agent tool when a specialist should handle
            the request. Choose `nutritionist` for diet questions and
            `fitness_trainer` for workout questions.""",
            tools=[AgentTool(), nutritionist, fitness_trainer],
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
        # pip install msgflux
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.tools.builtin import AgentTool

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.chat_completion("openai/gpt-5.6-luna")

        reviewer = nn.Agent(
            name="reviewer",
            model=model,
            description="Review answers for correctness and missing details.",
            system_prompt="Return a concise review with concrete corrections.",
        )

        planner = nn.Agent(
            name="planner",
            model=model,
            description="Break complex requests into ordered implementation steps.",
            system_prompt="Return a practical plan with risks and validation steps.",
        )

        coordinator = nn.Agent(
            name="coordinator",
            model=model,
            system_prompt="Delegate planning and review work through the agent tool.",
            tools=[AgentTool(), reviewer, planner],
        )

        response = coordinator("Plan and review a migration to a new provider")
        ```

        Because `AgentTool` is present, `reviewer` and `planner` are captured by
        the bucket. The model still sees only the single public `agent` tool.
