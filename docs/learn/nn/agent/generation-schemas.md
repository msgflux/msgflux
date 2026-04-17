# Generation Schemas

`generation_schema` guide the model to respond in a pre-established structured format. By defining a schema using `msgspec.Struct`, the agent automatically constrains the model's output to match the expected structure, ensuring type-safe and predictable responses.

!!! tip "Performance"
    `msgspec` is the fastest validation and serialization library, which is why it was chosen to deliver maximum performance. See the [benchmarks](https://jcristharif.com/msgspec/benchmarks.html).

???+ example

    ```python
    # pip install msgflux[openai]
    from msgspec import Struct
    import msgflux as mf
    import msgflux.nn as nn

    # mf.set_envs(OPENAI_API_KEY="...")

    class ContentCheck(Struct):
        reason: str
        is_safe: bool

    class Moderator(nn.Agent):
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        generation_schema = ContentCheck
        config = {"verbose": True}

    agent = Moderator()
    result = agent(
        "Analyze this message: 'You are amazing and I appreciate your help!'"
    )
    print(result.is_safe)  # True/False
    ```

## Reasoning Schemas

msgFlux provides built-in generation schemas that implement common reasoning strategies. These schemas guide the model through structured thinking patterns before producing a response.

All reasoning schemas produce a `final_answer: str` field containing the model's concluded response.

---

### Chain of Thought

> Wei et al., 2022 — [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

**Chain of Thought (CoT)** is the simplest and most widely used reasoning schema. It prompts the model to articulate its reasoning step-by-step before committing to a final answer. By making the thinking process explicit and structured, the model is less likely to jump to incorrect conclusions — especially on math, logic, and multi-step problems.

The schema adds a single `reasoning` field whose description hint (`"Let's think step by step in order to"`) nudges the model to elaborate before responding.

```
                    Input
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  reasoning:                                 │
│    "Step 1: Subtract 7 from both sides...   │
│     Step 2: Divide both sides by 8...       │
│     Step 3: x = -3.75"                      │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
              [ final_answer: "-3.75" ]
```

**Schema fields:**

| Field          | Type  | Description                                           |
|----------------|-------|-------------------------------------------------------|
| `reasoning`    | `str` | Step-by-step thinking that precedes the final answer  |
| `final_answer` | `str` | The concluded response based on the reasoning chain   |

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought

# mf.set_envs(OPENAI_API_KEY="...")

class Solver(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    generation_schema = ChainOfThought
    config = {"verbose": True}

agent = Solver()
result = agent("Solve: 8x + 7 = -23")

print(result.reasoning)    # "Step 1: Subtract 7... Step 2: Divide by 8..."
print(result.final_answer) # "x = -3.75"
```

!!! tip "When to use"
    CoT works best for problems that benefit from explicit decomposition: algebra, logic puzzles, multi-step reasoning, comparisons, or any task where the path to the answer matters as much as the answer itself.

---

### ReAct

> Yao et al., 2022 — [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

**ReAct** (Reasoning + Acting) is a dynamic schema designed for agents that need to use tools. Instead of reasoning in a single pass, the model interleaves `thought` (internal planning), `actions` (tool calls), and `observations` (tool results) in an iterative loop — repeating until it has enough information to produce a `final_answer`.

```
                    Input
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  thought:  "I need to look up the current   │
│             Python release on python.org"   │
│  actions:  [{name: "web_fetch",             │
│              args: [{name: "url",           │
│                      value: "..."}]}]       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
               [ Tool Execution ]
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  observations: [{tool: "web_fetch",         │
│                  result: "Python 3.13..."}] │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  thought:  "I now have the information I    │
│             need to answer the question"    │
└──────────────────────┬──────────────────────┘
                       │
                  Task complete?
                 /              \
               No               Yes
                │                 │
          Next cycle     [ final_answer: "3.14.x" ]
```

**Schema fields:**

| Field          | Type                   | Description                                      |
|----------------|------------------------|--------------------------------------------------|
| `thought`      | `str`                  | The agent's internal reasoning and plan          |
| `actions`      | `List[Action] \| None` | Tool calls to execute in this step               |
| `final_answer` | `str \| None`          | Set once the agent has all needed information    |

Each `Action` contains:

| Field       | Type              | Description                            |
|-------------|-------------------|----------------------------------------|
| `name`      | `str`             | The tool function to call              |
| `arguments` | `dict[str, Any]`  | Named arguments passed to the tool     |

!!! warning "Tools are serialized as text"
    Unlike standard tool calling, ReAct injects tool schemas into the system prompt as text descriptions rather than passing function definitions to the model's native `tools` parameter. This makes the loop more portable across models and providers, but changes how tools are represented internally.

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ReAct
from msgflux.tools.builtin import WebFetch

# mf.set_envs(OPENAI_API_KEY="...")

class WebResearcher(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    generation_schema = ReAct
    tools = [WebFetch]
    config = {"verbose": True}

agent = WebResearcher()
result = agent("What is the latest Python version from python.org?")

print(result.thought)       # "I need to fetch python.org to get the version..."
print(result.final_answer)  # "Python 3.14.x"
```

!!! tip "Default system_message"
    `ReAct` ships with a built-in `system_message` that instructs the model to follow the Thought → Action → Observation loop. You can inspect it with `ReAct.system_message`. It can be overridden by setting `system_message` on the agent, though that is generally not recommended — the default prompt is carefully tuned to keep the loop stable.

!!! tip "When to use"
    ReAct is the right choice when the agent needs external information to answer a question — web searches, API calls, database lookups, file reads, or any task requiring multi-turn tool interactions before an answer can be formed.

---

### Self Consistency

> Wang et al., 2022 — [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)

**Self-Consistency** reduces reasoning errors by generating multiple independent reasoning paths and selecting the most frequent answer through majority voting. Instead of relying on a single chain of thought, the model explores different approaches to the same problem and cross-checks its own conclusions — surface-level errors in one path get cancelled out by the others.

```
        Input
          │
          ├──▶ Path 1: "Distance ÷ Time = 120 ÷ 2..."  → answer: "60 km/h"
          │
          ├──▶ Path 2: "v = d/t, so 120/2 = ..."       → answer: "60 km/h"
          │
          └──▶ Path 3: "Speed formula: s × t = d..."   → answer: "65 km/h"
                                  │
                                  ▼
                           Majority Vote
                         ("60 km/h" wins 2/3)
                                  │
                                  ▼
                     [ final_answer: "60 km/h" ]
```

**Schema fields:**

| Field          | Type                   | Description                                         |
|----------------|------------------------|-----------------------------------------------------|
| `paths`        | `List[ReasoningPath]`  | Set of multiple independent reasoning paths         |
| `final_answer` | `str`                  | Answer chosen by majority vote across all paths     |

Each `ReasoningPath` contains:

| Field       | Type  | Description                             |
|-------------|-------|-----------------------------------------|
| `reasoning` | `str` | A single chain of thought for this path |
| `answer`    | `str` | The answer derived from this path       |

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import SelfConsistency

# mf.set_envs(OPENAI_API_KEY="...")

class Solver(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    generation_schema = SelfConsistency
    config = {"verbose": True}

agent = Solver()
result = agent("If a train travels 120km in 2 hours, what is its speed?")

for i, path in enumerate(result.paths, 1):
    print(f"Path {i}: {path.reasoning!r} → {path.answer!r}")

print(result.final_answer)  # "60 km/h"
```

!!! tip "When to use"
    Self-Consistency shines when accuracy is critical and the problem has a verifiable correct answer — math, science, logic, or any domain where multiple approaches can independently converge on the right result.

!!! warning "Token usage"
    Self-Consistency generates multiple reasoning paths in a single response, which increases output token consumption compared to CoT. The model decides how many paths to produce based on the complexity of the question.

---

## Extending Reasoning Schemas

You can extend any reasoning schema by inheriting from it and redefining the `final_answer` field with a custom type.

```
  ┌──────────────────────┐            ┌───────────────────────┐
  │   ChainOfThought     │            │    NumericAnswer      │
  ├──────────────────────┤   ─────▶   ├───────────────────────┤
  │ reasoning: str       │            │ reasoning: str        │  ← inherited
  │ final_answer: str    │            │ final_answer: int     │  ← overridden
  └──────────────────────┘            └───────────────────────┘

  ┌──────────────────────┐            ┌─────────────────────────────────────┐
  │   ChainOfThought     │            │       ReasonedDecision              │
  ├──────────────────────┤   ─────▶   ├─────────────────────────────────────┤
  │ reasoning: str       │            │ reasoning: str          ← inherited │
  │ final_answer: str    │            │ final_answer: Decision  ← overridden│
  └──────────────────────┘            └────────────────┬────────────────────┘
                                                       │
                                              ┌────────┴────────┐
                                              │    Decision     │
                                              ├─────────────────┤
                                              │ approved: bool  │
                                              │confidence: float│
                                              │justification:str│
                                              └─────────────────┘
```

!!! example

    === "Python Type"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ChainOfThought

        # mf.set_envs(OPENAI_API_KEY="...")

        class NumericAnswer(ChainOfThought):
            final_answer: int

        class Calculator(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = NumericAnswer
            config = {"verbose": True}

        agent = Calculator()
        result = agent("What is 25 + 17?")
        print(result.final_answer)  # 42 (int)
        ```

    === "Struct Type"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf
        import msgflux.nn as nn
        from msgspec import Struct
        from msgflux.generation.reasoning import ChainOfThought

        # mf.set_envs(OPENAI_API_KEY="...")

        class Decision(Struct):
            approved: bool
            confidence: float
            justification: str

        class ReasonedDecision(ChainOfThought):
            final_answer: Decision

        class Reviewer(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            generation_schema = ReasonedDecision
            config = {"verbose": True}

        agent = Reviewer()
        result = agent("Should we approve this budget request for $5000?")
        print(result.final_answer.approved)      # True/False
        print(result.final_answer.confidence)    # 0.85
        print(result.final_answer.justification) # "The request is within budget limits..."
        ```
