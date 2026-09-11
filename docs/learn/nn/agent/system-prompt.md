# System Prompt

An Agent has one canonical `system_prompt`. Put its stable role, instructions,
constraints, and expected output in this value:

```python
import msgflux as mf
import msgflux.nn as nn


class IncidentAnalyst(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-5.6-luna")
    system_prompt = """
    You analyze warehouse incidents.

    Identify the failure sequence, customer impact, and next actions.
    Return concise Markdown with one section for each topic.
    """


agent = IncidentAnalyst()
```

`system_prompt` is an `nn.Parameter`, so prompt optimizers can update one stable
source of truth. Runtime variables are rendered before extensions run:

```python
agent = nn.Agent(
    name="support_agent",
    model="openai/gpt-5.6-luna",
    system_prompt="You support {{ product_name }} customers.",
)

prompt = agent.get_system_prompt(vars={"product_name": "Warehouse Cloud"})
```

Extensions may add request-specific context to the model-facing prompt. They do
not mutate `agent.system_prompt`, so their content does not accumulate between
runs.

## Few-shot examples

`examples=` is a convenience that installs a removable
`FewShotExamplesExtension`. Examples are not another Agent parameter:

```python
examples = [
    mf.Example(
        inputs="Scanner A stopped sending updates.",
        labels="Reconcile inventory and pause affected reservations.",
        title="Scanner outage",
    )
]

agent = nn.Agent(
    name="incident_analyst",
    model="openai/gpt-5.6-luna",
    system_prompt="Recommend the next operational action.",
    examples=examples,
)

assert agent.has_extension("few_shot_examples")
assert "<examples>" in agent.get_system_prompt()
```

String examples are also accepted:

```python
agent = nn.Agent(
    name="classifier",
    model="openai/gpt-5.6-luna",
    system_prompt="Classify each request as billing or technical.",
    examples="""
    Input: I was charged twice.
    Output: billing

    Input: The scanner is offline.
    Output: technical
    """,
)
```

To manage the capability explicitly, install the same extension yourself:

```python
from msgflux.nn import FewShotExamplesExtension

extension = FewShotExamplesExtension(examples)
handle = agent.register_extension("few_shot_examples", extension)

handle.remove()
```

Do not pass both `examples=` and an explicitly registered extension named
`few_shot_examples`.
