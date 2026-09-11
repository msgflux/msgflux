# Model Gateway

Use a `ModelGateway` when an Agent needs named model deployments, automatic
fallback, or time-based availability rules. The Agent accepts the Gateway as
its normal `model` and routes `model_preference` to it.

```python
# pip install msgflux
import msgflux as mf
import msgflux.nn as nn

# mf.set_envs(OPENAI_API_KEY="...")

gateway = mf.ModelGateway(
    [
        {
            "model_name": "fast",
            "model": "openai/gpt-5.6-luna",
            "description": "Fast for clear, repeatable tasks.",
        },
        {
            "model_name": "deep",
            "model": "openai/gpt-5.6-sol",
            "description": "Deep reasoning for ambiguous, multi-step tasks.",
        },
    ]
)

agent = nn.Agent("analyst", gateway)

incident_log = """\
09:02 - Scanner A stopped sending inventory updates.
09:07 - Orders continued reserving stock from the last known snapshot.
09:23 - Two orders had overlapping reservations for SKU-1842.
"""

quick_result = agent(
    f"Extract the affected SKU from this log:\n\n{incident_log}",
    model_preference="fast",
)
deep_result = agent(
    f"Explain the likely failure sequence in this log:\n\n{incident_log}",
    model_preference="deep",
)
```

`model_preference` must match a configured `model_name`. With the default
`fallback=True`, the selected deployment is tried first; if it fails or is
unavailable because of a time constraint, the Gateway tries another available
deployment.

Set `fallback=False` when the selected model must be the only attempted model:

```python
strict_gateway = mf.ModelGateway(
    [
        {
            "model_name": "fast",
            "model": "openai/gpt-5.6-luna",
            "description": "Fast for clear, repeatable tasks.",
        },
        {
            "model_name": "deep",
            "model": "openai/gpt-5.6-sol",
            "description": "Deep reasoning for ambiguous, multi-step tasks.",
        },
    ],
    fallback=False,
)
```

Without an explicit preference, a strict Gateway attempts only its first
deployment. A missing alias, restricted selected model, or model failure raises
instead of transitioning to another deployment.

Descriptions are optional for direct Gateway use. They become required when a
Gateway-backed Agent is captured by [`AgentTool`](tools/agent-tool.md#selecting-a-subagent-model),
because the coordinator needs those descriptions to select an appropriate
model.
