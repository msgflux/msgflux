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

## Random selection

Pass the reserved value `model_preference="random"` to pick one deployment
uniformly at random instead of naming it:

```python
first = agent("Summarize the log:\n\n" + incident_log, model_preference="random")
second = agent("Draft the follow-up:\n\n" + incident_log, model_preference="random")
```

Every call draws again, so `first` and `second` can be served by different
deployments. The draw only considers deployments that are not restricted by
`time_constraints` at call time. With `fallback=True`, the drawn deployment is
tried first and the remaining ones follow in their configured order; with
`fallback=False`, only the drawn deployment is attempted. Direct Gateway
calls follow the same rule:

```python
response = gateway("Summarize this log.", model_preference="random")
```

The alias `random` is reserved and cannot name a deployment; a deployment
named `random` is rejected when the Gateway is created. The value is also not
a deployment alias for helpers such as `get_model_description` or
`set_reasoning_effort`, which keep validating real deployment names.

Streaming calls follow the same rule until the stream starts producing output.
If a deployment fails while setting up the stream, before emitting a response
type, content, commentary, reasoning, a tool call, or any stream event, the
Gateway treats that deployment as failed and tries the next one. Once any output
or activity has been exposed, the Gateway returns that stream as-is; a later
failure is reported by the Agent and is never retried, because retrying could
duplicate visible output or tool effects. Control failures such as abort, pause,
and interrupt are never routed to another deployment.

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
