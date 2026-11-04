# Guides

Use these guides when an agent needs to keep working beyond a single call.
For your first agent, start with the [Quickstart](../quickstart.md).

## Persistent threads and independent runs

[Agent Service](../learn/nn/agent/service.md) manages threads and executions
independently of a connected client. Choose the
[local Python client](../learn/nn/agent/service-local.md) when embedding the
service, or the [HTTP API](../learn/nn/agent/service-http.md) for a separate client.
The [CLI](../learn/nn/agent/coding-cli.md) provides a ready-to-use coding harness.

## Execution and recovery

[Runtime](../learn/nn/agent/runtime.md) explains execution state, checkpoints,
and recovery. [Resources](../learn/nn/agent/resources.md) covers shared resources
and their lifecycle.

## Context and operating costs

[Compaction](../learn/nn/agent/compaction.md) manages the context sent to the
model as a conversation grows. [Prompt Cache](../learn/nn/agent/prompt-cache.md)
explains provider caching and usage reporting. Compaction changes model context;
it does not replace durable storage or checkpoint recovery.

## Troubleshooting

[Debug](../learn/nn/agent/debug.md) explains how to inspect agent behavior.
For service execution IDs, status, and recent tool calls, see the observation
commands in [CLI](../learn/nn/agent/coding-cli.md).
