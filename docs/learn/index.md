# Learn

Start with the [Quickstart](../quickstart.md) to run an agent with a tool.
Use this section to understand each component and configure it for your application.
You can read the topics independently.

## Agents and tools

An [Agent](nn/agent/index.md) connects a model, instructions, and tools.
[Tools](nn/agent/tools/index.md) let it call your code;
[Signatures](nn/agent/signatures.md) describe typed inputs and outputs.

Start here when building an assistant or a task-oriented agent. For the coding
agent you can run from a terminal, see [CLI](nn/agent/coding-cli.md).

## Models and data

[Models](models/model.md) provide inference across supported providers and
modalities. [Retrievers](retrievers/retriever.md) find information,
[Parsers](parsers.md) extract content, and [Storage](storage.md) explains
how to integrate application-owned databases.

Read these topics when connecting your agent or application to a particular
provider or data source.

## Composition

[Modules](nn/module/index.md) organize application behavior in Python classes.
Use them to combine agents with custom logic, or compose specialized components
such as transcription and retrieval. [Message](nn/message.md) provides shared
fields for components that exchange structured data.

For operating persistent agents, recovery, and memory management, see
[Guides](../guides/index.md). Internal architecture is documented in
[Anatomy](../anatomy/index.md).
