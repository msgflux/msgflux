# nn.Agent


The `Agent` is a `Module` that uses language models to solve tasks. It can handle multimodal data, interact with environments through tool calls, and manage complex workflows with structured outputs.

## ✦₊⁺ Overview

An `Agent` combines a language model with instructions and tools to accomplish tasks. The Agent module adopts a task decomposition strategy, allowing each part of a task to be treated in isolation.

### Key Features


- **Multimodal Support**: Handle text, images, audio, video, and files
- **Tool Calling**: Execute functions to interact with external systems
- **Generation Schemas**: Guides the model to generate typed responses, with support for reasoning strategies: Chain of Thought, ReAct, Self-Consistency
- **Flexible Configuration**: Customize behavior through message fields and config options
- **Template System**: Use Jinja templates for prompts and responses
- **Modular System Prompt**: Compose system prompts from independent components
- **Task Decomposition**: Break down complex tasks into manageable parts

## 1. **Contents**

| Topic | Description |
|-------|-------------|
| [Quick Start](quickstart.md) | Get started with a minimal agent, class-based definition (AutoParams) |
| [Async](async.md) | Asynchronous agent execution |
| [Streaming](streaming.md) | Real-time response streaming |
| [Prompt Cache Warmup](prompt-cache.md) | Warm provider prompt caches with the system prompt and tool schemas |
| [Reasoning](reasoning.md) | Model-level reasoning, `reasoning_in_response`, dual-queue streaming |
| [How to Debug an Agent](debug.md) | Inspection and debugging tools |
| [System Prompt Components](system-prompt.md) | Compose system prompts from components |
| [Generation Schemas](generation-schemas.md) | Structured outputs and reasoning strategies |
| [Task and Context](task-and-context.md) | Input handling, templates, multimodal, chat history |
| [Vars](vars.md) | Unified execution variable space |
| [Tools](tools.md) | Tool calling, configuration, MCP, agent-as-tool |
| [Signatures](signatures.md) | Declarative input/output specifications |
| [Hooks & Guards](hooks.md) | Input and output safety checks with configurable policy |
| [Model Gateway](model-gateway.md) | Multi-model routing |
| [Prefilling](prefilling.md) | Guide response format with prefilling |

## 2. **See Also**

- [Module](../module/index.md) - Base class for all nn components
- [Message](../message.md) - Structured message passing
- [Model](../../models/model.md) - Model factory and types
