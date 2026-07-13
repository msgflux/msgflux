<p align="center">
  <img src="docs/assets/msgflux-wordmark.png" alt="msgFlux" width="720" />
</p>

<p align="center"><strong>Dynamic AI systems in Python.</strong></p>

<p align="center">
  <a href="https://msgflux.com/">Documentation</a> |
  <a href="https://msgflux.com/tutorials/">Tutorials</a> |
  <a href="https://msgflux.com/learn/nn/agent/quickstart/">Quickstart</a>
</p>

msgFlux is an open-source Python framework for building AI systems with pretrained models as composable software components. It is built for applications where the model is one part of a larger program: agents, tools, signatures, multimodal modules, shared messages, and explicit orchestration.

Its core mental model comes from PyTorch: modules compose into programs, execution is explicit, and system behavior lives in code structure. msgFlux also adopts typed signatures as a useful abstraction for LM workflows, while keeping prompting, tools, and data flow grounded in a broader module-oriented architecture.

## Why msgFlux

- Build AI systems, not isolated prompts.
- Treat signatures, prompts, tools, and message flow as first-class program structure.
- Compose `nn.Agent`, `nn.Module`, `nn.Transcriber`, `nn.Speaker`, `nn.Embedder`, and `nn.MediaMaker`.
- Choose imperative calls or declarative message binding per module.
- Run against OpenAI-compatible providers, hosted APIs, or self-hosted endpoints.

## Install

### Core

```bash
uv add msgflux
# or
pip install msgflux
```

### OpenAI and OpenAI-compatible providers

```bash
uv add "msgflux[openai]"
# or
pip install "msgflux[openai]"
```

More setup details: https://msgflux.com/dependency-management/

## Minimal Examples

### Imperative

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(OPENAI_API_KEY="...")


class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-5.6-luna")
    system_message = "You are a helpful support agent."
    instructions = "You are assisting {{ user_name }}."


agent = SupportAgent()
result = agent(
    "My dashboard is not loading after the last update.",
    vars={"user_name": "Alice"},
)

print(result)
```

### Declarative

```python
import msgflux as mf
import msgflux.nn as nn

mf.set_envs(OPENAI_API_KEY="...")


class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-5.6-luna")
    system_message = "You are a helpful support agent."
    instructions = "You are assisting {{ user_name }}."
    message_fields = {"task": "issue", "vars": "variables"}
    response_mode = "solution"


agent = SupportAgent()

msg = mf.Message()
msg.issue = "My dashboard is not loading after the last update."
msg.variables = {"user_name": "Alice"}

agent(msg)
print(msg.solution)
```

## Core Ideas

### AI systems, not ML systems

msgFlux targets software built with pretrained models as components inside larger applications. The focus is not training the model; it is designing the system around it.

### Declarative and imperative

A module can behave like a regular Python callable or bind itself to a shared message object. That choice lives at the module level, which makes it easier to mix direct control flow with pipeline-style orchestration.

### Programming and prompting

Prompting and programming are related, but not the same thing. msgFlux lets you write prompts directly when needed, while still treating signatures, schemas, and routing as explicit code-level structure.

### Modules compose into programs

Agents, transcribers, speakers, embedders, retrievers, and custom modules can be combined into larger pipelines, routers, and multimodal workflows.

## Learn More

- Documentation: https://msgflux.com/
- Agent quickstart: https://msgflux.com/learn/nn/agent/quickstart/
- Tutorials index: https://msgflux.com/tutorials/
- Dependency management: https://msgflux.com/dependency-management/
- Home page theory and architecture: https://msgflux.com/

## Example Tutorials

- Support Ticket Router: https://msgflux.com/tutorials/support-ticket-router/
- Meeting Assistant: https://msgflux.com/tutorials/meeting-assistant/
- PIX Assistant: https://msgflux.com/tutorials/pix-assistant/
- Plan Tool with Root Context: https://msgflux.com/tutorials/plan-tool-with-root-context/
