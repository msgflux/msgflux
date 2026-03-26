# Quick Start

An Agent combines a language model with instructions and tools to accomplish tasks. Creating one takes just a few lines: pick a model, describe what the agent should do, and call it.

## Minimal Agent

The simplest agent needs only a name and a model:

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn

# mf.set_envs(OPENAI_API_KEY="...")

model = mf.Model.chat_completion("openai/gpt-4.1-mini")

agent = nn.Agent("assistant", model)

response = agent("What is the capital of France?")
print(response)  # "The capital of France is Paris."
```

The agent sends the input to the model and returns the response as a plain string. No conversation history is kept between calls — each call is independent.

## Adding Behavior

Use `system_message` and `instructions` to shape how the agent responds:

```python
agent = nn.Agent(
    "translator",
    model,
    system_message="You are a professional translator.",
    instructions="Translate the user's message to Brazilian Portuguese.",
    expected_output="Return only the translation, nothing else.",
)

response = agent("The weather is beautiful today.")
print(response)  # "O clima está lindo hoje."
```

These components are assembled into a system prompt sent to the model. See [System Prompt Components](system-prompt.md) for the full list.

## Class-based Definition (AutoParams)

The **preferred** way to define agents is using class attributes. This approach, called **AutoParams**, automatically captures the class name as the agent `name` and the docstring as the agent `description`:

```python
class Translator(nn.Agent):
    """Professional translator for Brazilian Portuguese."""

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You are a professional translator."
    instructions = "Translate the user's message to Brazilian Portuguese."
    expected_output = "Return only the translation, nothing else."

agent = Translator()

print(agent.name)         # "Translator"
print(agent.description)  # "Professional translator for Brazilian Portuguese."

response = agent("The weather is beautiful today.")
print(response)  # "O clima está lindo hoje."
```

Any `__init__` parameter can be set as a class attribute: `model`, `system_message`, `instructions`, `expected_output`, `examples`, `tools`, `config`, `templates`, `generation_schema`, `signature`, and others.

The `description` is especially useful when using an [agent as a tool](tools.md#agent-as-tool) — the calling agent uses it to understand what the tool-agent does.

## Adding Tools

Agents can call functions to interact with external systems. Any callable with type annotations and a docstring can be used as a tool:

```python
import subprocess

def run_shell(command: str) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to execute.

    Returns:
        Command stdout, or stderr if it fails.
    """
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=10
    )
    return result.stdout or result.stderr

class DevAssistant(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    instructions = "Help the user with system tasks using the shell."
    tools = [run_shell]

agent = DevAssistant()
response = agent("How much disk space is available?")
print(response)
# "You have 142 GB available out of 256 GB on your main disk."
```

The model decides when and how to call the tool. The Agent handles the full loop: model suggests a call, the function executes, the result goes back to the model, and the model produces the final answer. See [Tools](tools.md) for advanced usage.

## What's Next

| Topic | Description |
|-------|-------------|
| [System Prompt Components](system-prompt.md) | Customize behavior with system_message, instructions, examples |
| [Async](async.md) | Non-blocking execution with `acall` |
| [Streaming](streaming.md) | Real-time token-by-token output |
| [Tools](tools.md) | Function calling, MCP, agent-as-tool |
| [Generation Schemas](generation-schemas.md) | Structured output and reasoning strategies |
| [Reasoning](reasoning.md) | Model-level reasoning with `response.reasoning` |
