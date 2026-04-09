---
date: 2026-04-09
authors:
  - vilsonrodrigues
title: Why msgFlux Exists
description: >
  msgFlux started from a simple goal: create an abstraction for AI systems that
  stays concise, explicit, and flexible enough to support both imperative and
  declarative development.
categories:
  - Architecture
---

# Why msgFlux Exists

I started msgFlux because I wanted an abstraction for AI systems that keeps code concise, easy to understand, and explicit about what it is doing.

The wrong abstraction gets expensive very quickly. Hide too much, and it becomes hard to see how data moves through the system. Expose too much, and every pipeline turns into prompts, wrappers, and glue code. I wanted something in between: a structure that helps organize the system without hiding it.

<!-- more -->

Calling a model is the easy part.

The real problem starts when the model becomes one component inside a larger application, with context, tools, multiple stages, structured inputs and outputs, and different modules collaborating.

At that point, the important questions are no longer "how do I generate text?" but:

- how do components connect to each other?
- where does each module read from?
- where does it write to?
- when should the flow stay explicit in code?
- when is it better to declare a contract and let the system orchestrate it?

I built msgFlux to make those questions part of the primary API, not an afterthought.

A good abstraction for AI systems should support both **imperative** and **declarative** development.

In the imperative style, I define the flow manually. I call a module, get a result back, transform it, route it, and decide the next step myself. This style is direct and useful when control flow should stay obvious at the call site.

In the declarative style, the module knows where to consume input from and where to produce output. Instead of manually passing every argument, I configure the input and output fields and let the component operate over a shared message object. This reduces wiring and makes composition easier.

To me, these two styles should not compete with each other. They solve different problems, so msgFlux treats this as a module-level choice.

In imperative mode, the code can look as direct as a function call:

```python
import msgflux as mf
import msgflux.nn as nn


class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You are a helpful support agent."


agent = SupportAgent()
result = agent("My dashboard is not loading.")
```

In declarative mode, the same kind of component can operate over a shared message:

```python
import msgflux as mf
import msgflux.nn as nn


class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You are a helpful support agent."
    message_fields = {"task": "issue"}
    response_mode = "solution"


msg = mf.Message()
msg.issue = "My dashboard is not loading."

agent = SupportAgent()
agent(msg)
print(msg.solution)

```

That duality is not incidental. It is one of the core properties I wanted from the abstraction.

Another idea that became important to me in msgFlux is `Inline`.

With `Inline`, the process logic lives separately from the modules and components that implement each step. Instead of hard-coding every transition from A to B and then to C in Python control flow, the route is described as a string expression that integrates with the declarative API.

That changes the cost of evolving a system.

If I want to alter the flow, I do not need to rewrite orchestration code across multiple files. In many cases, I only need to change one line. The modules remain the same, while the route between them can change independently.

That is useful for developers, because an application can change its processing logic without being tightly bound to hard-coded orchestration. But it is also useful for AI agents operating over msgFlux itself: an agent can change the logic of a pipeline without having to emit raw control-flow code every time.

`Inline` is not just a convenience feature. It is part of the broader goal of making flow explicit, editable, and decoupled from the implementation details of each component.

Another design requirement was separating two complementary ways of defining behavior.

The first is direct prompting: writing a system prompt, instructions, and an expected output format manually. This still matters. Not every problem needs to be formalized as a contract before it becomes useful.

The second is treating prompts as programmable contracts, isolating **what** a component should do from **how** that behavior is expressed to the model. This is the main idea I adopted from DSPy in msgFlux: **signatures** are a very good abstraction for explicitly defining inputs and outputs in LM workflows.

I also cared a lot about API cleanliness. I did not want users wasting time trying to discover which namespace contains a particular helper, component, or dependency.

Good libraries create familiarity. Once you learn the shape of the API, the rest should feel predictable. That kind of consistency reduces friction, especially as a project grows.

In msgFlux, that means grouping essential helpers in sensible places, keeping core components easy to find, and avoiding an unnecessarily fragmented API surface.

msgFlux did not appear fully formed in its current shape. It is the third rewrite of the project.

The first version was called `llmflow`. Then it became `msgflow`. Now it is `msgflux`.

That history matters because the current design is not the result of a single pass. It comes from repeatedly testing ideas, keeping what felt structurally right, and discarding what made the system harder to reason about.

Each rewrite pushed the project closer to the same goal: a framework that stays explicit, composable, and practical without collapsing into glue code or over-designed indirection.

msgFlux does not come from a single source. I built it by combining patterns that have already proven their value in other libraries.

- **NumPy**, for its cohesive helper surface and practical organization.
- **PyTorch**, for modules, composition, component hierarchy, and explicit execution.
- **Hugging Face Transformers**, for central classes that organize core components such as models and retrievers.
- **DSPy**, in a much more specific way, for the idea of **signatures** as programmable contracts.

If I had to summarize the conceptual center of the project in one sentence, it would be this: msgFlux is much closer to **PyTorch for AI systems**, with additional abstractions for prompts, contracts, tools, and multimodal workflows.

In the end, my ambition for msgFlux is simple to describe, even if it is difficult to execute: provide an abstraction strong enough to organize complex systems, but light enough that it does not hide what is happening.

I did not want a framework that forces the user to choose between clarity and convenience.

I wanted a framework where it feels natural to:

- write manual flow when manual control is the right tool;
- declare consumption and production when composition matters more;
- use direct prompting when that is enough;
- use programmable contracts when they add robustness;
- build larger systems without losing readability.

If msgFlux works well, it should not feel magical. It should feel natural. That is the point.

There is still a lot I want to build into msgFlux.

One major direction is evolving `Inline` beyond in-place message mutation into a model that can also work with **incremental returns and deltas**. I am especially interested in pushing it toward a more Erlang-inspired direction, where flow is not only composable, but also better suited to incremental state transitions.

Another important step is **durability**. I want both `Inline` and `Agent` to become more resilient in the face of failures, with the ability to recover execution instead of treating errors as the end of the run.

I also want to introduce **Environments** for code execution, making it easier to treat execution contexts as first-class runtime components.

On the generation side, there are several schemas I want to add, including:

- `CodeAct`
- `ProgramOfThought`
- `RLM`
- `LKI`

`LKI` is especially important to me. `LKI` stands for `Late Knowledge Injection`, and it is a line of work of my own that I have been developing since January 2025. It follows the idea of making `vars` more central and more powerful. In msgFlux, `vars` is already a unified object, and I think the framework should go further in allowing models to manipulate variables directly and intentionally. That opens up a powerful design space, and it deserves much more prominence in the system.

I also want to add **optimizers**, in the same broad spirit that has proven useful elsewhere for LM systems, while adapting them to msgFlux's module-oriented architecture.

There are many other ideas around these pieces, but the direction is consistent: make AI systems easier to compose, easier to evolve, and more capable of adapting their own flow without collapsing into opaque hard-coded behavior.

I also want to thank my friend **Yvson**, who spent one year and eight months listening to my ideas about how I wanted to build msgFlux.

Long before this version existed, he heard the repeated discussions, the redesigns, and the attempts to shape the project into something coherent. That kind of patience matters more than it seems when a project goes through multiple rewrites before it finds its real form.
