# Prefilling

Prefilling injects a partial assistant message at the end of the message stack before the request reaches the provider. The model then continues from that text rather than generating from scratch.

## How It Works

**Without prefilling:**

```
App                    msgFlux                 Provider
 |                        |                       |
 |-- call(user_msg) ----→ |                       |
 |                        |-- [user: "..."] ----→ |
 |                        |                       |-- generates freely
 |                        |←-- "The answer is..." |
 |←-- response -----------|                       |
```

**With prefilling:**

```
App                       msgFlux                    Provider
 |                           |                          |
 |-- call(user_msg,          |                          |
 |    prefilling="Step 1:") →|                          |
 |                           |-- [user: "..."]        → |
 |                           |   [assistant: "Step 1:"] |
 |                           |                          |-- continues from "Step 1:"
 |                           |←-- "Step 1: ..."         |
 |←-- response --------------|                          |
```

The provider never sees `prefilling` as a parameter — msgFlux silently appends `{"role": "assistant", "content": <prefilling>}` to the message list before the request is sent. The response always begins with the prefilled text.

## Usage

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn

# mf.set_envs(OPENAI_API_KEY="...")

class Assistant(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

# Encourage step-by-step reasoning
agent = Assistant(prefilling="Let me solve this step by step.")
response = agent("What is the derivative of x^(2/3)?")

# Force specific format
agent = Assistant(prefilling="Here are the planets:\n1.")
response = agent("List the planets in our solar system")
```

## Use Cases

| Goal | Prefilling example |
|------|--------------------|
| Step-by-step reasoning | `"Let me solve this step by step."` |
| Numbered list | `"Here are the results:\n1."` |
| JSON output | `"{"` |
| Specific tone | `"Sure! Here's a fun explanation:"` |

!!! note "Provider support"
    Prefilling is supported by most providers. Some (like OpenAI) require the prefilled assistant message to not end with trailing whitespace. msgFlux does not strip or modify the value you pass.

## See Also

- [chat_completion](../../models/chat_completion.md#10-prefilling) — lower-level prefilling via `Model.chat_completion`
- [Generation Schemas](generation-schemas.md) — structured outputs as an alternative way to control response format
