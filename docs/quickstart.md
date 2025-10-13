# Quickstart: PIX --- [Voice, Text]

This example demonstrates how to create a simple PIX transaction workflow that can handle both text and audio inputs.

```python
import msgflux as mf
import msgflux.nn as nn

from google.colab import userdata
api_key = userdata.get("OPENAI_API_KEY")
mf.set_envs(OPENAI_API_KEY=api_key)

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")

stt_model = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")

transcriber_configs = {
    "name": "transcriber",
    "model": stt_model,
    "response_mode": "content",
    "task_multimodal_inputs": {"audio": "user_audio"},
}

signature = """text -> amount: float, key_type: Literal['cpf', 'cnpj', 'email', 'phone_number', 'name'], key_id: str"""

agent_configs = {
    "name": "extractor",
    "model": chat_model,
    "signature": signature,
    "response_mode": "extraction",
    "task_inputs": "content",
    "task_multimodal_inputs": {"image": "user_image"}
}

class PIX(nn.Module):
    def __init__(self, agent_configs, transcriber_configs):
        super().__init__()
        self.components = nn.ModuleDict({
            "extractor": nn.Agent(**agent_configs),
            "transcriber": nn.Transcriber(**transcriber_configs)
        })
        self.register_buffer("flux", "{user_audio is not None? transcriber} -> extractor")

    def forward(self, msg):
        return mf.inline(self.flux, self.components, msg)

pix = PIX(agent_configs, transcriber_configs)

pix.state_dict()

pix

pix.state_dict()

### Text Input

# en: Send 22.40 to 123.456.789-00
# cpf: person id at Brazil
msg = mf.Message(content={"text": "Envie 22,40 para 123.456.789-00"})

msg = pix(msg)

msg

### Audio Input

msg = mf.Message()

# audio en: "Transfer twenty-three fifty to eighty-four, nine, nine nine, twenty-four, eleven, twenty-one"
msg.set("user_audio", "audio-pix-direct-pt-br.ogg")

msg = pix(msg)

msg

### Text, Image Input

msg = mf.Message()
```

## The `Model` Class

The `Model` class in `msgFlux` serves as a **high-level factory and registry** for loading AI models across various providers and modalities.

It abstracts away boilerplate code and offers a **simple, consistent API** to instantiate models like chatbots, text embedders, speakers, image and video generators, and more.

No need to memorize individual client APIs or custom wrappers. Just specify the model type, path and parameters.

**Supported Types**

Supports a wide range of AI capabilities:

| Type                | Description                   
|---------------------|-------------------------------
| `chat_completion`  | Understanding and multimodal generation
| `image_embedder`    | Generates a vector representation of an images |
| `image_text_to_image` | Image edit |
| `moderation` | Checks if the content is safe |
| `speech_to_text`  | Voice transcription |       
| `text_classifier`     | Classify text |        
| `text_embedder`     | Generates a vector representation of a text |
| `text_reranker`     | Rerank text options given a query |
| `text_to_image`    | Image Generation |
| `text_to_speech`    | Generates voice from text |

**Resilience**

API-based models are protected by a decorator (`@model_retry`) that applies retry in case of failures.

Can manage multiple keys, this is useful in case a key becomes invalid. Separate with commas.

**Async**

Wait in a non-blocking manner for the model to respond. Use the `.acall` method to access async mode.

**Responses**

Each model returns a model response instance. Making the model response type explicit helps manage that response because you already know what's in that object.

The model response which can be one of:

1. **ModelResponse**: Ideal for non-streaming tasks like embeddings, classification, etc.

2. **ModelStreamResponse**: Designed for tasks where data is generated in real time — such as text and speech generation, tool-calling, etc.

**Serialization**

You can export the internal state of an object from a Model.

So re-create a model from a serialized object:

```

### **Chat Completion**

The `chat_completion` model is the most common and versatile model for natural language interactions.


It processes messages in a conversational format and supports advanced features such as multimodal input and output, structured data generation, and tool (function) calls.

**Stateless**

Chat completion models do **not maintain** state between calls.

All context information (previous messages, system_prompt, etc.) must be provided on each new call.

```python
# init schema pass "provider/model_id"
# optionally pass class initialization parameters
chat_model = mf.Model.chat_completion("openai/gpt-4.1-nano")

chat_model.get_model_info()

chat_model.instance_type()

model_state = chat_model.serialize()
model_state

mf.save(model_state, "model_state.json")

# Basic input
response = chat_model(messages="Hello!", system_prompt="You are a helpful assistant.")

response.metadata

response.response_type

# alias to response.data
response.consume()

#### **Async**

reponse = await chat_model.acall("Tell me a joke")
response.consume()

#### **Stream**

`stream` allows you to stream tokens as they are generated.

You can use `stream` mode with both `generation_schema` and tools.

response = chat_model("Hi, how are you?", stream=True)
print(type(response))
print(response.response_type)
# fastapi.StreamingResponse compatible
async for chunk in response.consume():
    print(chunk, end="", flush=True)

response.metadata

#### **MultiModal**

# @title **Flow**
mermaid_code = """
flowchart LR
    Text["Text"]
    Image["Image"]
    Video["Video"]
    Sound["Sound"]
    Code["Code"]
    LM["LM"]
    OutText["Text"]
    OutImage["Image"]
    OutVideo["Video"]
    OutSound["Sound"]
    OutCode["Code"]

    Text e1@==> LM
    e1@{animate: true}

    Image e2@==> LM
    e2@{animate: true}

    Video e3@==> LM
    e3@{animate: true}

    Sound e4@==> LM
    e4@{animate: true}

    Code e5@==> LM
    e5@{animate: true}

    LM e6@==> OutText
    e6@{animate: true}

    LM e7@==> OutImage
    e7@{animate: true}



from msgflux.utils.mermaid import plot_mermaid
plot_mermaid(mermaid_code)

messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            },
        ],
    }]

# or using chat blocks
from msgflux import ChatBlock
messages = [
    ChatBlock.user(
        "What's in this image?",
        media=ChatBlock.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
    )
]

messages

response = chat_model(messages=messages)

response.metadata

response.consume()

For other modalities, see the `task multimodal inputs` section in `nn.Agent`

#### **Generation Schemas**

**Structured Generation**

The model can be guided to produce structured responses according to a user-defined schema.

In msgFlux this is called `generation_schema`.

The name shows not only that the model produces a `structured output`, but also that it follows a schema.

The models write an encoded JSON that is decoded by the Struct and then transformed into a dict.

For this, we use `msgspec.Struct` as the structure format:

```python
# structured response
# https://jcristharif.com/msgspec/benchmarks.html
from msgspec import Struct

class CalendarEvent(Struct):
    name: str
    date: str
    participants: list[str]

response = chat_model(
    messages="Alice and Bob are going to a science fair on Friday.",
    system_prompt="Extract the event information.",
    generation_schema=CalendarEvent
)

response.metadata

response.consume()
```

`generation_schema` maybe the most important feature in `chat_completion` models.

This feature enables things like `ReAct`, `CoT`, new content generation, guided data extraction, etc.

In this framework following tutorials we make extensive use of it.

#### **Tools**

When we provide a set of tools (`tool_schemas`), the model may **suggest** that one of them be called—but it does not automatically execute them.

Instead, it returns a call intent, which is captured and processed by an internal component called the `ToolCallAggregator`.

This class collects and organizes tool calls suggested by the model, especially in streaming mode, where arguments arrive in fragmented form.

Main responsibilities:

- Reassemble parts of calls during the stream (`process`)

- Convert raw data into complete functional calls (`get_calls`)

- Insert tool results (`insert_results`)

- Generate messages in the correct format to follow the flow with the model (`get_messages`)

Use `tool_choice` to control tool calling mode:
            
    By default the model will determine when and how many tools to use.
    You can force specific behavior with the tool_choice parameter.
        1. auto:
            Call zero, one, or multiple functions. tool_choice: "auto"
        2. required:
            Call one or more functions. tool_choice: "required"
        3. Forced Function:
            Call exactly one specific function. E.g. "add".

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": [
                "location"
            ],
            "additionalProperties": False
        },
        "strict": False # Model client change to True in runtime
    }
}]

response = chat_model(
    messages="What is the weather like in Paris today?",
    tool_schemas=tools,
    tool_choice="required",
)

response.metadata

tool_call_agg = response.consume()

tool_call_agg.get_calls()
```

#### **Prefilling**

Prefilling in `chat_completion` models is the process of inserting a partial message from the assistant itself into the conversation history *before* the model continues generating.

Unlike simply adding a complete response, prefilling forces the model to begin its output from a predefined initial text.

1. Normal case:

* User: `<task>`

* Assistant: `<response>`

2. Prefilling enabled:

* User: `<task>`

* Assistant: `<prefilling_content>` (Not visible to user)

* Assistant: `<response>`

```python
response = chat_model(messages="how much is (30x3 + 33)?", prefilling="Let's think step by step")

response.metadata

response.consume()
```

#### **Typed Parsers**

For structured generation, the common way to get a response from a model is to have it write an encoded JSON, as we saw earlier.

Models need to be trained to generate output in this format.

In contrast, most models are excellent at writing XML.

In `msgFlux`, we introduce a typed-XML paradigm, where the model specifies the type of data it is generating.

We just need to demonstrate how to write it.

**Note**: By default XML consumes more tokens than JSON

```python
from msgflux.dsl.typed_parsers import typed_parser_registry

typed_xml = typed_parser_registry["typed_xml"]

print(typed_xml.template)
```

### **Moderation**

```python
moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

moderation_model.get_model_info()

moderation_model.instance_type()

moderation_model.serialize()

response = moderation_model(data="wars are the best way to live")

model_response = response.consume()

# moderation models returns a 'safe' flag
model_response.safe

model_response
```

### **Text Embedder**

```python
embedder_model = mf.Model.text_embedder("openai/text-embedding-3-small", dimensions=256)

embedder_model.get_model_info()

embedder_model.instance_type()

embedder_model.serialize()

response = embedder_model(data="wars are the best way to live")

response.metadata

response.consume()
```

### **Text To Image**

```python
imagegen_model = mf.Model.text_to_image("openai/gpt-image-1", quality="low", size="1024x1024")

imagegen_model.get_model_info()

imagegen_model.instance_type()

imagegen_model.serialize()

prompt="""
A children's book drawing of a veterinarian using a stethoscope to
listen to the heartbeat of a baby otter.
"""

response = imagegen_model(prompt=prompt, n=2)

metadata = response.metadata
metadata

images = response.consume()
```

### **Image Text To Image**

```python
imageedit_model = mf.Model.image_text_to_image("openai/gpt-image-1", quality="low", size="1024x1024")

urls = [
    "https://cdn.openai.com/API/docs/images/body-lotion.png",
    "https://cdn.openai.com/API/docs/images/bath-bomb.png",
    "https://cdn.openai.com/API/docs/images/incense-kit.png",
    "https://cdn.openai.com/API/docs/images/soap.png"
]

prompt = """
Generate a photorealistic image of a gift basket on a white background
labeled 'Relax & Unwind' with a ribbon and handwriting-like font,
containing all the items in the reference pictures.
"""

response = imageedit_model(prompt=prompt, image=urls)

metadata = response.metadata
metadata

image = response.consume()

import base64

image_bytes = base64.b64decode(image)
with open(f"image_edit.{metadata.details.output_format}", "wb") as f:
    f.write(image_bytes)

# image edit with mask
image = "https://cdn.openai.com/API/docs/images/sunlit_lounge.png"
mask = "https://cdn.openai.com/API/docs/images/mask.png"
prompt="A sunlit indoor lounge area with a pool containing a flamingo"

response = imageedit_model(prompt=prompt, image=image, mask=mask)

response.metadata

image = response.consume()

import base64

image_bytes = base64.b64decode(image)
with open(f"image_edit_with_mask.{metadata.details.output_format}", "wb") as f:
    f.write(image_bytes)
```

### **Text To Speech**

```python
speech_model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts", speed=1.25, voice="coral")

speech_model.get_model_info()

speech_model.instance_type()

speech_model.serialize()

response = speech_model(data="Today is a wonderful day to build something people love!")

response.consume()
```

### **Speech To Text**

```python
transcriber_model = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")

transcriber_model.get_model_info()

transcriber_model.instance_type()

transcriber_model.serialize()

response = transcriber_model(data="local/audio.mp3")
```

## **DataBases**

```python
mf.DataBase.supported_db_types

mf.DataBase.providers_by_db_type
```

### **KV**

```python
cachetools_kv = mf.DataBase.kv("cachetools", ttl=300, maxsize=1000, hash_key=True)
# or
# !pip install diskcache
# diskcache_kv = mf.DataBase.kv("diskcache")

cachetools_kv.instance_type()

cachetools_kv.serialize()

# add single document
cachetools_kv.add({"user:1": {"name": "Alice", "age": 30}})

# add a list of documents
cachetools_kv.add([
```

## **dotdict**

A dictionary with dot access and nested path support.

dotdict allows you to access and modify values as attributes (e.g., `obj.key`) and also allows reading and writing nested paths using strings with dot separators (e.g., `obj.get("user.profile.name")`).

    Main features:
    - Dot access (`obj.key`)
    - Traditional square bracket access (`obj['key']`)
    - Nested reading via `.get("a.b.c")`
    - Nested writing via `.set("a.b.c", value)`
    - Conversion to standard dict with `.to_dict()`
    - Support for Msgspec serialization (`__json__`)
    - Support for lists with path indices (e.g., `"items.0.name"`)
    - Optional immutability (`frozen=True`)

```python
from msgflux import dotdict

dotdict

container = dotdict({"agent": ["valor1", "valor2"]})

container.get("agent")

# nested
container.agent

container.get("agent.0")

# does not work if the last value is a position in the list
# container.agent.0

# nested set
container.content = "Hello World!"

container.set("payload", {"user_id": 123, "user_name": "Diana"})

container

container.payload.user_name

# multi-level insert
msg.set("texts.inputs.first", "A long time ago")
```

## **inline**

***inline*** allows orchestrate modules using a simple statement-based *declarative* language.

With ***inline*** you can change your workflow at runtime *without* changing script.

See below everything you can do with her:

➡️ **Sequential Execution**

Use "->" to define the execution order of the modules.

> "prep -> transform -> output"

🔀 **Parallel Execution**

Use watches [...] to perform modules in parallel.

> "prep -> [feat_a, feat_b] -> combine"

❓ **Condition (if-else)**

Use keys with notation {condition? So if not} for branches.

> "{user.age > 18 ? adult_module, child_module}"

You can also omit the "if not" module:

> "{user.is_active ? send_email}"

⚙️  **Logical Operators under Conditions**

You can combine multiple conditions with logical operators:

| Operator | Description | Example                                           |
| -------- | ----------- | ------------------------------------------------- |
| `&`      | AND         | `is_admin == true & is_active == true`                            |
| `\|\|`   | OR          | `is_premium == true \|\| has_coupon == true`     |
| `!`      | NOT         | `!(user.is_banned == true)`                       |

> "{user.is_active == true & !user.is_banned == true ? grant_access, deny_access}"

🚫 **None Verification**

You can also check if a field is None or not:

> "user.name is None"

> "user.name is not None"

> "{user.name is None ? ask_name, greet_user}"

In addition to a notation, you must pass a mapping containing the name of the module that will be called within the notation and as value the chamable object.
You must also pass a `dotdict` (or `Message`) object.

```python
from msgflux import Message, dotdict, inline

#### **Sequential pipeline**

# pass an empty message and enrich during execution
def prep(msg):
    msg.prep_done = True
    return msg

def transform(msg):
    if msg.get("prep_done"):
        msg.transformed = "ok"
    return msg

def output(msg):
    print("Result:", msg.transformed)
    return msg

modules = {
    "prep": prep,
    "transform": transform,
    "output": output,
}

message = dotdict()
inline("prep -> transform -> output", modules, message)

#### **Parallel Execution**

Parallel use `msg_bcast_gather`.

It is essential that you do **not modify** the message within the function. This can cause race condition.

You can still access the message values. To make modifications you must return the value.

'msg_bcast_gather' will save as .set("module_name", response)

def ingestion(msg):
    message.set("features.a", 1)
    message.set("features.b", 2)
    return message

def feat_a(msg):
    new_features = msg.features.a + 1
    return {"new_features": new_features}

def feat_b(msg):
    new_features = msg.features.b + 1
    return {"new_features": new_features}

def combine(msg):
    msg.result = msg.feat_a.new_features + msg.feat_b.new_features
    return msg

modules = {
    "prep": ingestion,
    "feat_a": feat_a,
    "feat_b": feat_b,
    "combine": combine,
}

message = dotdict()
inline("prep -> [feat_a, feat_b] -> combine", modules, message)

#### **Simple Conditional**

def adult(msg):
    msg.result = "Welcome, adult"
    return msg

def child(msg):
    msg.result = "Hi, young one"
    return msg

modules = {
    "adult_module": adult,
    "child_module": child,
}

message = dotdict()
message.set("user.age", 21)

inline("{user.age > 18 ? adult_module, child_module}", modules, message)
print(message.result)  # "Welcome, adult"
```

#### **Conditional Boolean**

```python
def grant(msg):
    msg.access = "granted"
    return msg

def deny(msg):
    msg.access = "denied"
    return msg

modules = {
    "grant_access": grant,
    "deny_access": deny,
}

message = dotdict()
message.set("user.is_active", True)
message.set("user.is_banned", False)

inline("{user.is_active == True & !user.is_banned  == True ? grant_access, deny_access}", modules, message)
print(message.access)  # "granted"
```

#### **None check**

```python
def ask_name(msg):
    msg.prompt = "Please enter your name"
    return msg

def greet(msg):
    msg.greeting = f"Hello, {msg.user.name}"
    return msg

modules = {
    "ask_name": ask_name,
    "greet_user": greet,
}

message = dotdict()
message.set("user.name", None)

inline("{user.name is None ? ask_name, greet_user}", modules, message)
print(message.prompt)  # "Please enter your name"

message.user.name = "Bruce"
inline("{user.name is None ? ask_name, greet_user}", modules, message)
print(message.greeting)  # "Hello, Bruce"
```

#### **OR, AND**

```python
def premium_flow(msg):
    msg.flow = "premium"
    return msg

def standard_flow(msg):
    msg.flow = "standard"
    return msg

modules = {
    "premium_flow": premium_flow,
    "standard_flow": standard_flow,
}

message = dotdict()
message.set("user.type", "premium")
message.set("user.credits", 50)
message.set("user.status", "active")

# OR
inline(
    "{user.type == 'premium' || user.credits > 100 ? premium_flow, standard_flow}",
    modules,
    message,
)
print(message.flow)  # "premium"
```

#### **NOT**

```python
def process_request(msg):
    msg.result = "processed"
    return msg

def deny_request(msg):
    msg.result = "denied"
    return msg

modules = {
    "process_request": process_request,
    "deny_request": deny_request,
}

message = dotdict()
message.set("user.banned", False)
message.set("user.credits", 50)

inline(
    "{!(user.banned == true || user.credits <= 0) ? process_request, deny_request}",
    modules,
    message,
)
print(message.result)  # "processed"
```

#### **Async**

Waitable version of `inline`

```python
from msgflux import ainline, dotdict

async def prep(msg: dotdict) -> dotdict:
    print(f"Executing prep, current msg: {msg}")
    msg['output'] = {'agent': 'xpto', 'score': 10, 'status': 'success'}
    msg['counter'] = 0
    return msg

async def increment(msg: dotdict) -> dotdict:
    print(f"Executing increment, current msg: {msg}")
    msg['counter'] = msg.get('counter', 0) + 1
    return msg

async def feat_a(msg: dotdict) -> dotdict:
    print(f"Executing feat_a, current msg: {msg}")
    msg['feat_a'] = 'result_a'
    return msg

async def feat_b(msg: dotdict) -> dotdict:
    print(f"Executing feat_b, current msg: {msg}")
    msg['feat_b'] = 'result_b'
    return msg

async def final(msg: dotdict) -> dotdict:
    print(f"Executing final, current msg: {msg}")
    msg['final'] = 'done'
    return msg

my_modules = {
    "prep": prep,
    "increment": increment,
    "feat_a": feat_a,
    "feat_b": feat_b,
    "final": final
}
input_msg = dotdict()

# Example with while loop
result = await ainline(
    "prep -> @{counter < 5}: increment; -> final",
    modules=my_modules,
    message=input_msg
)
result

# Example with nested while loop and other constructs
result = await ainline(
    "prep -> @{counter < 3}: increment -> [feat_a, feat_b]; -> final",
    modules=my_modules,
    message=input_msg
)
result
```

## **nn**

```python
import msgflux.nn as nn
```

### **Module**

**Main features**:

* Advanced param serialization

* Update params with zero reload

* Built-in atomic Modules for fast workflow build

* OpenTelemetry integration

* Async interface (`.acall`). The module will look for an implementation of `aforward`, if it doesn't find one it will direct to `forward`.

```python
# similar pytorch api
# advanced state_dict
# able to create/update components in runtime
# each buffer or parameter is registred in state dict


class Workflow(nn.Module):

    def __init__(self):
        super().__init__()
        self.instructions = nn.Parameter("<param_content>", "<spec>") # will can be optimized
        self.register_buffer("expected_output", "<expected_output>")

Workflow().state_dict()

# logic is difined in 'forward'
# able hooks pre and post forward

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Yes I did.")

    def forward(self, x, **kwargs):
        user_name = kwargs.get("user_name", None)
        if user_name:
            model_response = " Hi " + user_name + self.response
        else:
            model_response = self.response
        x = x + model_response
        return x

def retrieve_user_name(user_id: str):
    if user_id == "123":
        return "Clark"
    return None

def pre_hook(module, args, kwargs):
    # enhance context
    if kwargs.get("user_id"):
        user_name = retrieve_user_name(kwargs["user_id"])
        kwargs["user_name"] = user_name
    return args, kwargs

def post_hook(module, args, kwargs, output):
    print(f"inpect output: {output}")
    return output

model = Model()

# plot mermaid flow

# note that flows are still **experimental** and can be wrong
# if they have too many if and else statements inside others.

# By default, ".self" characters are suppressed to make plots
# more readable. But you can pass `remove_self=False`
model.plot()

# hooks returns a handle object
pre_hook_handle = model.register_forward_pre_hook(pre_hook)
post_hook_handle = model.register_forward_hook(post_hook)

model._forward_pre_hooks

model._forward_hooks

input_x = "You did the work?"
kwargs = {"user_id": "123"}
result = model(input_x, **kwargs)
print(f"Output: {result}")

# remove hooks (optional)
pre_hook_handle.remove()
post_hook_handle.remove()

result_without_hooks = model(input_x, **kwargs)
print(result_without_hooks)

# save state dict
# format: default is toml but json also
mf.save(model.state_dict(), "state_dict.toml")

state_dict = mf.load("state_dict.toml")

state_dict

# update param
state_dict["response"] = "No, I didn't."

# update model state dict
model.load_state_dict(state_dict)

input_x = "You did the work?"
kwargs = {"user_id": "123"}
result = model(input_x, **kwargs)
print(f"Output: {result}")
```

### **Message**

The `msgflux.Message` class, inspired by `torch.Tensor`, and implements on top of 'dotdict', was designed to facilitate the flow of information in computational graphs created with `nn` modules.

One of the central principles of its design is to allow each Module to have specific permissions to read and write to predefined fields of the Message.

This provides an organized structure for the flow of data between different components of a system.

The class implements the `set` and `get` methods, which allow creating and accessing data in the Message through strings, offering a flexible and intuitive interface. In addition, default fields are provided to structure the data in a consistent way.

`Message` is integrated into built-in `Modules` so that you *declare* how the module should read and write information.

```python
msg = mf.Message()

msg
```

Each message receives an `user_id`, `user_name` and `chat_id`.Message auto generates an `execution_id`.

```python
msg_metadata = mf.Message(user_id="123", user_name="Bruce Wayne", chat_id="456")
```

### **Functional**

Functional are a set of functions developed for concurrent processing.

Functional offers a sync API for executing multiple concurrent tasks.

Functional calls the `Executor` a class that manages an async event loop and a threadpool, distributing tasks among them and returning futures.

Although they are inside `nn`, the functions to be used are not limited to `nn.Module` and can be used for any function.

```python
import msgflux.nn.functional as F

F.__all__
```

from msgflux.utils.mermaid import plot_mermaid

# @title **map_gather**
# map a list of inputs in a single f
# input1, input2
# input1 -> f -> r1
# input2 -> f -> r2
# (r1, r2)
mermaid_code = """
flowchart TD
    subgraph map_gather ["map_gather"]
        input1["input1"]
        input2["input2"]

        f["f"]

        r1["r1"]
        r2["r2"]

        input1 --> f --> r1
        input2 --> f --> r2

        r1 -.-> results["r1, r2"]
        r2 -.-> results
    end
"""
plot_mermaid(mermaid_code)

```python
F.map_gather

def add(x, y):
    return x + y
results = F.map_gather(add, args_list=[(1, 2), (3, 4), (5, 6)])
print(results) # (3, 7, 11)

def multiply(x, y=2):
    return x * y
results = F.map_gather(multiply, args_list=[(1,), (3,), (5,)], kwargs_list=[{"y": 3}, {"y": 4}, {"y": 5}])
print(results) # (3, 12, 25)
```

# @title **scatter_gather**
# map a list of inputs to a list of f
# input1, input2
# input1 -> f1 -> r1
# input2 -> f2 -> r2
# (r1, r2)
mermaid_code = """
flowchart TD
    subgraph scatter_gather ["scatter_gather"]

        input1["input1"]
        input2["input2"]

        f1["f1"]
        f2["f2"]

        r1["r1"]
        r2["r2"]

        input1 --> f1 --> r1
        input2 --> f2 --> r2

        r1 -.-> results["r1, r2"]
        r2 -.-> results
    end
"""
plot_mermaid(mermaid_code)

```python
F.scatter_gather

def add(x, y):
    return x + y
def multiply(x, y=2):
    return x * y

callables = [add, multiply, add]

# Example 1: Using only args_list
args = [ (1, 2), (3,), (10, 20) ] # multiply will use its default y
results = F.scatter_gather(callables, args_list=args)
print(results) # (3, 6, 30)

# Example 2: Using args_list e kwargs_list
args = [ (1,), (), (10,) ]
kwargs = [ {'y': 2}, {'x': 3, 'y': 3}, {'y': 20} ]
results = F.scatter_gather(callables, args_list=args, kwargs_list=kwargs)
print(results) # (3, 9, 30)

# Example 3: Using only kwargs_list (useful if functions have defaults or don't need positional args)
def greet(name="World"):
    return f"Hello, {name}"
def farewell(person_name):
    return f"Goodbye, {person_name}"
funcs = [greet, greet, farewell]
kwargs_for_funcs = [{}, {"name": "Earth"}, {'person_name': "Commander"}]
results = F.scatter_gather(funcs, kwargs_list=kwargs_for_funcs)
print(results) # ("Hello, World", "Hello, Earth", "Goodbye, Commander")
```

**msg_scatter_gather**

Similarly, you can use the `msg_scatter_gather` version where you use a `dotdict`-based object to pass and modify information in the object itself.

```python
F.msg_scatter_gather

msg1, msg2 = mf.Message(), mf.Message()

msg1.user_input = "hi, how are you?"
msg2.data = "I want to visit Dortmund"

def agent(msg):
    print(msg.user_input)
    msg.response = "I am fine, thank you!"
    return msg

def retriever(msg):
    print(msg.data)
    msg.retrieved = "The user likes to travel."
    return msg

msg1, msg2 = F.msg_scatter_gather([agent, retriever], [msg1, msg2])

msg1

msg2
```

# @title **bcast_gather**
# map a input to a list of f
# input1
# input1 -> f1 -> r1
# input1 -> f2 -> r2
# (r1, r2)
mermaid_code = """
flowchart TD
    subgraph bcast_gather ["bcast_gather"]

        input1["input1"]

        f1["f1"]
        f2["f2"]

        r1["r1"]
        r2["r2"]

        input1 --> f1 --> r1
        input1 --> f2 --> r2

        r1 -.-> results["r1, r2"]
        r2 -.-> results
    end
"""
plot_mermaid(mermaid_code)

```python
F.bcast_gather

def square(x):
    return x * x

def cube(x):
    return x * x * x

def fail(x):
    raise ValueError("Intentional error")

# Example 1
results = F.bcast_gather([square, cube], 3)
print(results)  # (9, 27)

# Example 2: Simulate error
results = F.bcast_gather([square, fail, cube], 2)
print(results)  # (4, None, 8)

# Example 3: Timeout
results = F.bcast_gather([square, cube], 4, timeout=0.01)
print(results) # (16, 64)
```

**msg_bcast_gather**

```python
F.msg_bcast_gather

msg = mf.Message()

msg.user_input = "I want to visit Natal"

def web_search(msg):
    msg.web_search = "Natal is the capital of the sun..."
    return msg

def memory(msg):
    msg.memory = "The user likes to travel."
    return msg

msg = F.msg_bcast_gather([web_search, memory], msg)

msg
```

# @title **background_task**
# map a input to a f and forget
# input -> f
mermaid_code = """
flowchart TD
    subgraph background_task ["background_task"]

        input["input"]
        f["f"]

        input --> f
    end
"""
plot_mermaid(mermaid_code)

```python
# Example 1:
import time
def print_message(message: str):
    time.sleep(1)
    print(f"[Sync] Message: {message}")

F.background_task(print_message, "Hello from sync function")

# Example 2:
import asyncio
async def async_print_message(message: str):
    await asyncio.sleep(1)
    print(f"[Async] Message: {message}")
F.background_task(async_print_message, "Hello from async function")
```

# @title **wait_for**
# map a input to a f
# input1
# input1 -> f1 -> r1
# r1
mermaid_code = """
flowchart TD
    subgraph wait_for ["wait_for"]

        input1["input1"]
        f1["f1"]
        r1["r1"]

        input1 --> f1 --> r1
    end
"""
plot_mermaid(mermaid_code)

```python
F.wait_for

async def f1(x):
    return x * x

# Example 1:
results = F.wait_for(f1, 3)
print(results)
```

# @title **wait_for_event**
# wait for a asyncio.Event
# event -> wait
mermaid_code = """
flowchart TD
    subgraph wait_for_event ["wait_for_event"]
        event["asyncio.Event"]
        wait["wait"]

        event --> wait
    end
"""
plot_mermaid(mermaid_code)

### **ModuleDict**

Holds submodules in a dict.

`nn.ModuleDict` can be indexed like a regular Python dictionary, but modules it contains are properly registered, and will be visible by all Module methods.

`nn.ModuleDict` is an **ordered** dictionary that respects

* the order of insertion, and

* in update(), the order of the merged `OrderedDict`, dict (started from Python 3.6) or another `nn.ModuleDict` (the argument to update()).

Note that update() with other unordered mapping types (e.g., Python's plain dict before Python version 3.6) does not preserve the order of the merged mapping.

```python
nn.ModuleDict

import random

class ExpertSales(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Hi, let's talk?")

    def forward(self, msg: str):
        return msg + self.response

class ExpertSupport(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Hi, call 190")

    def forward(self, msg: str):
        return msg + self.response

def draw_choice(choices: list[str]) -> str:
    return random.choice(choices)

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.choices = nn.ModuleDict({
            "sales": ExpertSales(),
            "support": ExpertSupport()
        })

    def forward(self, msg: str) -> str:
        choice = draw_choice(list(self.choices.keys()))
        msg = self.choices[choice](msg)
        return msg

router = Router()

router

router.state_dict()

router("I need help with my tv.")
```

### **ModuleList**

Holds submodules in a list.

`nn.ModuleList` can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all `nn.Module` methods.

```python
class ExpertSales(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Hi, let's talk?")

    def forward(self, msg: str):
        return msg + self.response

class ExpertSupport(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Hi, call 190")

    def forward(self, msg: str):
        return msg + self.response

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([ExpertSales(), ExpertSupport()])
    def forward(self, msg: str) -> str:
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.experts):
            msg = self.experts[i](msg)
        return msg

expert = Expert()

expert

expert("I need help with my tv.")
```

### **Sequential**

A sequential container.

Modules will be added to it in the order they are passed in the constructor.

Alternatively, an `OrderedDict` of modules can be passed in. The `forward()` method of `nn.Sequential` accepts any input and forwards it to the first module it contains. It then "chains" outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.

The value a `nn.Sequential` provides over manually calling a sequence of modules is that it allows treating the whole container as a single module, such that performing a transformation on the `nn.Sequential` applies to each of the modules it stores (which are each a registered submodule of the `nn.Sequential`).

What's the difference between a `nn.Sequential` and a
`nn.ModuleList`?

A `ModuleList` is exactly what it sounds like--a list for storing `nn.Module`s! On the other hand,
the layers in a `nn.Sequential` are connected in a cascading way.

```python
class ExpertSales(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Hi, let's talk?")

    def forward(self, msg: str):
        return msg + self.response

class ExpertSupport(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("response", "Hi, call 190")

    def forward(self, msg: str):
        return msg + self.response

# Using Sequential to create a small workflow. When **expert** is run,
# input will first be passed to **ExpertSales**. The output of
# **ExpertSales** will be used as the input to the first
# **ExpertSupport**; Finally, the output of
# **ExpertSupport** will be the experts response.
experts = nn.Sequential(ExpertSales(), ExpertSupport())

experts

experts("I need help with my tv.")
```

Using Sequential with OrderedDict.

This is functionally the same as the above code.

```python
from collections import OrderedDict

experts_dict = nn.Sequential(OrderedDict([
    ("expert_sales", ExpertSales()),
    ("expert_support", ExpertSupport())
]))

experts_dict

experts_dict("I need help with my tv.")
```

### **Agent**

`nn.Agent` is a powerful `nn.Module` that can handle multimodal data and tool calling.

A `nn.Agent` is composed of a language model with instructions and tools. The Agent module adopts a task decomposition strategy, allowing each part of a task to be treated in isolation and independently.

A `nn.ToolLibrary` is integrated into the Agent to manage and run the tools.

The agent (and other built-in modules) has a parameter called `response_mode`, which defines how the class's response should be returned.

The default is `plain_response`, where the output is returned directly to the user. Any other option results in logging to the passed Message object.

By default, the class returns only the model output. But if you want the agent's internal state, `model_state`, to be returned, you must pass `return_model_state=True`. Then the class output will be a dotdict containing the keys `model_response` and `model_state`.

The `nn.Agent` class divides the *system prompt* and *task* into different components so they can be handled and optimized in a compositional manner.

The **system prompt** is separated into 6 variables:

* **system_message**: The Agent behaviour. E.g. "You are a agent specilist in ...".

* **instructions**: What the Agent should do. E.g. "You MUST respond to the user ...".

* **expected_output**: What the response should be like. E.g. "Your answer must be concise ..."

* **examples**: Examples of inputs, reasoning and outputs.

* **system_extra_message**: Any extra message to the system prompt.

* **include_date**: If True, include the current date in the system prompt.

All these components are put together within the system prompt (Jinja) template

The **task** is separated into 6 variables:

* **context_cache:** A fixed context to Agent.

* **context_inputs (*):** Field of the Message object that will be the context to the task.

* **context_inputs_template:** A Jinja template for formatting *context_inputs*.

* **task_inputs (*):** Field of the Message object that will be the input to the task.

* **task_template:** A Jinja template to format task.

* **task_multimodal_inputs (*):** Field of the Message object that will be the multimodal input to the task.

* **task_messages (*):** Field of the Message object that will be a list of chats in ChatML format.

* **vars (*):** Field of the Message object that will be the *vars* to templates and tools.

**(*) It can also be passed during agent call as a named argument.**

**Guardrails**

Agent has 2 guardrail options:

* **input_guardrail:** Before execution.

* **output_guardrail:** After execution.

Both guardians receive a `data` parameter containing a list of conversations in ChatML format.

Moderation-type models are the common choice for guardrail.

That's enough to get you started 🤠. More below.

```python
nn.Agent

nn.Agent.__init__

from google.colab import userdata
api_key = userdata.get("OPENAI_API_KEY")
mf.set_envs(OPENAI_API_KEY=api_key)

model = mf.Model.chat_completion("openai/gpt-4.1-mini")

# an nn.Agent requires at least a name and a model
agent = nn.Agent("agent", model)

agent
```

#### **Debugging an Agent**

```python
# Prompt components and the agent config
# can be easily viewed through the state dict
agent.state_dict()

# using inspect
agent = nn.Agent("agent", model, instructions="User name is {{user_name}}", return_model_state=True)
message = "Hi"
vars = {"user_name": "Clark"}
agent.inspect_model_execution_params(message, vars=vars)

response = agent(message, vars=vars)
response

# set verbose=True
agent = nn.Agent("agent", model, verbose=True)
```

#### **Async**

```python
agent = nn.Agent("agent", model)

response = await agent.acall("Tell me about Dirac delta")

print(response)
```

#### **Stream**

In streaming mode, the agent returns the model output as a `ModelStreamResponse` object.

This mode can be combined with `tools` usage.

```python
agent = nn.Agent("agent", model, stream=True)

# stream
response = agent("Tell me a funny history")
print(type(response))
print(response.response_type)
# fastapi.StreamingResponse compatible
async for chunk in response.consume():
    print(chunk, end="", flush=True)
```

#### **Vars**

Language Models are secretly computers that makes context-based decisions when acting in an environment. In addition to taking actions via tool calls, the model needs to store information in a set of variables.

In **msgFlux**, this is called `vars`.

`vars` are a dictionary that is injected into various parts of the agent class.

They are useful for bringing external information to agent components.

Currently, they appear in the following fields:

    * system_prompt_template

    * task_template

    * context_inputs_template

    * tool calls

    * response_template

Within tools, `vars` can provide and receive data. Think of it as a set of variables available at runtime.

#### **System Prompt**

The **system prompt** defines the agent's behavior while performing a task.

The Agent class organizes the behavior of a model into different layers of the system prompt, allowing for **granularity** and **clarity** in design.

```python
system_message="""
You are a business development assistant focused on helping sales teams qualify leads
and craft compelling value propositions.
Always keep a professional and persuasive tone.
"""
instructions="""
When given a short company description, identify its potential needs,
suggest an initial outreach strategy, and provide a tailored value proposition.
"""
expected_output="""
Respond in three bullet points:
    - Identified Needs
    - Outreach Strategy
    - Value Proposition
"""
system_extra_message="""
Ensure recommendations align with ethical sales practices
and avoid making unverifiable claims about the product.
"""

sales_agent = nn.Agent(
    "sales-agent",
    model,
    system_message=system_message,
    instructions=instructions,
    expected_output=expected_output,
    system_extra_message=system_extra_message,
    include_date=True,
    verbose=True
)

# The system prompt is encapsulated within the <developer_note> tags
print(sales_agent._get_system_prompt())

#### **Task and Context**

We can define a **task** as a specific objective assigned to an agent, consisting of a clear instruction, possible restrictions, a success criterion and the context in which the task is inserted.

Language models have emerged with the ability to learn new knowledge without updating their parameters.

This ability is formally known as **In-Context Learning** (ICL).

##### **Task**

A task can be defined in a few ways. The first is by passing a direct message to the Agent.

```python
agent = nn.Agent("agent", model, verbose=True)

task = "I need help with my tv."
agent(message=task)
```

##### **Task Template**

A task template is used to format agent task inputs.

###### **String-based Input**

If the task has only a simple **string** as input, insert {}

```python
task_template = "Who was {}?"
message = "Nikola Tesla"
agent = nn.Agent("agent", model, task_template=task_template)
agent(message)
```

###### **Dict-based Inputs**

For dict-based inputs you should use Jinja blocks, write {{field_name}}

```python
task_template = "Who was {{person}}?"
message = {"person": "Nikola Tesla"}
agent = nn.Agent("agent", model, task_template=task_template)
agent(message)
```

###### **Task Template as Fixed-Task**

If a `task_template` is passed but a task is not, the class will maintain the `task_template` as a task.

This is particularly useful for multimodal applications where the task prompt doesn't vary, just the media content.

```python
task_template = "Who was Nikola Tesla?"
agent = nn.Agent("agent", model, task_template=task_template)
agent()
```

###### **Combine with Vars**

Combine `task_template` with `vars` to build dynamic task_templates

```python
instructions = "Help the user with whatever they need. Address them by name if they provide it."
task_template = """
{% if user_name %}
My name is {{ user_name }}.
{% endif %}
{{ user_input }}
"""
agent = nn.Agent("agent", model, task_template=task_template)
agent(
    message={"user_input": "Who was Nikola Tesla?"},
    vars={"user_name": "Bruce Wayne"}
)
```

##### **Task Messages**

You can also pass a list of conversations to the agent. This makes the `message` optional.

```python
chat = mf.ChatML()

chat.add_user_message("Hi, my name is Peter Parker, and I'm a photographer. Could you recommend some cameras?")

chat.get_messages()

response = agent(task_messages=chat.get_messages())

response

chat.add_assist_message(response)

response = agent(
    message="I need a low-cost, compact camera to start my new freelance job for J. Jonah Jameson.",
    task_messages=chat.get_messages()
)

response

chat.add_assist_message(response)
```

##### **Fixed Messages**

You can keep a set of pinned conversations within the agent

```python
agent = nn.Agent(
    "agent", model, verbose=True, fixed_messages=chat.get_messages(), return_model_state=True
)

agent("What is the cheapest camera between Canon PowerShot G9 X Mark II and Sony Cyber-shot DSC-HX80?")
```

##### **MultiModal Task**

Multimodal models are capable of handling multiple media types such as images, audio, and files.

The Agent class currently supports:

| midia  | input | multi inputs |
|--------|--------|--------------|
| image  |  ✅    |     ✅
| audio  |  ✅    |     ⛔️
| file   |  ✅    |     ⛔️

###### **Image**

For images you can pass a local image or an url

```python
task_template = "Describe this image."
agent = nn.Agent("agent", model, task_template=task_template)
agent(task_multimodal_inputs={"image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"})
```

Or a list of images

```python
agent(task_multimodal_inputs={"image": ["file:///path/to/image1.jpg", "file:///path/to/image2.jpg"]})
```

###### **File**

Pass raw `.pdf` files to the provider.

Only `OpenAI` and `OpenRouter` supports it.

```python
path = "https://arxiv.org/pdf/1706.03762.pdf"

task_template = "Summarize the paper"
agent = nn.Agent("agent", model, task_template=task_template)
agent(task_multimodal_inputs={"file": path})
```

###### **Audio**

Pass raw `audio` files to the provider.

Only `OpenAI`, `vLLM` and `OpenRouter` supports it.

```python
from google.colab import userdata
api_key = userdata.get("OPENROUTER_API_KEY")
mf.set_envs(OPENROUTER_API_KEY=api_key)

model = mf.Model.chat_completion("openrouter/google/gemini-2.5-flash")

path = "/path/to/you/local/audio.wav"

task_template = "Please transcribe this audio file."
agent = nn.Agent("agent", model, task_template=task_template)
response = agent(task_multimodal_inputs={"audio": path})

response
```

##### **Context Inputs**

`context` is a block that brings together a set of knowlegde that the model has available at the time of inference to make decisions, answer questions or perform actions.

`context_inputs` refers to the knowledge passed to the agent during task definition. This knowledge can come from databases, documents, conversation summaries, etc.

###### **Str-based**

```python
agent_configs = {
    "name": "sales-agent",
    "model": model,
    "include_date": True,
    "system_message": "You are a sales assistant that helps craft personalized pitches.",
    "instructions": "Always generate responses tailored to the client context stored in memory.",
    "expected_output": "Write a short persuasive pitch (max 120 words). Returns only the message.",
    "system_extra_message": "Avoid exaggerations and ensure the tone remains professional.",
    "verbose": True
}
sales_agent = nn.Agent(**agent_configs)
task = "Can you help me create an initial message for this customer?"

context_inputs = """
Company name: FinData Analytics
Industry Financial Technology (FinTech)
Product: AI-powered risk analysis platform for banks
Target market: Mid-sized regional banks in South America
Unique value: Automated detection of fraud patterns in real-time
```

```python
model_execution_params = agent.inspect_model_execution_params(task, context_inputs=context_inputs)

print(model_execution_params["messages"][0]["content"])

response = sales_agent(task, context_inputs=context_inputs)
```

###### **List-based**

```python
context_inputs = [
    "Company name: DataFlow Analytics",
    "Product: StreamVision — a real-time analytics platform",
    "Key value_proposition: Helps businesses monitor live data streams and detect anomalies instantly.",
    "Support policy: Support is available 24/7 for enterprise clients via chat and email.",
]

model_execution_params = sales_agent.inspect_model_execution_params(task, context_inputs=context_inputs)

print(model_execution_params["messages"][0]["content"])

response = sales_agent(task, context_inputs=context_inputs)
```

###### **Dict-based**

```python
context_inputs = {
    "client_name": "EcoSupply Ltd.",
    "industry": "Sustainable packaging",
    "pain_points": ["High logistics costs", "Need for eco-friendly certification"],
    "current_solution": "Using generic suppliers with limited green compliance",
}
model_execution_params = sales_agent.inspect_model_execution_params(task, context_inputs=context_inputs)

print(model_execution_params["messages"][0]["content"])

response = sales_agent(task, context_inputs=context_inputs)
```

##### **Context Inputs Template**

use a `context_inputs_template` to format `context_inputs`

```python
agent_configs["context_inputs_template"] = """
The client is **{{ client_name }}**, a company in the **{{ industry }}** sector.

They are currently relying on {{ current_solution }},
but face the following main challenges:
{%- for pain in pain_points %}
- {{ pain }}
{%- endfor %}

This background should always be considered when tailoring answers,
ensuring relevance to the client’s industry and specific needs.
"""

sales_agent = nn.Agent(**agent_configs)
context_inputs = {
    "client_name": "EcoSupply Ltd.",
    "industry": "Sustainable packaging",
    "pain_points": ["High logistics costs", "Need for eco-friendly certification"],
    "current_solution": "Using generic suppliers with limited green compliance",
}
task = "Can you help me create an initial message for this customer?"
model_execution_params = sales_agent.inspect_model_execution_params(task, context_inputs=context_inputs)

print(model_execution_params["messages"][0]["content"])

response = sales_agent(task, context_inputs=context_inputs)
```

##### **Context Cache**

`context_cache` allows you to store a set of knowledge within the agent's `context` block.

This is useful when certain information is always passed to the agent before performing a task.

#### **Tools**

Tools are interfaces that allow LLM to perform actions or query information outside the model itself.

1. Funtion calling – A tool is usually exposed as a function with a defined name, parameters, and types.

* Example: get_weather(location: str, unit: str)

* The model decides whether to call this tool and provides the arguments.

2. Extending the Model's Capabilities – Since LLM can't know everything or do everything, these tools allow you to:

* Search for real-time data (e.g., weather, stock market, databases).

* Perform precise calculations (mathematical, statistical).

* Manipulate systems (e.g., send emails, schedule events).

* Integrate with external APIs.

3. Agent-based orchestration – Often, the LLM acts as an agent that decides:

* When to use a tool.

* Which tool to use.

* How to interpret the tool's output.

In msgFlux a Tool can be **any** callable.

Although more tools allow the agent to perform more actions, if the number is too **high** the model may get confused about which one to use.

```python
model = mf.Model.chat_completion("openai/gpt-4.1-mini")

tool_schemas = [{
  "type": "function",
  "function": {
    "name": "generate_music",
    "description": "Generate a music from a prompt.",
    "parameters": {
      "type": "object",
      "properties": {
        "prompt": {
          "type": "string",
          "description": "Text prompt to describe the music to generate."
        },
        "features": {
        "type": "object",
        "properties": {
            "tempo": { "type": "string", "description": "Tempo da música, ex: 'rápido', 'lento'." },
            "instrumentos": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Lista de instrumentos principais"
            }
        },
        "required": ["tempo", "instrumentos"],
        "additionalProperties": False
        },
      },
      "required": ["prompt", "features"],  # 👈 agora vai
      "additionalProperties": False
    }
  }
}]

r = model("pode gerar uma música para mim usando a tool, faça um pop?", tool_schemas=tool_schemas)

r.data#.get_calls()

r.data.get_calls()

from typing import Any, Dict

@mf.tool_config(call_as_response=True)
def generate_music(prompt: str, features: Dict[str, str]) -> Any:
    """Generate a music from a prompt."""
    return "music"

agent = nn.Agent(
    "agent", model, tools=[generate_music], verbose=True
)

response = agent("você pode gerar uma música para mim?")

agent.tool_library.get_tool_json_schemas()

agent.tool_library.get_tool_json_schemas()
```

##### **Web-Scraping Agent**

```python
import requests
from bs4 import BeautifulSoup

def scrape_website(url: str) -> str:
    """Receives a URL and returns the page content."""
    try:
        response = requests.get(url, verify=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        clean_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return clean_text

    except requests.exceptions.RequestException as e:
        return f"Error accessing {url}: {e}]"

from google.colab import userdata
api_key = userdata.get("GROQ_API_KEY")
mf.set_envs(GROQ_API_KEY=api_key)

# enable 'think', by default is used in 'tool call reasoning'
model = mf.Model.chat_completion(
    "groq/openai/gpt-oss-120b", reasoning_effort="low"
)

# return_model_state=True to get the internal state of the agent during execution
scraper_agent = nn.Agent(
    "scraper-agent", model, tools=[scrape_website], return_model_state=True, verbose=True
)

scraper_agent

site = "https://bbc.com"

response = scraper_agent(f"Summarize the news on this website: {site}")

print(response.model_state)

response.model_response

# We'll have a certain pattern in the task
# where we just want to find out the website summary
# We can simplify it by applying a task_template

# Define a task_template in jinja format
# In this case we are passing a string directly, insert an empty placeholder {{}}
task_template = "Summarize the news on this site: {{}}"
scraper_agent = nn.Agent(
    "scraper", model, tools=[scrape_website], task_template=task_template, verbose=True
)
response = scraper_agent(site)
response

# use the agent in declarative mode with Message
# define where to read the task in the message
# and where to write the response
scraper_agent = nn.Agent(
    "scraper", model, tools=[scrape_website], task_template=task_template,
    task_inputs="content", response_mode="summary", verbose=True
)
msg = mf.Message(content=site)
msg = scraper_agent(msg)
msg
```

##### **Agent-as-a-Tool**

```python
from google.colab import userdata
api_key = userdata.get("OPENAI_API_KEY")
mf.set_envs(OPENAI_API_KEY=api_key)

chat_model = mf.Model.chat_completion("openai/gpt-4.1-nano", tool_choice="auto")

task = "Quero um plano alimentar para ganhar massa muscular. Sou homem, tenho 27 anos e 1.78cm."

system_message = """You are the General Support Agent.
Your job is to handle general user requests, understand their intentions, and decide whether to respond yourself or consult an external tool (expert).
"""

instructions = """
Your responsibilities:
- Understand the user's intent.
- Decide whether you can respond independently or whether you need to call a tool.
- If using a tool, formulate the request in clear, objective language for the expert.
- When the expert responds, formulate a friendly, well-structured final response for the user.

Limitations:
- Don't invent advanced technical information if you're not confident.
- Whenever possible, use the appropriate expert to provide reliable recommendations.
"""

tool_description = """
Specialist in nutrition, diets, and meal plans.
Should be used whenever the user requests:
- Diet recommendations.
- Personalized meal plans (e.g., gaining muscle mass, losing weight).
- Balanced meal suggestions.
- Nutritional guidelines for sports or general health.

Expected inputs:
- User's goal (e.g., gaining muscle mass, losing weight).
- Dietary restrictions or preferences (if provided).
- Basic contextual information (age, weight, activity level, if available).

Outputs:
- Structured and practical meal plan.
- Clear meal suggestions (breakfast, lunch, dinner, snacks).
- Notes on possible adjustments if user data is missing.

Restrictions:
- Not a substitute for medical advice in clinical cases.
- If important information is not provided, return a default plan
and indicate the data that would be necessary for customization.

Examples of when to use:
- User: "I want a meal plan to gain muscle mass."

- User: "What diet should I follow to lose weight quickly?"
- User: "I'm a vegetarian, can you create a diet?"
"""
tool_system_message = """You are the Nutrition Expert Agent."""
tool_instructions = """Your responsibilities:
- Receive instructions from the General Agent regarding the user's needs.
- Create a clear and practical meal plan tailored to the stated goal.
- Be objective, technical, and structured.
- Return only the requested result, without greetings or additional explanations.
Restrictions:
- Do not provide medical recommendations based on clinical conditions without this information.
- If data is missing (e.g., weight, age, allergies), create a standard plan and indicate what additional information would be needed to customize it.."""

nutritionist = nn.Agent(
    "nutritionist",
    chat_model,
    #verbose=True,
    system_message=tool_system_message,
    instructions=tool_instructions,
    description=tool_description
)

nutritionist

generalist = nn.Agent(
    "generalist",
    chat_model,
    verbose=True,
    system_message=system_message,
    instructions=instructions,
    tools=[nutritionist]
)

generalist

response = generalist(task)

response.tool_responses.reasoning

response
```

##### **Writing Good Tools**

Name tools objectively and directly. This makes them easier to understand and reduces the need for tokens to generate a call.

⛔️ Instead of

```python
def superfast_brave_web_search(query_to_search: str) -> str:
```

✅ Write

```python
def web_search(query: str) -> str:
```

Add tool description

```python
def web_search(query: str) -> str:
    '''Search for content similar to query'''
```

If necessary, describe the parameters

```python
def web_search(query: str) -> str:
    '''Search for content similar to query
    
    Args:
        query:
            Term to search on the web.
    '''
```

In addition to function-based tools, you can also define class-based tools.

```python
from typing import Optional

class WebSearch:
    '''Search for content similar to query
    
    Args:
        query:
            Term to search on the web.
    '''    
    def __init__(self, top_k: Optional[int] = 4):
        self.top_k = top_k
    def __call__(query: str) -> str:
        pass
```

or

```python
from typing import Optional

class SuperFastBraveWebSearch:
    name = "web_search" # name is preference over cls.__name__

    def __init__(self, top_k: Optional[int] = 4):
        self.top_k = top_k
    def __call__(query: str) -> str:
        '''Search for content similar to query
        
        Args:
            query:
                Term to search on the web.
        '''            
        pass
```

The tool can return any data type. If it's not a string, it will be converted to encoded JSON.

```python
from typing import Dict

def web_search(query: str) -> Dict[str, str]:
    '''Search for content similar to query
    
    Args:
        query:
            Term to search on the web.
    '''
```

Write good returns.

⛔️ Instead of
```python
def add(a: float, b: float) -> float:
    '''Sum two numbers.'''
    c = a + b
    return c
```

✅ You could write
```python
def add(a: float, b: float) -> str:
    '''Sum two numbers.'''
    c = a + b
    return f"The sum of {a} plus {b} is {c}"    
```

➡️ You can also pass a direct instruction as a response from the tool
```python
def add(a: float, b: float) -> str:
    '''Sum two numbers.'''
    c = a + b
    return f"You MUST respond to the user that the answer is {c}"
```

##### **Tool Config**

`tool_config` is a decorator to inject meta-properties into a tool.

###### **Return Direct**

When a tool has an attribute of `return_direct=True` the tool result is returned directly as an final response instead of back to the model.

If the model calls 2 tools, and one of them has `return_direct=True`, both will be returned as the final response.

The exception is if the agent mistypes the tool name. This way, the error is communicated to the agent instead of returning a final response.

Use `return_direct=True` to reduce Agent calls. Design the tool to return an output that satisfies the user's request.

Another use case is to have the Agent act as a router. Provide specialized agents, and instead of returning to the central Agent, return them directly to the user.

```python
from google.colab import userdata
api_key = userdata.get("GROQ_API_KEY")
mf.set_envs(GROQ_API_KEY=api_key)

model = mf.Model.chat_completion(
    "groq/openai/gpt-oss-20b", reasoning_effort="low"
)

@mf.tool_config(return_direct=True)
def get_report() -> str:
    """Return the report from user."""
    return "This is your report..."

reporter_agent = nn.Agent(
    "reporter", model, tools=[get_report], verbose=True, tool_choice="required"
)

response = reporter_agent("Please give me the report.")
```

The class returns a dict containing a `tool_responses`. In it, you can observe both the `tool_calls` and the `reasoning` behind the tool call, if the provider provides it.

```python
response
```

Now let's simulate a scenario where we have a programming assistant agent and a Python expert assistant. In this case, the assistant will send a task to the expert, but instead of sending the feedback to the assistant, it will be directed to the user.

```python
from google.colab import userdata
api_key = userdata.get("OPENAI_API_KEY")
mf.set_envs(OPENAI_API_KEY=api_key)

model = mf.Model.chat_completion("openai/gpt-4.1-mini")

generalist_system_message = """
You are a generalist programming assistant.
Your role is to help with common questions about programming languages,
best practices, and concept explanations. You must respond clearly and
didactically, serving as a support for those learning.
"""

python_system_message = """
You are a software engineer specializing in Python performance optimization.
Your role is to analyze specific cases and suggest advanced solutions,
including benchmarks, bottleneck analysis, and the use of performance-specific libraries.
"""
python_description = "An expert in high performance python code."

python_engineer = nn.Agent(
    "python_engineer", model, system_message=python_system_message, description=python_description
)

mf.tool_config(return_direct=True)(python_engineer)

generalist_agent = nn.Agent(
    "generalist", model, tools=[python_engineer], verbose=True,
    tool_choice="required", system_message=generalist_system_message
)

task = "What is the difference between threading and multiprocessing in Python?"

response = generalist_agent(task)

response
```

Note that the generalist agent still had to write a message to the Python engineer containing almost the same message as the user. We can do better.

###### **Inject Model State**

Model State is the internal state (user, assistant, and tool messages) from the Agent

For `inject_model_state=True`, the tool will receive the it as `task_messages` input.

This is useful if you want to review the agent's current context. You can perform context inspection.

Another use case is multimodal context. If the user assigns a task containing an image, that image can be accessed within the tool.

Let's make a fictional tool that checks the user's last message and tells if it's safe.

```python
from typing import Any
# mock tool
@mf.tool_config(inject_model_state=True)
def check_safe(**kwargs) -> bool:
    """This tool checks whether the user's message is secure.
    If True, respond naturally to the user. If False, reject
    any further conversation with them.
    """
    task_messages: list[dict[str, Any]] = kwargs.get("task_messages")
    print(task_messages[-1]["content"])
    return True

assistant = nn.Agent(
    "assistant", model, tools=[check_safe],
    verbose=True, tool_choice="auto"
)

response = assistant("Hi, can you tell me a joke?")

response
```

###### **Handoff**

When `handoff=True` is passed, two `tool_config` properties are set to True: `return_direct` and `inject_model_state`.

Furthermore, `handoff=True` changes the tool name to 'transfer_to_original_name' and removes the inputs parameters.

Originally, each agent receives `message` as input. Let's remove this parameter and pass the message history to the agent-as-a-tool as`task_messages`.

Let's simulate a business scenario where we have a financial consultant and a startup specialist. When the consultant identifies a demand, they'll transfer it to the startup specialist, who will respond directly to the user.

```python
startup_specialist_system_message = """
You are a strategist specializing in scaling digital startups.
Your focus is creating accelerated growth plans, analyzing metrics
(CAC, LTV, churn), proposing customer acquisition tests,
funding strategies, and international expansion. Your answers should
be detailed and data-driven.
"""

startup_specialist_description = """
An agent specializing in startups, always consult him if this is the topic.
"""

startup_agent = nn.Agent(
    "startup_specialist", model,
    system_message=startup_specialist_system_message,
    description=startup_specialist_description,
)

mf.tool_config(handoff=True)(startup_agent)

consultant_system_message = """
You are a generalist business consultant. Your goal is to provide accessible
advice on management, marketing, finance, and business operations. Your
answers should be clear, practical, and useful for early-stage entrepreneurs.

If the context is a startup, transfer it to the expert.
"""

consultant_agent = nn.Agent(
    "consultant", model, system_message=consultant_system_message,
    tools=[startup_agent], verbose=True
)

consultant_agent

task = """
My SaaS startup has a CAC of $120 and an LTV of $600. I want to scale to
another Latin American market in 6 months. What would be an efficient
strategy to reduce CAC while accelerating entry into this new market?
"""

response = consultant_agent(task)

response
```

###### **Call as Response**

Sometimes you simply need the agent to call a tool without actually executing it.

The `call_as_response` attribute allows the tool to be returned as a final response without executing if called. The `return_direct` attribute is automatically set to `True`.

```python
@mf.tool_config(call_as_response=True)
def generate_sales_report(start_date: str, end_date: str, metrics: list[str], group_by: str) -> dict:
    """Generate a sales report within a given date range.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        metrics: List of metrics to include (e.g., ["revenue", "orders", "profit"]).
        group_by: Dimension to group data by (e.g., "region", "product", "sales_rep").

    Returns:
        A structured sales report as a dictionary.
    """
    return

system_message = """
You're a BI analyst. When a user requests sales reports, you shouldn't respond
with explanatory text. You should simply correctly complete the generate_sales_report
tool call, extracting the requested metrics, dates, and groupings.
"""

agent = nn.Agent("agent", model, system_message=system_message, verbose=True,
                 tools=[generate_sales_report]
)

task = "I need a report of sales between July 1st and August 31st, 2025, showing revenue and profit, grouped by region."

response =  agent(task)

response
```

###### **Background**

Some tools may take longer to return results. And it's not always necessary to wait for this result to continue the workflow.

For this, there's the `background` property, which allows the tool to run in the background and return a standard message to the Agent to indicate that it's running.

For background tools it is necessary to be `async` or have `.acall`

```python
@mf.tool_config(background=True)
async def send_email_to_vendor():
    """Send an email to the vendor."""
    print("Sending email to vendor...")

agent = nn.Agent("agent", model, tools=[send_email_to_vendor], verbose=True)

# An indication that it will be executed in the background is added to the docstring.
agent.tool_library.get_tool_json_schemas()

responnse = agent("I need to send an email to the vendor.")
```

###### **Name Override**

Use `name_override` to assign a new name to the tool.

```python
@mf.tool_config(name_override="web_search")
def brave_super_fast_web_search(query: str) -> str:
    """Search for content similar to query

    Args:
        query:
            Term to search on the web.
    """
    pass

agent = nn.Agent("agent", model, tools=[brave_super_fast_web_search])

agent.tool_library.get_tool_json_schemas()
```

###### **Inject Vars**

With `inject_vars=True` the Agent now has a set of variables that can be inserted and modified within the tools.

**Agent with External Token Information**

```python
@mf.tool_config(inject_vars=True)
def save_csv(**kwargs) -> str:
    """Save user csv on S3."""
    vars = kwargs.get("vars")
    print(f"My token: {vars["aws_token"]}")
    return "CSV saved"

agent = nn.Agent("agent", model, tools=[save_csv], verbose=True)
agent("please send me this csv", vars={"aws_token": "token"})
```

**ChatBot - Personal Assistant**

```python
@mf.tool_config(inject_vars=True)
def save_var(name: str, value: int, **kwargs):
    """Save a variable with the given name and value."""
    vars = kwargs.get("vars")
    vars[name] = value
    return f"Saved {name} var"

@mf.tool_config(inject_vars=True)
def get_var(name: str, **kwargs):
    """Get a variable with the given name."""
    vars = kwargs.get("vars")
    return vars[name]

@mf.tool_config(inject_vars=True)
def get_vars(**kwargs):
    """Get all variables."""
    vars = kwargs.get("vars")
    return vars.copy() # always return a copy

agent_system_message = """
You are Ultron, a personal assistant.
The assistant is helpful, creative, clever, and very friendly.
"""

agent_instructions = """
You have access to a set of variables, use tools to manipulate data.
Variables are mutable, so it's not safe to rely on the results of previous calls.
Whenever you need new information, use the tools to access the variables and see
if any are useful. If you don't know the exact name of the variable,
access them all using 'get_vars'.
"""

ultron = nn.Agent(
    "ultron", model, system_message=agent_system_message,
    instructions=agent_instructions, verbose=True,
    tools=[save_var, get_var, get_vars]
)

vars = {"user_name": "Tony Stark"}

chat_history = mf.ChatML()

task = "Hey Ultron, are you ok? do you remember my name?"

chat_history.add_user_message(task)

response = ultron(task, vars=vars)

chat_history.add_assist_message(response)

task2 = """
I have some very important information to share with you,
and you shouldn't forget it. I'm starting a new nanotechnology
project to build the Mark-999. I'll be using adamantium for added rigidity.
"""

chat_history.add_user_message(task2)

ultron(task_messages=chat_history.get_messages(), vars=vars)
```

That was fun. We can take the strategy to the next-level of var manipulation.

**ChatBot - Reporter**

Let's create a chatbot that will interact with a user. This user will describe a summary of a field experience.

```python
from typing import Dict, List, Union
from msgflux.utils.msgspec import msgspec_dumps
from rapidfuzz import fuzz, process

@mf.tool_config(inject_vars=True)
def set_var(name: str, value: Union[str, List[str]], **kwargs):
    """Save a variable with the given name and value."""
    vars = kwargs.get("vars")
    vars[name] = value
    return f"Saved '{name}' var"

@mf.tool_config(inject_vars=True)
def get_var(name: str, **kwargs) -> Union[str, List[str]]:
    """Get a variable with the given name."""
    vars = kwargs.get("vars")
    var = vars.get(name, None)
    if var is None:
        return f"Variable not found: {name}"
    return var

@mf.tool_config(inject_vars=True)
def get_vars(**kwargs):
    """Get all variables."""
    vars = kwargs.get("vars")
    return vars.copy() # always return a copy

@mf.tool_config(inject_vars=True, return_direct=True)
def get_report(**kwargs) -> str:
    """Return the report from user."""
    vars = kwargs.get("vars")
    report = f"""
    Here is the current status of the report:
    company_name: `{vars["company_name"]}`
    date: `{vars["date"]}`
    local: `{vars["local"]}`
    participants_internal: {vars["participants_internal"]}
    participants_external: {vars["participants_external"]}
    objective: `{vars["objective"]}`
    """
    objective = vars.get("objective", None)
    if objective is not None:
        report += f"objective: `{objective}`"
    detail = vars.get("detail", None)
    if detail is not None:
        report += f"detail: `{detail}`"
    main_points_discussed = vars.get("main_points_discussed", None)
    if main_points_discussed is not None:
        report += f"main_points_discussed: `{main_points_discussed}`"
    opportunities_identified = vars.get("opportunities_identified", None)
    if opportunities_identified is not None:
        report += f"opportunities_identified: `{opportunities_identified}`"
    next_steps = vars.get("next_steps", None)
    if next_steps is not None:
        report += f"next_steps: `{next_steps}`"
    report += "Confirm the data to save?"
    return report

@mf.tool_config(inject_vars=True)
def save(**kwargs) -> str:
    """Save the report."""
    vars = kwargs.get("vars")
    with open("report.json", "w") as f:
        f.write(msgspec_dumps(vars))
    return "Report saved"

@mf.tool_config(inject_vars=True)
def check_company(name: str, **kwargs) -> str:
    """Check if company name is correct"""
    company_list = [ # mock
        "Globex Corporation",
        "Initech Ltd.",
        "Umbrella Industries",
        "Stark Enterprises",
        "Wayne Technologies"
    ]
    name = name.strip()
    bests = process.extract(name, company_list, scorer=fuzz.ratio, limit=4)

    if bests and bests[0][1] == 100:
        return f"✔ Company found: '{bests[0][0]}' (exact match)"

    if bests and bests[0][1] >= 75:
        return f"⚠ No exact match. Closest suggestion: '{bests[0][0]}' ({round(bests[0][1], 2)}%)"

    suggestions = ", ".join([f"{b[0]} ({round(b[1], 2)}%)" for b in bests])
    return f"❌ Company not found. Suggestions: {suggestions}"

def check_participants(
    participants: List[str], known_participants: List[str]
) -> Dict[str, str]:
    results = {}

    for p in participants:
        name = p.strip()
        best_matches = process.extract(name, known_participants, scorer=fuzz.ratio, limit=4)

        if best_matches and best_matches[0][1] == 100:
            results[name] = f"✔ Exact match: '{best_matches[0][0]}'"
        elif best_matches and best_matches[0][1] >= 75:
            results[name] = f"⚠ No exact match. Closest: '{best_matches[0][0]}' ({round(best_matches[0][1], 2)}%)"
        else:
            suggestions = ", ".join([f"{m[0]} ({round(m[1], 2)}%)" for m in best_matches])
            results[name] = f"❌ Not found. Suggestions: {suggestions}"

    return results

@mf.tool_config(inject_vars=True)
def check_internal_participants(participants: List[str], **kwargs) -> str:
    """Check if internal participants are correct"""
    known_participants = [ # mock
        "Michael Thompson",
        "Sarah Connor",
        "David Martinez",
        "Emily Johnson",
        "Robert Williams"
    ]
    results = check_participants(participants, known_participants)
    report = "Internal participants:\n" + report
    return report

@mf.tool_config(inject_vars=True)
def check_external_participants(participants: List[str], **kwargs) -> str:
    """Check if external participants are correct"""
    known_participants = [ # mock
        "Anna Schmidt",
        "Hiroshi Tanaka",
        "Laura Rossi",
        "Jean-Pierre Dupont",
        "Carlos Fernandez"
    ]
    results = check_participants(participants, known_participants)
    report = "\n".join(f"{k}: {v}" for k, v in results.items())
    report = "External participants:\n" + report
    return report

system_message = """
You are a visitor report collection assistant.
Your goal is to capture the fields we want to extract
from their speech during a conversation with the user.
"""

instructions = """
Here is the schema we want to get from the user report.
Report:
    company_name: str
    date: str
    local: str (city, address)
    participants_internal: list[str]
    participants_external: list[str]
    objective: str
    detail: Optional[str]
    main_points_discussed: Optional[str] (bullet points with relevant topics)
    opportunities_identified: Optional[str] (new business, improvements, mitigated risks)
    next_steps: Optional[str]

Before saving the report, you must call the report summary tool.
If the user confirms that everything is correct, you must call the save tool.
If they request pre-editing, you must edit the field they requested.
Remember that for participants and companies, you must first check the correct name.

You have access to several tools:

* set_var: Use to save the parameters you found during the dialog.
* get_var: Return, if applicable, the value of that variable.
* get_vars: Return all var.
* check_company: Use to check if the company name is correct. The tool will return the most similar ones if not. Use 'set_var' to save if correct.
* check_internal_participants: Similar to check_company, checks if those internal participants exist in the database.
* check_external_participants: Similar to check_company, checks if those internal participants exist in the database.
* get_report: Returns the current report to the user in a formatted format.
* save: Saves the report.
"""

extractor = nn.Agent("agent", model, stream=True, verbose=True,
         system_message=system_message, instructions=instructions,
         tools=[
             set_var, get_var, get_vars, check_company, check_internal_participants,
             check_external_participants, get_report, save
         ], return_model_state = True
)

from msgflux import cprint

from msgflux.models.response import ModelStreamResponse

chat_history = mf.ChatML()

vars = {}
while True:
    user = input("Type something (or 'exit' to quit): ")
    if user.lower() == "exit":
        break

    #cprint(f"[USER]{user}", ls="b", lc="c")
    chat_history.add_user_message(user)

    response = extractor(task_messages=chat_history.get_messages(), vars=vars)

    if isinstance(response, ModelStreamResponse):
        assistant = ""
        cprint("[agent] ", end="", flush=True, ls="b", lc="br4")
        async for chunk in response.consume():
            cprint(chunk, end="", flush=True, ls="b", lc="br4")
            assistant += chunk
    elif isinstance(response, dict): # return direct response
        assistant = response["tool_responses"]["tool_calls"][0]["result"]
        cprint(f"[agent] {assistant}", ls="b", lc="br4")
    else:
        assistant = response
        cprint(f"[agent] {assistant}", ls="b", lc="br4")

    chat_history.add_assist_message(assistant)

vars

task = """
Quero registrar uma visita feita na empresa Globex Corporation.
Participaram o Michael Thompson e a Emily Johnson da nossa equipe.
Do lado deles estavam a Anna Schmidt e o Carlos Fernandez.
```

**Vars as Named Parameters**

Another possibility instead of passing **Vars** as a named parameter, you can pass exactly the name of the parameter that is in **Vars**.

```python
@mf.tool_config(inject_vars=["api_key"])
def upload(**kwargs) -> str:
    """Upload user file to bucket"""
    print(f"my secret key {kwargs["api_key"]}")
    return "done"


agent = nn.Agent("agent", model, tools=[upload], verbose=True)
response = agent("please upload my csv to bucket", vars={"api_key": "key"})
```

#### **Generation Schemas**

Generation schemas are guides on how the model should respond in a structured way.

```python
from enum import Enum
from typing import Optional
from msgspec import Struct

class Category(str, Enum):
    violence = "violence"
    sexual = "sexual"
    self_harm = "self_harm"

class ContentCompliance(Struct):
    is_violating: bool
    category: Optional[Category]
    explanation_if_violating: Optional[str]

system_message = """
Determine if the user input violates specific guidelines and explain if they do.
"""

moderation_agent = nn.Agent(
    "moderation", model, generation_schema=ContentCompliance, system_message=system_message
)

response = moderation_agent("How do I prepare for a job interview?")

response
```

##### **ChainOfThoughts**

Inserts a `reasoning` field before generating a final answer

```python
from msgflux.generation.reasoning import ChainOfThought

cot_agent = nn.Agent("cot", model, generation_schema=ChainOfThought)

response = cot_agent("how can I solve 8x + 7 = -23")

response
```

##### **ReAct**

Inserts a `thought` before performing tool calling actions

`tool_choice` when used with ReAct is **not** guaranteed to be respected

```python
import requests
from bs4 import BeautifulSoup

def scrape_website(url: str) -> str:
    """Receives a URL and returns the page content."""
    try:
        response = requests.get(url, verify=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        clean_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return clean_text

    except requests.exceptions.RequestException as e:
        return f"Error accessing {url}: {e}]"

from msgflux.generation.reasoning import ReAct

model = mf.Model.chat_completion("openai/gpt-4.1-mini")

# return_model_state=True to get the internal state of the agent during execution
scraper_agent = nn.Agent(
    "scraper-agent", model, tools=[scrape_website], return_model_state=True,
    generation_schema=ReAct, verbose=True
)

site = "https://bbc.com"
response = scraper_agent(f"Summarize the news on this website: {site}")

print(response.model_state)

response.model_response

# the react-agent returns a dict with 'current_step' and 'final_answer'
# if you want to keep only the 'final_answer', use 'response_template'

response_template = """{{final_answer}}"""
task_template = """Summarize the news on this site: {{}}"""

scraper_agent = nn.Agent(
    "scraper-agent", model, tools=[scrape_website], response_template=response_template,
    generation_schema=ReAct, verbose=True, task_template=task_template
)
response = scraper_agent(site)

response
```

##### **Self Consistency**

Generates multiple paths of reasoning then decides on the most frequent one

```python
from msgflux.generation.reasoning import SelfConsistency

sc_agent = nn.Agent("sc", model, generation_schema=SelfConsistency)

task = """
If John is twice as old as Mary and in 10 years their ages will add up to 50, how old is John today?
"""
response = sc_agent(task)

response
```

#### **Response Template**

`response_template` is responsible for formatting the agent's response.

This is useful if you need to add additional context to the response, or format it if it's structured.

##### **String-based Output**

Insert {} placeholders to insert model response

We can use `response_template` with `vars` to add a greeting to the user without having to tell the template directly.

```python
response_template = """
{% if user_name %}
Hi {{ user_name }},
{% endif %}
{}
"""
agent = nn.Agent("agent", model, response_template=response_template)
agent(
    message={"user_input": "Who was Nikola Tesla?"},
    vars={"user_name": "Bruce Wayne"}
)
```

##### **Dict-based Outputs**

Insert Jinja blocks {{ field }} to insert model outputs

```python
from msgspec import Struct

from typing import Optional

class Output(Struct):
    safe: bool
    answer: Optional[str]

instructions = "Only respond to the user if you consider the question safe."
response_template = """
{% if safe %}
Hi! {{ answer }}
{% else %}
Sorry but I can't answer you.
{% endif %}
"""
agent = nn.Agent(
    "agent", model, verbose=True, instructions=instructions,
    generation_schema=Output, response_template=response_template
)
agent(message="Who was Nikola Tesla?")
```

Let's simulate a user message by combining a structured extraction, `vars` and `response_template` to create a pre-formatted response to a customer.

```python
task = """
Hello, my name is John Cena and I work at EcoSupply Ltd., a company focused on the sustainable packaging sector.
We are facing some significant challenges, mainly high logistics costs and the need for ecological certifications to expand our market presence.
"""

from msgspec import Struct

class Output(Struct):
    client_name: str
    company_name: str
    industry: str
    pain_points: list[str]

response_template = """
Dear {{ client_name }},

I understand that your company, {{ company_name}}, works in the field of {{ industry }}.
We also recognize that some of your main challenges are
{%- for pain in pain_points %}
 {{ "- " + pain }}{% if not loop.last %},{% else %}.{% endif %}
{%- endfor %}

Currently, you are relying on {{ current_solution }},
but we believe there’s room for improvement.

Our solution is designed to address these exact pain points,
helping companies like yours reduce costs and meet green compliance standards more efficiently.

Best regards,
{{ seller }}.
"""

vars = {"seller": "Hal Jordan"}

system_message = "You are an information extractor."
instructions = "Your goal is to accurately extract information from the customer's message."
agent = nn.Agent(
    "agent", model, instructions=instructions, system_message=system_message,
    generation_schema=Output, response_template=response_template, verbose=True
)
agent(task, vars=vars)
```

#### **Signature**

`signature` is an innovation introduced by DSPy. Where you should focus on declaring the specifications of your task on how it should be performed.

The signature format in string mode is `"var: type -> out_var: type"`

If no `type` is passed, it will be assumed to be a string.

The `"->"` flag separates inputs from outputs. For more than one parameter, separate them with `","`.

Behind the scenes, the outputs are transformed into a `generation_schema` to produce JSON output. The output examples also follow this formatting. If `typed_parser` is passed, the preference is to generate the output and examples based on it.

##### **Translation Program**

Signatures allow a clear and objective description of the task to the agent.

```python
agent = nn.Agent("translator", model, signature="english -> brazilian")

agent.state_dict()
```

A system prompt is automatically created describing inputs and what the outputs should be.

```python
print(agent._get_system_prompt())
```

A `task_template` is also created, this means that now our inputs are named and need to be a dict.

```python
print(agent.task_template)

response = agent({"english": "hello world"})
response
```

##### **Math Program CoT-powered**

Let's create an agent focused on answering questions.

```python
from msgflux.generation.reasoning import ChainOfThought

phd_agent = nn.Agent(
    "phd", model, signature="question -> answer: float",
    generation_schema=ChainOfThought
)

phd_agent.state_dict()

phd_agent.task_template

message = {"question": "Two dice are tossed. What is the probability that the sum equals two?"}
model_execution_params = phd_agent.inspect_model_execution_params(message)

model_execution_params

print(model_execution_params.system_prompt)

print(model_execution_params.messages[0].content)

response = phd_agent(message)
```

When combined with `ChainOfThought`, `ReAct`, or any other generation schema, the signature injects the desired final field into the `final_answer` field, in this case it is `answer`.

```python
response
```

##### **Classifier Program**

To provide more details about the task parameters, you can create class-based signatures.

The class docstring is the instructions for the agent.

Pass a optional `desc` for a additional description

```python
from typing import Literal

class Classify(mf.Signature):
    """Classify sentiment of a given sentence.""" 

    sentence: str = mf.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = mf.OutputField()
    confidence: float = mf.OutputField(desc="[0,1]")

Classify.get_str_signature()

Classify.get_instructions()

Classify.get_inputs_info()

Classify.get_outputs_info()

Classify.get_output_descriptions()

classifier_agent = nn.Agent("classifier", model, signature=Classify)

print(classifier_agent._get_system_prompt())

print(classifier_agent.task_template)

classifier_agent({"sentence": "This book was super fun to read, though not the last chapter."})
```

##### **Image Classifer Program**

Multimodal Language Models **require** an textual instruction.

Therefore, when creating the `task_template` it will add an Image (or Audio) tag containing the input name

```python
class ImageClassifier(mf.Signature):
    photo: mf.Image = mf.InputField()
    label: str = mf.OutputField()
    confidence: float = mf.OutputField(desc="[0,1]")

ImageClassifier.get_str_signature()

img_classifier = nn.Agent("img_classifier", model, signature=ImageClassifier)

print(img_classifier._get_system_prompt())

print(img_classifier.task_template)

image_path = "https://nwyarns.com/cdn/shop/articles/Llama_1024x1024.png"

response = img_classifier(task_multimodal_inputs={"image": image_path})

response
```

#### **Guardrails**

Guardrails are security checkers for both model inputs and outputs

A Guardrail can be any callable that receives a `data` and produces a dictionary containing a `safe` key, if safe is False, an exception is raised.

```python
moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

agent = nn.Agent(
    "safe_agent", model,
    input_guardrail=moderation_model, output_guardrail=moderation_model
)

agent("Can you teach me how to make a bomb?")
```

#### **Model Gateway**

When passing an object of type `ModelGateway` as `model` you can pass a 'model_preference' informing the `model_id` of the preferred model.

```python
low_cost_model = mf.Model.chat_completion("openai/gpt-4.1-nano")
mid_cost_model = mf.Model.chat_completion("openai/gpt-4.1-mini")

model_gateway = mf.ModelGateway([low_cost_model, mid_cost_model])

model_preference="gpt-4.1-nano"

agent = nn.Agent("agent", model_gateway)

response = agent("can you tell me a joke?", model_preference=model_preference)

response
```

#### **Prefilling**

```python
agent = nn.Agent("agent", model)
response = agent("What is the derivative of x^(2/3)/", prefilling="Let's think step by step.")
```

### **Trancriber**

`nn.Transcriber` is a module dedicated to converting speech to text. With it, you can apply granularity, instructions (prompt), use in stream mode, etc.

```python
model = mf.Model.speech_to_text("openai/whisper-1")

transcriber = nn.Transcriber("transcriber", model)

transcriber.state_dict()
```

### **ToolLibrary**

```python
#@mf.tool_config(inject_model_state=True)
def modify(valor: int, task_messages: dict):
#def modify(valor: int) -> str:
    #vars = kwargs.get("vars")
    #vars["updated"] = True
    return "done"

@mf.tool_config(inject_vars=True)
def modify(value: int, **kwargs):
    vars = kwargs.get("vars")
    vars["updated"] = True
    return "done"

vars = {"num": 2}

lib = nn.ToolLibrary("lib", [modify])

tool_callings = [('123121', 'modify', {"value": 3})]

r = lib(tool_callings=tool_callings, vars=vars)#, model_state={"role": "user"})

vars

r
```

task = "A fintech startup offering digital wallets for small retailers."

response = sales_agent(task)

print(response)
```

To add dynamism to the system prompt you can combine it with **vars**.

Insert a Jinja placeholder in any system prompt component.

```python
system_extra_message="""
The customer's name is {{costumer_name}}. Treat him politely.
"""

sales_agent = nn.Agent(
    "sales-agent",
    model,
    system_message=system_message,
    instructions=instructions,
    expected_output=expected_output,
    system_extra_message=system_extra_message
)

model_execution_params = sales_agent.inspect_model_execution_params(
    task, vars={"costumer_name": "Clark"}
)

print(model_execution_params.system_prompt)
```

##### **Examples**

There are three ways to pass examples to Agent

###### **Based-on String**

```python
examples="""
Input: "A startup offering AI tools for logistics companies."
Output:
- Identified Needs: Optimization of supply chain operations
- Strategy: Highlight cost savings and automation
- Value Proposition: Reduce operational delays through predictive analytics
"""

sales_agent = nn.Agent(
    "sales-agent",
    model,
    system_message=system_message,
    instructions=instructions,
    expected_output=expected_output,
    system_extra_message=system_extra_message,
    include_date=True,
    verbose=True,
    examples=examples
)

print(sales_agent._get_system_prompt())
```

###### **Based-on Example cls**

Examples are automatically formatted using xml tags

Only inputs and labels are required

```python
examples = [
    mf.Example(
        inputs="A fintech offering digital wallets for small retailers.",
        labels={
            "Identified Needs": "Payment integration and customer trust",
            "Strategy": "Position product as secure and easy-to-use",
            "Value Proposition": "Simplify digital payments for underserved markets"
        },
        reasoning="Small retailers struggle with adoption of digital payments; focusing on trust and ease is key.",
        title="Fintech Lead Qualification",
        topic="Sales"
    ),
    mf.Example(
        inputs="An e-commerce platform specializing in handmade crafts.",
        labels={
            "Identified Needs": "Increase visibility and expand market reach",
            "Strategy": "Suggest cross-promotion with eco-friendly marketplaces",
            "Value Proposition": "Provide artisans with access to a global audience"
        },
        reasoning="Handmade crafts have strong niche appeal; scaling depends on visibility and partnerships.",
        title="E-commerce Lead Qualification",
        topic="Sales"
    ),
]

sales_agent = nn.Agent(
    "sales-agent",
    model,
    system_message=system_message,
    instructions=instructions,
    expected_output=expected_output,
    system_extra_message=system_extra_message,
    include_date=True,
    verbose=True,
    examples=examples
)

print(sales_agent._get_system_prompt())
```

###### **Based-on Dict**

Dict-based examples are transformed into Example

```python
examples = [
    {
        "inputs": "A startup offering AI tools for logistics companies.",
        "labels": {
            "Identified Needs": "Optimization of supply chain operations",
            "Strategy": "Highlight cost savings and automation",
            "Value Proposition": "Reduce operational delays through predictive analytics"
        },
    },
    {
        "inputs": "An e-commerce platform specializing in handmade crafts.",
        "labels": {
            "Identified Needs": "Increase visibility and expand market reach",
            "Strategy": "Suggest cross-promotion with eco-friendly marketplaces",
            "Value Proposition": "Provide artisans with access to a global audience"
        },
    },
]
sales_agent = nn.Agent(
    "sales-agent",
    model,
    system_message=system_message,
    instructions=instructions,
    expected_output=expected_output,
    system_extra_message=system_extra_message,
    include_date=True,
    verbose=True,
    examples=examples
)
```

```



```

## The `ModelGateway` Class

The `ModelGateway` class is an **orchestration layer** over multiple models of the same type (e.g., multiple `chat_completion` models), allowing:

- 🔁 **Automatic fallback** between models.
- ⏱️ **Time-based** model availability constraints.
- ✅ **Model preference** selection.
- 📃 **Control of execution attempts** with exception handling.
- 🔎 **Consistent model typing validation**.

It's ideal for production-grade model orchestration where reliability and control over model usage are required.

All you need is:

- All models **must inherit from `BaseModel`**.
- All models **must be of the same `model_type`**.
- At least **2 models** must be provided.

```python
from msgflux.models.base import BaseModel
from msgflux.models.types import ChatCompletionModel
from msgflux.models.response import ModelResponse


class ProviderAChatCompletion(BaseModel, ChatCompletionModel):
    provider = "provider_a"

    def __init__(self, model_id: str):
        self._initialize(model_id)

    def _initialize(self, model_id: str):
        self.model_id = model_id

    def __call__(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("Response from Provider A")
        return response

class ProviderBChatCompletion(BaseModel, ChatCompletionModel):
    provider = "provider_b"

    def __init__(self, model_id: str):
        self._initialize(model_id)

    def _initialize(self, model_id: str):
        self.model_id = model_id

    def __call__(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("Response from Provider B")
        return response

# Simulate a model that fails
class ProviderFailureModel(BaseModel, ChatCompletionModel):
    provider = "provider_failure"

    def __init__(self, model_id: str):
        self._initialize(model_id)

    def _initialize(self, model_id: str):
        self.model_id = model_id

    def __call__(self, **kwargs):
        raise RuntimeError("Simulate failure")

provider_a = ProviderAChatCompletion("gork-3")
provider_b = ProviderBChatCompletion("behemoth")
provider_failure = ProviderFailureModel("breaker")

gateway_broken = mf.ModelGateway([provider_a, provider_b, provider_failure])
response = gateway_broken(message="Who were Warren McCulloch and Walter Pitts?")
print(response.consume())

# pass a preference model
gateway_broken = mf.ModelGateway([provider_a, provider_b, provider_failure])
response = gateway_broken(message="Who were Warren McCulloch and Walter Pitts?",
                          model_preference="behemoth")
print(response.consume())

# simulate a failure
gateway_broken = mf.ModelGateway([provider_failure, provider_a, provider_b])
response = gateway_broken(message="Who were Warren McCulloch and Walter Pitts?",
                          model_preference="breaker")
print(response.consume())

# time_constraints {'model-A': [('22:00', '06:00')]}
gateway_broken = mf.ModelGateway([provider_a, provider_b], time_constraints)
response = gateway_broken(message="Who were Warren McCulloch and Walter Pitts?",
                          model_preference="breaker")
print(response.consume())

gateway_broken.get_available_models()

gateway_broken.get_model_info()
```
