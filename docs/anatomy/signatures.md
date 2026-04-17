# Signatures

In msgFlux, a `Signature` is not only a nicer prompt syntax.

It is a contract compiler for `Agent`.

When a signature is present, the agent derives multiple internal pieces from a
single declaration:

- input annotations
- task template
- instructions
- expected output description
- output struct used as `generation_schema`

This is why signatures matter architecturally. They are part of how the agent
builds its runtime contract before any provider call happens.

## Two Input Forms

msgFlux supports two signature sources:

- inline string signatures
- class-based `Signature` subclasses

Examples:

```python
"question: str -> answer: str"
```

```python
class Translate(Signature):
    text: str = InputField()
    translation: str = OutputField()
```

These two forms enter the same compilation path in `Agent._set_signature(...)`.

## What `Agent._set_signature(...)` Does

Once `signature` is provided, `Agent` stops treating it as a lightweight helper.

It performs a structured compilation step:

```text
signature
  -> parse input fields
  -> parse output fields
  -> build task template
  -> build instructions
  -> build output struct
  -> build expected output text
  -> build annotations for agent/tool integration
```

This is why signatures influence more than prompting.

They shape both the user-facing API of the agent and the internal structured
output contract.

## String Signatures

Inline signatures are parsed from a compact DSL:

```text
input_a: str, input_b: int -> output_x: bool
```

That DSL is handled by `StructFactory._parse_annotations(...)` and
`StructFactory._parse_type_string(...)`.

The parsing step extracts field names and type strings, then turns the output
side into a real `msgspec.Struct`.

Conceptually:

```text
"text: str -> sentiment: str, confidence: float"
  -> parse fields
  -> output signature string
  -> StructFactory.from_signature(...)
  -> msgspec.Struct
```

## Class-Based Signatures

Class-based signatures provide more metadata than the inline string form.

They can contribute:

- field descriptions through `InputField(...)` and `OutputField(...)`
- instructions from the class docstring
- examples through `__examples__`

The class is read through `Signature` and `SignatureFactory`, which expose:

- `get_inputs_info()`
- `get_outputs_info()`
- `get_instructions()`
- `get_output_descriptions()`
- `get_examples_from_signature(...)`

So the class-based path is not just typed. It is also richer in metadata.

## The Role Of `StructFactory`

`StructFactory` is the bridge between signature notation and `msgspec.Struct`.

Its job is:

- parse field declarations
- parse nested type strings
- build runtime structs from signature outputs

That makes it a contract-construction utility, not just a parser.

It is especially important because the output struct created here becomes the
canonical structured output contract unless another outer `generation_schema`
wraps it.

## The Role Of `SignatureFactory`

`SignatureFactory` handles the prompt-facing and description-facing side of the
signature.

It derives:

- the task template from inputs
- the expected output description from outputs
- example data from the signature class

So the split is:

- `StructFactory`: build runtime types
- `SignatureFactory`: build prompt and expectation artifacts

Together, they turn a declarative signature into an executable agent contract.

## Signature And Agent Annotations

An easy detail to miss is that signatures also affect the agent's own
annotations.

`Agent._set_signature(...)` calls `generate_annotations_from_signature(...)` to
derive input annotations for agent-as-tool integration.

That means a signature changes both:

- how the model is prompted
- how the agent itself is exposed as a callable module/tool

This is one of the reasons signatures should be seen as contract builders, not
just prompt sugar.

## Signature And `generation_schema`

The most important special case is combining a signature with an explicit
`generation_schema`.

When that happens, msgFlux does not discard either side.

Instead:

- the signature output is converted to a struct
- the explicit `generation_schema` remains the outer schema
- the signature output becomes the type of `final_answer`

Conceptually:

```text
signature outputs
  -> OutputStruct

generation_schema
  -> has final_answer

merge
  -> final_answer: OutputStruct
  -> fused generation schema
```

If the outer schema uses `Optional[final_answer]`, the signature output is
wrapped accordingly.

This is what makes combinations such as:

- signature + `ChainOfThought`
- signature + `ReAct`
- signature + other reasoning schemas

work without changing the meaning of the outer reasoning flow.

## Contract Compilation View

The easiest way to understand signatures is as a pre-runtime compilation step.

```text
Signature
  -> inputs_info
  -> outputs_info
  -> task template
  -> instructions
  -> output struct
  -> expected output text
  -> generated annotations
  -> Agent buffers
```

Only after that compilation finishes does the usual runtime start:

```text
Agent
  -> prepare inputs
  -> build provider params
  -> execute model
  -> process response
```

So signatures live before provider execution, but they shape almost everything
that follows.

## ASCII Diagram

This is the full signature compilation path:

```text
inline string or Signature class
  -> Agent._set_signature(...)
       |
       +--> StructFactory
       |     -> parse outputs
       |     -> build msgspec.Struct
       |
       +--> SignatureFactory
       |     -> task template
       |     -> expected output
       |     -> examples
       |
       +--> signature instructions
       +--> generated input annotations
       +--> optional fusion with generation_schema
  -> Agent buffers updated
  -> runtime execution can start
```

And the fusion case:

```text
signature output struct
          +
outer generation_schema
          |
          v
 fused schema with typed final_answer
```

## Why This Shape Matters

If signatures were treated as plain prompt shortcuts, the system would need
separate configuration paths for:

- task templates
- expected output text
- annotations
- output structs

Instead, msgFlux lets one declaration produce all of those consistently.

That reduces duplication and keeps the public API declarative, while still
feeding a structured internal contract to the runtime.

## Related Pages

- [Agent](agent.md)
- [msgspec Transport Lowering](msgspec-transport-lowering.md)
- [Logical vs Provider Schema](logical-vs-provider-schema.md)
