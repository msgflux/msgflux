# Anatomy

The pages in this section document architectural choices that are easy to miss
when reading a single file in isolation.

They are not API tutorials. They describe the internal contracts that keep
`Agent`, `ToolFlowControl`, provider adapters, transport schemas, and runtime
normalization aligned.

Use this section when you need to answer questions like:

- Why does a flow control exist instead of extending the default tool loop?
- What is the difference between the logical generation schema and the provider
  response format?
- Why are some values lowered before they are sent to a model and restored
  afterward?
- Why does ReAct build a provider-specific schema instead of exposing its
  runtime shape directly?

## Pages

- [Agent](agent.md): the main orchestration module that ties together input
  preparation, provider execution, tool loops, and final response shaping.
- [Signatures](signatures.md): how a signature compiles templates,
  annotations, output structs, and optional reasoning-schema fusion.
- [ToolLibrary](tool-library.md): the execution boundary that registers tools,
  exposes schemas, restores typed arguments, and collects tool results.
- [OpenAI Chat Completion](openai-chat-completion.md): the provider adapter that
  translates msgFlux chat contracts into OpenAI request, schema, and decoding
  rules.
- [Chat Schema Utils](chat-schema-utils.md): schema-envelope helpers for
  `response_format`, tool JSON schema generation, and shared ChatML blocks.
- [msgspec Transport Lowering](msgspec-transport-lowering.md): the type
  translation layer that lowers logical schemas for providers and restores
  transport payloads afterward.
- [ToolFlowControl](tool-flow-control.md): extension point for custom tool loops
  without modifying the default `Agent` flow.
- [Task Runtime](task-runtime.md): the background-task contract for task state,
  progress reporting, and notification delivery.
- [Checkpoints And Replay](checkpoints-and-replay.md): the durability contract
  for `thread_id`, `run_id`, subagent recovery, and parallel worker
  replay.
- [Agent Inbox](agent-inbox.md): the notification primitive for runtime
  signals delivered to the model.
- [Logical vs Provider Schema](logical-vs-provider-schema.md): why msgFlux
  separates runtime shape from provider-facing schema.
- [Dict Lowering and Restoration](dict-lowering-and-restoration.md): how
  `dict[K, V]` is encoded for strict structured outputs and restored at runtime.
- [ReAct Provider Schemas](react-provider-schemas.md): how ReAct uses tool
  schemas to build a dynamic provider-facing response format.
- [LLM-as-a-Verifier](llm-as-a-verifier.md): how verifier prompts, concurrent
  execution, score-token parsing, and weighted aggregation fit together.
