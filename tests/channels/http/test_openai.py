import msgspec
import pytest

from msgflux.channels import ChannelRegistry
from msgflux.channels.http.openai import (
    create_chat_completion,
    create_chat_completion_stream,
    decode_chat_completion_request,
)
from msgflux.channels.http.schemas import ChatCompletionRequest
from msgflux.models.response import ModelStreamResponse


class FakeAgent:
    name = "support"

    def __init__(self, output):
        self.output = output
        self.calls = []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def test_decode_chat_completion_request_ignores_openai_fields():
    request = decode_chat_completion_request(
        msgspec.json.encode(
            {
                "model": "support",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.2,
                "run_config": {"vars": {"tenant": "acme"}},
            }
        )
    )

    assert request.model == "support"
    assert request.messages == [{"role": "user", "content": "hello"}]
    assert request.run_config == {"vars": {"tenant": "acme"}}


@pytest.mark.asyncio
async def test_create_chat_completion_calls_agent_with_run_config():
    registry = ChannelRegistry()
    agent = FakeAgent("hello")
    registry.agent(agent)
    request = ChatCompletionRequest(
        model="support",
        messages=[{"role": "user", "content": "hi"}],
        run_config={
            "vars": {"tenant": "acme"},
            "model_preference": "fast",
            "tool_filter": {"block": "*"},
        },
    )

    response = await create_chat_completion(registry, request)

    assert response.object == "chat.completion"
    assert response.model == "support"
    assert response.choices[0].message.content == "hello"
    assert agent.calls == [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "vars": {"tenant": "acme"},
            "model_preference": "fast",
            "tool_filter": {"block": "*"},
            "stream": False,
        }
    ]


@pytest.mark.asyncio
async def test_create_chat_completion_does_not_accept_run_config_kwargs():
    registry = ChannelRegistry()
    agent = FakeAgent("hello")
    registry.agent(agent)
    request = ChatCompletionRequest(
        model="support",
        messages=[{"role": "user", "content": "hi"}],
        run_config={
            "kwargs": {
                "stream": True,
                "messages": [],
                "task": "ignored",
            },
        },
    )

    await create_chat_completion(registry, request)

    assert agent.calls == [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "vars": {},
            "model_preference": None,
            "tool_filter": None,
            "stream": False,
        }
    ]


@pytest.mark.asyncio
async def test_create_chat_completion_applies_global_and_agent_defaults():
    registry = ChannelRegistry()
    agent = FakeAgent("hello")
    registry.agent(agent)
    registry.defaults(
        vars={"tenant": "default", "region": "us"},
        model_preference="global-fast",
        tool_filter={"block": "*"},
        kwargs={"task_context": "global context"},
        reasoning_policy={"effort": "low"},
    )
    registry.defaults(
        "support",
        vars={"tenant": "support"},
        model_preference="support-fast",
        tool_filter={"allow": ["search"]},
        kwargs={"task_context": "support context"},
    )
    seen_policies = []

    @registry.pre("support")
    def collect_policies(_request, _context, run):
        seen_policies.append(run.policies)

    request = ChatCompletionRequest(
        model="support",
        messages=[{"role": "user", "content": "hi"}],
        run_config={"vars": {"region": "br"}},
    )

    await create_chat_completion(registry, request)

    assert agent.calls == [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "vars": {"tenant": "support", "region": "br"},
            "model_preference": "support-fast",
            "tool_filter": {"allow": ["search"]},
            "stream": False,
            "task_context": "support context",
        }
    ]
    assert seen_policies == [{"reasoning": {"effort": "low"}}]


@pytest.mark.asyncio
async def test_create_chat_completion_request_overrides_defaults():
    registry = ChannelRegistry()
    agent = FakeAgent("hello")
    registry.agent(agent)
    registry.defaults(
        vars={"tenant": "default", "region": "us"},
        model_preference="global-fast",
        tool_filter={"block": "*"},
    )
    request = ChatCompletionRequest(
        model="support",
        messages=[{"role": "user", "content": "hi"}],
        run_config={
            "vars": {"tenant": "acme"},
            "model_preference": "request-fast",
            "tool_filter": {"allow": ["calculator"]},
        },
    )

    await create_chat_completion(registry, request)

    assert agent.calls == [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "vars": {"tenant": "acme", "region": "us"},
            "model_preference": "request-fast",
            "tool_filter": {"allow": ["calculator"]},
            "stream": False,
        }
    ]


@pytest.mark.asyncio
async def test_create_chat_completion_applies_pre_and_post_processors():
    registry = ChannelRegistry()
    agent = FakeAgent("hello")
    registry.agent(agent)

    @registry.pre("support")
    async def pre(request, context, run):
        context.state["seen"] = request.model
        return {
            "messages": [*run.messages, {"role": "system", "content": "extra"}],
            "vars": {"tenant": "acme"},
        }

    @registry.post("support")
    def post(output, context):
        return {"answer": f"{output} {context.state['seen']}", "reasoning": "because"}

    request = ChatCompletionRequest(
        model="support",
        messages=[{"role": "user", "content": "hi"}],
    )

    response = await create_chat_completion(registry, request)

    assert agent.calls[0]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "extra"},
    ]
    assert agent.calls[0]["vars"] == {"tenant": "acme"}
    message = response.choices[0].message
    assert message.content == "hello support"
    assert message.reasoning_content == "because"


@pytest.mark.asyncio
async def test_create_chat_completion_runs_request_hooks():
    registry = ChannelRegistry()
    agent = FakeAgent("hello")
    registry.agent(agent)
    events = []

    @registry.on_request_start
    def start(request, context):
        events.append(("start", request.model, context.agent_name))
        context.state["started"] = True

    @registry.on_request_end
    def end(request, context, run, response, error):
        events.append(
            (
                "end",
                context.state["started"],
                run.stream,
                response.model,
                error,
            )
        )

    request = ChatCompletionRequest(
        model="support",
        messages=[{"role": "user", "content": "hi"}],
    )

    response = await create_chat_completion(registry, request)

    assert response.model == "support"
    assert events == [
        ("start", "support", "support"),
        ("end", True, False, "support", None),
    ]


@pytest.mark.asyncio
async def test_create_chat_completion_serializes_structured_answer_with_reasoning():
    registry = ChannelRegistry()
    agent = FakeAgent(
        {
            "answer": {"final_answer": "42"},
            "reasoning": "Step by step",
        }
    )
    registry.agent(agent)
    request = ChatCompletionRequest(
        model="support",
        messages=[{"role": "user", "content": "hi"}],
    )

    response = await create_chat_completion(registry, request)

    message = response.choices[0].message
    assert message.content == '{"final_answer":"42"}'
    assert message.reasoning_content == "Step by step"


@pytest.mark.asyncio
async def test_create_chat_completion_stream_yields_openai_sse_chunks():
    stream_response = ModelStreamResponse(mode="async")
    stream_response.set_response_type("text_generation")
    stream_response.add_reasoning("thinking")
    stream_response.add_reasoning(None)
    stream_response.add("hello")
    stream_response.add(None)
    stream_response.set_metadata({"finish_reason": "stop"})

    registry = ChannelRegistry()
    agent = FakeAgent(stream_response)
    registry.agent(agent)
    seen_chunks = []

    @registry.on_stream_chunk
    def on_chunk(chunk):
        delta = chunk.choices[0].delta
        seen_chunks.append(delta.role or delta.reasoning_content or delta.content)

    request = ChatCompletionRequest(
        model="support",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    chunks = [
        chunk.decode("utf-8")
        async for chunk in create_chat_completion_stream(registry, request)
    ]

    assert chunks[0].startswith("data: ")
    assert '"role":"assistant"' in chunks[0]
    assert any('"reasoning_content":"thinking"' in chunk for chunk in chunks)
    assert any('"content":"hello"' in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"
    assert agent.calls[0]["stream"] is True
    assert "assistant" in seen_chunks
    assert "thinking" in seen_chunks
    assert "hello" in seen_chunks
