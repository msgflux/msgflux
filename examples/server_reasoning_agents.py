# /// script
# dependencies = []
# ///
#
# Run this server with:
#
#   uv run --with 'msgflux[server,openai]' msgflux server \
#     examples/server_reasoning_agents.py:registry --host 127.0.0.1 --port 8010
#
# Exposed agents:
#   - model="groq_reasoning"
#   - model="openai_react"

import msgflux as mf
from msgflux import nn
from msgflux.generation.reasoning import ReAct
from msgflux.tools.builtin import WebFetch

mf.load_dotenv()

registry = mf.ChannelRegistry()
registry.settings(
    title="msgFlux Reasoning Channel",
    subtitle="Reasoning and ReAct agents behind one HTTP boundary",
    description="OpenAI-compatible reasoning server with per-agent rate limits.",
)
registry.rate_limit(
    name="groq-reasoning-api-key-minute",
    agent="groq_reasoning",
    requests=20,
    window_s=60,
    by="api_key",
)
registry.rate_limit(
    name="openai-react-ip-minute",
    agent="openai_react",
    requests=10,
    window_s=60,
    by="ip",
)


@registry.agent(name="groq_reasoning")
class GroqReasoningAgent(nn.Agent):
    """Reasoning validation with Groq gpt-oss through HTTP channel."""

    model = mf.Model.chat_completion(
        "groq/openai/gpt-oss-120b",
        reasoning_effort="low",
    )
    instructions = "Solve step-by-step and keep the final answer concise."
    config = {"reasoning_in_response": True}


@registry.agent(name="openai_react")
class OpenAIReActAgent(nn.Agent):
    """ReAct validation with OpenAI (non-streaming only)."""

    model = "openai/gpt-4.1-mini"
    generation_schema = ReAct
    tools = [WebFetch]
    instructions = """
    Use WebFetch when external verification is needed.
    Keep the final answer short and practical.
    """
    # ReAct (generation_schema) is not compatible with stream=True.
    config = {"reasoning_in_response": True}
