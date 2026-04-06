"""Tests for tool_config(inject_message=True).

Unit tests use Agents with mock models.
"""

import pytest
from unittest.mock import MagicMock

import msgflux as mf
from msgflux.nn import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_model(text: str = "ok") -> MagicMock:
    model = MagicMock()
    model.model_type = "chat_completion"
    resp = MagicMock()
    resp.response_type = "text_generation"
    resp.consume.return_value = text
    model.return_value = resp
    return model


# ---------------------------------------------------------------------------
# tool_config: inject_message stored correctly
# ---------------------------------------------------------------------------

class TestToolConfigInjectMessage:
    def test_inject_message_default_false(self):
        @mf.tool_config(return_direct=False)
        def my_tool(x: str) -> str:
            return x

        assert my_tool.tool_config.inject_message is False

    def test_inject_message_true_on_function(self):
        @mf.tool_config(inject_message=True)
        def my_tool(x: str) -> str:
            return x

        assert my_tool.tool_config.inject_message is True

    def test_inject_message_on_agent_class(self):
        @mf.tool_config(inject_message=True, return_direct=True)
        class SubAgent(Agent):
            """A sub-agent used as a tool."""

        assert SubAgent.tool_config.inject_message is True
        assert SubAgent.tool_config.return_direct is True


# ---------------------------------------------------------------------------
# Agent-as-tool: unit tests (mocked model)
# ---------------------------------------------------------------------------

class TestInjectMessageAnnotations:
    """Validate that inject_message=True strips 'message' from the tool schema."""

    def test_message_excluded_from_tool_schema(self):
        """With inject_message=True, LLM-facing schema must NOT have 'message'."""

        @mf.tool_config(inject_message=True, return_direct=True)
        class TranslationAgent(Agent):
            """Translate text to a target language."""

            model = _mock_model()
            annotations = {"target_language": str}

        orchestrator = Agent(
            name="Orchestrator",
            model=_mock_model(),
            tools=[TranslationAgent],
        )

        schemas = orchestrator.tool_library.get_tool_json_schemas()
        schema = next(s for s in schemas if s["function"]["name"] == "TranslationAgent")
        props = schema["function"]["parameters"].get("properties", {})

        # message is injected by framework — must NOT appear in the LLM schema
        assert "message" not in props
        # task parameter IS in schema for the LLM to fill
        assert "target_language" in props

    def test_default_annotations_message_stripped(self):
        """Agent with default annotations (message: str) — message must be stripped."""

        @mf.tool_config(inject_message=True, return_direct=True)
        class SubAgent(Agent):
            """Sub-agent with default annotations."""

            model = _mock_model()
            # No custom annotations — defaults to {"message": str, "return": str}

        orchestrator = Agent(
            name="Orchestrator",
            model=_mock_model(),
            tools=[SubAgent],
        )

        schemas = orchestrator.tool_library.get_tool_json_schemas()
        schema = next(s for s in schemas if s["function"]["name"] == "SubAgent")
        props = schema["function"]["parameters"].get("properties", {})

        assert "message" not in props


class TestAgentAsToolInjectMessage:
    """Validate inject_message=True end-to-end through ToolLibrary."""

    def test_message_not_present_when_flag_false(self):
        """Without inject_message, forward() does not receive message kwarg."""
        received = {}

        class SubAgent(Agent):
            """Agent without inject_message."""

            def forward(self, message=None, **kwargs):
                received["message"] = message
                return "done"

        sub = SubAgent(model=_mock_model())
        orchestrator = Agent(
            name="Orchestrator",
            model=_mock_model(),
            tools=[sub],
        )

        input_msg = mf.dotdict(text="hello")
        orchestrator.tool_library(
            tool_callings=[("id1", "SubAgent", {})],
            message=input_msg,
        )

        # message kwarg was not injected → stays at default None
        assert received.get("message") is None

    def test_message_not_leaked_in_tool_call_parameters(self):
        """message must not appear in the logged ToolCall.parameters."""

        @mf.tool_config(inject_message=True, return_direct=True)
        class SubAgent(Agent):
            """Sub."""

            def forward(self, message=None, **kwargs):
                return "ok"

        sub = SubAgent(model=_mock_model())
        orchestrator = Agent(
            name="Orchestrator",
            model=_mock_model(),
            tools=[sub],
        )

        responses = orchestrator.tool_library(
            tool_callings=[("id1", "SubAgent", {"query": "test"})],
            message=mf.dotdict(text="ctx"),
        )

        call = responses.tool_calls[0]
        assert "message" not in (call.parameters or {})
