"""Tests for tool_config(inject_message=True).

Unit tests use Agents with mock models.
Integration test (marked with `integration`) uses real Groq API via .env.

Run integration tests:
    pytest tests/test_inject_message.py -v -m integration
"""

import os

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

            def forward(self, message=None, **kwargs):  # noqa: ARG002
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

            def forward(self, message=None, **kwargs):  # noqa: ARG002
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


# ---------------------------------------------------------------------------
# Integration test — real Groq model
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInjectMessageIntegration:
    """End-to-end test: Router uses SubAgent as tool with inject_message.

    Pattern from docs "Router Pattern" — extended with inject_message so the
    specialist receives the original structured input directly.

    Requires GROQ_API_KEY in the environment (.env).
    Run with: pytest tests/test_inject_message.py -v -m integration
    """

    def test_router_injects_message_into_specialist(self):
        """
        Scenario:
          - Router receives mf.dotdict(question="...", customer_id="cust_42").
          - Router model (forced via tool_choice) calls SupportSpecialist.
          - SupportSpecialist has inject_message=True, return_direct=True.
          - Specialist receives the original dotdict and can read customer_id
            without the orchestrator needing to repeat it in tool kwargs.
        """
        mf.set_envs(GROQ_API_KEY=os.environ.get("GROQ_API_KEY", ""))
        _model = mf.Model.chat_completion("groq/llama-3.1-8b-instant")

        captured = {}

        @mf.tool_config(inject_message=True, return_direct=True)
        class SupportSpecialist(Agent):
            """
            Handles customer support questions.
            Call this tool for any support-related question.
            No parameters needed.
            """

            model = _model

            def forward(self, message=None, **kwargs):  # noqa: ARG002
                captured["message"] = message
                cid = message.customer_id if message else "unknown"
                return f"Support answer for customer {cid}: resolved."

        class Router(Agent):
            """Routes support questions to the right specialist."""

            model = _model
            system_message = "Route support questions to SupportSpecialist."
            tools = [SupportSpecialist]
            config = {"tool_choice": "required"}
            message_fields = {"task_inputs": "question"}

        router = Router()

        input_msg = mf.dotdict(
            question="How do I reset my password?",
            customer_id="cust_42",
        )
        result = router(input_msg)

        # SupportSpecialist must have received the original dotdict
        assert captured.get("message") is not None
        assert captured["message"].customer_id == "cust_42"
        # return_direct=True wraps result in tool_responses dotdict
        tool_result = result.tool_responses.tool_calls[0].result
        assert "cust_42" in tool_result
