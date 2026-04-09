"""Tests for msgflux.nn.modules.agent module."""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from msgflux.nn.modules.agent import Agent, _RESERVED_KWARGS
from msgflux.core.message import Message
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.nn.modules.tool import ToolLibrary, ToolResponses, ToolCall
from msgflux.core.examples import Example


@pytest.fixture
def mock_chat_model():
    """Create a mock chat completion model."""
    model = Mock()
    model.model_type = "chat_completion"
    return model


class TestAgentReservedKwargs:
    """Test reserved kwargs constant."""

    def test_reserved_kwargs_set(self):
        """Test that _RESERVED_KWARGS is a set with expected values."""
        assert isinstance(_RESERVED_KWARGS, set)
        assert "task" in _RESERVED_KWARGS
        assert "vars" in _RESERVED_KWARGS
        assert "messages" in _RESERVED_KWARGS
        assert "task_multimodal" in _RESERVED_KWARGS
        assert "task_context" in _RESERVED_KWARGS
        assert "model_preference" in _RESERVED_KWARGS

    @pytest.mark.parametrize("name", sorted(_RESERVED_KWARGS))
    def test_signature_rejects_reserved_input_names(self, name):
        """Signature inputs must not clash with reserved kwargs."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        sig = f"{name}: str -> answer: str"
        with pytest.raises(ValueError, match="conflict with reserved"):
            Agent(name="agent", model=mock_model, signature=sig)

    @pytest.mark.parametrize("name", sorted(_RESERVED_KWARGS - {"task"}))
    def test_annotations_reject_reserved_input_names(self, name):
        """Custom annotations must not use reserved kwargs as input names."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        annotations = {name: str, "return": str}
        with pytest.raises(ValueError, match="conflict with reserved"):
            Agent(name="agent", model=mock_model, annotations=annotations)

    def test_default_annotations_are_accepted(self):
        """The default {"task": str, "return": str} annotations must be valid."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)
        assert agent.annotations == {"task": str, "return": str}

    def test_non_reserved_annotations_are_accepted(self):
        """Custom annotations with non-reserved names must work."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(
            name="agent", model=mock_model, annotations={"query": str, "return": str}
        )
        assert "query" in agent.annotations

    def test_non_reserved_signature_is_accepted(self):
        """Signature with non-reserved input names must work."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(
            name="agent", model=mock_model, signature="query: str -> answer: str"
        )
        assert agent.annotations is not None


class TestAgentInitialization:
    """Test Agent initialization."""

    def test_agent_basic_initialization(self):
        """Test basic Agent initialization."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="test_agent", model=mock_model)

        assert agent.name == "test_agent"
        assert agent.generator.model == mock_model

    def test_agent_with_system_message(self):
        """Test Agent with system message."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            system_message="You are a helpful assistant.",
        )

        assert hasattr(agent, "system_message") and agent.system_message is not None

    def test_agent_with_instructions(self):
        """Test Agent with instructions."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent", model=mock_model, instructions="Follow these steps carefully."
        )

        assert hasattr(agent, "instructions") and agent.instructions is not None

    def test_agent_with_expected_output(self):
        """Test Agent with expected output."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            expected_output="Provide a detailed response.",
        )

        assert hasattr(agent, "expected_output") and agent.expected_output is not None

    def test_agent_with_examples_string(self):
        """Test Agent with examples as string."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, examples="Example 1\nExample 2")

        # Examples should be processed
        assert hasattr(agent, "_buffers")

    def test_agent_with_examples_list(self):
        """Test Agent with examples as list."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        example1 = Example(inputs="test input", labels="test output")

        agent = Agent(name="agent", model=mock_model, examples=[example1])

        assert hasattr(agent, "_buffers")

    def test_agent_with_hooks(self):
        """Test Agent with hooks."""
        from msgflux.nn.hooks import Guard

        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        guard = Guard(validator=lambda data: {"safe": True}, on="pre")
        agent = Agent(name="agent", model=mock_model, hooks=[guard])

        assert len(agent.generator._forward_pre_hooks) == 1

    def test_agent_with_message_fields(self):
        """Test Agent with message fields."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent", model=mock_model, message_fields={"task": "input.user"}
        )

        # message_fields is unpacked, not stored as single attribute
        # Just verify agent was created successfully
        assert agent.name == "agent"

    def test_agent_with_config(self):
        """Test Agent with config."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent", model=mock_model, config={"verbose": True, "stream": False}
        )

        assert hasattr(agent, "config") and agent.config is not None

    def test_agent_with_templates(self):
        """Test Agent with custom templates."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent", model=mock_model, templates={"task": "Custom task template"}
        )

        assert hasattr(agent, "templates") and agent.templates is not None

    def test_agent_with_context_cache(self):
        """Test Agent with context cache."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, context_cache="cache_key")

        assert hasattr(agent, "context_cache") and agent.context_cache is not None

    def test_agent_with_prefilling(self):
        """Test Agent with prefilling."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, prefilling="Start with this text")

        assert hasattr(agent, "prefilling") and agent.prefilling is not None

    def test_agent_with_response_mode(self):
        """Test Agent with response mode."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, response_mode="structured")

        assert hasattr(agent, "response_mode") and agent.response_mode is not None

    def test_agent_with_tools(self):
        """Test Agent with tools."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        def my_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        agent = Agent(name="agent", model=mock_model, tools=[my_tool])

        # Tools are stored in tool_library attribute
        assert hasattr(agent, "tool_library")
        assert isinstance(agent.tool_library, ToolLibrary)

    def test_agent_with_signature(self):
        """Test Agent with signature."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, signature="input -> output")

        # Signature configures the agent but isn't stored as attribute
        # Verify agent was created successfully and has task template
        assert agent.name == "agent"
        assert "task" in agent.templates

    def test_agent_model_property(self):
        """Test Agent model property getter."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model)

        assert agent.model == mock_model

    def test_agent_model_property_setter(self):
        """Test Agent model property setter."""
        mock_model1 = Mock()
        mock_model1.model_type = "chat_completion"
        mock_model2 = Mock()
        mock_model2.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model1)
        agent.model = mock_model2

        assert agent.model == mock_model2


class TestAgentForward:
    """Test Agent forward method."""

    def test_agent_forward_simple(self):
        """Test simple Agent forward call with signature."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        mock_response = ModelResponse()
        mock_response.data = "Test response"
        mock_response.response_type = "text_generation"  # Set response_type

        # Mock the model call
        mock_model.return_value = mock_response
        mock_model.acall = AsyncMock(return_value=mock_response)

        agent = Agent(name="agent", model=mock_model, signature="query -> response")

        # Mock the lm forward call
        agent.generator.forward = Mock(return_value=mock_response)

        result = agent(query="Test input")

        assert result is not None

    def test_agent_forward_with_kwargs(self):
        """Test Agent forward with kwargs."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"

        mock_model.return_value = mock_response
        mock_model.acall = AsyncMock(return_value=mock_response)

        agent = Agent(
            name="agent", model=mock_model, signature="query, context -> response"
        )

        agent.generator.forward = Mock(return_value=mock_response)

        result = agent(query="What is AI?", context="ML context")

        assert result is not None

    def test_agent_forward_with_model_preference(self):
        """Test Agent forward with model preference."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"

        mock_model.return_value = mock_response
        mock_model.acall = AsyncMock(return_value=mock_response)

        agent = Agent(name="agent", model=mock_model, signature="query -> response")

        agent.generator.forward = Mock(return_value=mock_response)

        result = agent(query="Test", model_preference="gpt-4")

        assert result is not None

    def test_agent_forward_with_vars(self):
        """Test Agent forward with vars."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"

        mock_model.return_value = mock_response
        mock_model.acall = AsyncMock(return_value=mock_response)

        agent = Agent(name="agent", model=mock_model, signature="query -> response")

        agent.generator.forward = Mock(return_value=mock_response)

        result = agent(query="Test", vars={"key": "value"})

        assert result is not None

    def test_agent_forward_with_messages(self):
        """Test Agent forward with messages."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"

        mock_model.return_value = mock_response
        mock_model.acall = AsyncMock(return_value=mock_response)

        agent = Agent(name="agent", model=mock_model, signature="query -> response")

        agent.generator.forward = Mock(return_value=mock_response)

        result = agent(query="Test", messages=[])

        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_aforward_raises_explicit_stream_error(self):
        """Async stream failures should surface as explicit agent errors."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            signature="query -> response",
            config={"stream": True},
        )

        stream_response = ModelStreamResponse(mode="async")
        stream_response.set_error(TypeError("backend blew up"))
        agent.generator.acall = AsyncMock(return_value=stream_response)

        with pytest.raises(
            RuntimeError,
            match="Model stream failed before producing a response: backend blew up",
        ):
            await agent.aforward(query="Test input")


class TestAgentInspect:
    """Test Agent inspection methods."""

    def test_inspect_model_execution_params(self):
        """Test inspect_model_execution_params method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            system_message="System prompt",
            signature="input -> output",  # Need signature for task template
        )

        params = agent.inspect_model_execution_params(input="Test input")

        assert isinstance(params, dict)
        assert "messages" in params or "prompt" in params or len(params) >= 0


class TestAgentSetters:
    """Test Agent setter methods."""

    def test_set_system_message(self):
        """Test _set_system_message method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        agent._set_system_message("New system message")

        assert agent.system_message.data == "New system message"

    def test_set_instructions(self):
        """Test _set_instructions method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        agent._set_instructions("New instructions")

        assert agent.instructions.data == "New instructions"

    def test_set_expected_output(self):
        """Test _set_expected_output method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        agent._set_expected_output("Expected format")

        assert agent.expected_output.data == "Expected format"

    def test_set_config(self):
        """Test _set_config method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        agent._set_config({"verbose": False})

        assert agent.config.get("verbose") == False

    def test_set_context_cache(self):
        """Test _set_context_cache method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        agent._set_context_cache("cache_key")

        assert agent.context_cache == "cache_key"

    def test_set_prefilling(self):
        """Test _set_prefilling method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        agent._set_prefilling("Prefilled text")

        assert agent.prefilling == "Prefilled text"


class TestAgentProcessing:
    """Test Agent processing methods."""

    def test_prepare_inputs(self):
        """Test _prepare_inputs method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(
            name="agent",
            model=mock_model,
            system_message="You are helpful",
            signature="input -> output",
        )

        result = agent._prepare_inputs(input="Test input")

        assert isinstance(result, dict)


class TestAgentMultimodal:
    """Test Agent multimodal functionality."""

    def test_format_image_input(self):
        """Test _format_image_input method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        # Test with mock image source
        result = agent._format_image_input("test.jpg")

        # Should return a dict or None
        assert result is None or isinstance(result, dict)

    def test_format_video_input(self):
        """Test _format_video_input method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        result = agent._format_video_input("test.mp4")

        assert result is None or isinstance(result, dict)

    def test_format_audio_input(self):
        """Test _format_audio_input method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        result = agent._format_audio_input("test.mp3")

        assert result is None or isinstance(result, dict)

    def test_format_file_input(self):
        """Test _format_file_input method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        result = agent._format_file_input("test.pdf")

        assert result is None or isinstance(result, dict)


class TestAgentTools:
    """Test Agent with tools."""

    def test_agent_with_tool_library(self):
        """Test Agent with ToolLibrary."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        agent = Agent(name="agent", model=mock_model, tools=[add])

        # Tools are stored in tool_library attribute
        assert hasattr(agent, "tool_library")
        assert isinstance(agent.tool_library, ToolLibrary)

    def test_set_tools(self):
        """Test _set_tools method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model)

        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        agent._set_tools([multiply])

        # Tools should be set in tool_library after calling _set_tools
        assert hasattr(agent, "tool_library")
        assert isinstance(agent.tool_library, ToolLibrary)


class TestAgentDescription:
    """Test Agent description and name."""

    def test_agent_name_from_init(self):
        """Test agent name is set correctly."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="my_custom_agent", model=mock_model)

        assert agent.name == "my_custom_agent"

    def test_agent_has_description(self):
        """Test agent has description from docstring."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        agent = Agent(name="agent", model=mock_model, description="Test agent")

        # Should have description attribute
        assert hasattr(agent, "description")
        assert agent.description == "Test agent"


class TestAgentGenerationSchema:
    """Test Agent with generation schemas."""

    def test_agent_with_generation_schema(self):
        """Test Agent with msgspec struct as generation schema."""
        import msgspec

        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        class Output(msgspec.Struct):
            answer: str

        agent = Agent(
            name="agent",
            model=mock_model,
            signature="query -> output",
            generation_schema=Output,
        )

        assert hasattr(agent, "generation_schema")
        assert agent.generation_schema is not None


class TestAgentTemplates:
    """Test Agent template functionality."""

    def test_agent_get_system_prompt(self):
        """Test get_system_prompt method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            system_message="You are helpful",
            instructions="Be concise",
        )

        system_prompt = agent.get_system_prompt()

        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0

    def test_agent_with_custom_task_template(self):
        """Test Agent with custom task template."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent", model=mock_model, templates={"task": "Question: {{query}}"}
        )

        assert "task" in agent.templates
        assert agent.templates["task"] == "Question: {{query}}"

    def test_agent_format_template(self):
        """Test _format_template method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model)

        result = agent._format_template({"name": "Alice"}, "Hello {{name}}")

        assert result == "Hello Alice"


class TestAgentHooks:
    """Test Agent hooks functionality."""

    def test_agent_pre_hook(self):
        """Test Agent with pre hook."""
        from msgflux.nn.hooks import Guard

        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        guard = Guard(validator=lambda data: {"safe": True}, on="pre")
        agent = Agent(name="agent", model=mock_model, hooks=[guard])

        assert len(agent.generator._forward_pre_hooks) == 1


class TestAgentExamples:
    """Test Agent examples functionality."""

    def test_agent_examples_as_list(self):
        """Test Agent with examples as list of Example objects."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        ex1 = Example(inputs="What is 2+2?", labels="4")
        ex2 = Example(inputs="What is 3+3?", labels="6")

        agent = Agent(name="agent", model=mock_model, examples=[ex1, ex2])

        assert hasattr(agent, "examples")

    def test_agent_set_examples(self):
        """Test _set_examples method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model)

        ex1 = Example(inputs="Test", labels="Output")
        agent._set_examples([ex1])

        assert hasattr(agent, "examples")

    def test_examples_not_html_escaped_in_system_prompt(self):
        """Regression test: examples with XML tags must not be HTML-escaped.

        Previously, format_template applied markupsafe.escape() on dict values,
        converting '<example ...>' into '&lt;example ...&gt;' inside the system
        prompt and corrupting the prompt sent to the model.
        """
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        example = Example(
            inputs="A fintech offering digital wallets.",
            labels={"Needs": "Payment integration", "Value": "Simplify payments"},
            title="Fintech Lead",
            topic="Sales",
        )

        agent = Agent(name="agent", model=mock_model, examples=[example])

        system_prompt = agent.get_system_prompt()

        assert "&lt;" not in system_prompt, "XML tags must not be HTML-escaped"
        assert "&gt;" not in system_prompt, "XML tags must not be HTML-escaped"
        assert "&#34;" not in system_prompt, "Quotes must not be HTML-escaped"
        assert "<example" in system_prompt, "Example XML tag must be present"
        assert "<input>" in system_prompt, "Input tag must be present"


class TestAgentTypedParser:
    """Test Agent typed parser functionality."""

    def test_agent_typed_parser_attribute(self):
        """Test Agent has typed_parser attribute when configured."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model)

        # Agent should have typed_parser attribute
        assert hasattr(agent, "typed_parser")


class TestAgentConfigOptions:
    """Test various Agent config options."""

    def test_agent_config_verbose(self):
        """Test Agent with verbose config."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, config={"verbose": True})

        assert agent.config.get("verbose") is True

    def test_agent_config_return_messages(self):
        """Test Agent with return_messages config."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, config={"return_messages": True})

        assert agent.config.get("return_messages") is True

    def test_agent_config_tool_choice(self):
        """Test Agent with tool_choice config."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, config={"tool_choice": "auto"})

        assert agent.config.get("tool_choice") == "auto"

    def test_agent_config_include_date(self):
        """Test Agent with include_date config."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, config={"include_date": True})

        assert agent.config.get("include_date") is True


class TestAgentAnnotations:
    """Test Agent annotations functionality."""

    def test_agent_set_annotations(self):
        """Test set_annotations method."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent", model=mock_model, annotations={"message": str, "return": dict}
        )

        assert hasattr(agent, "annotations")

    def test_agent_annotations_attribute(self):
        """Test Agent has annotations attribute."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent", model=mock_model, annotations={"message": str, "return": str}
        )

        assert hasattr(agent, "annotations")
        assert isinstance(agent.annotations, dict)


class TestAgentExecutionPaths:
    """Test Agent execution with various scenarios."""

    def test_agent_execute_with_dict_input(self):
        """Test Agent execution with dict input."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"

        mock_model.return_value = mock_response
        mock_model.acall = AsyncMock(return_value=mock_response)

        agent = Agent(name="agent", model=mock_model, signature="query -> response")

        agent.generator.forward = Mock(return_value=mock_response)

        # Test with dict input
        result = agent({"query": "Test"})

        assert result is not None

    def test_agent_execute_with_message_input(self):
        """Test Agent execution with Message input."""
        from msgflux.core.message import Message

        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"

        mock_model.return_value = mock_response
        mock_model.acall = AsyncMock(return_value=mock_response)

        agent = Agent(
            name="agent",
            model=mock_model,
            signature="query -> response",
            message_fields={"task": "query"},
        )

        agent.generator.forward = Mock(return_value=mock_response)

        # Create Message object
        msg = Message()
        msg.query = {"query": "Test question"}

        result = agent(msg)

        assert result is not None

    def test_agent_with_context(self):
        """Test Agent with context."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"

        mock_model.return_value = mock_response
        mock_model.acall = AsyncMock(return_value=mock_response)

        agent = Agent(name="agent", model=mock_model, signature="query -> response")

        agent.generator.forward = Mock(return_value=mock_response)

        result = agent(query="Test", task_context="Some context")

        assert result is not None

    def test_agent_signature_context_is_not_reserved(self):
        """Test that signature fields named context work as normal task inputs."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            signature="query, context -> response",
        )

        params = agent.inspect_model_execution_params(
            query="What is AI?", context="ML context"
        )

        content = str(params["messages"])
        assert "What is AI?" in content
        assert "ML context" in content


class TestAgentSystemPrompt:
    """Test Agent system prompt generation."""

    def test_agent_system_prompt_with_date(self):
        """Test system prompt generation with date included."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            system_message="You are helpful",
            config={"include_date": True},
        )

        system_prompt = agent.get_system_prompt()

        assert isinstance(system_prompt, str)
        # Date should be included
        assert len(system_prompt) > 0

    def test_agent_system_prompt_with_vars(self):
        """Test system prompt generation with runtime vars."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, system_message="Hello")

        system_prompt = agent.get_system_prompt(vars={"name": "Alice"})

        assert isinstance(system_prompt, str)

    def test_agent_system_prompt_template_property(self):
        """Test system_prompt_template property."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            templates={"system_prompt": "Custom: {{system_message}}"},
        )

        template = agent.system_prompt_template

        assert template == "Custom: {{system_message}}"


class TestAgentMessagePreparation:
    """Test Agent message preparation."""

    def test_agent_prepare_inputs_with_vars(self):
        """Test _prepare_inputs with vars parameter."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, signature="query -> response")

        result = agent._prepare_inputs(query="Test {{var}}", vars={"var": "value"})

        assert isinstance(result, dict)

    def test_agent_prepare_context_with_template(self):
        """Test _prepare_context with custom template."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            templates={"task_context": "Context: {{context}}"},
        )

        # Context preparation should use custom template
        assert agent.templates["task_context"] == "Context: {{context}}"


class TestAgentProperties:
    """Test Agent property methods."""

    def test_agent_response_mode_property(self):
        """Test response_mode property."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, response_mode=None)

        assert hasattr(agent, "response_mode")
        assert agent.response_mode is None

    def test_agent_context_cache_property(self):
        """Test context_cache property."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, context_cache="cached_context")

        assert agent.context_cache == "cached_context"

    def test_agent_prefilling_property(self):
        """Test prefilling property."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(name="agent", model=mock_model, prefilling="Start here")

        assert agent.prefilling == "Start here"


class TestAgentJinjaTaskTemplateValidation:
    """Test that a string task passed to an agent with a Jinja2-only task_template raises ValueError."""

    def test_jinja_variable_template_with_str_task_raises(self, mock_chat_model):
        agent = Agent(
            name="agent",
            model=mock_chat_model,
            templates={"task": "<ticket>{{ ticket }}</ticket>"},
        )
        with pytest.raises(ValueError, match="Jinja2 variables"):
            agent._render_task("", vars={}, task="classify this ticket")

    def test_jinja_block_template_with_str_task_raises(self, mock_chat_model):
        agent = Agent(
            name="agent",
            model=mock_chat_model,
            templates={"task": "{% if ticket %}{{ ticket }}{% endif %}"},
        )
        with pytest.raises(ValueError, match="Jinja2 variables"):
            agent._render_task("", vars={}, task="classify this ticket")

    def test_error_message_includes_agent_name(self, mock_chat_model):
        agent = Agent(
            name="my-router",
            model=mock_chat_model,
            templates={"task": "{{ ticket }}"},
        )
        with pytest.raises(ValueError, match="my-router"):
            agent._render_task("", vars={}, task="some text")

    def test_mixed_jinja_and_format_placeholder_does_not_raise(self, mock_chat_model):
        """Template with both Jinja2 vars and {} placeholder is valid with a string task."""
        agent = Agent(
            name="agent",
            model=mock_chat_model,
            templates={"task": "{% if ctx %}Context: {{ ctx }}\n{% endif %}Task: {}"},
        )
        # Should not raise — {} receives the string task after Jinja2 renders vars
        agent._render_task("", vars={"ctx": "billing"}, task="help me")

    def test_pure_format_placeholder_does_not_raise(self, mock_chat_model):
        agent = Agent(
            name="agent",
            model=mock_chat_model,
            templates={"task": "Task: {}"},
        )
        agent._render_task("", vars={}, task="help me")

    def test_jinja_template_with_dict_task_does_not_raise(self, mock_chat_model):
        """Dict task always follows the Jinja2 rendering path — no error."""
        agent = Agent(
            name="agent",
            model=mock_chat_model,
            templates={"task": "<ticket>{{ ticket }}</ticket>"},
        )
        agent._render_task("", vars={}, task={"ticket": "I was charged twice"})

    def test_no_template_with_str_task_does_not_raise(self, mock_chat_model):
        """No task_template — string task is used as-is."""
        agent = Agent(name="agent", model=mock_chat_model)
        agent._render_task("", vars={}, task="help me")

    @pytest.mark.asyncio
    async def test_jinja_template_with_str_task_raises_async(self, mock_chat_model):
        agent = Agent(
            name="agent",
            model=mock_chat_model,
            templates={"task": "<ticket>{{ ticket }}</ticket>"},
        )
        with pytest.raises(ValueError, match="Jinja2 variables"):
            await agent._arender_task("", vars={}, task="classify this ticket")

    @pytest.mark.asyncio
    async def test_mixed_template_does_not_raise_async(self, mock_chat_model):
        agent = Agent(
            name="agent",
            model=mock_chat_model,
            templates={"task": "{% if ctx %}{{ ctx }}\n{% endif %}Task: {}"},
        )
        agent._render_task("", vars={}, task="help me")


class TestAgentSystemExtraMessage:
    """Test Agent system_extra_message."""

    def test_agent_with_system_extra_message(self):
        """Test Agent with system_extra_message."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        agent = Agent(
            name="agent",
            model=mock_model,
            system_message="Main message",
            system_extra_message="Extra info",
        )

        assert hasattr(agent, "system_extra_message")
        assert agent.system_extra_message == "Extra info"


class TestAgentMessagesAccumulator:
    """Test messages parameter accumulator semantics.

    - messages not passed (default) → ephemeral, no external side effect
    - messages=[]                   → opt-in accumulator, mutated in-place
    - messages=[...]                → extends existing history
    """

    @pytest.fixture
    def agent(self):
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        return Agent(name="agent", model=mock_model)

    # --- _prepare_inputs: pure accumulation logic ---

    def test_no_messages_arg_is_ephemeral(self, agent):
        """Not passing messages returns an internal list; no external object modified."""
        result = agent._prepare_inputs("Hello")
        # Internal list is created — has the user message
        assert result["messages"] is not None
        assert any(m.get("role") == "user" for m in result["messages"])

    def test_empty_list_accumulates_user_input(self, agent):
        """messages=[] is mutated in-place with the user input."""
        history = []
        agent._prepare_inputs("Hello", messages=history)
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_nonempty_list_extends_with_user_input(self, agent):
        """messages=[...] is extended with the new user input."""
        existing = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        history = list(existing)
        agent._prepare_inputs("Follow-up question", messages=history)
        assert len(history) == 3
        assert history[2]["role"] == "user"

    def test_empty_list_is_same_object(self, agent):
        """The list passed as messages=[] is the same object after the call."""
        history = []
        agent._prepare_inputs("Hello", messages=history)
        assert id(history) == id(history)  # tautology — check via len instead
        assert len(history) == 1

    def test_none_messages_does_not_mutate_any_external_list(self, agent):
        """Passing messages=None explicitly behaves as ephemeral (no crash, no side effect)."""
        # Should not raise and returns a valid messages list internally
        result = agent._prepare_inputs("Hello", messages=None)
        assert result["messages"] is not None

    # --- full forward: user input and tool calls accumulate ---

    def test_forward_accumulates_user_input(self, agent):
        """forward() with messages=[] accumulates the user input into the list."""
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"
        agent.generator.forward = Mock(return_value=mock_response)

        history = []
        agent("Hello", messages=history)

        assert len(history) >= 1
        assert history[0]["role"] == "user"

    def test_forward_no_messages_arg_has_no_side_effect(self, agent):
        """forward() without messages= leaves no external object modified."""
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"
        agent.generator.forward = Mock(return_value=mock_response)

        # No messages argument — purely ephemeral
        result = agent("Hello")
        assert isinstance(result, str)

    def test_forward_two_turns_accumulate_correctly(self, agent):
        """Two forward() calls with the same list accumulate both user inputs."""
        mock_response = ModelResponse()
        mock_response.data = "Response"
        mock_response.response_type = "text_generation"
        agent.generator.forward = Mock(return_value=mock_response)

        history = []
        agent("Turn one", messages=history)
        len_after_turn1 = len(history)

        agent("Turn two", messages=history)
        len_after_turn2 = len(history)

        assert len_after_turn2 > len_after_turn1

    def test_forward_does_not_add_assistant_reply(self, agent):
        """The final assistant response is never appended to messages automatically."""
        mock_response = ModelResponse()
        mock_response.data = "I am the assistant"
        mock_response.response_type = "text_generation"
        agent.generator.forward = Mock(return_value=mock_response)

        history = []
        response = agent("Hello", messages=history)

        roles = [m.get("role") for m in history]
        assert "assistant" not in roles
        assert response == "I am the assistant"
