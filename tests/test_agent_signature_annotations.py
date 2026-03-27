"""
Test for Agent automatic annotations generation from signatures.

This test validates that when a signature is provided, the agent automatically
generates annotations from the signature inputs, excluding multimodal types.
"""

from typing import Optional
from unittest.mock import MagicMock

import pytest

import msgflux as mf
from msgflux.data.types import Audio, Image, Video
from msgflux.dsl.signature import Signature
from msgflux.nn.modules.agent import Agent


def create_mock_model():
    """Create a mock chat completion model."""
    model = MagicMock()
    model.model_type = "chat_completion"

    # Mock the model response
    mock_response = MagicMock()
    mock_response.response_type = "text_generation"
    mock_response.consume.return_value = "Mocked response"
    model.return_value = mock_response

    return model


def test_signature_class_simple():
    """Test annotations generation from simple class-based signature."""
    model = create_mock_model()

    class SimpleSignature(Signature):
        """Simple signature without multimodal types."""

        name: str = mf.InputField()
        age: int = mf.InputField()
        result: str = mf.OutputField()

    agent = Agent(name="test", model=model, signature=SimpleSignature)

    # Should have generated annotations from inputs
    assert agent.annotations == {"name": str, "age": int, "return": str}
    print("✓ Test 1 passed: Simple class signature generates correct annotations")


def test_signature_class_with_multimodal():
    """Test that multimodal types are excluded from annotations."""
    model = create_mock_model()

    class MultimodalSignature(Signature):
        """Signature with multimodal types."""

        text: str = mf.InputField()
        image: Image = mf.InputField()
        audio: Audio = mf.InputField()
        answer: str = mf.OutputField()

    agent = Agent(name="test", model=model, signature=MultimodalSignature)

    # Image and Audio should NOT appear in annotations
    assert agent.annotations == {"text": str, "return": str}
    assert "image" not in agent.annotations
    assert "audio" not in agent.annotations
    print(
        "✓ Test 2 passed: Multimodal types excluded from annotations (class signature)"
    )


def test_signature_class_with_optional_multimodal():
    """Test that Optional multimodal types are excluded."""
    model = create_mock_model()

    class OptionalMultimodal(Signature):
        """Signature with optional multimodal."""

        text: str = mf.InputField()
        image: Optional[Image] = mf.InputField()
        video: Optional[Video] = mf.InputField()
        result: str = mf.OutputField()

    agent = Agent(name="test", model=model, signature=OptionalMultimodal)

    # Optional multimodal types should also be excluded
    assert agent.annotations == {"text": str, "return": str}
    assert "image" not in agent.annotations
    assert "video" not in agent.annotations
    print("✓ Test 3 passed: Optional multimodal types excluded from annotations")


def test_signature_string_simple():
    """Test annotations generation from string signature."""
    model = create_mock_model()

    agent = Agent(
        name="test",
        model=model,
        signature="query: str, count: int -> result: str",
    )

    # Should parse and generate annotations
    assert agent.annotations == {"query": str, "count": int, "return": str}
    print("✓ Test 4 passed: String signature generates correct annotations")


def test_signature_string_with_complex_types():
    """Test string signature with complex types."""
    from typing import List as TypingList

    model = create_mock_model()

    agent = Agent(
        name="test",
        model=model,
        signature="name: str, scores: list, flag: bool -> output: str",
    )

    # Should handle different types
    # Note: StructFactory._parse_type_string returns typing.List, not builtin list
    assert agent.annotations["name"] == str
    assert agent.annotations["scores"] == TypingList
    assert agent.annotations["flag"] == bool
    assert agent.annotations["return"] == str
    print("✓ Test 5 passed: String signature with complex types")


def test_signature_and_custom_annotations_raises_error():
    """Test that providing both signature and custom annotations raises error."""
    model = create_mock_model()

    class TestSignature(Signature):
        text: str = mf.InputField()
        result: str = mf.OutputField()

    # Should raise ValueError
    with pytest.raises(
        ValueError,
        match="Cannot specify both 'signature' and custom 'annotations'",
    ):
        Agent(
            name="test",
            model=model,
            signature=TestSignature,
            annotations={"custom": str, "return": str},
        )

    print("✓ Test 6 passed: Error when both signature and annotations provided")


def test_no_signature_uses_provided_annotations():
    """Test that without signature, provided annotations are used."""
    model = create_mock_model()

    agent = Agent(
        name="test",
        model=model,
        annotations={"custom_input": str, "return": str},
    )

    # Should use provided annotations
    assert agent.annotations == {"custom_input": str, "return": str}
    print("✓ Test 7 passed: Custom annotations used when no signature")


def test_no_signature_no_annotations_uses_default():
    """Test that without signature or annotations, default is used."""
    model = create_mock_model()

    agent = Agent(name="test", model=model)

    # Should use default annotations
    assert agent.annotations == {"task": str, "return": str}
    print("✓ Test 8 passed: Default annotations when neither signature nor annotations")


def test_signature_with_only_multimodal_inputs():
    """Test signature with only multimodal inputs."""
    model = create_mock_model()

    class OnlyMultimodal(Signature):
        """Signature with only multimodal inputs."""

        image: Image = mf.InputField()
        audio: Audio = mf.InputField()
        result: str = mf.OutputField()

    agent = Agent(name="test", model=model, signature=OnlyMultimodal)

    # Should only have return annotation
    assert agent.annotations == {"return": str}
    assert "image" not in agent.annotations
    assert "audio" not in agent.annotations
    print("✓ Test 9 passed: Signature with only multimodal inputs")


def test_annotations_work_with_named_kwargs():
    """Test that generated annotations work with named kwargs feature."""
    model = create_mock_model()

    class QuerySignature(Signature):
        query: str = mf.InputField()
        max_results: int = mf.InputField()
        answer: str = mf.OutputField()

    agent = Agent(name="test", model=model, signature=QuerySignature)

    # Should have correct annotations
    assert agent.annotations == {"query": str, "max_results": int, "return": str}

    # Should work with named kwargs (requires task template from signature)
    # This validates integration with the named kwargs feature
    result = agent(query="test query", max_results=5)
    assert result is not None
    print("✓ Test 10 passed: Generated annotations work with named kwargs feature")


if __name__ == "__main__":
    # Run all tests
    test_signature_class_simple()
    test_signature_class_with_multimodal()
    test_signature_class_with_optional_multimodal()
    test_signature_string_simple()
    test_signature_string_with_complex_types()
    test_signature_and_custom_annotations_raises_error()
    test_no_signature_uses_provided_annotations()
    test_no_signature_no_annotations_uses_default()
    test_signature_with_only_multimodal_inputs()
    test_annotations_work_with_named_kwargs()

    print("\n✅ All tests passed! Signature auto-annotations feature validated.")
    print("\nFeature summary:")
    print("  1. Annotations generated automatically from signature inputs")
    print("  2. Multimodal types (Image, Audio, Video, File) excluded from annotations")
    print("  3. Works with both class-based and string signatures")
    print("  4. Error when both signature and custom annotations provided")
    print("  5. Backward compatible with manual annotations")
    print("  6. Integrates with named kwargs feature")
