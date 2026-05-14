import msgspec
import pytest
from typing import Optional

from msgflux.utils.chat import (
    ChatBlock,
    ChatML,
    adapt_messages_for_vllm_audio,
    clean_docstring,
    generate_json_schema,
    generate_tool_json_schema,
    hint_to_schema,
    parse_docstring_args,
    response_format_from_msgspec_struct,
)


class TestChatBlock:
    def test_user_message(self):
        assert ChatBlock.user("hello") == {"role": "user", "content": "hello"}

    def test_user_message_with_media(self):
        media = {"type": "image", "url": "http://example.com/img.png"}
        message = ChatBlock.user("hello", media=media)
        assert message == {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "url": "http://example.com/img.png"},
            ],
        }

    def test_assist_message(self):
        assert ChatBlock.assist("world") == {"role": "assistant", "content": "world"}

    def test_system_message(self):
        assert ChatBlock.system("system prompt") == {
            "role": "system",
            "content": "system prompt",
        }

    def test_tool_call(self):
        assert ChatBlock.tool_call("123", "my_tool", "{}") == {
            "id": "123",
            "type": "function",
            "function": {"name": "my_tool", "arguments": "{}"},
        }

    def test_assist_tool_calls(self):
        tool_calls = [ChatBlock.tool_call("123", "my_tool", "{}")]
        assert ChatBlock.assist_tool_calls(tool_calls) == {
            "role": "assistant",
            "tool_calls": tool_calls,
        }

    def test_tool_message(self):
        assert ChatBlock.tool("123", "tool output") == {
            "role": "tool",
            "tool_call_id": "123",
            "content": "tool output",
        }

    def test_image_without_detail(self):
        """Test image block without detail parameter."""
        result = ChatBlock.image("http://example.com/image.jpg")
        assert result == {
            "type": "image_url",
            "image_url": {"url": "http://example.com/image.jpg"},
        }

    def test_image_with_high_detail(self):
        """Test image block with high detail parameter."""
        result = ChatBlock.image("http://example.com/image.jpg", detail="high")
        assert result == {
            "type": "image_url",
            "image_url": {"url": "http://example.com/image.jpg", "detail": "high"},
        }

    def test_image_with_low_detail(self):
        """Test image block with low detail parameter."""
        result = ChatBlock.image("http://example.com/image.jpg", detail="low")
        assert result == {
            "type": "image_url",
            "image_url": {"url": "http://example.com/image.jpg", "detail": "low"},
        }

    def test_image_list_with_detail(self):
        """Test multiple images with detail parameter."""
        urls = ["http://example.com/img1.jpg", "http://example.com/img2.jpg"]
        result = ChatBlock.image(urls, detail="high")
        assert result == [
            {
                "type": "image_url",
                "image_url": {"url": "http://example.com/img1.jpg", "detail": "high"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "http://example.com/img2.jpg", "detail": "high"},
            },
        ]

    def test_image_list_without_detail(self):
        """Test multiple images without detail parameter."""
        urls = ["http://example.com/img1.jpg", "http://example.com/img2.jpg"]
        result = ChatBlock.image(urls)
        assert result == [
            {"type": "image_url", "image_url": {"url": "http://example.com/img1.jpg"}},
            {"type": "image_url", "image_url": {"url": "http://example.com/img2.jpg"}},
        ]


class TestChatML:
    def test_add_messages(self):
        chat = ChatML()
        chat.add_user_message("hello")
        chat.add_assist_message("world")
        assert chat.get_messages() == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_clear(self):
        chat = ChatML()
        chat.add_user_message("hello")
        chat.clear()
        assert chat.get_messages() == []


class MyStruct(msgspec.Struct):
    a: int
    b: str


def test_response_format_from_msgspec_struct():
    schema = response_format_from_msgspec_struct(MyStruct)
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["name"] == "mystruct"
    assert schema["json_schema"]["schema"]["properties"]["a"]["type"] == "integer"


def test_hint_to_schema():
    assert hint_to_schema(int) == {"type": "integer"}
    assert hint_to_schema(str) == {"type": "string"}
    assert hint_to_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}


class TodoItem(msgspec.Struct):
    content: str
    active_form: str
    status: str


def test_hint_to_schema_supports_msgspec_struct():
    schema = hint_to_schema(TodoItem)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["active_form", "content", "status"]
    assert schema["properties"]["content"]["type"] == "string"


def test_hint_to_schema_supports_list_of_msgspec_struct():
    schema = hint_to_schema(list[TodoItem])

    assert schema["type"] == "array"
    assert schema["items"]["type"] == "object"
    assert schema["items"]["properties"]["active_form"]["type"] == "string"


DOCSTRING = """
My function.

Args:
    a: first param.
    b (int): second param.
"""


def test_clean_docstring():
    assert clean_docstring(DOCSTRING) == "My function."


def test_parse_docstring_args():
    args = parse_docstring_args(DOCSTRING)
    assert args["a"] == "first param."
    assert args["b"] == "second param."


class MyTool:
    def get_module_name(self):
        return "my_tool"

    def get_module_description(self):
        return DOCSTRING

    def get_module_annotations(self):
        return {"a": str, "b": int}


def test_generate_json_schema():
    schema = generate_json_schema(MyTool())
    assert schema["name"] == "my_tool"
    assert schema["description"] == "My function."
    assert schema["parameters"]["properties"]["a"]["description"] == "first param."


class MyOptionalTool:
    def get_module_name(self):
        return "my_optional_tool"

    def get_module_description(self):
        return DOCSTRING

    def get_module_annotations(self):
        return {"a": str, "b": Optional[int]}

    def forward(self, a: str, b: Optional[int] = None):
        return a, b


class MyDefaultTool:
    def get_module_name(self):
        return "my_default_tool"

    def get_module_description(self):
        return DOCSTRING

    def get_module_annotations(self):
        return {"a": str, "limit": int}

    def forward(self, a: str, limit: int = 5):
        return a, limit


def test_generate_json_schema_optional_tool_param_is_required_and_nullable():
    schema = generate_json_schema(MyOptionalTool())
    params = schema["parameters"]

    assert sorted(params["required"]) == ["a", "b"]
    assert params["properties"]["b"]["anyOf"] == [
        {"type": "integer"},
        {"type": "null"},
    ]


def test_generate_json_schema_default_tool_param_is_required_and_nullable():
    schema = generate_json_schema(MyDefaultTool())
    params = schema["parameters"]

    assert sorted(params["required"]) == ["a", "limit"]
    assert params["properties"]["limit"]["anyOf"] == [
        {"type": "integer"},
        {"type": "null"},
    ]


def test_generate_tool_json_schema():
    schema = generate_tool_json_schema(MyTool())
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "my_tool"


def test_adapt_messages_for_vllm_audio():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "base64_data", "format": "mp3"},
                },
            ],
        }
    ]
    adapted = adapt_messages_for_vllm_audio(messages)
    assert adapted[0]["content"][1]["type"] == "audio_url"
    assert (
        "data:audio/mpeg;base64,base64_data"
        in adapted[0]["content"][1]["audio_url"]["url"]
    )
