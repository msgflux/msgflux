"""Unit tests for msgflux.tools.builtin.web_search."""

from typing import Optional
from unittest.mock import MagicMock

import pytest

from msgflux.core.dotdict import dotdict
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools.builtin.web_search import WebSearch


class TestRetrieverAlias:
    def test_web_search_alias_calls_web_factory(self, mocker):
        mock_factory = mocker.patch(
            "msgflux.data.retrievers.retriever.Retriever.web"
        )

        sentinel = object()
        mock_factory.return_value = sentinel

        from msgflux.data.retrievers.retriever import Retriever

        result = Retriever.web_search("wikipedia", language="pt")

        assert result is sentinel
        mock_factory.assert_called_once_with("wikipedia", language="pt")


class TestWebSearchInit:
    def test_engine_can_be_loaded_from_env(self, mocker):
        mock_retriever = MagicMock()
        mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=mock_retriever,
        )
        mocker.patch.dict(
            "os.environ",
            {"MSGFLUX_WEB_SEARCH_ENGINE": "retriever/wikipedia"},
            clear=False,
        )

        tool = WebSearch()

        assert tool.engine == "retriever/wikipedia"
        assert tool.engine_kind == "retriever"

    def test_retriever_engine_sets_query_only_schema(self, mocker):
        mock_retriever = MagicMock()
        mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=mock_retriever,
        )

        tool = WebSearch("retriever/wikipedia")

        assert tool.name == "web_search"
        assert tool.engine_kind == "retriever"
        assert tool.engine_target == "wikipedia"
        assert tool.annotations == {"query": str, "return": str}
        assert "wikipedia" in tool.description.lower()
        assert "retriever backend" in tool.description.lower()

    def test_model_engine_adds_prompt_annotation(self, mocker):
        mock_factory = mocker.patch(
            "msgflux.tools.builtin.web_search.Model.chat_completion",
            return_value=MagicMock(),
        )

        tool = WebSearch(
            "model/openai/gpt-4o-search-preview",
            web_search_options={"search_context_size": "low"},
        )

        assert tool.engine_kind == "model"
        assert tool.engine_target == "openai/gpt-4o-search-preview"
        assert tool.annotations["query"] == str
        assert tool.annotations["prompt"] == Optional[str]
        assert tool.annotations["return"] == str
        assert "openai/gpt-4o-search-preview" in tool.description
        assert "web_search_options" in tool.description
        mock_factory.assert_called_once_with(
            "openai/gpt-4o-search-preview",
            web_search_options={"search_context_size": "low"},
        )

    def test_invalid_engine_format_raises(self):
        with pytest.raises(ValueError, match="engine format"):
            WebSearch("invalid")

    def test_retriever_engine_rejects_prompt(self, mocker):
        mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=MagicMock(),
        )

        tool = WebSearch("retriever/wikipedia")

        with pytest.raises(ValueError, match="prompt"):
            tool("python", prompt="Guide the model.")

    def test_missing_engine_raises(self, mocker):
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(ValueError, match="MSGFLUX_WEB_SEARCH_ENGINE"):
            WebSearch()


class TestWebSearchCall:
    def test_retriever_engine_returns_dict(self, mocker):
        mock_retriever = MagicMock()
        mock_retriever.return_value = dotdict(
            {
                "response_type": "web_search",
                "data": [
                    {
                        "results": [
                            {
                                "data": {
                                    "title": "Example",
                                    "content": "Snippet",
                                }
                            }
                        ]
                    }
                ],
            }
        )
        mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=mock_retriever,
        )

        tool = WebSearch("retriever/wikipedia", top_k=3)
        result = tool("python")

        mock_retriever.assert_called_once_with("python", top_k=3)
        assert result["data"] == [
            {
                "results": [
                    {
                        "data": {
                            "title": "Example",
                            "content": "Snippet",
                        }
                    }
                ]
            }
        ]
        assert result["annotations"] == []

    def test_model_engine_forwards_prompt_to_model(self, mocker):
        mock_response = MagicMock()
        mock_response.consume.return_value = "final answer"
        mock_response.metadata = dotdict(
            {
                "annotations": [
                    {"url_citation": {"url": "https://example.com"}}
                ]
            }
        )
        mock_model = MagicMock(return_value=mock_response)
        mocker.patch(
            "msgflux.tools.builtin.web_search.Model.chat_completion",
            return_value=mock_model,
        )

        tool = WebSearch(
            "model/openai/gpt-4o-search-preview",
            web_search_options={"search_context_size": "low"},
        )

        result = tool("What is the latest release?", prompt="Use concise style.")

        assert result["data"] == "final answer"
        assert result["annotations"] == [
            {"url_citation": {"url": "https://example.com"}}
        ]
        mock_model.assert_called_once_with(
            messages="What is the latest release?",
            system_prompt="Use concise style.",
        )
        mock_response.consume.assert_called_once()


class TestWebSearchToolLibraryIntegration:
    def test_retriever_mode_exposes_dynamic_schema(self, mocker):
        mock_retriever = MagicMock()
        mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=mock_retriever,
        )

        tool = WebSearch("retriever/wikipedia")
        library = ToolLibrary(name="search", tools=[tool])
        schemas = library.get_tool_json_schemas()

        assert schemas[0]["function"]["name"] == "web_search"
        assert "wikipedia" in schemas[0]["function"]["description"].lower()
        assert "query" in schemas[0]["function"]["parameters"]["properties"]
        assert "prompt" not in schemas[0]["function"]["parameters"]["properties"]

    def test_model_mode_exposes_prompt_in_schema(self, mocker):
        mock_model = MagicMock()
        mocker.patch(
            "msgflux.tools.builtin.web_search.Model.chat_completion",
            return_value=mock_model,
        )

        tool = WebSearch("model/openai/gpt-4o-search-preview")
        library = ToolLibrary(name="search", tools=[tool])
        schemas = library.get_tool_json_schemas()

        props = schemas[0]["function"]["parameters"]["properties"]
        assert "query" in props
        assert "prompt" in props
        assert "web_search_options" not in props
