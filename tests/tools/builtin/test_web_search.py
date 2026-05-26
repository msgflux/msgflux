"""Unit tests for msgflux.tools.builtin.web_search."""

from typing import Optional
from unittest.mock import MagicMock

import pytest

from msgflux.core.dotdict import dotdict
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools.builtin.web_search import WebSearch


class TestRetrieverAlias:
    def test_web_search_alias_calls_web_factory(self, mocker):
        mock_factory = mocker.patch("msgflux.data.retrievers.retriever.Retriever.web")

        sentinel = object()
        mock_factory.return_value = sentinel

        from msgflux.data.retrievers.retriever import Retriever

        result = Retriever.web_search("wikipedia", language="pt")

        assert result is sentinel
        mock_factory.assert_called_once_with("wikipedia", language="pt")


class TestWebSearchInit:
    def test_name_and_display_name_attributes(self):
        assert WebSearch.name == "web_search"
        assert WebSearch.display_name == "Web Search"

    def test_engine_can_be_loaded_from_env(self, mocker):
        mock_retriever = MagicMock()
        mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=mock_retriever,
        )
        mocker.patch.dict(
            "os.environ",
            {"MSGFLUX_TOOL_WEB_SEARCH_ENGINE": "retriever/wikipedia"},
            clear=False,
        )

        tool = WebSearch()

        assert tool.engine == "retriever/wikipedia"
        assert tool.engine_kind == "retriever"

    def test_params_can_be_loaded_from_env_json(self, mocker):
        mock_retriever = MagicMock()
        mock_factory = mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=mock_retriever,
        )
        mocker.patch.dict(
            "os.environ",
            {
                "MSGFLUX_TOOL_WEB_SEARCH_ENGINE": "retriever/wikipedia",
                "MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS": '{"language": "pt"}',
                "MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS": '{"top_k": 2}',
            },
            clear=True,
        )

        tool = WebSearch()
        tool("python")

        assert tool.init_params == {"language": "pt"}
        assert tool.call_params == {"top_k": 2}
        mock_factory.assert_called_once_with("wikipedia", language="pt")
        mock_retriever.assert_called_once_with("python", top_k=2)

    def test_explicit_params_override_env_json(self, mocker):
        mock_retriever = MagicMock()
        mock_factory = mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=mock_retriever,
        )
        mocker.patch.dict(
            "os.environ",
            {
                "MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS": '{"language": "pt"}',
                "MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS": '{"top_k": 1}',
            },
            clear=True,
        )

        tool = WebSearch(
            "retriever/wikipedia",
            init_params={"language": "en"},
            call_params={"top_k": 3},
        )
        tool("python")

        mock_factory.assert_called_once_with("wikipedia", language="en")
        mock_retriever.assert_called_once_with("python", top_k=3)

    def test_env_params_must_be_json_objects(self, mocker):
        mocker.patch.dict(
            "os.environ",
            {
                "MSGFLUX_TOOL_WEB_SEARCH_ENGINE": "retriever/wikipedia",
                "MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS": '["invalid"]',
            },
            clear=True,
        )

        with pytest.raises(ValueError, match="MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS"):
            WebSearch()

    def test_explicit_params_must_be_dicts(self, mocker):
        mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=MagicMock(),
        )

        with pytest.raises(TypeError, match="init_params"):
            WebSearch("retriever/wikipedia", init_params=[])

        with pytest.raises(TypeError, match="call_params"):
            WebSearch("retriever/wikipedia", call_params=[])

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
        assert "current information" in tool.description.lower()
        assert "concise search phrase" in tool.description.lower()
        assert "wikipedia" not in tool.description.lower()

    def test_model_engine_adds_goal_annotation(self, mocker):
        mock_factory = mocker.patch(
            "msgflux.tools.builtin.web_search.Model.chat_completion",
            return_value=MagicMock(),
        )

        tool = WebSearch(
            "model/openai/gpt-4o-search-preview",
            init_params={"web_search_options": {"search_context_size": "low"}},
        )

        assert tool.engine_kind == "model"
        assert tool.engine_target == "openai/gpt-4o-search-preview"
        assert tool.annotations["query"] == str
        assert tool.annotations["goal"] == Optional[str]
        assert tool.annotations["return"] == str
        assert "synthesized" in tool.description
        assert "answer grounded in web results" in tool.description
        assert "goal" in tool.description
        assert "openai/gpt-4o-search-preview" not in tool.description
        mock_factory.assert_called_once_with(
            "openai/gpt-4o-search-preview",
            web_search_options={"search_context_size": "low"},
        )

    def test_model_engine_rejects_call_params(self):
        with pytest.raises(ValueError, match="call_params"):
            WebSearch(
                "model/openai/gpt-4o-search-preview",
                call_params={"top_k": 2},
            )

    def test_model_engine_rejects_env_call_params(self, mocker):
        mocker.patch.dict(
            "os.environ",
            {"MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS": '{"top_k": 2}'},
            clear=True,
        )

        with pytest.raises(ValueError, match="call_params"):
            WebSearch("model/openai/gpt-4o-search-preview")

    def test_invalid_engine_format_raises(self):
        with pytest.raises(ValueError, match="engine format"):
            WebSearch("invalid")

    def test_retriever_engine_rejects_goal(self, mocker):
        mocker.patch(
            "msgflux.tools.builtin.web_search.Retriever.web_search",
            return_value=MagicMock(),
        )

        tool = WebSearch("retriever/wikipedia")

        with pytest.raises(ValueError, match="goal"):
            tool("python", goal="Guide the model.")

    def test_missing_engine_raises(self, mocker):
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(ValueError, match="MSGFLUX_TOOL_WEB_SEARCH_ENGINE"):
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

        tool = WebSearch("retriever/wikipedia", call_params={"top_k": 3})
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

    def test_model_engine_forwards_goal_to_model(self, mocker):
        mock_response = MagicMock()
        mock_response.consume.return_value = "final answer"
        mock_response.metadata = dotdict(
            {"annotations": [{"url_citation": {"url": "https://example.com"}}]}
        )
        mock_model = MagicMock(return_value=mock_response)
        mocker.patch(
            "msgflux.tools.builtin.web_search.Model.chat_completion",
            return_value=mock_model,
        )

        tool = WebSearch(
            "model/openai/gpt-4o-search-preview",
            init_params={"web_search_options": {"search_context_size": "low"}},
        )

        result = tool("What is the latest release?", goal="Use concise style.")

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
        assert "current information" in schemas[0]["function"]["description"].lower()
        assert "query" in schemas[0]["function"]["parameters"]["properties"]
        assert "prompt" not in schemas[0]["function"]["parameters"]["properties"]
        assert "goal" not in schemas[0]["function"]["parameters"]["properties"]

    def test_model_mode_exposes_goal_in_schema(self, mocker):
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
        assert "goal" in props
        assert "prompt" not in props
        assert "web_search_options" not in props
        assert "call_params" not in props
