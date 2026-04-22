from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import msgflux.data.retrievers.providers.serpapi as serpapi_provider
from msgflux.data.retrievers.providers.serpapi import SerpApiWebRetriever


@pytest.fixture
def mock_serpapi_client():
    client_cls = MagicMock()
    fake_serpapi = SimpleNamespace(Client=client_cls)
    with patch.object(serpapi_provider, "serpapi", fake_serpapi):
        yield client_cls


@pytest.fixture
def mock_serpapi_client_instance(mock_serpapi_client):
    client_instance = MagicMock()
    mock_serpapi_client.return_value = client_instance
    return client_instance


@pytest.fixture
def retriever(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        return SerpApiWebRetriever()


def test_init_defaults(retriever):
    assert retriever.engine == "google"
    assert retriever.location is None
    assert retriever.gl is None
    assert retriever.hl is None
    assert retriever.safe is None
    assert retriever.tbm is None


def test_init_custom_params(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(
            engine="google",
            location="Austin,Texas",
            gl="us",
            hl="en",
            safe="active",
            tbm="nws",
        )
        assert retriever.location == "Austin,Texas"
        assert retriever.gl == "us"
        assert retriever.hl == "en"
        assert retriever.safe == "active"
        assert retriever.tbm == "nws"


def test_init_uses_official_serpapi_client(mock_serpapi_client):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        SerpApiWebRetriever()

    mock_serpapi_client.assert_called_once_with(api_key="test_key")


def test_init_accepts_legacy_env_names(mock_serpapi_client):
    with patch.dict("os.environ", {"SERP_API_KEY": "test_key"}, clear=True):
        SerpApiWebRetriever()

    mock_serpapi_client.assert_called_once_with(api_key="test_key")


@pytest.mark.asyncio
async def test_organic_search(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever()

        mock_response = {
            "organic_results": [
                {
                    "title": "Test Title",
                    "link": "https://example.com",
                    "snippet": "Test snippet content",
                    "date": "2024-01-15",
                }
            ]
        }

        mock_serpapi_client_instance.search.return_value = mock_response

        results = await retriever.acall("test query", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert len(results.data[0].results) == 1

        result_data = results.data[0].results[0]["data"]
        assert result_data["title"] == "Test Title"
        assert result_data["url"] == "https://example.com"
        assert result_data["content"] == "Test snippet content"
        assert result_data["date"] == "2024-01-15"


@pytest.mark.asyncio
async def test_news_search(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(tbm="nws")

        mock_response = {
            "news_results": [
                {
                    "title": "News Title",
                    "link": "https://news.com",
                    "snippet": "News snippet",
                    "date": "2 hours ago",
                }
            ]
        }

        mock_serpapi_client_instance.search.return_value = mock_response

        results = await retriever.acall("news query", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data[0].results) == 1
        result_data = results.data[0].results[0]["data"]
        assert result_data["title"] == "News Title"


@pytest.mark.asyncio
async def test_search_with_location_and_engine(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(
            engine="bing",
            location="Austin,Texas",
            gl="us",
        )

        mock_response = {"organic_results": []}
        mock_serpapi_client_instance.search.return_value = mock_response

        await retriever.acall("local query", top_k=5)

        call_args = mock_serpapi_client_instance.search.call_args[0][0]
        assert call_args["engine"] == "bing"
        assert call_args["q"] == "local query"
        assert call_args["location"] == "Austin,Texas"
        assert call_args["gl"] == "us"
        assert call_args["num"] == 5
        assert "api_key" not in call_args


def test_search_uses_engine_specific_query_param(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(engine="yahoo")

    params = retriever._build_search_params("query", top_k=1)

    assert params["engine"] == "yahoo"
    assert params["p"] == "query"
    assert "q" not in params


def test_image_search(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(tbm="isch")

    mock_response = {
        "images_results": [
            {
                "title": "Image Title",
                "link": "https://example.com/page",
                "original": "https://example.com/image.jpg",
                "thumbnail": "https://example.com/thumb.jpg",
                "source": "Example",
            }
        ]
    }
    mock_serpapi_client_instance.search.return_value = mock_response

    results = retriever("image query", top_k=1)

    result = results.data[0].results[0]
    assert result["data"]["title"] == "Image Title"
    assert result["data"]["url"] == "https://example.com/page"
    assert result["images"] == ["https://example.com/thumb.jpg"]


def test_shopping_search(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(tbm="shop")

    mock_response = {
        "shopping_results": [
            {
                "title": "Product Title",
                "link": "https://shop.example/product",
                "source": "Shop",
                "price": "$10.00",
            }
        ]
    }
    mock_serpapi_client_instance.search.return_value = mock_response

    results = retriever("shopping query", top_k=1)

    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "Product Title"
    assert result_data["url"] == "https://shop.example/product"
    assert result_data["price"] == "$10.00"


def test_results_are_limited_to_top_k(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever()

    mock_serpapi_client_instance.search.return_value = {
        "organic_results": [
            {"title": "First", "link": "https://example.com/1"},
            {"title": "Second", "link": "https://example.com/2"},
        ]
    }

    results = retriever("query", top_k=1)

    assert len(results.data[0].results) == 1
    assert results.data[0].results[0]["data"]["title"] == "First"


def test_sync_search(mock_serpapi_client_instance):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever()

        mock_response = {
            "organic_results": [
                {
                    "title": "Sync Test",
                    "link": "https://sync.com",
                    "snippet": "Sync content",
                }
            ]
        }

        mock_serpapi_client_instance.search.return_value = mock_response

        results = retriever(["sync query"], top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert results.data[0].results[0]["data"]["title"] == "Sync Test"


def test_init_raises_without_api_key(mock_serpapi_client):
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            SerpApiWebRetriever()
