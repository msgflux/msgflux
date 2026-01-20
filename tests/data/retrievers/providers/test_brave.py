import pytest
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Mock the module before import to ensure all symbols are available
mock_brave_module = MagicMock()
sys.modules["brave_search_python_client"] = mock_brave_module

from msgflux.data.retrievers.providers.brave import BraveWebRetriever
from msgflux.dotdict import dotdict

@pytest.fixture
def mock_brave_client():
    # Setup the client instance that BraveSearch() will return
    client_instance = AsyncMock()
    mock_brave_module.BraveSearch.return_value = client_instance
    return client_instance

@pytest.fixture
def retriever(mock_brave_client):
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}):
        # BraveSearch is already mocked via sys.modules
        return BraveWebRetriever()



@pytest.mark.asyncio
async def test_search_mode(retriever, mock_brave_client):
    # Mock return value for web search
    mock_result = MagicMock()
    # Structure based on library: response.web.results
    mock_web_response = MagicMock()
    
    mock_item = MagicMock()
    mock_item.title = "Test Title"
    mock_item.description = "Test Content"
    mock_item.url = "http://test.com"
    mock_item.thumbnail.src = "img.jpg"
    
    # Needs to be iterable
    mock_web_response.results = [mock_item]
    mock_result.web = mock_web_response
    
    mock_brave_client.web.return_value = mock_result
    
    retriever.return_images = True

    results = await retriever.acall("query", top_k=1)
    
    assert results.response_type == "web_search"
    assert len(results.data[0].results) == 1
    result_data = results.data[0].results[0]['data']
    assert result_data['title'] == "Test Title"
    assert result_data['content'] == "Test Content"
    assert results.data[0].results[0]['images'][0] == "img.jpg"
    
    # Verify WebSearchRequest was initialized correctly
    mock_brave_module.WebSearchRequest.assert_called_with(q="query", count=1)
    # Verify client.web was called (implying the request object was passed)
    mock_brave_client.web.assert_called_once()

@pytest.mark.asyncio
async def test_news_mode(mock_brave_client):
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}):
        retriever = BraveWebRetriever(mode="news")
        
        mock_result = MagicMock()
        mock_item = MagicMock()
        mock_item.title = "News Title"
        mock_item.description = "News Content"
        mock_item.url = "http://news.com"
        mock_item.age = "2h"
        # Mock thumbnail to imply no image
        mock_item.thumbnail = None
        
        mock_result.results = [mock_item]
        mock_brave_client.news.return_value = mock_result
        
        results = await retriever.acall("news query", top_k=2)
        
        assert len(results.data[0].results) == 1
        assert results.data[0].results[0]['data']['title'] == "News Title"
        
        # Verify NewsSearchRequest was initialized correctly
        mock_brave_module.NewsSearchRequest.assert_called_with(q="news query", count=2)
        mock_brave_client.news.assert_called_once()

@pytest.mark.asyncio
async def test_image_mode(mock_brave_client):
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}):
        retriever = BraveWebRetriever(mode="image")
        
        mock_result = MagicMock()
        mock_item = MagicMock()
        mock_item.title = "Image Title"
        mock_item.url = "http://img.com"
        
        mock_result.results = [mock_item]
        mock_brave_client.images.return_value = mock_result
        
        results = await retriever.acall("image query", top_k=3)
        
        assert len(results.data[0].results) == 1
        assert results.data[0].results[0]['data']['title'] == "Image Title"
        assert results.data[0].results[0]['images'][0] == "http://img.com"
        
        # Verify ImagesSearchRequest was initialized correctly
        mock_brave_module.ImagesSearchRequest.assert_called_with(q="image query", count=3)
        mock_brave_client.images.assert_called_once()
