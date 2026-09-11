"""Tests for msgflux.models.httpx module."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


class TestHTTPXModelClientImport:
    """Test HTTPXModelClient import and initialization."""

    def test_httpx_import_available(self):
        """Test that HTTPXModelClient imports with the native HTTPX2 dependency."""
        from msgflux.models.httpx import HTTPXModelClient

        assert HTTPXModelClient is not None


class TestHTTPXModelClient:
    """Test suite for HTTPXModelClient."""

    @pytest.fixture
    def concrete_httpx_client(self):
        """Create concrete HTTPXModelClient for testing."""
        from msgflux.models.httpx import HTTPXModelClient
        from msgflux.models.response import ModelResponse

        class ConcreteHTTPXClient(HTTPXModelClient):
            """Concrete implementation for testing."""

            model_type = "test_model"
            provider = "test_provider"

            def __call__(self, **kwargs):
                return self._execute(**kwargs)

            async def acall(self, **kwargs):
                return await self._aexecute(**kwargs)

        return ConcreteHTTPXClient

    @pytest.fixture
    def mock_httpx(self):
        """Mock the HTTPX2 module."""
        with patch("msgflux.models.httpx.httpx2") as mock_httpx:
            # Mock Client and AsyncClient
            mock_client = MagicMock()
            mock_async_client = MagicMock()

            mock_httpx.Client.return_value = mock_client
            mock_httpx.AsyncClient.return_value = mock_async_client
            mock_httpx.Limits = MagicMock()
            mock_httpx.HTTPTransport = MagicMock()
            mock_httpx.AsyncHTTPTransport = MagicMock()

            yield {
                "httpx2": mock_httpx,
                "client": mock_client,
                "async_client": mock_async_client,
            }

    def test_httpx_client_initialization(self, concrete_httpx_client, mock_httpx):
        """Test HTTPXModelClient initialization."""
        client = concrete_httpx_client()
        client._initialize()

        # Verify clients were created
        assert client.client is not None
        assert client.aclient is not None
        assert client.current_key_index == 0

    def test_httpx_client_headers(self, concrete_httpx_client, mock_httpx):
        """Test HTTPXModelClient has correct headers."""
        client = concrete_httpx_client()

        assert "accept" in client.headers
        assert client.headers["accept"] == "application/json"
        assert "Content-Type" in client.headers
        assert client.headers["Content-Type"] == "application/json"

    def test_execute_basic(self, concrete_httpx_client, mock_httpx):
        """Test _execute method with basic parameters."""
        client = concrete_httpx_client()
        client.model_id = "test-model"
        client.endpoint = "/test"
        client.sampling_params = {"base_url": "http://test.example.com"}
        client._initialize()

        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        client.client.post.return_value = mock_response

        result = client._execute(param1="value1")

        # Verify POST was called with correct parameters
        client.client.post.assert_called_once()
        call_args = client.client.post.call_args

        assert call_args[0][0] == "http://test.example.com/test"
        assert "json" in call_args[1]
        assert call_args[1]["json"]["model"] == "test-model"
        assert call_args[1]["json"]["param1"] == "value1"

    def test_execute_with_api_key(self, concrete_httpx_client, mock_httpx):
        """Test _execute method includes API key in headers."""
        client = concrete_httpx_client()
        client.model_id = "test-model"
        client.endpoint = "/test"
        client.sampling_params = {"base_url": "http://test.example.com"}
        client._api_key = ["test-key-123"]
        client._initialize()

        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        client.client.post.return_value = mock_response

        result = client._execute()

        # Verify Authorization header was set
        call_args = client.client.post.call_args
        assert "headers" in call_args[1]
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key-123"

    def test_execute_with_sampling_run_params(self, concrete_httpx_client, mock_httpx):
        """Test _execute method merges sampling_run_params."""
        client = concrete_httpx_client()
        client.model_id = "test-model"
        client.endpoint = "/test"
        client.sampling_params = {"base_url": "http://test.example.com"}
        client.sampling_run_params = {"temperature": 0.7, "max_tokens": 100}
        client._initialize()

        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        client.client.post.return_value = mock_response

        result = client._execute(input="test")

        # Verify sampling_run_params were included
        call_args = client.client.post.call_args
        assert call_args[1]["json"]["temperature"] == 0.7
        assert call_args[1]["json"]["max_tokens"] == 100
        assert call_args[1]["json"]["input"] == "test"

    @pytest.mark.asyncio
    async def test_aexecute_basic(self, concrete_httpx_client, mock_httpx):
        """Test async _aexecute method."""
        client = concrete_httpx_client()
        client.model_id = "test-model"
        client.endpoint = "/test"
        client.sampling_params = {"base_url": "http://test.example.com"}
        client._initialize()

        # Mock async response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        client.aclient.post = AsyncMock(return_value=mock_response)

        result = await client._aexecute(param1="value1")

        # Verify async POST was called
        client.aclient.post.assert_called_once()
        call_args = client.aclient.post.call_args

        assert call_args[0][0] == "http://test.example.com/test"
        assert call_args[1]["json"]["model"] == "test-model"
        assert call_args[1]["json"]["param1"] == "value1"

    @pytest.mark.asyncio
    async def test_aexecute_with_api_key(self, concrete_httpx_client, mock_httpx):
        """Test async _aexecute method includes API key."""
        client = concrete_httpx_client()
        client.model_id = "test-model"
        client.endpoint = "/test"
        client.sampling_params = {"base_url": "http://test.example.com"}
        client._api_key = ["async-key-456"]
        client._initialize()

        # Mock async response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        client.aclient.post = AsyncMock(return_value=mock_response)

        result = await client._aexecute()

        # Verify Authorization header was set
        call_args = client.aclient.post.call_args
        assert "headers" in call_args[1]
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Authorization"] == "Bearer async-key-456"

    def test_execute_raises_for_status(self, concrete_httpx_client, mock_httpx):
        """Test _execute calls raise_for_status on response."""
        client = concrete_httpx_client()
        client.model_id = "test-model"
        client.endpoint = "/test"
        client.sampling_params = {"base_url": "http://test.example.com"}
        client._initialize()

        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        client.client.post.return_value = mock_response

        result = client._execute()

        # Verify raise_for_status was called
        mock_response.raise_for_status.assert_called_once()

    def test_httpx_limits_configuration(self, concrete_httpx_client, mock_httpx):
        """Test HTTPXModelClient configures connection limits."""
        client = concrete_httpx_client()
        client._initialize()

        # Verify Limits was called with correct parameters
        mock_httpx["httpx2"].Limits.assert_called()
        call_args = mock_httpx["httpx2"].Limits.call_args

        # Check that limits were configured
        assert "max_connections" in call_args[1] or call_args[0]
        assert "max_keepalive_connections" in call_args[1] or call_args[0]

    def test_httpx_transport_retries(self, concrete_httpx_client, mock_httpx):
        """Test HTTPXModelClient configures transport with retries."""
        client = concrete_httpx_client()
        client._initialize()

        # Verify HTTPTransport was configured
        mock_httpx["httpx2"].HTTPTransport.assert_called()
        mock_httpx["httpx2"].AsyncHTTPTransport.assert_called()
