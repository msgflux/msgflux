from os import getenv

try:
    import httpx
except Exception as e:
    raise ImportError(
        "`httpx` is not detected, please installusing `pip install msgflux[httpx]`"
    ) from e
from msgflux.envs import envs
from msgflux.models.base import BaseModel
from msgflux.utils.tenacity import model_retry


class HTTPXModelClient(BaseModel):
    """HTTPX interface for routes not supported by the OpenAI client."""

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    def _initialize(self):
        """Initialize the HTTPX client with empty API key."""
        self.current_key_index = 0
        timeout = getenv("OPENAI_TIMEOUT", None)
        self.client = httpx.Client(
            limits=httpx.Limits(max_connections=2000, max_keepalive_connections=100),
            timeout=timeout,
            transport=httpx.HTTPTransport(retries=envs.httpx_max_retries),
        )
        self.aclient = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=2000, max_keepalive_connections=100),
            timeout=timeout,
            transport=httpx.AsyncHTTPTransport(retries=envs.httpx_max_retries),
        )

    @model_retry
    def _execute(self, **kwargs):
        params = {"model": self.model_id, **kwargs}
        if hasattr(self, "sampling_run_params"):
            params.update(self.sampling_run_params)
        url = self.sampling_params["base_url"] + self.endpoint
        headers = self.headers
        if hasattr(self, "_api_key"):
            api_key = self._api_key[self.current_key_index] if isinstance(self._api_key, list) else self._api_key
            headers["Authorization"] = f"Bearer {api_key}"
        response = self.client.post(url, headers=headers, json=params)
        response.raise_for_status()
        model_output = response.json()
        return model_output

    @model_retry
    async def _aexecute(self, **kwargs):
        params = {"model": self.model_id, **kwargs}
        if hasattr(self, "sampling_run_params"):
            params.update(self.sampling_run_params)
        url = self.sampling_params["base_url"] + self.endpoint
        headers = self.headers
        if hasattr(self, "_api_key"):
            api_key = self._api_key[self.current_key_index] if isinstance(self._api_key, list) else self._api_key
            headers["Authorization"] = f"Bearer {api_key}"
        response = await self.aclient.post(url, headers=headers, json=params)
        response.raise_for_status()
        model_output = response.json()
        return model_output
