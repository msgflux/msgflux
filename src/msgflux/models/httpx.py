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
        """Initialize the OpenAI client with empty API key."""
        self.current_key_index = 0
        timeout = getenv("OPENAI_TIMEOUT", None)
        self.client = httpx.Client(
            limits=httpx.Limits(max_connections=2000, max_keepalive_connections=100),
            timeout=timeout,
            transport=httpx.HTTPTransport(retries=envs.httpx_max_retries),
        )

    @model_retry
    def _execute(self, **kwargs):
        params = {"model": self.model_id, **kwargs}
        if hasattr(self, "sampling_run_params"):
            params.update(self.sampling_run_params)
        url = self.sampling_params["base_url"] + self.url_path
        headers = self.headers
        if hasattr(self, "_api_key"):
            headers["Authorization"] = (
                f"Bearer {self._api_key[0]}"  # Not rotate for now
            )
        response = self.client.post(url, headers=headers, json=params)
        response.raise_for_status()
        model_output = response.json()
        return model_output
