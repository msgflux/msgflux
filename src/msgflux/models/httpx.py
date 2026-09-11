from os import getenv

import httpx2

from msgflux.envs import envs
from msgflux.models.base import BaseModel
from msgflux.models.profiles import ensure_profiles_loaded
from msgflux.utils.tenacity import apply_retry, default_model_retry


class HTTPXModelClient(BaseModel):
    """HTTPX2 interface for routes not supported by the OpenAI client."""

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    def _initialize(self):
        """Initialize the HTTPX2 client with empty API key."""
        self.current_key_index = 0
        timeout = getenv("OPENAI_TIMEOUT", None)
        self.client = httpx2.Client(
            limits=httpx2.Limits(max_connections=2000, max_keepalive_connections=100),
            timeout=timeout,
            transport=httpx2.HTTPTransport(retries=envs.httpx_max_retries),
        )
        self.aclient = httpx2.AsyncClient(
            limits=httpx2.Limits(max_connections=2000, max_keepalive_connections=100),
            timeout=timeout,
            transport=httpx2.AsyncHTTPTransport(retries=envs.httpx_max_retries),
        )

        # Trigger lazy load of model profiles in background
        ensure_profiles_loaded(background=True)

        # Apply retry
        retry_config = getattr(self, "retry", None)
        self._execute = apply_retry(
            self._execute, retry_config, default=default_model_retry
        )
        self._aexecute = apply_retry(
            self._aexecute, retry_config, default=default_model_retry
        )

    def _execute(self, **kwargs):
        params = {"model": self.model_id, **kwargs}
        if hasattr(self, "sampling_run_params"):
            params.update(self.sampling_run_params)
        url = self.sampling_params["base_url"] + self.endpoint
        headers = self.headers
        if hasattr(self, "_api_key"):
            api_key = (
                self._api_key[self.current_key_index]
                if isinstance(self._api_key, list)
                else self._api_key
            )
            headers["Authorization"] = f"Bearer {api_key}"
        response = self.client.post(url, headers=headers, json=params)
        response.raise_for_status()
        model_output = response.json()
        return model_output

    async def _aexecute(self, **kwargs):
        params = {"model": self.model_id, **kwargs}
        if hasattr(self, "sampling_run_params"):
            params.update(self.sampling_run_params)
        url = self.sampling_params["base_url"] + self.endpoint
        headers = self.headers
        if hasattr(self, "_api_key"):
            api_key = (
                self._api_key[self.current_key_index]
                if isinstance(self._api_key, list)
                else self._api_key
            )
            headers["Authorization"] = f"Bearer {api_key}"
        response = await self.aclient.post(url, headers=headers, json=params)
        response.raise_for_status()
        model_output = response.json()
        return model_output
