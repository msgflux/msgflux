"""Shared HTTP client lifecycle for REST-backed retrievers."""

import httpx2


class HTTPRetrieverMixin:
    """Rebuild runtime-only clients from a retriever's serializable timeout."""

    def _initialize(self) -> None:
        self.client = httpx2.Client(timeout=self.timeout)
        try:
            self.aclient = httpx2.AsyncClient(timeout=self.timeout)
        except BaseException:
            self.client.close()
            raise

    def close(self) -> None:
        """Close the synchronous client after synchronous usage."""
        self.client.close()

    async def aclose(self) -> None:
        """Close both clients when the async host releases this retriever."""
        try:
            await self.aclient.aclose()
        finally:
            self.client.close()
