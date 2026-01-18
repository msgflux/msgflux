from typing import Union

from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import ChatCompletionModel
from msgflux.nn.modules.module import Module


class LM(Module):
    """Language Model wrapper - enables hooks and composition for LM calls."""

    def __init__(self, model: Union[ChatCompletionModel, ModelGateway]):
        """Initialize LM wrapper.

        Args:
            model:
                A msgflux Model instance (e.g., from mf.Model.chat_completion()).

        Raises:
            TypeError:
                If model is not a chat completion model.
        """
        super().__init__()
        self._set_model(model)

    def forward(self, **kwargs) -> Union[ModelResponse, ModelStreamResponse]:
        """Forward call to chat model.

        Args:
            **kwargs:
                Arguments passed to the model (messages, tool_schemas, etc.)

        Returns:
            Model response.
        """
        return self.model(**kwargs)

    async def aforward(self, **kwargs) -> Union[ModelResponse, ModelStreamResponse]:
        """Async forward call to chat model."""
        return await self.model.acall(**kwargs)

    def _set_model(self, model: Union[ChatCompletionModel, ModelGateway]):
        if not hasattr(model, "model_type") or model.model_type != "chat_completion":
            raise TypeError(
                f"`model` must be a `chat_completion` model, given `{type(model)}`"
            )
        self.register_buffer("model", model)
