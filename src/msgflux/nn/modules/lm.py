"""Language Model wrapper module for msgflux.nn.

This module provides a Module wrapper around language models, enabling:
- Forward hooks for intercepting/modifying LM calls
- Composition patterns (fallback, retry, caching)
- Consistent interface with other nn.Modules
"""

from typing import Any

from msgflux.nn.modules.module import Module


class LM(Module):
    """Language Model wrapper - enables hooks and composition for LM calls.

    This wrapper allows language models to be used as nn.Modules, which enables:
    - Forward hooks for pre/post-processing
    - Composition patterns (e.g., FallbackLM, CachedLM, RetryLM)
    - Symmetry with other Agent components like ToolLibrary

    Example:
        Basic usage (auto-wrapped by Agent):
        >>> model = mf.Model.chat_completion("openai/gpt-4o")
        >>> agent = nn.Agent(model=model)  # Auto-wraps as LM(model)

        Advanced usage with hooks:
        >>> lm = nn.LM(model=model)
        >>> lm.register_forward_pre_hook(lambda m, a, kw: adjust_temp(kw))
        >>> agent = nn.Agent(model=lm)  # Uses LM directly

        Composition:
        >>> class FallbackLM(nn.LM):
        ...     def forward(self, **kwargs):
        ...         try:
        ...             return self.model(**kwargs)
        ...         except:
        ...             return self.fallback_model(**kwargs)
    """

    def __init__(self, model):
        """Initialize LM wrapper.

        Args:
            model: A msgflux Model instance (e.g., from mf.Model.chat_completion())

        Raises:
            TypeError: If model is not a chat completion model
        """
        super().__init__()

        # Validate model type
        if not hasattr(model, 'model_type') or model.model_type != "chat_completion":
            raise TypeError(
                f"`model` must be a `chat_completion` model, given `{type(model)}`"
            )

        self.model = model

    def forward(self, **kwargs) -> Any:
        """Forward call to underlying model.

        Args:
            **kwargs: Arguments passed to the model (messages, tools, etc.)

        Returns:
            Model response
        """
        return self.model(**kwargs)

    async def aforward(self, **kwargs) -> Any:
        """Async forward call to underlying model.

        Args:
            **kwargs: Arguments passed to the model (messages, tools, etc.)

        Returns:
            Model response
        """
        return await self.model.acall(**kwargs)
