"""Language Model wrapper module for msgflux.nn.

This module provides a Module wrapper around language models, enabling:
- Forward hooks for intercepting/modifying LM calls
- Composition patterns (fallback, retry, caching)
- Consistent interface with other nn.Modules
"""

from typing import Any, Dict, Optional

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
        """
        super().__init__()
        self.model = model

        # Set module name based on model (sanitize for snake_case)
        if hasattr(model, 'model_name'):
            # Convert model name to snake_case (replace hyphens and dots with underscores)
            sanitized_name = model.model_name.replace("-", "_").replace(".", "_").replace("/", "_")
            self.set_name(f"lm_{sanitized_name}")
        else:
            self.set_name("lm")

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

    def _set_model(self, model):
        """Update the wrapped model.

        Args:
            model: New model instance to wrap
        """
        self.model = model
        if hasattr(model, 'model_name'):
            # Convert model name to snake_case (replace hyphens and dots with underscores)
            sanitized_name = model.model_name.replace("-", "_").replace(".", "_").replace("/", "_")
            self.set_name(f"lm_{sanitized_name}")

    @property
    def model_name(self) -> Optional[str]:
        """Get the name of the wrapped model.

        Returns:
            Model name if available, None otherwise
        """
        return getattr(self.model, 'model_name', None)

    @property
    def model_type(self) -> Optional[str]:
        """Get the type of the wrapped model.

        Returns:
            Model type if available, None otherwise
        """
        return getattr(self.model, 'model_type', None)

    def __repr__(self) -> str:
        """String representation of LM."""
        model_name = self.model_name or "unknown"
        return f"LM(model={model_name})"
