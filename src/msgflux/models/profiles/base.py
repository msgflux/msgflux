"""Model profile data structures.

Provides dataclasses for model and provider profiles from models.dev.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelCapabilities:
    """Model capabilities and features."""

    tool_call: bool
    structured_output: bool
    reasoning: bool
    attachment: bool
    temperature: bool


@dataclass
class ModelModalities:
    """Input/output modalities supported by the model."""

    input: List[str]  # e.g., ["text", "image"]
    output: List[str]  # e.g., ["text"]


@dataclass
class ModelCost:
    """Model pricing information.

    All costs are in USD.
    """

    # Cost per 1M tokens (standard from API)
    input_per_million: float
    output_per_million: float
    cache_read_per_million: Optional[float] = None

    @property
    def input_per_token(self) -> float:
        """Cost per single input token in USD."""
        return self.input_per_million / 1_000_000

    @property
    def output_per_token(self) -> float:
        """Cost per single output token in USD."""
        return self.output_per_million / 1_000_000

    @property
    def cache_read_per_token(self) -> Optional[float]:
        """Cost per single cached token in USD."""
        if self.cache_read_per_million is None:
            return None
        return self.cache_read_per_million / 1_000_000

    def calculate(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate total cost in USD.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens (if supported)

        Returns:
            Total cost in USD
        """
        cost = (
            input_tokens * self.input_per_token + output_tokens * self.output_per_token
        )
        if cached_tokens > 0 and self.cache_read_per_token:
            cost += cached_tokens * self.cache_read_per_token
        return cost


@dataclass
class ModelLimits:
    """Model context and output limits."""

    context: int  # Maximum context window size
    output: int  # Maximum output tokens


@dataclass
class ModelProfile:
    """Complete profile for a model.

    Contains all metadata, capabilities, pricing, and limits.
    """

    id: str
    name: str
    provider_id: str
    capabilities: ModelCapabilities
    modalities: ModelModalities
    cost: ModelCost
    limits: ModelLimits
    knowledge: Optional[str]  # Knowledge cutoff date (e.g., "2024-04")
    release_date: Optional[str]
    last_updated: Optional[str]
    open_weights: bool


@dataclass
class ProviderProfile:
    """Complete profile for a provider.

    Contains provider metadata and all models offered.
    """

    id: str
    name: str
    api_base: Optional[str]
    env_vars: List[str]
    doc_url: Optional[str]
    models: Dict[str, ModelProfile]
