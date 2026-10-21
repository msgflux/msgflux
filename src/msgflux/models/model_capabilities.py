"""Built-in model capability rules for OpenAI-compatible Responses APIs.

Rules describe known model behavior and apply only where the provider checks
them. Unknown IDs remain unsupported unless the caller supplies an override.
"""

import re

from msgflux.models.chat_capabilities import ChatModelCapabilities

_GPT_MODEL_ID = re.compile(
    r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?"
    r"(?:-(?:astra|sol|terra|luna))?"
    r"(?:-\d{4}-\d{2}-\d{2})?$"
)


def built_in_model_capabilities(model_id: str) -> ChatModelCapabilities:
    """Infer known capabilities from a model ID, including routed GPT IDs."""
    name = model_id.rsplit("/", maxsplit=1)[-1]
    match = _GPT_MODEL_ID.fullmatch(name)
    if match is None:
        return ChatModelCapabilities()
    version = (int(match["major"]), int(match["minor"] or 0))
    return ChatModelCapabilities(
        hosted_tool_search=version >= (5, 4),
        reasoning_updates=version >= (6, 0),
    )
