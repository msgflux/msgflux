"""Built-in model capability rules for OpenAI-compatible Responses APIs.

Rules describe model families rather than the service that distributes them.
Unknown IDs remain unsupported unless the caller supplies an explicit override.
"""

import re

from msgflux.models.chat_capabilities import ChatModelCapabilities

_GPT_MODEL_ID = re.compile(
    r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?"
    r"(?:-(?:astra|sol|terra|luna))?"
    r"(?:-\d{4}-\d{2}-\d{2})?$"
)


def built_in_model_capabilities(model_id: str) -> ChatModelCapabilities:
    """Infer known capabilities from a GPT model ID, including routed IDs."""
    name = model_id.rsplit("/", maxsplit=1)[-1]
    match = _GPT_MODEL_ID.fullmatch(name)
    if match is None:
        return ChatModelCapabilities()
    version = (int(match["major"]), int(match["minor"] or 0))
    return ChatModelCapabilities(
        hosted_tool_search=version >= (5, 4),
        reasoning_updates=version >= (6, 0),
    )
