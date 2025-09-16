from typing import Literal

from msgflux._private.response import BaseResponse, StreamResponse


class ModelResponse(BaseResponse):
    response_type: Literal[
        "audio_embedding",
        "audio_generation",
        "audio_text_generation",
        "image_embedding",
        "image_generation",
        "image_text_generation",
        "moderation",
        "structured",
        "reasoning_structured",
        "reasoning_text_generation",
        "tool_call",
        "transcript",
        "text_classification",
        "text_embedding",
        "text_generation",
        "text_reranked",
    ]

class ModelStreamResponse(StreamResponse):
    response_type: Literal[
        "audio_generation", "reasoning_text_generation", "text_generation", "tool_call"
    ]
