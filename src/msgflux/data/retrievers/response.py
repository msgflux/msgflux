from typing import Literal

from msgflux._private.response import BaseResponse


class RetrieverResponse(BaseResponse):
    response_type: Literal[
        "lexical_search",
        "vector_search",
        "web_search"
    ]
