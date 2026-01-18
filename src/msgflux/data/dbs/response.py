from typing import Literal

from msgflux._private.response import BaseResponse


class DBResponse(BaseResponse):
    response_type: Literal[
        "key_search",
        "vector_search",
    ]
