from typing import Literal

from msgflux._private.response import BaseResponse


class ParserResponse(BaseResponse):
    response_type: Literal[
        "pdf_parse",
        "pptx_parse",
        "xlsx_parse"
    ]
