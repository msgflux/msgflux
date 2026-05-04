from typing import ClassVar

from msgspec import Meta, Struct
from typing_extensions import Annotated


class ChainOfThought(Struct):
    extract_reasoning: ClassVar[bool] = True
    reasoning_field: ClassVar[str] = "reasoning"

    reasoning: Annotated[str, Meta(description="Let's think step by step in order to")]
    final_answer: str
