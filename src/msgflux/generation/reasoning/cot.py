from msgspec import Meta, Struct
from typing_extensions import Annotated


class ChainOfThought(Struct):
    reasoning: Annotated[str, Meta(description="Let's think step by step in order to")]
    final_answer: str
