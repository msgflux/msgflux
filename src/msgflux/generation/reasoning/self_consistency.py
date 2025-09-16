from typing import List, Optional
from msgspec import Struct, Meta
from typing_extensions import Annotated


class ReasoningPath(Struct):
    reasoning: Annotated[str, Meta(description="A possible path of reasoning")]
    answer: str


class SelfConsistency(Struct):
    paths: Annotated[
        List[ReasoningPath], Meta(description="Set of multiple reasoning paths")
    ]
    final_answer: Annotated[str, Meta(description="Answer chosen between the paths")]
