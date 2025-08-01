from enum import Enum
from typing import Dict, List, Optional

from msgspec import Struct
from typing_extensions import Generic, TypeVar

T = TypeVar("T", default=str)


class ThoughtState(Enum):
    PROMISING = "promising"
    NEUTRAL = "neutral"
    DEAD_END = "dead_end"


class Thought(Struct):
    content: str
    evaluation: ThoughtState
    score: float
    reasoning: str


class ThoughtNode(Struct, kw_only=True):
    thought: Thought
    children: Optional[List["ThoughtNode"]] = []
    depth: int
    branch_id: str


class TreeExploration(Struct):
    current_node: ThoughtNode
    promising_paths: List[List[Thought]]
    dead_ends: List[List[Thought]]
    evaluation_metrics: Dict[str, float]


class TreeOfThoughts(Struct, Generic[T]):
    initial_thoughts: List[Thought]
    exploration_tree: ThoughtNode
    best_path: List[Thought]
    confidence_score: float
    reasoning_summary: str
    final_answer: T


TOT_SYSTEM_MESSAGE = """
You must structure your response using the provided 'TreeOfThoughts' schema.

Start by generating `initial_thoughts`.
Construct the `exploration_tree` by exploring reasoning paths using `ThoughtNode`s.
Each node must contain a `Thought`.
For every `Thought`, provide its `content`, evaluate it as `PROMISING`, `NEUTRAL`, or `DEAD_END` in `evaluation`, assign a `score`, and give `reasoning`.
Assign `depth` and a unique `branch_id` to each `ThoughtNode`. Track children nodes.
After exploration, identify and list the `best_path` (sequence of Thoughts).
Provide a `reasoning_summary` of the exploration process and an overall `confidence_score`.

Finally, deliver the `final_answer` according to the expected type.
""" # noqa: E501
