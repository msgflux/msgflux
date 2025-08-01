from typing import Dict, List

from msgspec import Struct
from typing_extensions import Generic, TypeVar

T = TypeVar("T", default=str)


class Solution(Struct):
    reasoning_steps: List[str]
    answer: str
    confidence_score: float


class SelfConsistency(Struct, Generic[T]):
    solutions: List[Solution]
    most_common_answer: str
    confidence_distribution: Dict[str, float]
    final_answer: T


SELF_CONSISTENCY_SYSTEM_MESSAGE = """
You must structure your response using the provided 'SelfConsistency' schema.

Generate multiple diverse solution paths to the problem.
For each attempt, create a 'Solution' object containing:
- 'reasoning_steps': A list of strings detailing the derivation.
- 'answer': The conclusion reached by this path.
- 'confidence_score': Your confidence in this specific solution.
Populate the 'solutions' list with all generated 'Solution' objects.
Analyze the 'solutions' list to determine the 'most_common_answer'.
Calculate and provide the 'confidence_distribution' (e.g., frequency of each unique answer).

Based on the analysis, provide the consolidated 'final_answer'.
""" # noqa: E501
