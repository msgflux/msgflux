from enum import Enum
from typing import Dict, List, Optional, Union

from msgspec import Struct
from typing_extensions import Generic, TypeVar

T = TypeVar("T", default=str)


class StepType(Enum):
    OBSERVATION = "observation"
    CALCULATION = "calculation"
    DEDUCTION = "deduction"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"


class Difficulty(Enum):
    BASIC = "basic"
    MIDDLE = "middle"
    HIGH = "high"


class Context(Struct):
    domain: str
    difficulty: Difficulty
    required_knowledge: str


class Step(Struct):
    step_type: StepType
    explanation: str
    output: str
    confidence: float
    intermediate_results: Optional[Dict[str, Union[str, float, int]]]
    steps_dependencies_idx: Optional[List[int]] = []


class ValidationStep(Struct):
    method: str
    result: bool
    error_margin: Optional[float]
    explanation: str


class ChainOfThoughts(Struct, Generic[T], kw_only=True):
    context: Optional[Context]
    assumptions: List[str] = []
    steps: List[Step]
    validation: Optional[ValidationStep]
    final_answer: T
    confidence_score: float
    alternative_approaches: Optional[List[str]] = []


COT_SYSTEM_MESSAGE = """
You must structure your response using the provided 'ChainOfThoughts' schema.

Define the 'Context' including domain, difficulty, and required knowledge.
List any 'Assumptions' made.
Detail each reasoning step in the 'Steps' list, specifying 'step_type', 'explanation',
'output', and 'confidence'. Include 'intermediate_results' and 'steps_dependencies_idx'
where applicable.
Provide the 'final_answer' according to the expected type.
State the overall 'confidence_score'.
Optionally, list 'alternative_approaches' or include a 'ValidationStep' detailing the
validation method, result, and explanation.

Adhere strictly to the defined types and structure.
"""
