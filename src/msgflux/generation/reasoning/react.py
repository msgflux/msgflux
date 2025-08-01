from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from msgspec import Struct
from typing_extensions import Generic, TypeVar

from msgflux.utils.tool import ToolFlowControl

T = TypeVar("T", default=str)


class ToolCall(Struct, kw_only=True):
    id: Optional[UUID] = uuid4()
    name: str
    arguments: Optional[Dict[str, Any]]
    justification: Optional[str]
    result: Optional[str] = None


class Thought(Struct):
    reasoning: str
    plan: Optional[str] = None


class ReActStep(Struct):
    thought: Thought
    actions: List[ToolCall] = []


class ReAct(Struct, ToolFlowControl, Generic[T]):
    current_step: Optional[ReActStep] = None
    final_answer: Optional[T] = None


REACT_SYSTEM_MESSAGE = """
You must follow the ReAct schema for your response.

Generate a `Thought` containing your reasoning and plan.
Identify and define necessary `actions` by creating a list of `ToolCall` objects.
You MUST use the available tools when needed to achieve the objective.
Include the function `name`, `arguments`, and `justification` for each call.
Await the results for the tool calls.
Analyze the results and repeat the thought-action cycle if necessary.
Once the objective is met using the tools, provide the `final_answer` using the
`ReActResult` structure, including the `answer` and `explanation`.

Do NOT provide the `final_answer` before completing the required tool calls.
"""

REACT_TOOLS_TEMPLATE = """X + y"""
