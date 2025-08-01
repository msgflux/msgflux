from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RetrieverResult:
    data: str
    score: Optional[float]
    images: Optional[List[str]]

@dataclass
class RetrieverResults:
    results: List[RetrieverResult]
