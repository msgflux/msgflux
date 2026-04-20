import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import combinations
from statistics import fmean
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import msgflux.nn.functional as F
from msgflux.core.dotdict import dotdict
from msgflux.exceptions import TaskError
from msgflux.models.gateway import ModelGateway
from msgflux.models.model import Model
from msgflux.models.types import ChatCompletionModel

ModelLike = Union[str, ChatCompletionModel, ModelGateway]

_TAG_PATTERN = re.compile(r"</?[^>]+>")
_TOKEN_STRIP_CHARS = " \t\r\n.,:;!?()[]{}\"'`"  # noqa: S105


@dataclass(frozen=True)
class VerificationCriterion:
    id: str
    name: str
    description: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("`weight` must be greater than 0")


@dataclass(frozen=True)
class ScoreScale:
    description: str
    score_format: str
    token_values: Mapping[str, float]

    @classmethod
    def letter(cls, granularity: int = 20) -> "ScoreScale":
        if granularity < 2 or granularity > 26:
            raise ValueError("`granularity` must be between 2 and 26")
        final_token = chr(64 + granularity)
        token_values = {
            chr(65 + index): float(granularity - index) for index in range(granularity)
        }
        description = (
            f"Rate on a {granularity}-point scale using letters A through "
            f"{final_token}. A is best and {final_token} is worst."
        )
        return cls(
            description=description,
            score_format=f"LETTER_A_TO_{final_token}",
            token_values=token_values,
        )

    @property
    def suggested_top_logprobs(self) -> int:
        return len(self.token_values)

    @property
    def bounds(self) -> tuple[float, float]:
        values = tuple(self.token_values.values())
        return min(values), max(values)

    @property
    def ordered_tokens(self) -> tuple[str, ...]:
        return tuple(self.token_values.keys())

    def normalize_token(self, token: str) -> Optional[str]:
        cleaned = token.strip().upper()
        if cleaned in self.token_values:
            return cleaned
        return None

    def extract_token(self, token: str) -> Optional[str]:
        normalized = self.normalize_token(token)
        if normalized is not None:
            return normalized

        cleaned = _TAG_PATTERN.sub(" ", token).strip(_TOKEN_STRIP_CHARS)
        normalized = self.normalize_token(cleaned)
        if normalized is not None:
            return normalized

        for part in cleaned.split():
            normalized = self.normalize_token(part.strip(_TOKEN_STRIP_CHARS))
            if normalized is not None:
                return normalized

        for part in re.split(r"[^A-Z0-9_]+", cleaned.upper()):
            if not part:
                continue
            normalized = self._extract_attached_token(part)
            if normalized is not None:
                return normalized
        return None

    def _extract_attached_token(self, token: str) -> Optional[str]:
        for candidate in sorted(self.token_values, key=len, reverse=True):
            if token.endswith(candidate) and token[: -len(candidate)].isdigit():
                return candidate
            if token.startswith(candidate) and token[len(candidate) :].isdigit():
                return candidate
        return None

    def to_normalized_score(self, token_probabilities: Mapping[str, float]) -> float:
        if not token_probabilities:
            return 0.5
        min_value, max_value = self.bounds
        total_probability = sum(token_probabilities.values())
        if total_probability <= 0:
            return 0.5
        expected = (
            sum(
                self.token_values[token] * probability
                for token, probability in token_probabilities.items()
            )
            / total_probability
        )
        if max_value == min_value:
            return 0.5
        return (expected - min_value) / (max_value - min_value)


@dataclass(frozen=True)
class VerificationPromptInput:
    task: Any
    criterion: VerificationCriterion
    candidates: Mapping[str, Any]
    score_scale: ScoreScale
    ground_truth_note: Optional[str] = None
    extra_instructions: Optional[str] = None
    context: Optional[Mapping[str, Any]] = None


@dataclass
class ScoreEvidence:
    score: float
    method: str
    raw_token: Optional[str] = None
    token_probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class VerificationAttempt:
    criterion_id: str
    repetition: int
    prompt_text: Optional[str]
    response_text: str
    scores: dict[str, float]
    evidence: dict[str, ScoreEvidence]
    metadata: dict[str, Any]


@dataclass
class CriterionVerification:
    criterion: VerificationCriterion
    scores: dict[str, float]
    attempts: list[VerificationAttempt]


@dataclass(frozen=True)
class VerificationRequest:
    criterion_index: int
    repetition: int
    prompt_text: str
    request_kwargs: dict[str, Any]


@dataclass(frozen=True)
class TournamentRequest:
    candidate_a_label: str
    candidate_b_label: str
    candidates: dict[str, Any]


@dataclass
class VerifierResult:
    scores: dict[str, float]
    criteria_results: list[CriterionVerification]
    verdict: str
    winner: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if len(self.scores) != 1:
            raise ValueError("`score` is only available for single-candidate results")
        return next(iter(self.scores.values()))


@dataclass
class TournamentMatch:
    candidate_a_label: str
    candidate_b_label: str
    result: VerifierResult


@dataclass
class TournamentResult:
    winner: str
    wins: dict[str, float]
    average_scores: dict[str, float]
    matches: list[TournamentMatch]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ranking(self) -> list[str]:
        return sorted(
            self.wins,
            key=lambda label: (self.wins[label], self.average_scores[label], label),
            reverse=True,
        )


def default_prompt_builder(prompt_input: VerificationPromptInput) -> str:
    candidate_items = list(prompt_input.candidates.items())
    score_tokens = prompt_input.score_scale.ordered_tokens
    best_token = score_tokens[0]
    worst_token = score_tokens[-1]
    sections = [
        "You are an expert verifier. Evaluate the candidate strictly on the "
        "requested criterion.",
    ]
    if prompt_input.ground_truth_note:
        sections.append(prompt_input.ground_truth_note)
    if prompt_input.extra_instructions:
        sections.append(prompt_input.extra_instructions)
    sections.append(f"Task:\n{_render_prompt_value(prompt_input.task)}")
    if prompt_input.context:
        sections.append(f"Context:\n{_format_context(prompt_input.context)}")
    sections.append(
        f"Criterion — {prompt_input.criterion.name}:\n"
        f"{prompt_input.criterion.description}"
    )
    if len(candidate_items) == 1:
        _, candidate = candidate_items[0]
        sections.append(f"Candidate:\n{_render_prompt_value(candidate)}")
        sections.append(
            f"Rating scale:\n{prompt_input.score_scale.description}\n\n"
            "Provide a short analysis and then end with exactly one score token "
            "inside the tag.\n"
            f"Use a single token from {best_token} to {worst_token}. "
            f"{best_token} is best and {worst_token} is worst.\n"
            "Do not output the scale name, numbers attached to the token, "
            "or extra text inside the tag.\n"
            f"Example final line:\n<score>{best_token}</score>"
        )
    else:
        (label_a, candidate_a), (label_b, candidate_b) = candidate_items
        sections.append(
            f"Candidate A ({label_a}):\n{_render_prompt_value(candidate_a)}"
        )
        sections.append(
            f"Candidate B ({label_b}):\n{_render_prompt_value(candidate_b)}"
        )
        sections.append(
            f"Rating scale:\n{prompt_input.score_scale.description}\n\n"
            "Provide a short analysis and then end with exactly one score token "
            "inside each tag.\n"
            f"Use a single token from {best_token} to {worst_token}. "
            f"{best_token} is best and {worst_token} is worst.\n"
            "Do not output the scale name, numbers attached to the token, "
            "or extra text inside the tag.\n"
            "Example final lines:\n"
            f"<score_A>{best_token}</score_A>\n"
            f"<score_B>{worst_token}</score_B>"
        )
    return "\n\n".join(section.strip() for section in sections if section)


TRAJECTORY_ANALYSIS_GROUND_TRUTH_NOTE = (
    "Prioritize observed evidence in the trajectory. Do not trust self-reported "
    "claims of success when commands, outputs, or final verification steps "
    "contradict them."
)

TRAJECTORY_ANALYSIS_CRITERIA = (
    VerificationCriterion(
        id="task_completion",
        name="Task Completion Evidence",
        description=(
            "Decide whether the trajectory shows convincing evidence that the task "
            "was actually completed, not merely attempted."
        ),
    ),
    VerificationCriterion(
        id="verification_quality",
        name="Verification Quality",
        description=(
            "Judge whether the agent ran meaningful verification steps, observed "
            "expected outcomes, and avoided untested final changes."
        ),
    ),
    VerificationCriterion(
        id="error_signals",
        name="Error Signal Detection",
        description=(
            "Look for unresolved errors, failed commands, broken tests, or "
            "contradictory outputs that indicate the final state is not reliable."
        ),
    ),
)

ANSWER_RERANKING_EXTRA_INSTRUCTIONS = (
    "Prefer candidates that directly satisfy the task, avoid unsupported claims, "
    "and do not reward verbosity unless it improves usefulness."
)

ANSWER_RERANKING_CRITERIA = (
    VerificationCriterion(
        id="correctness",
        name="Correctness",
        description="Judge whether the candidate is factually and logically correct.",
    ),
    VerificationCriterion(
        id="instruction_following",
        name="Instruction Following",
        description=(
            "Check whether the candidate follows the task requirements and output "
            "constraints."
        ),
    ),
    VerificationCriterion(
        id="completeness",
        name="Completeness",
        description=(
            "Assess whether the candidate covers the essential parts of the task "
            "without important omissions."
        ),
    ),
    VerificationCriterion(
        id="clarity",
        name="Clarity",
        description=(
            "Assess whether the candidate is clear, concise, and easy to act on."
        ),
    ),
)

GROUNDED_ANSWER_VERIFICATION_GROUND_TRUTH_NOTE = (
    "Treat the provided context as the source of truth. Penalize unsupported "
    "claims, contradictions, or details that cannot be grounded in the context."
)

GROUNDED_ANSWER_VERIFICATION_CRITERIA = (
    VerificationCriterion(
        id="grounding",
        name="Grounding",
        description=(
            "Check whether the candidate is supported by the provided context."
        ),
    ),
    VerificationCriterion(
        id="unsupported_claims",
        name="Unsupported Claims",
        description=(
            "Look for claims, numbers, or specifics that are not justified by the "
            "provided context."
        ),
    ),
    VerificationCriterion(
        id="answer_completeness",
        name="Answer Completeness",
        description=(
            "Assess whether the candidate answers the task fully using only the "
            "grounded information available."
        ),
    ),
)

PATCH_SELECTION_GROUND_TRUTH_NOTE = (
    "Prefer patches that satisfy the task with the smallest justified change "
    "surface. Penalize speculative edits, missing coverage, and obvious "
    "regression risk."
)

PATCH_SELECTION_CRITERIA = (
    VerificationCriterion(
        id="requirement_coverage",
        name="Requirement Coverage",
        description=(
            "Judge whether the patch appears to address the stated task or bug "
            "without leaving core requirements unmet."
        ),
    ),
    VerificationCriterion(
        id="correctness_risk",
        name="Correctness Risk",
        description=(
            "Look for signs that the patch may be logically wrong, incomplete, or "
            "likely to fail in expected scenarios."
        ),
    ),
    VerificationCriterion(
        id="regression_risk",
        name="Regression Risk",
        description=(
            "Estimate whether the patch is likely to break adjacent behavior or "
            "introduce unnecessary side effects."
        ),
    ),
    VerificationCriterion(
        id="minimality",
        name="Minimality",
        description=(
            "Prefer focused patches that solve the problem without unrelated or "
            "overly broad changes."
        ),
    ),
)

TOOL_TRACE_VERIFICATION_GROUND_TRUTH_NOTE = (
    "Prioritize observed tool outputs and trace evidence over confident "
    "self-reports. Penalize final answers that ignore failed actions or "
    "contradict tool results."
)

TOOL_TRACE_VERIFICATION_CRITERIA = (
    VerificationCriterion(
        id="tool_grounding",
        name="Tool Grounding",
        description=(
            "Check whether the final candidate is consistent with the tool outputs "
            "and execution trace."
        ),
    ),
    VerificationCriterion(
        id="unresolved_errors",
        name="Unresolved Errors",
        description=(
            "Look for failed actions, warnings, or contradictions that were not "
            "properly resolved before the final answer."
        ),
    ),
    VerificationCriterion(
        id="final_answer_quality",
        name="Final Answer Quality",
        description=(
            "Judge whether the final candidate answers the task adequately given "
            "the evidence gathered."
        ),
    ),
    VerificationCriterion(
        id="action_efficiency",
        name="Action Efficiency",
        description=(
            "Prefer trajectories that use tools purposefully and avoid needless "
            "steps or repetitive actions."
        ),
    ),
)

SYNTHETIC_DATA_FILTERING_GROUND_TRUTH_NOTE = (
    "Prefer examples that are internally consistent, unambiguous, and useful for "
    "training or evaluation. Penalize noisy, contradictory, or weakly labeled "
    "examples."
)

SYNTHETIC_DATA_FILTERING_CRITERIA = (
    VerificationCriterion(
        id="consistency",
        name="Consistency",
        description=(
            "Check whether the example is internally consistent and free from "
            "contradictory statements or labels."
        ),
    ),
    VerificationCriterion(
        id="label_quality",
        name="Label Quality",
        description=(
            "Assess whether the label, target, or expected output is well-formed "
            "and appropriate for the example."
        ),
    ),
    VerificationCriterion(
        id="ambiguity",
        name="Ambiguity",
        description=(
            "Penalize examples where the task, label, or expected answer is too "
            "ambiguous to be a reliable training signal."
        ),
    ),
    VerificationCriterion(
        id="usefulness",
        name="Usefulness",
        description=(
            "Judge whether the example is informative and worth keeping in a "
            "dataset or evaluation set."
        ),
    ),
)


class LLMAsVerifier:
    def __init__(
        self,
        model: ModelLike,
        *,
        criteria: Sequence[VerificationCriterion],
        prompt_builder: Optional[Callable[[VerificationPromptInput], str]] = None,
        score_scale: Optional[ScoreScale] = None,
        n_verifications: int = 1,
        top_logprobs: Optional[int] = None,
        ground_truth_note: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        model_request_kwargs: Optional[Mapping[str, Any]] = None,
        strict_logprobs: bool = False,
        pass_threshold: float = 0.5,
        verbose: bool = False,
    ):
        self.model = self._resolve_model(model)
        self.criteria = self._normalize_criteria(criteria)
        self.prompt_builder = prompt_builder or default_prompt_builder
        self.score_scale = score_scale or ScoreScale.letter()
        self.n_verifications = n_verifications
        self.top_logprobs = top_logprobs or self.score_scale.suggested_top_logprobs
        self.ground_truth_note = ground_truth_note
        self.extra_instructions = extra_instructions
        self.model_request_kwargs = dict(model_request_kwargs or {})
        self.strict_logprobs = strict_logprobs
        self.pass_threshold = pass_threshold
        self.verbose = verbose

        if self.n_verifications < 1:
            raise ValueError("`n_verifications` must be at least 1")
        if self.top_logprobs < 1:
            raise ValueError("`top_logprobs` must be at least 1")
        if not 0 <= self.pass_threshold <= 1:
            raise ValueError("`pass_threshold` must be between 0 and 1")

        self._validate_model_request_kwargs()

    @classmethod
    def trajectory_analysis(
        cls,
        model: ModelLike,
        *,
        criteria: Sequence[VerificationCriterion] = TRAJECTORY_ANALYSIS_CRITERIA,
        **kwargs: Any,
    ) -> "LLMAsVerifier":
        return cls._from_preset(
            model=model,
            criteria=criteria,
            default_ground_truth_note=TRAJECTORY_ANALYSIS_GROUND_TRUTH_NOTE,
            **kwargs,
        )

    @classmethod
    def answer_reranking(
        cls,
        model: ModelLike,
        *,
        criteria: Sequence[VerificationCriterion] = ANSWER_RERANKING_CRITERIA,
        **kwargs: Any,
    ) -> "LLMAsVerifier":
        return cls._from_preset(
            model=model,
            criteria=criteria,
            default_extra_instructions=ANSWER_RERANKING_EXTRA_INSTRUCTIONS,
            **kwargs,
        )

    @classmethod
    def grounded_answer_verification(
        cls,
        model: ModelLike,
        *,
        criteria: Sequence[VerificationCriterion] = (
            GROUNDED_ANSWER_VERIFICATION_CRITERIA
        ),
        **kwargs: Any,
    ) -> "LLMAsVerifier":
        return cls._from_preset(
            model=model,
            criteria=criteria,
            default_ground_truth_note=GROUNDED_ANSWER_VERIFICATION_GROUND_TRUTH_NOTE,
            **kwargs,
        )

    @classmethod
    def patch_selection(
        cls,
        model: ModelLike,
        *,
        criteria: Sequence[VerificationCriterion] = PATCH_SELECTION_CRITERIA,
        **kwargs: Any,
    ) -> "LLMAsVerifier":
        return cls._from_preset(
            model=model,
            criteria=criteria,
            default_ground_truth_note=PATCH_SELECTION_GROUND_TRUTH_NOTE,
            **kwargs,
        )

    @classmethod
    def tool_trace_verification(
        cls,
        model: ModelLike,
        *,
        criteria: Sequence[VerificationCriterion] = TOOL_TRACE_VERIFICATION_CRITERIA,
        **kwargs: Any,
    ) -> "LLMAsVerifier":
        return cls._from_preset(
            model=model,
            criteria=criteria,
            default_ground_truth_note=TOOL_TRACE_VERIFICATION_GROUND_TRUTH_NOTE,
            **kwargs,
        )

    @classmethod
    def synthetic_data_filtering(
        cls,
        model: ModelLike,
        *,
        criteria: Sequence[VerificationCriterion] = (SYNTHETIC_DATA_FILTERING_CRITERIA),
        **kwargs: Any,
    ) -> "LLMAsVerifier":
        return cls._from_preset(
            model=model,
            criteria=criteria,
            default_ground_truth_note=SYNTHETIC_DATA_FILTERING_GROUND_TRUTH_NOTE,
            **kwargs,
        )

    def __call__(
        self,
        *,
        task: Any,
        candidates: Union[Sequence[Any], Mapping[str, Any]],
        criteria: Optional[Sequence[VerificationCriterion]] = None,
        context: Optional[Mapping[str, Any]] = None,
        ground_truth_note: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> VerifierResult:
        active_criteria = self._get_active_criteria(criteria)
        candidates_map = self._normalize_candidates(
            candidates, min_count=1, max_count=2
        )
        labels = tuple(candidates_map.keys())
        request_batch = self._build_request_batch(
            task=task,
            criteria=active_criteria,
            candidates=candidates_map,
            context=context,
            ground_truth_note=ground_truth_note,
            extra_instructions=extra_instructions,
        )
        criterion_results = self._execute_request_batch_sync(
            criteria=active_criteria,
            request_batch=request_batch,
            labels=labels,
        )

        return self._build_result(criterion_results, labels)

    async def acall(
        self,
        *,
        task: Any,
        candidates: Union[Sequence[Any], Mapping[str, Any]],
        criteria: Optional[Sequence[VerificationCriterion]] = None,
        context: Optional[Mapping[str, Any]] = None,
        ground_truth_note: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> VerifierResult:
        active_criteria = self._get_active_criteria(criteria)
        candidates_map = self._normalize_candidates(
            candidates, min_count=1, max_count=2
        )
        labels = tuple(candidates_map.keys())
        request_batch = self._build_request_batch(
            task=task,
            criteria=active_criteria,
            candidates=candidates_map,
            context=context,
            ground_truth_note=ground_truth_note,
            extra_instructions=extra_instructions,
        )
        criterion_results = await self._execute_request_batch_async(
            criteria=active_criteria,
            request_batch=request_batch,
            labels=labels,
        )

        return self._build_result(criterion_results, labels)

    def select_best(
        self,
        *,
        task: Any,
        candidates: Union[Sequence[Any], Mapping[str, Any]],
        criteria: Optional[Sequence[VerificationCriterion]] = None,
        context: Optional[Mapping[str, Any]] = None,
        ground_truth_note: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> TournamentResult:
        candidates_map = self._normalize_candidates(candidates, min_count=2)
        labels = list(candidates_map.keys())
        active_criteria = self._get_active_criteria(criteria)
        match_batch = self._build_match_batch(candidates_map)

        if len(match_batch) == 1:
            matches = (
                self._evaluate_match_sync(
                    match_batch[0],
                    task=task,
                    criteria=active_criteria,
                    context=context,
                    ground_truth_note=ground_truth_note,
                    extra_instructions=extra_instructions,
                ),
            )
        else:
            match_responses = F.map_gather(
                self._evaluate_match_sync,
                args_list=[(match_request,) for match_request in match_batch],
                kwargs_list=[
                    {
                        "task": task,
                        "criteria": active_criteria,
                        "context": context,
                        "ground_truth_note": ground_truth_note,
                        "extra_instructions": extra_instructions,
                    }
                    for _ in match_batch
                ],
            )
            matches = tuple(
                self._unwrap_parallel_response(response) for response in match_responses
            )

        return self._build_tournament_result(
            labels=labels,
            matches=matches,
            criteria=active_criteria,
        )

    async def aselect_best(
        self,
        *,
        task: Any,
        candidates: Union[Sequence[Any], Mapping[str, Any]],
        criteria: Optional[Sequence[VerificationCriterion]] = None,
        context: Optional[Mapping[str, Any]] = None,
        ground_truth_note: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> TournamentResult:
        candidates_map = self._normalize_candidates(candidates, min_count=2)
        labels = list(candidates_map.keys())
        active_criteria = self._get_active_criteria(criteria)
        match_batch = self._build_match_batch(candidates_map)

        if len(match_batch) == 1:
            matches = (
                await self._evaluate_match_async(
                    match_batch[0],
                    task=task,
                    criteria=active_criteria,
                    context=context,
                    ground_truth_note=ground_truth_note,
                    extra_instructions=extra_instructions,
                ),
            )
        else:
            match_responses = await F.amap_gather(
                self._evaluate_match_async,
                args_list=[(match_request,) for match_request in match_batch],
                kwargs_list=[
                    {
                        "task": task,
                        "criteria": active_criteria,
                        "context": context,
                        "ground_truth_note": ground_truth_note,
                        "extra_instructions": extra_instructions,
                    }
                    for _ in match_batch
                ],
            )
            matches = tuple(
                self._unwrap_parallel_response(response) for response in match_responses
            )

        return self._build_tournament_result(
            labels=labels,
            matches=matches,
            criteria=active_criteria,
        )

    def _build_match_batch(
        self, candidates: Mapping[str, Any]
    ) -> list[TournamentRequest]:
        labels = list(candidates.keys())
        match_batch = []

        # Pairwise tournament matches are independent, so this layer can also run
        # concurrently on top of the per-attempt concurrency inside `__call__`.
        for index_a, index_b in combinations(range(len(labels)), 2):
            label_a = labels[index_a]
            label_b = labels[index_b]
            match_batch.append(
                TournamentRequest(
                    candidate_a_label=label_a,
                    candidate_b_label=label_b,
                    candidates={
                        label_a: candidates[label_a],
                        label_b: candidates[label_b],
                    },
                )
            )

        return match_batch

    def _evaluate_match_sync(
        self,
        match_request: TournamentRequest,
        *,
        task: Any,
        criteria: Sequence[VerificationCriterion],
        context: Optional[Mapping[str, Any]],
        ground_truth_note: Optional[str],
        extra_instructions: Optional[str],
    ) -> TournamentMatch:
        result = self(
            task=task,
            candidates=match_request.candidates,
            criteria=criteria,
            context=context,
            ground_truth_note=ground_truth_note,
            extra_instructions=extra_instructions,
        )
        return TournamentMatch(
            candidate_a_label=match_request.candidate_a_label,
            candidate_b_label=match_request.candidate_b_label,
            result=result,
        )

    async def _evaluate_match_async(
        self,
        match_request: TournamentRequest,
        *,
        task: Any,
        criteria: Sequence[VerificationCriterion],
        context: Optional[Mapping[str, Any]],
        ground_truth_note: Optional[str],
        extra_instructions: Optional[str],
    ) -> TournamentMatch:
        result = await self.acall(
            task=task,
            candidates=match_request.candidates,
            criteria=criteria,
            context=context,
            ground_truth_note=ground_truth_note,
            extra_instructions=extra_instructions,
        )
        return TournamentMatch(
            candidate_a_label=match_request.candidate_a_label,
            candidate_b_label=match_request.candidate_b_label,
            result=result,
        )

    def _build_request_batch(
        self,
        *,
        task: Any,
        criteria: Sequence[VerificationCriterion],
        candidates: Mapping[str, Any],
        context: Optional[Mapping[str, Any]],
        ground_truth_note: Optional[str],
        extra_instructions: Optional[str],
    ) -> list[VerificationRequest]:
        request_batch = []

        # Each criterion/repetition pair is independent, so we can fan them out
        # concurrently and regroup the attempts deterministically afterward.
        for criterion_index, criterion in enumerate(criteria):
            for repetition in range(self.n_verifications):
                prompt_input = self._build_prompt_input(
                    task=task,
                    criterion=criterion,
                    candidates=candidates,
                    context=context,
                    ground_truth_note=ground_truth_note,
                    extra_instructions=extra_instructions,
                )
                request_kwargs = self._build_request_kwargs(prompt_input)
                request_batch.append(
                    VerificationRequest(
                        criterion_index=criterion_index,
                        repetition=repetition,
                        prompt_text=self._coerce_response_text(
                            request_kwargs["messages"]
                        ),
                        request_kwargs=request_kwargs,
                    )
                )

        return request_batch

    def _execute_request_batch_sync(
        self,
        *,
        criteria: Sequence[VerificationCriterion],
        request_batch: Sequence[VerificationRequest],
        labels: Sequence[str],
    ) -> list[CriterionVerification]:
        if len(request_batch) == 1:
            responses = (self._call_model_sync(request_batch[0].request_kwargs),)
        else:
            responses = F.map_gather(
                self._call_model_sync,
                args_list=[(request.request_kwargs,) for request in request_batch],
            )

        return self._build_criterion_results_from_responses(
            criteria=criteria,
            request_batch=request_batch,
            responses=responses,
            labels=labels,
        )

    async def _execute_request_batch_async(
        self,
        *,
        criteria: Sequence[VerificationCriterion],
        request_batch: Sequence[VerificationRequest],
        labels: Sequence[str],
    ) -> list[CriterionVerification]:
        if len(request_batch) == 1:
            responses = (await self._call_model_async(request_batch[0].request_kwargs),)
        else:
            responses = await F.amap_gather(
                self._call_model_async,
                args_list=[(request.request_kwargs,) for request in request_batch],
            )

        return self._build_criterion_results_from_responses(
            criteria=criteria,
            request_batch=request_batch,
            responses=responses,
            labels=labels,
        )

    def _build_criterion_results_from_responses(
        self,
        *,
        criteria: Sequence[VerificationCriterion],
        request_batch: Sequence[VerificationRequest],
        responses: Sequence[Any],
        labels: Sequence[str],
    ) -> list[CriterionVerification]:
        attempts_by_criterion = [[] for _ in criteria]

        # `map_gather` and `amap_gather` preserve input order, which lets us keep
        # repetition ordering stable even though the work ran concurrently.
        for request, response in zip(request_batch, responses):
            resolved_response = self._unwrap_parallel_response(response)
            criterion = criteria[request.criterion_index]
            attempts_by_criterion[request.criterion_index].append(
                self._build_attempt(
                    response=resolved_response,
                    repetition=request.repetition,
                    criterion=criterion,
                    labels=labels,
                    prompt_text=request.prompt_text,
                )
            )

        return [
            self._aggregate_criterion(criteria[index], attempts)
            for index, attempts in enumerate(attempts_by_criterion)
        ]

    def _call_model_sync(self, request_kwargs: Mapping[str, Any]) -> Any:
        return self.model(**request_kwargs)

    async def _call_model_async(self, request_kwargs: Mapping[str, Any]) -> Any:
        return await self.model.acall(**request_kwargs)

    @staticmethod
    def _unwrap_parallel_response(response: Any) -> Any:
        if isinstance(response, TaskError):
            raise response.exception
        return response

    def _build_request_kwargs(
        self, prompt_input: VerificationPromptInput
    ) -> dict[str, Any]:
        kwargs = dict(self.model_request_kwargs)
        kwargs["messages"] = self.prompt_builder(prompt_input)
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = self.top_logprobs
        return kwargs

    def _build_prompt_input(
        self,
        *,
        task: Any,
        criterion: VerificationCriterion,
        candidates: Mapping[str, Any],
        context: Optional[Mapping[str, Any]],
        ground_truth_note: Optional[str],
        extra_instructions: Optional[str],
    ) -> VerificationPromptInput:
        return VerificationPromptInput(
            task=task,
            criterion=criterion,
            candidates=candidates,
            score_scale=self.score_scale,
            ground_truth_note=ground_truth_note or self.ground_truth_note,
            extra_instructions=extra_instructions or self.extra_instructions,
            context=context,
        )

    def _build_attempt(
        self,
        *,
        response: Any,
        repetition: int,
        criterion: VerificationCriterion,
        labels: Sequence[str],
        prompt_text: Optional[str],
    ) -> VerificationAttempt:
        metadata = dotdict(getattr(response, "metadata", None) or {})
        response_text = self._coerce_response_text(response.consume())
        tags = self._score_tags(labels)
        evidence = {
            label: self._extract_score_evidence(
                response_text=response_text,
                metadata=metadata,
                tag=tag,
            )
            for label, tag in zip(labels, tags)
        }
        return VerificationAttempt(
            criterion_id=criterion.id,
            repetition=repetition,
            prompt_text=prompt_text,
            response_text=response_text,
            scores={label: item.score for label, item in evidence.items()},
            evidence=evidence,
            metadata=metadata.to_dict()
            if isinstance(metadata, dotdict)
            else dict(metadata),
        )

    def _extract_score_evidence(
        self,
        *,
        response_text: str,
        metadata: Mapping[str, Any],
        tag: str,
    ) -> ScoreEvidence:
        raw_text_token, normalized_text_token = self._extract_text_token(
            response_text, tag
        )
        logprobs = metadata.get("logprobs")
        if logprobs is not None:
            evidence = self._extract_from_logprobs(
                logprobs,
                tag,
                expected_token=normalized_text_token,
            )
            if evidence is not None:
                return evidence
        if self.strict_logprobs:
            raise ValueError(
                f"Unable to extract logprobs for tag `{tag}` from verifier response"
            )
        return self._build_text_evidence(raw_text_token, normalized_text_token)

    def _extract_from_logprobs(
        self,
        logprobs: Mapping[str, Any],
        tag: str,
        expected_token: Optional[str] = None,
    ) -> Optional[ScoreEvidence]:
        entries = logprobs.get("content")
        if not isinstance(entries, list):
            return None
        target_entry = self._find_score_entry(entries, tag)
        if target_entry is None and expected_token is not None:
            target_entry = self._find_score_entry_by_token(
                entries,
                expected_token,
                rank=self._score_rank(tag),
            )
        if target_entry is None:
            return None
        token_probabilities = self._collect_token_probabilities(target_entry)
        if not token_probabilities:
            return None
        raw_token = target_entry.get("token")
        return ScoreEvidence(
            score=self.score_scale.to_normalized_score(token_probabilities),
            method="logprobs",
            raw_token=raw_token,
            token_probabilities=token_probabilities,
        )

    def _extract_text_token(
        self, response_text: str, tag: str
    ) -> tuple[Optional[str], Optional[str]]:
        tag_name = tag.strip("<>")
        pattern = rf"<{re.escape(tag_name)}>\s*(.+?)\s*</{re.escape(tag_name)}>"
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if not match:
            # Real provider outputs can occasionally mangle the opening tag while
            # still preserving the score token and the closing tag.
            malformed_pattern = rf"<\s*(.+?)\s*</{re.escape(tag_name)}>"
            match = re.search(
                malformed_pattern, response_text, re.IGNORECASE | re.DOTALL
            )
        raw_token = match.group(1).strip() if match else None
        normalized = self.score_scale.extract_token(raw_token or "")
        return raw_token, normalized

    def _build_text_evidence(
        self,
        raw_token: Optional[str],
        normalized: Optional[str],
    ) -> ScoreEvidence:
        if normalized is None:
            return ScoreEvidence(score=0.5, method="default", raw_token=raw_token)
        token_probabilities = {normalized: 1.0}
        return ScoreEvidence(
            score=self.score_scale.to_normalized_score(token_probabilities),
            method="text",
            raw_token=raw_token,
            token_probabilities=token_probabilities,
        )

    def _find_score_entry(
        self, entries: Sequence[Mapping[str, Any]], tag: str
    ) -> Optional[Mapping[str, Any]]:
        text_so_far = ""
        tag_found = False

        for entry in entries:
            token = str(entry.get("token", ""))
            next_text = text_so_far + token

            if not tag_found and tag not in next_text:
                text_so_far = next_text
                continue

            tag_found = True
            if self._collect_token_probabilities(entry):
                return entry
            text_so_far = next_text

        return None

    def _find_score_entry_by_token(
        self,
        entries: Sequence[Mapping[str, Any]],
        expected_token: str,
        *,
        rank: int,
    ) -> Optional[Mapping[str, Any]]:
        closing_matches = []
        fallback_matches = []

        for index, entry in enumerate(entries):
            probabilities = self._collect_token_probabilities(entry)
            if expected_token not in probabilities:
                continue

            fallback_matches.append(entry)
            next_token = ""
            if index + 1 < len(entries):
                next_token = str(entries[index + 1].get("token", ""))
            if "</" in next_token:
                closing_matches.append(entry)

        # Prefer the score token that is immediately followed by a closing tag,
        # but keep a fallback for providers that tokenize the tag boundary oddly.
        ranked_matches = closing_matches or fallback_matches
        if not ranked_matches:
            return None
        if rank <= len(ranked_matches):
            return ranked_matches[rank - 1]
        return ranked_matches[-1]

    def _collect_token_probabilities(
        self, entry: Mapping[str, Any]
    ) -> dict[str, float]:
        token_probabilities = {}
        for candidate in self._iter_score_candidates(entry):
            normalized = self.score_scale.extract_token(str(candidate.get("token", "")))
            if normalized is None:
                continue
            probability = math.exp(candidate.get("logprob", float("-inf")))
            token_probabilities[normalized] = max(
                token_probabilities.get(normalized, 0.0),
                probability,
            )
        return token_probabilities

    @staticmethod
    def _iter_score_candidates(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        top_logprobs = entry.get("top_logprobs")
        candidates = list(top_logprobs) if isinstance(top_logprobs, list) else []
        chosen_token = entry.get("token")
        chosen_logprob = entry.get("logprob")
        if chosen_token is not None and chosen_logprob is not None:
            candidates.append({"token": chosen_token, "logprob": chosen_logprob})
        return candidates

    def _aggregate_criterion(
        self,
        criterion: VerificationCriterion,
        attempts: Sequence[VerificationAttempt],
    ) -> CriterionVerification:
        labels = attempts[0].scores.keys()
        scores = {
            label: fmean(attempt.scores[label] for attempt in attempts)
            for label in labels
        }
        return CriterionVerification(
            criterion=criterion,
            scores=scores,
            attempts=list(attempts),
        )

    def _build_tournament_metadata(
        self,
        *,
        matches: Sequence[TournamentMatch],
        criteria: Sequence[VerificationCriterion],
        n_candidates: int,
    ) -> dict[str, Any]:
        metadata = {
            "n_candidates": n_candidates,
            "n_matches": len(matches),
            "criteria_ids": [criterion.id for criterion in criteria],
            "n_verifications": self.n_verifications,
            "verbose": self.verbose,
            **self._model_metadata(),
        }
        if self.verbose:
            metadata["raw_outputs"] = [
                {
                    "candidate_a_label": match.candidate_a_label,
                    "candidate_b_label": match.candidate_b_label,
                    "outputs": match.result.metadata.get("raw_outputs", []),
                }
                for match in matches
            ]
        return metadata

    def _build_tournament_result(
        self,
        *,
        labels: Sequence[str],
        matches: Sequence[TournamentMatch],
        criteria: Sequence[VerificationCriterion],
    ) -> TournamentResult:
        wins = dict.fromkeys(labels, 0.0)
        score_history = {label: [] for label in labels}

        for match in matches:
            label_a = match.candidate_a_label
            label_b = match.candidate_b_label
            score_a = match.result.scores[label_a]
            score_b = match.result.scores[label_b]
            score_history[label_a].append(score_a)
            score_history[label_b].append(score_b)

            if score_a > score_b:
                wins[label_a] += 1.0
            elif score_b > score_a:
                wins[label_b] += 1.0
            else:
                wins[label_a] += 0.5
                wins[label_b] += 0.5

        average_scores = {
            label: (fmean(history) if history else 0.5)
            for label, history in score_history.items()
        }
        winner = max(
            labels,
            key=lambda label: (wins[label], average_scores[label], label),
        )
        return TournamentResult(
            winner=winner,
            wins=wins,
            average_scores=average_scores,
            matches=list(matches),
            metadata=self._build_tournament_metadata(
                matches=matches,
                criteria=criteria,
                n_candidates=len(labels),
            ),
        )

    def _build_result(
        self,
        criterion_results: Sequence[CriterionVerification],
        labels: Sequence[str],
    ) -> VerifierResult:
        total_weight = sum(result.criterion.weight for result in criterion_results)
        overall_scores = {
            label: sum(
                result.scores[label] * result.criterion.weight
                for result in criterion_results
            )
            / total_weight
            for label in labels
        }

        winner = None
        if len(labels) == 2:
            score_a = overall_scores[labels[0]]
            score_b = overall_scores[labels[1]]
            if score_a > score_b:
                winner = labels[0]
                verdict = winner
            elif score_b > score_a:
                winner = labels[1]
                verdict = winner
            else:
                verdict = "tie"
        else:
            score = overall_scores[labels[0]]
            if score > self.pass_threshold:
                verdict = "pass"
            elif score < self.pass_threshold:
                verdict = "fail"
            else:
                verdict = "uncertain"

        metadata = {
            "criteria_ids": [result.criterion.id for result in criterion_results],
            "criteria_weights": {
                result.criterion.id: result.criterion.weight
                for result in criterion_results
            },
            "mode": "single" if len(labels) == 1 else "pairwise",
            "n_criteria": len(criterion_results),
            "n_verifications": self.n_verifications,
            "score_format": self.score_scale.score_format,
            "top_logprobs": self.top_logprobs,
            "verbose": self.verbose,
            **self._model_metadata(),
        }
        if self.verbose:
            metadata["raw_outputs"] = self._build_verbose_outputs(criterion_results)

        return VerifierResult(
            scores=overall_scores,
            criteria_results=list(criterion_results),
            verdict=verdict,
            winner=winner,
            metadata=metadata,
        )

    def _build_verbose_outputs(
        self,
        criterion_results: Sequence[CriterionVerification],
    ) -> list[dict[str, Any]]:
        raw_outputs = []
        # Keep the prompt/response pairs per attempt so verifier calibration and
        # prompt debugging can happen without re-running the model call.
        for result in criterion_results:
            for attempt in result.attempts:
                raw_outputs.append(
                    {
                        "criterion_id": result.criterion.id,
                        "criterion_name": result.criterion.name,
                        "repetition": attempt.repetition,
                        "prompt": attempt.prompt_text,
                        "response_text": attempt.response_text,
                        "scores": dict(attempt.scores),
                        "evidence": {
                            label: {
                                "method": evidence.method,
                                "raw_token": evidence.raw_token,
                                "token_probabilities": dict(
                                    evidence.token_probabilities
                                ),
                            }
                            for label, evidence in attempt.evidence.items()
                        },
                    }
                )
        return raw_outputs

    @classmethod
    def _from_preset(
        cls,
        *,
        model: ModelLike,
        criteria: Sequence[VerificationCriterion],
        default_ground_truth_note: Optional[str] = None,
        default_extra_instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> "LLMAsVerifier":
        if default_ground_truth_note is not None:
            kwargs.setdefault("ground_truth_note", default_ground_truth_note)
        if default_extra_instructions is not None:
            kwargs.setdefault("extra_instructions", default_extra_instructions)
        return cls(model=model, criteria=criteria, **kwargs)

    @staticmethod
    def _resolve_model(model: ModelLike) -> Union[ChatCompletionModel, ModelGateway]:
        if isinstance(model, str):
            model = Model.chat_completion(model)
        model_type = getattr(model, "model_type", None)
        if model_type != "chat_completion":
            raise TypeError(
                "`model` must be a `chat_completion` model, a `ModelGateway`, "
                f"or a provider/model-id string. Given `{type(model)}`"
            )
        return model

    @staticmethod
    def _coerce_response_text(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=True, sort_keys=True)
        return str(payload)

    @staticmethod
    def _score_tags(labels: Sequence[str]) -> tuple[str, ...]:
        if len(labels) == 1:
            return ("<score>",)
        if len(labels) == 2:
            return ("<score_A>", "<score_B>")
        raise ValueError("Score tags support only one or two candidates")

    @staticmethod
    def _score_rank(tag: str) -> int:
        if tag == "<score_B>":
            return 2
        return 1

    @staticmethod
    def _normalize_criteria(
        criteria: Sequence[VerificationCriterion],
    ) -> list[VerificationCriterion]:
        normalized = list(criteria)
        if not normalized:
            raise ValueError("`criteria` must contain at least one criterion")
        return normalized

    def _get_active_criteria(
        self, criteria: Optional[Sequence[VerificationCriterion]]
    ) -> list[VerificationCriterion]:
        if criteria is None:
            return list(self.criteria)
        return self._normalize_criteria(criteria)

    def _normalize_candidates(
        self,
        candidates: Union[Sequence[Any], Mapping[str, Any]],
        *,
        min_count: int,
        max_count: Optional[int] = None,
    ) -> dict[str, Any]:
        if isinstance(candidates, Mapping):
            normalized = dict(candidates)
        else:
            values = list(candidates)
            normalized = {
                f"candidate_{index + 1}": value for index, value in enumerate(values)
            }

        if len(normalized) < min_count:
            raise ValueError(f"`candidates` must contain at least {min_count} item(s)")
        if max_count is not None and len(normalized) > max_count:
            raise ValueError(
                f"`candidates` supports at most {max_count} item(s) in this method"
            )
        return normalized

    def _validate_model_request_kwargs(self) -> None:
        forbidden_keys = {
            "generation_schema",
            "logprobs",
            "messages",
            "stream",
            "tool_definitions",
            "top_logprobs",
            "typed_parser",
        }
        overlapping = forbidden_keys.intersection(self.model_request_kwargs)
        if overlapping:
            blocked = ", ".join(sorted(overlapping))
            raise ValueError(
                "`model_request_kwargs` cannot override verifier-managed keys: "
                f"{blocked}"
            )

    def _model_metadata(self) -> dict[str, Any]:
        if isinstance(self.model, ModelGateway):
            return {
                "model_provider": "gateway",
                "model_id": None,
            }
        return {
            "model_provider": getattr(self.model, "provider", None),
            "model_id": getattr(self.model, "model_id", None),
        }


def _render_prompt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=True, indent=2)
    return str(value)


def _format_context(context: Mapping[str, Any]) -> str:
    rendered = []
    for key, value in context.items():
        rendered.append(f"{key}:\n{_render_prompt_value(value)}")
    return "\n\n".join(rendered)
