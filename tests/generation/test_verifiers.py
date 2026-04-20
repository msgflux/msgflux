import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from msgflux.core.dotdict import dotdict
from msgflux.generation.verifiers import (
    ANSWER_RERANKING_CRITERIA,
    GROUNDED_ANSWER_VERIFICATION_CRITERIA,
    LLMAsVerifier,
    PATCH_SELECTION_CRITERIA,
    ScoreScale,
    SWE_BENCH_VERIFIED_CRITERIA,
    SYNTHETIC_DATA_FILTERING_CRITERIA,
    TERMINAL_BENCH_CRITERIA,
    TOOL_TRACE_VERIFICATION_CRITERIA,
    TRAJECTORY_ANALYSIS_CRITERIA,
    VerificationCriterion,
    VerificationPromptInput,
    default_prompt_builder,
)
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse
from msgflux.models.types import ChatCompletionModel


CRITERION = VerificationCriterion(
    id="correctness",
    name="Correctness",
    description="Assess whether the candidate is correct.",
)

SECOND_CRITERION = VerificationCriterion(
    id="completeness",
    name="Completeness",
    description="Assess whether the candidate fully answers the task.",
)


def _make_response(
    text,
    *,
    token,
    alternatives=None,
    usage=None,
):
    response = ModelResponse()
    response.add(text)
    metadata = {}
    if token is not None:
        metadata["logprobs"] = {
            "content": [
                {"token": "Analysis", "logprob": -0.2, "top_logprobs": []},
                {"token": "<score>", "logprob": -0.1, "top_logprobs": []},
                {
                    "token": token,
                    "logprob": -0.05,
                    "top_logprobs": alternatives or [],
                },
            ]
        }
    if usage is not None:
        metadata["usage"] = usage
    response.set_metadata(dotdict(metadata))
    response.set_response_type("text_generation")
    return response


def _make_pairwise_response(
    text,
    *,
    token_a,
    token_b,
):
    response = ModelResponse()
    response.add(text)
    response.set_metadata(
        dotdict(
            {
                "logprobs": {
                    "content": [
                        {"token": "Analysis", "logprob": -0.2, "top_logprobs": []},
                        {"token": "<score_A>", "logprob": -0.1, "top_logprobs": []},
                        {
                            "token": token_a,
                            "logprob": -0.05,
                            "top_logprobs": [{"token": token_a, "logprob": -0.05}],
                        },
                        {"token": "<score_B>", "logprob": -0.1, "top_logprobs": []},
                        {
                            "token": token_b,
                            "logprob": -0.05,
                            "top_logprobs": [{"token": token_b, "logprob": -0.05}],
                        },
                    ]
                }
            }
        )
    )
    response.set_response_type("text_generation")
    return response


def _make_weird_tagged_response():
    response = ModelResponse()
    response.add("Looks correct\n<score>A</score>")
    response.set_metadata(
        dotdict(
            {
                "logprobs": {
                    "content": [
                        {"token": "<", "logprob": -0.01, "top_logprobs": []},
                        {
                            "token": "20",
                            "logprob": -0.57,
                            "top_logprobs": [{"token": "20", "logprob": -0.57}],
                        },
                        {
                            "token": "A",
                            "logprob": -0.0001,
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.0001},
                                {"token": "B", "logprob": -2.0},
                            ],
                        },
                        {"token": "</", "logprob": -0.01, "top_logprobs": []},
                        {
                            "token": "20",
                            "logprob": -0.57,
                            "top_logprobs": [{"token": "20", "logprob": -0.57}],
                        },
                        {
                            "token": "A",
                            "logprob": -0.0001,
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.0001},
                                {"token": "B", "logprob": -2.0},
                            ],
                        },
                        {"token": ">", "logprob": 0.0, "top_logprobs": []},
                    ]
                }
            }
        )
    )
    response.set_response_type("text_generation")
    return response


def _make_malformed_score_response():
    response = ModelResponse()
    response.add("Looks correct\n<20A</score>")
    response.set_metadata(
        dotdict(
            {
                "logprobs": {
                    "content": [
                        {"token": "Analysis", "logprob": -0.2, "top_logprobs": []},
                        {"token": "<", "logprob": -0.1, "top_logprobs": []},
                        {"token": "20", "logprob": -0.5, "top_logprobs": []},
                        {
                            "token": "A",
                            "logprob": -0.01,
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.01},
                                {"token": "B", "logprob": -2.0},
                            ],
                        },
                        {"token": "</score>", "logprob": -0.1, "top_logprobs": []},
                    ]
                }
            }
        )
    )
    response.set_response_type("text_generation")
    return response


class MockChatModel(BaseModel, ChatCompletionModel):
    def __init__(self, responses, model_id="mock-model", provider="mock"):
        self.model_id = model_id
        self.provider = provider
        self.responses = list(responses)
        self.calls = []

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("No more responses available")
        return self.responses.pop(0)

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("No more responses available")
        return self.responses.pop(0)

    def serialize(self):
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "model_type": self.model_type,
        }


class DynamicPairwiseModel(BaseModel, ChatCompletionModel):
    def __init__(self):
        self.model_id = "dynamic-model"
        self.provider = "mock"
        self.calls = []

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"]
        token_a = "T"
        token_b = "T"
        if "Candidate A (best):\nbest" in prompt:
            token_a = "A"
        elif "Candidate B (best):\nbest" in prompt:
            token_b = "A"
        elif "Candidate A (okay):\nokay" in prompt:
            token_a = "H"
        elif "Candidate B (okay):\nokay" in prompt:
            token_b = "H"
        return _make_pairwise_response(
            f"Analysis\n<score_A>{token_a}</score_A>\n<score_B>{token_b}</score_B>",
            token_a=token_a,
            token_b=token_b,
        )

    async def acall(self, **kwargs):
        return self.__call__(**kwargs)

    def serialize(self):
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "model_type": self.model_type,
        }


class ConcurrentTrackingModel(BaseModel, ChatCompletionModel):
    def __init__(self, responses, delay=0.05):
        self.model_id = "concurrent-model"
        self.provider = "mock"
        self.responses = list(responses)
        self.delay = delay
        self.calls = []
        self.active_calls = 0
        self.max_active_calls = 0
        self._lock = threading.Lock()

    def _initialize(self):
        pass

    def _enter_call(self, kwargs):
        self.calls.append(kwargs)
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)

    def _exit_call(self):
        with self._lock:
            self.active_calls -= 1

    def _pop_response(self):
        if not self.responses:
            raise RuntimeError("No more responses available")
        return self.responses.pop(0)

    def __call__(self, **kwargs):
        self._enter_call(kwargs)
        try:
            time.sleep(self.delay)
            return self._pop_response()
        finally:
            self._exit_call()

    async def acall(self, **kwargs):
        self._enter_call(kwargs)
        try:
            await asyncio.sleep(self.delay)
            return self._pop_response()
        finally:
            self._exit_call()

    def serialize(self):
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "model_type": self.model_type,
        }


class TestLLMAsVerifier:
    def test_default_prompt_builder_uses_single_score_token_examples(self):
        prompt = default_prompt_builder(
            VerificationPromptInput(
                task="What is 2 + 2?",
                criterion=CRITERION,
                candidates={"answer": "The answer is 4."},
                score_scale=ScoreScale.letter(),
            )
        )

        assert "<score>A</score>" in prompt
        assert "Do not output the scale name" in prompt
        assert "Use a single token from A to T." in prompt

    def test_trajectory_analysis_preset_sets_default_note(self):
        verifier = LLMAsVerifier.trajectory_analysis(
            model=MockChatModel([]),
        )

        assert verifier.ground_truth_note is not None
        assert [criterion.id for criterion in verifier.criteria] == [
            criterion.id for criterion in TRAJECTORY_ANALYSIS_CRITERIA
        ]

    def test_preset_respects_explicit_overrides(self):
        verifier = LLMAsVerifier.patch_selection(
            model=MockChatModel([]),
            ground_truth_note="Use only the patch diff as evidence.",
        )

        assert verifier.ground_truth_note == "Use only the patch diff as evidence."

    @pytest.mark.parametrize(
        ("factory_name", "expected_criteria"),
        [
            ("answer_reranking", ANSWER_RERANKING_CRITERIA),
            (
                "grounded_answer_verification",
                GROUNDED_ANSWER_VERIFICATION_CRITERIA,
            ),
            ("patch_selection", PATCH_SELECTION_CRITERIA),
            ("terminal_bench", TERMINAL_BENCH_CRITERIA),
            ("swe_bench_verified", SWE_BENCH_VERIFIED_CRITERIA),
            ("tool_trace_verification", TOOL_TRACE_VERIFICATION_CRITERIA),
            ("synthetic_data_filtering", SYNTHETIC_DATA_FILTERING_CRITERIA),
        ],
    )
    def test_presets_expose_expected_criteria(self, factory_name, expected_criteria):
        verifier = getattr(LLMAsVerifier, factory_name)(model=MockChatModel([]))

        assert [criterion.id for criterion in verifier.criteria] == [
            criterion.id for criterion in expected_criteria
        ]

    def test_single_candidate_uses_logprobs(self):
        model = MockChatModel(
            [
                _make_response(
                    "Looks correct\n<score>A</score>",
                    token="A",
                    alternatives=[
                        {"token": "A", "logprob": -0.05},
                        {"token": "B", "logprob": -2.0},
                    ],
                    usage={"total_tokens": 12},
                )
            ]
        )
        verifier = LLMAsVerifier(model=model, criteria=[CRITERION])

        result = verifier(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        assert result.verdict == "pass"
        assert result.score > 0.9
        assert result.metadata["mode"] == "single"
        attempt = result.criteria_results[0].attempts[0]
        assert attempt.evidence["answer"].method == "logprobs"
        assert attempt.metadata["usage"]["total_tokens"] == 12
        assert model.calls[0]["logprobs"] is True
        assert model.calls[0]["top_logprobs"] == 20

    def test_call_executes_attempts_concurrently(self):
        model = ConcurrentTrackingModel(
            [
                _make_response(
                    "Looks correct\n<score>A</score>",
                    token="A",
                    alternatives=[{"token": "A", "logprob": -0.01}],
                )
                for _ in range(4)
            ]
        )
        verifier = LLMAsVerifier(
            model=model,
            criteria=[CRITERION, SECOND_CRITERION],
            n_verifications=2,
        )

        result = verifier(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        assert result.verdict == "pass"
        assert model.max_active_calls > 1
        assert len(model.calls) == 4

    def test_verbose_mode_exposes_prompt_and_raw_outputs(self):
        model = MockChatModel(
            [
                _make_response(
                    "Looks correct\n<score>A</score>",
                    token="A",
                    alternatives=[{"token": "A", "logprob": -0.01}],
                )
            ]
        )
        verifier = LLMAsVerifier(
            model=model,
            criteria=[CRITERION],
            verbose=True,
        )

        result = verifier(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        attempt = result.criteria_results[0].attempts[0]
        assert attempt.prompt_text == model.calls[0]["messages"]
        assert result.metadata["verbose"] is True
        assert result.metadata["raw_outputs"][0]["prompt"] == model.calls[0]["messages"]
        assert (
            result.metadata["raw_outputs"][0]["response_text"]
            == "Looks correct\n<score>A</score>"
        )

    def test_single_candidate_falls_back_to_text_without_logprobs(self):
        model = MockChatModel(
            [
                _make_response(
                    "Looks correct\n<score>B</score>",
                    token=None,
                )
            ]
        )
        verifier = LLMAsVerifier(model=model, criteria=[CRITERION])

        result = verifier(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        assert result.verdict == "pass"
        assert result.criteria_results[0].attempts[0].evidence["answer"].method == (
            "text"
        )

    def test_single_candidate_strict_logprobs_requires_metadata(self):
        model = MockChatModel(
            [
                _make_response(
                    "Looks correct\n<score>A</score>",
                    token=None,
                )
            ]
        )
        verifier = LLMAsVerifier(
            model=model,
            criteria=[CRITERION],
            strict_logprobs=True,
        )

        with pytest.raises(ValueError, match="Unable to extract logprobs"):
            verifier(
                task="What is 2 + 2?",
                candidates={"answer": "The answer is 4."},
            )

    def test_pairwise_comparison_sets_winner(self):
        model = MockChatModel(
            [
                _make_pairwise_response(
                    "Analysis\n<score_A>A</score_A>\n<score_B>T</score_B>",
                    token_a="A",
                    token_b="T",
                )
            ]
        )
        verifier = LLMAsVerifier(model=model, criteria=[CRITERION])

        result = verifier(
            task="Which answer is correct?",
            candidates={"correct": "4", "wrong": "5"},
        )

        assert result.verdict == "correct"
        assert result.winner == "correct"
        assert result.scores["correct"] > result.scores["wrong"]

    def test_single_candidate_matches_realistic_tag_tokenization(self):
        model = MockChatModel([_make_weird_tagged_response()])
        verifier = LLMAsVerifier(
            model=model,
            criteria=[CRITERION],
            strict_logprobs=True,
        )

        result = verifier(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        assert result.verdict == "pass"
        assert result.criteria_results[0].attempts[0].evidence["answer"].method == (
            "logprobs"
        )

    def test_single_candidate_handles_malformed_score_tag_with_logprobs(self):
        model = MockChatModel([_make_malformed_score_response()])
        verifier = LLMAsVerifier(
            model=model,
            criteria=[CRITERION],
            strict_logprobs=True,
        )

        result = verifier(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        assert result.verdict == "pass"
        assert result.criteria_results[0].attempts[0].evidence["answer"].method == (
            "logprobs"
        )

    def test_accepts_model_gateway(self):
        gateway = ModelGateway(
            models=[
                {
                    "model_name": "primary",
                    "model": MockChatModel(
                        [
                            _make_response(
                                "Looks correct\n<score>A</score>",
                                token="A",
                                alternatives=[{"token": "A", "logprob": -0.01}],
                            )
                        ]
                    ),
                }
            ]
        )
        verifier = LLMAsVerifier(model=gateway, criteria=[CRITERION])

        result = verifier(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        assert result.verdict == "pass"
        assert result.metadata["model_provider"] == "gateway"

    def test_resolves_string_model(self):
        model = MockChatModel(
            [
                _make_response(
                    "Looks correct\n<score>A</score>",
                    token="A",
                    alternatives=[{"token": "A", "logprob": -0.01}],
                )
            ]
        )

        with patch(
            "msgflux.generation.verifiers.llm_as_a_verifier.Model.chat_completion"
        ) as mock_chat_completion:
            mock_chat_completion.return_value = model
            verifier = LLMAsVerifier(
                model="openai/gpt-4.1-mini",
                criteria=[CRITERION],
            )

        result = verifier(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        assert result.verdict == "pass"
        mock_chat_completion.assert_called_once_with("openai/gpt-4.1-mini")

    @pytest.mark.asyncio
    async def test_acall_uses_async_model_path(self):
        model = MockChatModel(
            [
                _make_response(
                    "Looks correct\n<score>A</score>",
                    token="A",
                    alternatives=[{"token": "A", "logprob": -0.01}],
                )
            ]
        )
        verifier = LLMAsVerifier(model=model, criteria=[CRITERION])

        result = await verifier.acall(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        assert result.verdict == "pass"
        assert model.calls[0]["logprobs"] is True

    @pytest.mark.asyncio
    async def test_acall_executes_attempts_concurrently(self):
        model = ConcurrentTrackingModel(
            [
                _make_response(
                    "Looks correct\n<score>A</score>",
                    token="A",
                    alternatives=[{"token": "A", "logprob": -0.01}],
                )
                for _ in range(4)
            ]
        )
        verifier = LLMAsVerifier(
            model=model,
            criteria=[CRITERION, SECOND_CRITERION],
            n_verifications=2,
        )

        result = await verifier.acall(
            task="What is 2 + 2?",
            candidates={"answer": "The answer is 4."},
        )

        assert result.verdict == "pass"
        assert model.max_active_calls > 1
        assert len(model.calls) == 4

    def test_select_best_runs_round_robin(self):
        verifier = LLMAsVerifier(
            model=DynamicPairwiseModel(),
            criteria=[CRITERION],
        )

        result = verifier.select_best(
            task="Pick the strongest final answer.",
            candidates={
                "best": "best",
                "okay": "okay",
                "bad": "bad",
            },
        )

        assert result.winner == "best"
        assert result.ranking[0] == "best"
        assert len(result.matches) == 3
        assert result.wins["best"] == 2.0

    def test_select_best_executes_matches_concurrently(self):
        model = ConcurrentTrackingModel(
            [
                _make_pairwise_response(
                    "Analysis\n<score_A>A</score_A>\n<score_B>T</score_B>",
                    token_a="A",
                    token_b="T",
                )
                for _ in range(3)
            ]
        )
        verifier = LLMAsVerifier(
            model=model,
            criteria=[CRITERION],
        )

        result = verifier.select_best(
            task="Pick the strongest final answer.",
            candidates={
                "best": "best",
                "okay": "okay",
                "bad": "bad",
            },
        )

        assert result.winner == "best"
        assert model.max_active_calls > 1
        assert len(model.calls) == 3

    def test_select_best_verbose_mode_aggregates_match_outputs(self):
        verifier = LLMAsVerifier(
            model=DynamicPairwiseModel(),
            criteria=[CRITERION],
            verbose=True,
        )

        result = verifier.select_best(
            task="Pick the strongest final answer.",
            candidates={
                "best": "best",
                "okay": "okay",
                "bad": "bad",
            },
        )

        assert result.metadata["verbose"] is True
        assert len(result.metadata["raw_outputs"]) == 3
        assert result.metadata["raw_outputs"][0]["outputs"]

    @pytest.mark.asyncio
    async def test_aselect_best_executes_matches_concurrently(self):
        model = ConcurrentTrackingModel(
            [
                _make_pairwise_response(
                    "Analysis\n<score_A>A</score_A>\n<score_B>T</score_B>",
                    token_a="A",
                    token_b="T",
                )
                for _ in range(3)
            ]
        )
        verifier = LLMAsVerifier(
            model=model,
            criteria=[CRITERION],
        )

        result = await verifier.aselect_best(
            task="Pick the strongest final answer.",
            candidates={
                "best": "best",
                "okay": "okay",
                "bad": "bad",
            },
        )

        assert result.winner == "best"
        assert model.max_active_calls > 1
        assert len(model.calls) == 3

    def test_call_rejects_more_than_two_candidates(self):
        verifier = LLMAsVerifier(
            model=DynamicPairwiseModel(),
            criteria=[CRITERION],
        )

        with pytest.raises(ValueError, match="at most 2 item"):
            verifier(
                task="Pick the strongest final answer.",
                candidates={
                    "best": "best",
                    "okay": "okay",
                    "bad": "bad",
                },
            )

    def test_model_request_kwargs_cannot_override_verifier_keys(self):
        model = MockChatModel([])

        with pytest.raises(ValueError, match="verifier-managed keys"):
            LLMAsVerifier(
                model=model,
                criteria=[CRITERION],
                model_request_kwargs={"stream": True},
            )
