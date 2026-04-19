from unittest.mock import patch

import pytest

from msgflux.core.dotdict import dotdict
from msgflux.generation.verifiers import LLMAsVerifier, VerificationCriterion
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse
from msgflux.models.types import ChatCompletionModel


CRITERION = VerificationCriterion(
    id="correctness",
    name="Correctness",
    description="Assess whether the candidate is correct.",
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


class TestLLMAsVerifier:
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
