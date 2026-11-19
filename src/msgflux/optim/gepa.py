"""GEPA search backend for msgFlux text optimizers."""

import sys
from typing import Literal

from msgflux.models.response import ModelResponse
from msgflux.optim.optimizer import Optimizer, ProgramResult, TextResult


class GEPA(Optimizer):
    """Search for instructions with GEPA's reflective mutation algorithm."""

    def __init__(
        self,
        reflection_model,
        max_metric_calls: int = 100,
        seed: int = 0,
        *,
        workers: int = 1,
        run_dir=None,
        module_selector: Literal["round_robin", "all"] = "round_robin",
    ):
        if not callable(reflection_model):
            raise TypeError("reflection_model must be a msgFlux chat model or callable")
        if type(max_metric_calls) is not int or max_metric_calls < 1:
            raise ValueError("max_metric_calls must be a positive integer")
        self.reflection_model = reflection_model
        self.max_metric_calls = max_metric_calls
        self.seed = seed
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = workers
        self.run_dir = run_dir
        if module_selector not in ("round_robin", "all"):
            raise ValueError("module_selector must be round_robin or all")
        self.module_selector = module_selector

    def _reflect(self, prompt):
        if getattr(self.reflection_model, "model_type", None) == "chat_completion":
            messages = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt
            )
            response = self.reflection_model(messages=messages)
        else:
            response = self.reflection_model(prompt)
        if isinstance(response, ModelResponse):
            response = response.consume()
        if not isinstance(response, str):
            raise TypeError(
                "The reflection model must return text or a text ModelResponse"
            )
        return response

    def _search(self, seed, trainset, valset, evaluator) -> TextResult:
        return TextResult(*self._run(seed, trainset, valset, evaluator))

    def _search_program(self, seeds, trainset, valset, evaluator) -> ProgramResult:
        return ProgramResult(*self._run(seeds, trainset, valset, evaluator))

    def _run(self, seed, trainset, valset, evaluator):
        try:
            from gepa.optimize_anything import (  # noqa: PLC0415
                EngineConfig,
                GEPAConfig,
                ReflectionConfig,
                optimize_anything,
            )
        except ModuleNotFoundError as error:
            if error.name != "gepa":
                raise
            if sys.version_info >= (3, 15):
                raise ImportError(
                    "GEPA requires Python 3.11-3.14 with msgFlux"
                ) from error
            raise ImportError("Install msgflux[optimizers] to use GEPA") from error

        if self.max_metric_calls < len(valset):
            raise ValueError("max_metric_calls must cover the initial validation set")
        available = self.max_metric_calls - 2 * len(valset)
        batch_size = max(1, min(3, len(trainset), available // 2))
        iteration_cost = 2 * batch_size + len(valset)

        def stop_before_incomplete_iteration(state):
            # GEPA restores this counter when resuming from its checkpoint.
            return state.total_num_evals + iteration_cost > self.max_metric_calls

        result = optimize_anything(
            seed_candidate=seed,
            dataset=trainset,
            valset=valset,
            evaluator=evaluator,
            config=GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=self.max_metric_calls,
                    seed=self.seed,
                    parallel=self.workers > 1,
                    max_workers=self.workers,
                    run_dir=str(self.run_dir) if self.run_dir is not None else None,
                    use_cloudpickle=False,
                ),
                reflection=ReflectionConfig(
                    reflection_lm=self._reflect,
                    reflection_minibatch_size=batch_size,
                    module_selector=self.module_selector,
                ),
                stop_callbacks=[stop_before_incomplete_iteration],
            ),
        )
        return (
            result.best_candidate,
            result.val_aggregate_scores[0],
            result.val_aggregate_scores[result.best_idx],
            result.total_metric_calls,
        )
