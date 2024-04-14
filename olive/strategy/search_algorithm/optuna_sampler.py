# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple

import optuna

from olive.common.config_utils import ConfigParam
from olive.common.utils import hash_dict
from olive.strategy.search_algorithm.search_algorithm import SearchAlgorithm
from olive.strategy.search_parameter import Categorical, Conditional, SpecialParamValue

if TYPE_CHECKING:
    from olive.evaluator.metric import MetricResult


optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaSearchAlgorithm(SearchAlgorithm):
    """Optuna sampler for search algorithms."""

    name = "optuna_sampler"

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return {
            "num_samples": ConfigParam(type_=int, default_value=1, description="Number of samples to suggest."),
            "seed": ConfigParam(type_=int, default_value=1, description="Seed for the rng."),
        }

    def initialize(self):
        """Initialize the searcher."""
        self._sampler = self._create_sampler()
        directions = ["maximize" if higher_is_better else "minimize" for higher_is_better in self._higher_is_betters]
        self._study = optuna.create_study(directions=directions, sampler=self._sampler)
        self._trial_ids = {}
        self._num_samples_suggested = 0

    def should_stop(self):
        return (
            (self._search_space.empty() and self._num_samples_suggested > 0)
            or (self._num_samples_suggested >= self._config.num_samples)
            or super().should_stop()
        )

    @abstractmethod
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create the sampler."""

    def suggest(self) -> Dict[str, Dict[str, Any]]:
        """Suggest a new configuration to try."""
        if self.should_stop():
            return None

        trial, search_point, invalid = self._get_trial()
        if invalid:
            self._study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
            return self.suggest()

        # save history
        search_point_hash = hash_dict(search_point)
        self._trial_ids[search_point_hash] = trial.number

        self._num_samples_suggested += 1

        return search_point

    def _get_trial(self) -> Tuple[optuna.trial.Trial, Dict[str, Dict[str, Any]]]:
        """Get a trial from the study."""
        trial = self._study.ask()
        search_point = self._search_space.empty_search_point()
        invalid_trial = False
        for space_name, param_name, param in self._search_space.iter_params():
            if space_name not in search_point:
                search_point[space_name] = {}
            suggestion_name = f"{space_name}___{param_name}"
            if isinstance(param, Categorical):
                search_point[space_name][param_name] = trial.suggest_categorical(suggestion_name, param.get_support())
            elif isinstance(param, Conditional):
                parent_vals = {parent: search_point[space_name][parent] for parent in param.parents}
                options = param.get_support(parent_vals)
                parent_vals_name = "_".join([f"{v}" for _, v in parent_vals.items()])
                suggestion_name = f"{space_name}___{param_name}___{parent_vals_name}"
                search_point[space_name][param_name] = trial.suggest_categorical(suggestion_name, options)
            else:
                raise ValueError(f"Unsupported parameter type: {type(param)}")
            invalid_trial = invalid_trial or (search_point[space_name][param_name] == SpecialParamValue.INVALID)
        return trial, search_point, invalid_trial

    def report(self, search_point: Dict[str, Dict[str, Any]], result: "MetricResult", should_prune: bool = False):
        search_point_hash = hash_dict(search_point)
        trial_id = self._trial_ids[search_point_hash]
        if should_prune:
            self._study.tell(trial_id, state=optuna.trial.TrialState.PRUNED)
        else:
            objectives = [result[objective].value for objective in self._objectives]
            self._study.tell(trial_id, objectives)
