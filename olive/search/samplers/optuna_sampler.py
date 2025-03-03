# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from abc import abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import optuna

from olive.common.config_utils import ConfigBase, ConfigParam
from olive.search.samplers.search_sampler import SearchSampler
from olive.search.search_parameter import Categorical, Conditional, SearchParameter
from olive.search.search_point import SearchPoint
from olive.search.search_space import SearchSpace

if TYPE_CHECKING:
    from optuna.trial import Trial

    from olive.evaluator.metric_result import MetricResult


optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaSampler(SearchSampler):
    """Optuna sampler for search sampling."""

    name = "optuna"

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return {
            **super()._default_config(),
            "seed": ConfigParam(type_=int, default_value=1, description="Seed for the rng."),
        }

    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Union[Dict[str, Any], ConfigBase]] = None,
        objectives: Dict[str, Dict[str, Any]] = None,
    ):
        super().__init__(search_space, config, objectives)

        # Initialize the searcher
        self._sampler = self._create_sampler()
        directions = ["maximize" if self._higher_is_betters[name] else "minimize" for name in self._objectives]
        self._study = optuna.create_study(directions=directions, sampler=self._sampler)

        self._num_samples_suggested = 0
        self._search_point_index_to_trail_id = {}

    @property
    def num_samples_suggested(self) -> int:
        """Returns the number of samples suggested so far."""
        return self._num_samples_suggested

    @abstractmethod
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create the sampler."""

    def suggest(self) -> SearchPoint:
        """Suggest a new configuration to try."""
        if self.should_stop:
            return None

        # The optuna.BaseSampler seems to be returning duplicates. Avoid returning
        # duplicates by querying recurrently to find the next sample. It won't be
        # an infinite loop because if all samples had been processed, the
        # self.should_stop check above would succeed.
        while True:
            trial = self._study.ask()
            values: Dict[str, Tuple[int, Any]] = OrderedDict()
            spi, _, values = self._get_search_point_values("", "", self._search_space, trial, values)
            if spi not in self._search_point_index_to_trail_id:
                break

        self._search_point_index_to_trail_id[spi] = trial.number
        self._num_samples_suggested += 1
        return SearchPoint(spi, values)

    def _get_search_point_values(
        self,
        prefix: str,
        name: str,
        param: Union[SearchParameter, SearchSpace],
        trial: "Trial",
        values: Dict[str, Tuple[int, Any]],
    ) -> Tuple[int, int, Union[Dict[str, Any], Any]]:
        if isinstance(param, SearchParameter):
            suggestion_name = f"{prefix}__{name}" if prefix else name

            if isinstance(param, Categorical):
                suggestions = param.get_support()
            elif isinstance(param, Conditional):
                parent_values = {parent: values[parent][1] for parent in param.parents}
                suggestions = param.get_support_with_args(parent_values)
                max_length = max(len(support.get_support()) for support in param.support.values())
                suggestions += param.default.get_support() * (max_length - len(suggestions))
                suggestion_name += "___" + " ".join(str(v) for v in parent_values.values())

            suggestion_len = len(suggestions)
            suggestion_index = trial.suggest_categorical(suggestion_name, list(range(suggestion_len)))
            suggestion = suggestions[suggestion_index]

            if isinstance(suggestion, (SearchParameter, SearchSpace)):
                suggestion_index, suggestion_len, _ = self._get_search_point_values(
                    prefix, name, suggestion, trial, values
                )
            else:
                values[name] = suggestion_index, suggestion

            return suggestion_index, suggestion_len, suggestion

        elif isinstance(param, SearchSpace):
            child_values = OrderedDict()
            indices_lengths = []
            for child_name, child_param in param.parameters:
                child_index, suggestions_len, _ = self._get_search_point_values(
                    prefix, child_name, child_param, trial, child_values
                )
                indices_lengths.append((child_index, suggestions_len))
            values[name] = (0, child_values)

            spi = 0
            for child_index, suggestions_len in reversed(indices_lengths):
                spi *= suggestions_len
                spi += child_index

            return spi, len(param), child_values

        else:
            raise ValueError(f"Unsupported parameter type: {type(param)}")

    def record_feedback_signal(self, search_point_index: int, signal: "MetricResult", should_prune: bool = False):
        trial_id = self._search_point_index_to_trail_id[search_point_index]
        if should_prune:
            self._study.tell(trial_id, state=optuna.trial.TrialState.PRUNED)
        else:
            values = []
            for name in self._objectives:
                if name in signal:
                    values.append(signal[name].value)
                elif self._higher_is_betters[name]:
                    values.append(-sys.maxsize - 1)
                else:
                    values.append(sys.maxsize)

            self._study.tell(trial_id, values)
