# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult


class SearchResults:
    def __init__(self, objectives: Dict[str, Dict[str, Any]] = None):
        # Order the objectives based on priority, and then by name
        objectives = objectives or {}
        self._objectives = OrderedDict(sorted(objectives.items(), key=lambda entry: (entry[1]["priority"], entry[0])))

        self._goals = {}
        self._multipliers = {}
        self._higher_is_betters = {}
        for name, objective in self._objectives.items():
            if objective.get("goal") is not None:
                self._goals[name] = objective["goal"]

            self._higher_is_betters[name] = objective.get("higher_is_better") or False
            self._multipliers[name] = 1 if self._higher_is_betters[name] else -1

        self._results: Tuple[MetricResult, List[str]] = []
        self._sorted_indices: List[int] = []

    def record_feedback_signal(self, search_point_index: int, result: "MetricResult", model_ids: List[str]):
        """Record the evaluation result of a search point."""
        self._results += [None] * ((search_point_index + 1) - len(self._results))
        self._results[search_point_index] = (result, model_ids)

    def meets_goals(self, search_point_index: int) -> bool:
        """Check if the result satisfies the constraints."""
        if search_point_index >= len(self._results):
            return False

        if not self._results[search_point_index]:
            return False

        if not self._goals:
            return True  # if goals are not set, always return True

        result, _ = self._results[search_point_index]
        return all(
            (self._multipliers[name] * result[name].value) >= (self._multipliers[name] * goal)
            for name, goal in self._goals.items()
            if name in result
        )

    def sort(self, apply_goals: bool = False):
        indices, results = self._get_results_list(apply_goals)
        if not results:
            self._sorted_indices = indices
            return False

        # sort by objectives, left most objective has highest priority
        # flip the order of the objectives since np.lexsort prioritizes the last column
        # negate the results since np.lexsort sorts in ascending order
        results = -np.flip(np.array(results), 1)
        sorted_indices = np.lexsort(results.T)
        self._sorted_indices = [indices[i] for i in sorted_indices]

        return True

    def get_next_best_result(self, start_index: int) -> Tuple[int, int, List[str]]:
        assert start_index is not None, "Expecting an index, got None"

        if start_index < -1:
            return None, None, None

        next_best_index = start_index + 1
        if next_best_index >= len(self._sorted_indices):
            return None, None, None

        _, model_ids = self._results[self._sorted_indices[next_best_index]]
        return next_best_index, self._sorted_indices[next_best_index], model_ids

    def _get_results_list(self, apply_goals: bool = False) -> Tuple[List[int], List[float]]:
        """Return the results as a tuple of indices and values."""
        values = []
        indices = []
        if not self._objectives:
            # If no objectives, then use the indices of the valid results in no specific order
            indices = [spi for spi, entry in enumerate(self._results) if entry]
            return indices, values

        # NOTE: values array need to be packed but a simple loop thru' each entry could
        # possibly create a jagged array if the number of actual objectives in the signal
        # are different from the expected ones. To circumvent the issue, we use min/max
        # depending on the higher_is_better values for the missing expected objectives
        # to deprioritize that objective while sorting.

        for spi, entry in enumerate(self._results):
            if entry and (not apply_goals or self.meets_goals(spi)):
                result, _ = entry

                v = []
                for name in self._objectives:
                    if name in result:
                        # Values are scaled for comparison such that higher is better for all objectives.
                        v.append(self._multipliers[name] * result[name].value)
                    else:
                        v.append(-sys.maxsize - 1 if self._higher_is_betters[name] else sys.maxsize)

                values.append(v)
                indices.append(spi)

        return indices, values

    def to_json(self):
        """Return a json representation of the search results."""
        return {"objectives": self._objectives, "results": self._results}

    @classmethod
    def from_json(cls, json_dict):
        """Create a SearchResults object from a json representation."""
        search_results = cls(json_dict["objectives"])
        search_results._results = json_dict["results"]
        return search_results
