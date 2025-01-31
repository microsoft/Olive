# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult


class SearchResults:
    def __init__(self):
        self._results: Tuple[MetricResult, List[str], Dict[str, Any]] = []
        self._sorted_indices: List[int] = []

    def record_feedback_signal(
        self, search_point_index: int, objectives: Dict[str, dict], result: "MetricResult", model_ids: List[str]
    ):
        """Record the evaluation result of a search point."""
        self._results += [None] * ((search_point_index + 1) - len(self._results))
        self._results[search_point_index] = (result, model_ids, objectives)

    def meets_goals(self, search_point_index: int) -> bool:
        """Check if the result satisfies the constraints."""
        if search_point_index >= len(self._results):
            return False

        if not self._results[search_point_index]:
            return False

        result, _, objectives = self._results[search_point_index]
        goals = {name: obj["goal"] for name, obj in objectives.items() if obj.get("goal") is not None}
        if not goals:
            return True  # if goals are not set, always return True

        # multiplier for each objective and goals
        multipliers = {
            name: 1 if objective.get("higher_is_better", False) else -1 for name, objective in objectives.items()
        }
        return all((multipliers[obj] * result[obj].value) >= (multipliers[obj] * goal) for obj, goal in goals.items())

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

        _, model_ids, _ = self._results[self._sorted_indices[next_best_index]]
        return next_best_index, self._sorted_indices[next_best_index], model_ids

    def _get_results_list(self, apply_goals: bool = False) -> Tuple[List[int], List[float]]:
        """Return the results as a tuple of indices and values.

        Values are multiplied by the objective multiplier so that higher is better for all objectives.
        """
        all_objectives = {}
        for spi, entry in enumerate(self._results):
            if entry and (not apply_goals or self.meets_goals(spi)):
                _, _, objectives = entry
                for name in objectives:
                    if name in all_objectives:
                        assert all_objectives[name] == objectives[name].get(
                            "higher_is_better", False
                        ), "Conflicting values for higher_is_better across same named objectives"
                    else:
                        all_objectives[name] = objectives[name].get("higher_is_better", False)

        indices = []
        values = []
        if not all_objectives:
            # If no objectives, then use the indices of the valid results in no specific order
            indices = [spi for spi, entry in enumerate(self._results) if entry]
            return indices, values

        # NOTE: values array need to be packed but a simple loop thru' each entry could
        # possibly create a zagged array if the number of objectives are different.

        for spi, entry in enumerate(self._results):
            if entry and (not apply_goals or self.meets_goals(spi)):
                result, _, objectives = entry
                if objectives:
                    indices.append(spi)
                    v = []
                    for name, hib in all_objectives.items():
                        if name in objectives:
                            v.append((1 if hib else -1) * result[name].value)
                        else:
                            v.append(-sys.maxsize - 1 if hib else sys.maxsize)
                    values.append(v)

        return indices, values

    def to_json(self):
        """Return a json representation of the search results."""
        return {"results": self._results}

    @classmethod
    def from_json(cls, json_dict):
        """Create a SearchResults object from a json representation."""
        search_results = cls()
        search_results._results = json_dict["results"]
        return search_results
