# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from olive.common.utils import hash_dict

if TYPE_CHECKING:
    from olive.evaluator.metric import MetricResult


class SearchResults:
    def __init__(
        self,
        objective_dict: Dict[str, dict],
        init_model_history: Dict[str, Any] = None,
    ):
        self.objective_dict = objective_dict
        # objectives and directions of optimization
        self.objectives = list(objective_dict.keys())
        self.higher_is_betters = [objective_dict[obj]["higher_is_better"] for obj in self.objectives]
        # multiplier for each objective
        self.obj_mul = {obj: 1 if hib else -1 for obj, hib in zip(self.objectives, self.higher_is_betters)}

        # objective goal values
        self.goals = {}
        for name, obj in self.objective_dict.items():
            if obj["goal"] is not None:
                self.goals[name] = obj["goal"]

        # Record of the search path that led to the init model
        # Of the form {"search_point": ..., "result": ..., "model_ids": ...}
        self.init_model_history = init_model_history

        # search results state
        self.search_point_hash_table = {}
        self.results = {}
        self.model_ids = {}

    def record(self, search_point: Dict[str, Dict[str, Any]], result: "MetricResult", model_ids: List[str]):
        """Report the result of a configuration."""
        search_point_hash = hash_dict(search_point)
        self.search_point_hash_table[search_point_hash] = deepcopy(search_point)
        self.results[search_point_hash] = deepcopy(result)
        self.model_ids[search_point_hash] = model_ids

    def check_goals(self, result: "MetricResult") -> bool:
        """Check if the result satisfies the constraints."""
        # if goals are not set, return True always
        if not self.goals:
            return True

        for obj, goal in self.goals.items():
            if self.obj_mul[obj] * result[obj].value < self.obj_mul[obj] * goal:
                return False
        return True

    def sort_search_points(self, objectives: List[str] = None, apply_goals: bool = False) -> List[str]:
        """Return the search points sorted by the objectives."""
        # TODO(trajep): this function only works for pass-by-pass execution order, but only return with the first model
        # with the best latency results which is not correct. One pass may generate multiple models.
        if objectives is None:
            objectives = self.objectives
        else:
            assert set(objectives).issubset(self.objectives)

        results, search_point_hashes = self._get_results_list(objectives, apply_goals)
        if not results:
            return None, None, None

        # sort by objectives, left most objective has highest priority
        # flip the order of the objectives since np.lexsort prioritizes the last column
        # negate the results since np.lexsort sorts in ascending order
        results = -np.flip(np.array(results), 1)
        sorted_indices = np.lexsort(results.T)
        sorted_hashes = [search_point_hashes[i] for i in sorted_indices]

        # get model numbers
        sorted_model_ids = [self.model_ids[point_hash] for point_hash in sorted_hashes]
        sorted_results = [self.results[point_hash] for point_hash in sorted_hashes]
        # TODO(jambayk): this will be done using helper later
        sorted_search_points = [self.search_point_hash_table[point_hash] for point_hash in sorted_hashes]
        return sorted_model_ids, sorted_search_points, sorted_results

    def _get_results_list(
        self, objectives: List[str] = None, apply_goals: bool = False
    ) -> Tuple[List[List[float]], List[str]]:
        """Return the results as a list of lists.

        Values are multiplied by the objective multiplier so that higher is better for all objectives.
        """
        if objectives is None:
            objectives = self.objectives
        else:
            assert set(objectives).issubset(self.objectives)

        search_point_hashes = []
        results = []
        for search_point_hash in self.results:
            result = self.results[search_point_hash]
            if not result:
                continue
            if apply_goals and not self.check_goals(result):
                continue
            search_point_hashes.append(search_point_hash)
            results.append([self.obj_mul[obj] * result[obj].value for obj in objectives])

        return results, search_point_hashes

    def to_json(self):
        """Return a json representation of the search results."""
        return {
            "objective_dict": self.objective_dict,
            "init_model_history": self.init_model_history,
            "results": self.results,
            "model_ids": self.model_ids,
            "search_point_hash_table": self.search_point_hash_table,
        }

    @classmethod
    def from_json(cls, json_dict):
        """Create a SearchResults object from a json representation."""
        search_results = cls(json_dict["objective_dict"], json_dict["init_model_history"])
        search_results.search_point_hash_table = json_dict["search_point_hash_table"]
        search_results.results = json_dict["results"]
        search_results.model_ids = json_dict["model_ids"]
        return search_results
