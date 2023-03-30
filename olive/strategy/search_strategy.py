# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import validator

from olive.common.config_utils import ConfigBase, validate_config
from olive.strategy.search_algorithm import REGISTRY, SearchAlgorithm
from olive.strategy.search_parameter import SearchParameter
from olive.strategy.search_results import SearchResults

logger = logging.getLogger(__name__)

_VALID_EXECUTION_ORDERS = ["joint", "pass-by-pass"]


class SearchStrategyConfig(ConfigBase):
    execution_order: str
    search_algorithm: str
    search_algorithm_config: ConfigBase = None
    stop_when_goals_met: bool = False
    max_iter: int = None
    max_time: int = None

    @validator("execution_order", pre=True)
    def _validate_execution_order(cls, v):
        if v not in _VALID_EXECUTION_ORDERS:
            raise ValueError(f"Unknown execution order: {v}")
        return v

    @validator("search_algorithm", pre=True)
    def _validate_search_algorithm(cls, v):
        if v not in REGISTRY:
            raise ValueError(f"Unknown search algorithm: {v}")
        return v

    @validator("search_algorithm_config", pre=True, always=True)
    def _validate_search_algorithm_config(cls, v, values):
        if "search_algorithm" not in values:
            raise ValueError("Invalid search_algorithm")

        config_class = REGISTRY[values["search_algorithm"]].get_config_class()
        return validate_config(v, ConfigBase, config_class)

    @validator("stop_when_goals_met", "max_iter", "max_time", pre=True)
    def _validate_stop_when_goals_met(cls, v, values, field):
        if "execution_order" not in values:
            raise ValueError("Invalid execution_order")
        if v and values["execution_order"] != "joint":
            logger.info(f"{field.name} is only supported for joint execution order. Ignoring...")
            return field.default
        return v


class SearchStrategy(ABC):
    """
    Search strategy
    """

    def __init__(self, config: Union[Dict[str, Any], SearchStrategyConfig]):
        self._config = validate_config(config, SearchStrategyConfig)
        self._initialized = False
        self.exit_criteria_met = False

    def initialize(
        self,
        search_spaces_list: List[Tuple[str, Dict[str, SearchParameter]]],
        init_model_id: str,
        objective_dict: Dict[str, dict],
    ):
        """
        Initialize the search strategy.

        search_spaces_list: list of tuples of format (search_space_name, {param_name: SearchParameter})
        objective_dict: dictionary of format {objective_name: {"higher_is_better": bool, "goal": float}}
        """
        self._objective_dict = objective_dict

        # search spaces
        self._spaces_order = [search_space[0] for search_space in search_spaces_list]
        self._spaces_dict = {search_space[0]: search_space[1] for search_space in search_spaces_list}

        # search space dictionaries for pass are grouped based on execution_order
        self._spaces_groups = self._group_search_spaces(self._spaces_order)
        self._done_spaces_groups = []
        self._active_spaces_group = None

        # state
        self._searchers = {}
        self._search_results = {}
        self._init_model_ids = {}
        self._best_search_points = {}

        # initialize the first search space
        self._next_search_group(init_model_id)

        self._initialized = True

    def _group_search_spaces(self, search_space_names: List[str]):
        """
        Group search spaces based on execution order.
        """
        # joint: all passes grouped together
        # pass-by-pass: each pass is a separate group
        if self._config.execution_order == "joint":
            search_spaces_groups = [search_space_names]
        elif self._config.execution_order == "pass-by-pass":
            search_spaces_groups = [[search_space_name] for search_space_name in search_space_names]
        else:
            raise ValueError(f"Unknown execution order: {self._config.execution_order}")

        return search_spaces_groups

    def _next_search_group(self, init_model_id: Optional[str] = None) -> Optional[SearchAlgorithm]:
        """
        Get the next search space group and initialize the search algorithm.
        """
        # TODO: organize the state better and make execution order more flexible using a graph
        if self._active_spaces_group is not None:
            self._done_spaces_groups.append(self._active_spaces_group)
            # legacy, will update once search results has info function
            sorted_model_ids, sorted_search_points, sorted_results = self._search_results[
                tuple(self._active_spaces_group)
            ].sort_search_points(apply_goals=True)
            # TODO: this is a hack to get the best search point for the current search space group
            #      it totally work for joint execution order, but not for pass-by-pass
            if sorted_search_points and sorted_results:
                best_search_point = (sorted_search_points[0], list(sorted_results[0].values()), sorted_model_ids[0])
                self._best_search_points[tuple(self._active_spaces_group)] = best_search_point
                init_model_id = best_search_point[2][-1]

        if len(self._spaces_groups) == 0:
            self._active_spaces_group = None
            return None

        self._active_spaces_group = self._spaces_groups.pop(0)
        self._searchers[tuple(self._active_spaces_group)] = self._create_searcher(self._active_spaces_group)
        self._search_results[tuple(self._active_spaces_group)] = SearchResults(self._objective_dict)
        self._init_model_ids[tuple(self._active_spaces_group)] = init_model_id

        return self._active_spaces_group

    def _create_searcher(self, search_space_names: List[str]) -> SearchAlgorithm:
        """
        Create a search algorithm.
        """
        search_spaces_dict = {space_name: deepcopy(self._spaces_dict[space_name]) for space_name in search_space_names}
        objectives = list(self._objective_dict.keys())
        higher_is_betters = [self._objective_dict[objective]["higher_is_better"] for objective in objectives]
        if self._config.search_algorithm in REGISTRY:
            searcher = REGISTRY[self._config.search_algorithm](
                search_spaces_dict, objectives, higher_is_betters, self._config.search_algorithm_config
            )
        else:
            raise ValueError(f"Unknown search algorithm: {self._config.search_algorithm}")
        return searcher

    def next_step(self) -> Optional[Dict[str, Any]]:
        """
        Get the next step in the search
        """
        if not self._initialized:
            raise ValueError("Search strategy is not initialized")

        if self.exit_criteria_met:
            self._next_search_group()

        # if there is no active searcher, we are done
        if self._active_spaces_group is None:
            return None

        # get the next search point from the active searcher
        search_point = self._searchers[tuple(self._active_spaces_group)].suggest()
        # if there are no more search points, move to the next search space group
        if search_point is None:
            self._next_search_group()
            return self.next_step()

        return {
            "search_point": search_point,
            "model_id": self._init_model_ids[tuple(self._active_spaces_group)],
            "passes": [(space_name, search_point[space_name]) for space_name in self._active_spaces_group],
        }

    def record_feedback_signal(
        self,
        search_point: Dict[str, Dict[str, Any]],
        signal: Dict[str, float],
        model_ids: List[str],
        should_prune: bool = False,
    ):
        """
        Record the feedback signal for the given search point.
        """
        if not self._initialized:
            raise ValueError("Search strategy is not initialized")
        self._search_results[tuple(self._active_spaces_group)].record(search_point, signal, model_ids)
        self._searchers[tuple(self._active_spaces_group)].report(search_point, signal, should_prune)

    def check_exit_criteria(self, iter_num, time_diff, metric_signal):
        """
        Check if the olive search_strategy should exit.
        """
        self.exit_criteria_met = False
        if not self._config.stop_when_goals_met:
            # stop early stopping when stop_when_goals_met is False, but still apply goals check without stopping
            return
        # early exit is not supported for pass-by-pass execution order currently
        if self._config.execution_order == "pass-by-pass":
            return
        if self._config.max_iter is not None and iter_num > self._config.max_iter:
            self.exit_criteria_met = True
            return
        if self._config.max_time is not None and time_diff > self._config.max_time:
            self.exit_criteria_met = True
            return
        if metric_signal == {}:
            return
        self.exit_criteria_met = self._config.stop_when_goals_met and self._search_results[
            tuple(self._active_spaces_group)
        ].check_goals(metric_signal)

    def get_best_execution(self) -> Dict[str, Any]:
        """
        Get the best execution found so far.
        """
        best_search_points = {}
        best_metric = None
        best_model_ids = []

        # legacy will update once search results has info function
        for spaces_group in self._done_spaces_groups:
            annotated_search_point = self._best_search_points.get(tuple(spaces_group), None)
            if not annotated_search_point:
                continue
            for space_name in spaces_group:
                best_search_points[space_name] = annotated_search_point[0][space_name]
            best_metric = annotated_search_point[1]
            best_model_ids += annotated_search_point[2]

        return {
            "search_points": best_search_points,
            "metric": best_metric,
            "model_ids": best_model_ids,
        }
