# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from olive.common.config_utils import ConfigBase, validate_config
from olive.common.pydantic_v1 import validator
from olive.strategy.search_algorithm import REGISTRY, SearchAlgorithm
from olive.strategy.search_results import SearchResults

if TYPE_CHECKING:
    from olive.evaluator.metric import MetricResult
    from olive.strategy.search_parameter import SearchParameter

logger = logging.getLogger(__name__)

_VALID_EXECUTION_ORDERS = ("joint", "pass-by-pass")

# pylint: disable=attribute-defined-outside-init


class SearchStrategyConfig(ConfigBase):
    execution_order: str
    search_algorithm: str
    search_algorithm_config: ConfigBase = None
    output_model_num: int = None
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
        return validate_config(v, config_class)

    @validator("stop_when_goals_met", "max_iter", "max_time", pre=True)
    def _validate_stop_when_goals_met(cls, v, values, field):
        if "execution_order" not in values:
            raise ValueError("Invalid execution_order")
        if v and values["execution_order"] != "joint":
            logger.info("%s is only supported for joint execution order. Ignoring...", field.name)
            return field.default
        return v


class SearchStrategy:
    def __init__(self, config: Union[Dict[str, Any], SearchStrategyConfig]):
        self._config = validate_config(config, SearchStrategyConfig)
        self._initialized = False
        self.exit_criteria_met = False

    def initialize(
        self,
        pass_flows_search_spaces: List[List[Tuple[str, Dict[str, "SearchParameter"]]]],
        init_model_id: str,
        objective_dict: Dict[str, dict],
    ):
        """Initialize the search strategy.

        pass_flows_search_spaces: list of list of tuples of format (search_space_name, {param_name: SearchParameter})
        objective_dict: dictionary of format {objective_name: {"higher_is_better": bool, "goal": float}}
        """
        self._objective_dict = objective_dict

        # search spaces
        self._spaces_order = [[pass_ss[0] for pass_ss in pass_flow_ss] for pass_flow_ss in pass_flows_search_spaces]
        self._spaces_dict = {}
        for pass_flow_ss in pass_flows_search_spaces:
            for pass_ss in pass_flow_ss:
                self._spaces_dict[pass_ss[0]] = pass_ss[1]

        # search space dictionaries for pass are grouped based on execution_order
        self._spaces_groups = self._group_search_spaces(self._spaces_order)
        # sub spaces group in pass-by-pass execution order
        self._pass_by_pass_sg = None

        self._done_spaces_groups = []
        self._active_spaces_group = None

        # state
        self._searchers: Dict[Any, SearchAlgorithm] = {}
        self._search_results: Dict[Any, SearchResults] = {}
        self._init_model_ids: Dict[Any, str] = {}
        self.init_model_id = init_model_id
        self._best_search_points = {}

        # initialize the first search space
        self._next_search_group(init_model_id)

        self._initialized = True

    def _group_search_spaces(self, search_space_names: List[List]):
        """Group search spaces based on execution order."""
        # joint: all passes grouped together
        # pass-by-pass: each pass is a separate group
        if self._config.execution_order == "joint":
            search_spaces_groups = search_space_names
        elif self._config.execution_order == "pass-by-pass":
            # run pass-by-pass for each pass flow which is defined as a list of registered passes
            search_spaces_groups = []
            for pass_flow_ss in search_space_names:
                pass_flow_groups = [[pass_ss] for pass_ss in pass_flow_ss]
                search_spaces_groups.append(pass_flow_groups)
        else:
            raise ValueError(f"Unknown execution order: {self._config.execution_order}")

        return search_spaces_groups

    def _next_search_group(self, init_model_id: Optional[str] = None) -> Optional[SearchAlgorithm]:
        """Get the next search space group and initialize the search algorithm."""
        # if there is no more search space group, return None
        # 1. joint: no more flows(self._space_groups)
        # 2. pass-by-pass: no more flows(self._space_groups) and no more passes(self._pass_by_pass_sg)
        if not (self._spaces_groups or self._pass_by_pass_sg):
            self._active_spaces_group = None
            return None

        # for the fist search group, init_model_id must be provided
        if init_model_id is None and self._active_spaces_group is None:
            raise ValueError("init_model_id must be provided for the first search group")

        if self._config.execution_order == "joint":
            next_sg = self._next_search_group_joint(init_model_id)
        elif self._config.execution_order == "pass-by-pass":
            next_sg = self._next_search_group_pass_by_pass(init_model_id)
        return next_sg

    def _next_search_group_pass_by_pass(self, init_model_id: Optional[str] = None) -> Optional[SearchAlgorithm]:
        # passes are exhausted or empty for current flow, try next pass flow
        if not self._pass_by_pass_sg:
            self._pass_by_pass_sg = self._spaces_groups.pop(0)
            self._active_spaces_group = None
            init_model_id = self.init_model_id

        # get the best model from last space group
        if self._active_spaces_group is not None:
            self._done_spaces_groups.append(self._active_spaces_group)
            # legacy, will update once search results has info function
            sorted_model_ids, sorted_search_points, sorted_results = self._search_results[
                tuple(self._active_spaces_group)
            ].sort_search_points(apply_goals=True)
            if sorted_model_ids is None:
                logger.warning(
                    "No models in this search group %s met the goals. Sorting the models without applying goals...",
                    self._active_spaces_group,
                )
                sorted_model_ids, sorted_search_points, sorted_results = self._search_results[
                    tuple(self._active_spaces_group)
                ].sort_search_points(apply_goals=False)
            # TODO(trajep): this is a hack to get the best search point for the current search space group
            #      it totally work for joint execution order, but not for pass-by-pass
            if sorted_search_points and sorted_results:
                best_search_point = (
                    sorted_search_points[0],
                    list(sorted_results[0].values()),
                    sorted_model_ids[0],
                )
                self._best_search_points[tuple(self._active_spaces_group)] = best_search_point
                init_model_id = best_search_point[2][-1]

        if init_model_id is None and self._active_spaces_group is not None:
            raise ValueError(
                f"The previous search group {self._active_spaces_group} has no output models that were created and"
                " evaluated successfully. Cannot continue."
            )

        # set up next search group
        # if it is the first run in this flow, init_model_id should be input model id
        # otherwise, it should be the best model id from last search group
        self._active_spaces_group = self._pass_by_pass_sg.pop(0)
        self._searchers[tuple(self._active_spaces_group)] = self._create_searcher(self._active_spaces_group)
        self._search_results[tuple(self._active_spaces_group)] = SearchResults(self._objective_dict)
        self._init_model_ids[tuple(self._active_spaces_group)] = init_model_id
        return self._active_spaces_group

    def _next_search_group_joint(self, init_model_id: Optional[str] = None) -> Optional[SearchAlgorithm]:
        init_model_id = init_model_id or self.init_model_id
        # get the first pass flow
        # for "joint" model, init_model_id should be input_model_id
        sg = self._spaces_groups.pop(0)
        self._searchers[tuple(sg)] = self._create_searcher(sg)
        self._search_results[tuple(sg)] = SearchResults(self._objective_dict)
        self._init_model_ids[tuple(sg)] = init_model_id
        self._active_spaces_group = sg
        return self._active_spaces_group

    def _create_searcher(self, search_space_names: List[str]) -> SearchAlgorithm:
        """Create a search algorithm."""
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
        """Get the next step in the search."""
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
        signal: "MetricResult",
        model_ids: List[str],
        should_prune: bool = False,
    ):
        """Record the feedback signal for the given search point."""
        if not self._initialized:
            raise ValueError("Search strategy is not initialized")
        self._search_results[tuple(self._active_spaces_group)].record(search_point, signal, model_ids)
        self._searchers[tuple(self._active_spaces_group)].report(search_point, signal, should_prune)

    def check_exit_criteria(self, iter_num, time_diff, metric_signal):
        """Check if the olive search_strategy should exit."""
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

    def get_output_model_num(self):
        return self._config.output_model_num
