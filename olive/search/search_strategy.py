# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Tuple, Union

from olive.common.config_utils import CaseInsensitiveEnum, ConfigBase, NestedConfig, validate_config, validate_enum
from olive.common.pydantic_v1 import validator
from olive.search.samplers import REGISTRY, SearchSampler
from olive.search.search_parameter import Categorical
from olive.search.search_results import SearchResults
from olive.search.search_sample import SearchSample
from olive.search.search_space import SearchSpace

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult
    from olive.search.search_parameter import SearchParameter

logger = logging.getLogger(__name__)

# ruff: noqa: PD011


class SearchStrategyExecutionOrder(CaseInsensitiveEnum):
    JOINT = "joint"
    PASS_BY_PASS = "pass-by-pass"


class SearchStrategyConfig(NestedConfig):
    _nested_field_name = "sampler_config"
    execution_order: Union[str, SearchStrategyExecutionOrder] = None
    sampler: str = None
    sampler_config: ConfigBase = None
    output_model_num: int = None
    stop_when_goals_met: bool = False
    max_iter: int = None
    max_time: int = None
    include_pass_params: bool = True

    @validator("execution_order", pre=True)
    def _validate_execution_order(cls, v):
        return validate_enum(SearchStrategyExecutionOrder, v)

    @validator("sampler", pre=True)
    def _validate_sampler(cls, v):
        if v not in REGISTRY:
            raise ValueError(f"Unknown sampler: {v}")
        return v

    @validator("sampler_config", pre=True, always=True)
    def _validate_sampler_config(cls, v, values):
        if "sampler" not in values:
            raise ValueError("Invalid sampler")

        config_class = REGISTRY[values["sampler"]].get_config_class()
        return validate_config(v, config_class)

    @validator("stop_when_goals_met", "max_iter", "max_time", pre=True)
    def _validate_stop_when_goals_met(cls, v, values, field):
        if "execution_order" not in values:
            raise ValueError("Invalid execution_order")
        if v and values["execution_order"] != SearchStrategyExecutionOrder.JOINT:
            logger.info("%s is only supported for joint execution order. Ignoring...", field.name)
            return field.default
        return v


@dataclass
class SearchWalkState:
    """A simple data class to hold the state while traversing the search space.

    Each instance of this class holds data (sampler to use, results of evaluation, etc.)
    for a single search space.
    """

    def __init__(self, path: List[int], model_ids: List[str], sampler: SearchSampler, results: SearchResults):
        # Unique identification for the state.
        self.path: List[int] = deepcopy(path)

        # Sampler to use for the relevant/owning search space
        self.sampler: SearchSampler = sampler

        # Result of evaluating generated samples
        self.results: SearchResults = results

        # Input model ids to be used for processing the generated sample
        self.model_ids: List[str] = model_ids

        # Once the search space has exhausted all its samples, the results
        # are sorted to find the order in which to move to the next search
        # space. This is the index in the those sorted results.
        self.best_result_index = -1


class SearchStrategy:
    def __init__(self, config: Union[Dict[str, Any], SearchStrategyConfig]):
        self.config: SearchStrategyConfig = validate_config(config, SearchStrategyConfig)

        # Initialization variables
        self._search_spaces: List[SearchSpace] = None
        self._objectives: Dict[str, Dict[str, Any]] = None
        self._init_model_id: str = None

        # State variables
        self._path: List[int] = None
        self._state: Dict[Tuple, SearchWalkState] = None

        # self._iteration_count and self._num_samples_suggested include counts across all search spaces.
        # For specific counts, query the sampler corresponding to the specific search space
        # i.e. SearchWalkState.sampler.num_samples_suggested.
        # Also, note that the iteration count includes invalid search points, but num_samples_suggested doesn't.
        # Invalid search points are automatically discarded during iteration.

        self._start_time: float = 0
        self._iteration_count: int = 0
        self._num_samples_suggested: int = 0

        self._initialized: bool = False

    def initialize(
        self,
        space_config: Dict[str, List[Dict[str, "SearchParameter"]]],
        init_model_id: str,
        objectives: Dict[str, Dict[str, Dict[str, Any]]],
    ):
        """Initialize the search strategy.

        space_config: Ordered dictionary of format {pass_name, [{param_name: SearchParameter}]}
        init_model_id: Input model id to use to start searching.
        objectives: dictionary of format
                    {pass_name: {objective_name: {"higher_is_better": bool, "goal": float, "priority": int}}}

        Depending on the execution order, we could generate either a single search space (for joint mode)
        or multiple search spaces (for pass-by-pass mode). However, the logic in how we process these
        search spaces does not differ.
        """
        # for the fist search group, init_model_id must be provided
        if not init_model_id:
            raise ValueError("init_model_id must be provided for search")

        if self.config.execution_order == SearchStrategyExecutionOrder.JOINT:
            self._search_spaces = [
                SearchSpace(
                    [
                        (pass_name, Categorical([SearchSpace(list(params.items())) for params in passes]))
                        for pass_name, passes in space_config.items()
                    ]
                )
            ]
        elif self.config.execution_order == SearchStrategyExecutionOrder.PASS_BY_PASS:
            self._search_spaces = [
                SearchSpace([(pass_name, Categorical([SearchSpace(list(params.items())) for params in passes]))])
                for pass_name, passes in space_config.items()
            ]
        else:
            raise ValueError(f"Unsupported execution order: {self.config.execution_order}")

        objectives = objectives or {}
        self._objectives = {pass_name: objectives.get(pass_name) or {} for pass_name in space_config}
        self._init_model_id = init_model_id

        # Note that the state variables will be initialized at start of iteration.
        self._initialized = True

    @property
    def search_spaces(self):
        """Returns the list of search spaces."""
        return self._search_spaces

    @property
    def iteration_count(self) -> int:
        """Returns the number of iterations so far across all search spaces."""
        return self._iteration_count

    @property
    def start_time(self) -> float:
        """Returns the start time of current iteration."""
        return self._start_time

    @property
    def elapsed_time(self) -> float:
        """Returns elapsed time of the current iteration."""
        return (time.time() - self._start_time) if self._start_time else 0

    @property
    def num_samples_suggested(self) -> int:
        """Returns the number of samples suggested so far across all search spaces."""
        return self._num_samples_suggested

    @property
    def max_samples(self) -> int:
        """Returns the maximum number of samples."""
        count = 1
        for space in self._search_spaces:
            count *= len(space)
        return count

    def __iter__(self) -> Generator[SearchSample, None, None]:
        # Initialize the state variables
        self._path = []
        search_space = self._search_spaces[len(self._path)]
        objectives = self._get_objectives(search_space)
        self._state = {
            tuple(self._path): SearchWalkState(
                self._path,
                [self._init_model_id],
                self._create_sampler(search_space, objectives),
                self._create_results(objectives),
            )
        }

        self._start_time = time.time()
        self._iteration_count = 0
        self._num_samples_suggested = 0

        while True:
            state = self._state[tuple(self._path)]

            while not state.sampler.should_stop:
                self._iteration_count += 1
                search_point = state.sampler.suggest()

                # Discard invalid search points
                if not search_point.is_valid():
                    continue

                self._num_samples_suggested += 1
                yield SearchSample(search_point, state.model_ids)

                # If this is the last pass in the walk, evaluate the model to see if all goals are met.
                if (
                    self.config.stop_when_goals_met
                    and (len(self._path) == (len(self._search_spaces) - 1))
                    and state.results.meets_goals(search_point.index)
                ):
                    return None

                # Check is any of the global stop criteria are met.
                # NOTE: Search will run at least one step before stopping.
                if self.should_stop:
                    return None

            # Try stepping down the tree, and if that fails, try stepping up. If both fails, we are done
            if not self._step_down() and not self._step_up():
                return None

    def _get_objectives(self, search_space: SearchSpace) -> Dict[str, Any]:
        """Return search space specific objectives."""
        return {
            name: objective
            for pass_name, _ in search_space.parameters
            for name, objective in self._objectives[pass_name].items()
        }

    def _create_sampler(self, search_space: SearchSpace, objectives: Dict[str, Any]) -> SearchSampler:
        """Create a search sampler."""
        if self.config.sampler not in REGISTRY:
            raise ValueError(f"Unsupported search sampler: {self.config.sampler}")

        return REGISTRY[self.config.sampler](search_space, self.config.sampler_config, objectives)

    def _create_results(self, objectives: Dict[str, Any]) -> SearchResults:
        """Create and return a search result."""
        return SearchResults(objectives)

    def _initialize_step(self) -> bool:
        state = self._state[tuple(self._path)]

        # NOTE: Two possible scenarios for pass-by-pass mode -
        #   1. All search points are evaluated for each search space before moving down the tree.
        #   2. Evaluate search points until we find a suitable candidate and move down. If failed at end, move
        #      up to continue finding next candidate.
        # Implementing option 1 currently i.e. all search points are evaluated for search space before moving down.
        # Logic here can be customized to support the other if need be.

        # Get the next best result index
        state.best_result_index, next_search_point, next_model_ids = state.results.get_next_best_result(
            state.best_result_index
        )
        if state.best_result_index is not None:
            self._path.append(next_search_point)
            search_space = self._search_spaces[len(self._path)]
            objectives = self._get_objectives(search_space)
            self._state[tuple(self._path)] = SearchWalkState(
                self._path,
                next_model_ids,
                self._create_sampler(search_space, objectives),
                self._create_results(objectives),
            )
            return True

        return False

    def _step_up(self) -> bool:
        """Step back to the previous search space on stack to evaluate based on the next best sample."""
        if not self._path:
            return False

        self._path.pop()

        # Results here are already sorted, don't have to do it again!
        return self._initialize_step()

    def _step_down(self) -> bool:
        """Step down to the next search space in queue."""
        if len(self._path) == (len(self._search_spaces) - 1):
            return False

        state = self._state[tuple(self._path)]

        # Current state is potentially modified, sort the collected results again!
        if not state.results.sort(apply_goals=True):
            logger.warning(
                "No models in path %s met the goals. Sorting the models without applying goals...", self._path
            )
            state.results.sort(apply_goals=False)

        return self._initialize_step()

    def record_feedback_signal(
        self,
        search_point_index: int,
        signal: "MetricResult",
        model_ids: List[str],
        should_prune: bool = False,
    ):
        """Record the feedback signal for the given search point."""
        assert self._initialized, "Search strategy is not initialized"

        state = self._state[tuple(self._path)]
        state.results.record_feedback_signal(search_point_index, signal, model_ids)
        state.sampler.record_feedback_signal(search_point_index, signal, should_prune)

    @property
    def should_stop(self):
        """Check if the search should stop."""
        # NOTE: Individual goal criteria is checked at the end of each step in the iteration loop
        return ((self.config.max_iter is not None) and (self._iteration_count > self.config.max_iter)) or (
            (self.config.max_time is not None) and (self.elapsed_time > self.config.max_time)
        )
