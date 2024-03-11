# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from random import Random
from typing import Any, Dict, Iterator, List, Optional, Tuple

from olive.strategy.search_parameter import Categorical, Conditional, SearchParameter, SpecialParamValue
from olive.strategy.utils import order_search_parameters


class SearchSpace:
    """Search space for a search algorithm."""

    def __init__(self, search_space: Dict[str, Dict[str, SearchParameter]], seed: Optional[int] = 1):
        # search_space is dictionary of format: {"pass_id/space_name": {"param_name": SearchParameter}
        self._search_space = deepcopy(search_space)
        self._iter_order = self._order_search_space(self._search_space)
        self._empty_search_point = {space_name: {} for space_name in self._search_space}
        self._seed = seed
        self.rng = Random(self._seed)

    def _order_search_space(self, search_space) -> List[Tuple[str, str]]:
        """Order the search space by topological order of parameters for each pass_id/space_name."""
        full_iter_order = []
        for space_name, space_item in search_space.items():
            iter_order = order_search_parameters(space_item)
            full_iter_order.extend([(space_name, param_name) for param_name in iter_order])
        return full_iter_order

    def set_seed(self, seed: int):
        """Set the random seed for the search space."""
        self._seed = seed
        self.reset_rng()

    def reset_rng(self):
        """Reset the random number generator."""
        self.rng = Random(self._seed)

    def random_sample(self) -> Dict[str, Dict[str, Any]]:
        """Sample a random configuration from the search space."""
        # initialize search point
        search_point = deepcopy(self._empty_search_point)

        # sample from search space
        for space_name, param_name in self._iter_order:
            param = self._search_space[space_name][param_name]
            if isinstance(param, Conditional):
                parent_vals = {parent: search_point[space_name][parent] for parent in param.parents}
                options = param.get_support(parent_vals)
            elif isinstance(param, Categorical):
                options = param.get_support()
            search_point[space_name][param_name] = self.rng.choice(options)
            if search_point[space_name][param_name] == SpecialParamValue.INVALID:
                return self.random_sample()

        return search_point

    def _iterate_util(
        self, full_iter_order: List[Tuple[str, str]], search_point: Dict[str, Dict[str, Any]], index: int
    ) -> Iterator[Dict[str, Dict[str, Any]]]:
        if index == len(full_iter_order):
            yield deepcopy(search_point)
            return

        space_name, param_name = full_iter_order[index]
        param = self._search_space[space_name][param_name]

        if isinstance(param, Conditional):
            parent_vals = {parent: search_point[space_name][parent] for parent in param.parents}
            options = param.get_support(parent_vals)
        elif isinstance(param, Categorical):
            options = param.get_support()
        for option in options:
            if option == SpecialParamValue.INVALID:
                continue
            search_point[space_name][param_name] = option
            yield from self._iterate_util(full_iter_order, search_point, index + 1)

    def iterate(self) -> Iterator[Dict[str, Dict[str, Any]]]:
        """Iterate over all possible configurations in the search space."""
        # initialize search point
        search_point = deepcopy(self._empty_search_point)

        # iterate over search space
        yield from self._iterate_util(self._iter_order, search_point, 0)

    def empty(self) -> bool:
        """Check if the search space is empty."""
        return all(not v for v in self._search_space.values())

    def size(self) -> int:
        """Get the size of the search space."""
        size = 0
        for _ in self.iterate():
            size += 1
        return size

    def empty_search_point(self) -> Dict[str, Dict[str, Any]]:
        """Get an empty search point."""
        return deepcopy(self._empty_search_point)

    def iter_params(self) -> Iterator[Tuple[str, str, SearchParameter]]:
        """Iterate over the search parameters in topological order."""
        for space_name, param_name in self._iter_order:
            yield space_name, param_name, self._search_space[space_name][param_name]
