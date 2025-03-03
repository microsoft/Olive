# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict
from typing import Any, Dict, Generator, List, Tuple, Union

from olive.search.search_parameter import Categorical, Conditional, SearchParameter
from olive.search.search_point import SearchPoint
from olive.search.utils import order_search_parameters


class SearchSpace:
    """Search space for sampling.

    While a SearchParameter represents a tree (leaf nodes are the possible choices),
    SearchSpace represents multiple trees (each tree with its own possible choices).
    Also, each intermediate node in the SearchSpace could also either be a SearchParameter
    or a SearchSpace, generating its own possible choices.

    Note that the length in this context is the same as the number of possible unique choices.

    A "index" represent the index across all of the possible permutations of the leaf nodes.
    An result of indexing is deterministic only if the length of all the intermediate nodes
    in the tree are constant.

    Catch: Conditional search parameter don't have a constant length since the length depend on
    the fixed values chosen so far. To circumvent the limitation, the maximum of all of
    the possible lengths of the choices is used i.e. ignoring the parents. The default value
    of the parameter is used to pad the suggestions. This will generate duplicate search points
    which will be discarded if the default is invalid, or would early-out during evaluation from
    the cache.

    Indexing logic:
    Consider the following search space,
        [
            ["A", ["a"]],
            ["B", ["b", "c"]],
            ["C", ["d", "e", "f"]]
        ]

    The length of this search space would be 6 i.e. len(A) * len(B) * len(C) = 1 * 2 * 3 = 6.

    Given the index, to compute the values of each search point at that index,
      - for each param, p, in search space
        - Index of choice for that parameter would be index % len(p)
        - Update index to index / len(p)

    Here's a list of all possible choices -
        index | values
          0   | {"A": [0, "a"], "B": [0, "b"], "C": [0, "d"]}
          1   | {"A": [0, "a"], "B": [1, "c"], "C": [0, "d"]}
          2   | {"A": [0, "a"], "B": [0, "b"], "C": [1, "e"]}
          3   | {"A": [0, "a"], "B": [1, "c"], "C": [1, "e"]}
          4   | {"A": [0, "a"], "B": [0, "b"], "C": [2, "f"]}
          5   | {"A": [0, "a"], "B": [1, "c"], "C": [2, "f"]}

    For index = 0,
        At parameter "A", index = 0 % len(A) = 0, carry-forward = 0 / len(A) = 0  => [0, "a"]
        At parameter "B", index = 0 % len(B) = 0, carry-forward = 0 / len(B) = 0  => [0, "b"]
        At parameter "C", index = 0 % len(C) = 0, carry-forward = 0 / len(C) = 0  => [0, "d"]

    For index = 1,
        At parameter "A", index = 1 % len(A) = 0, carry-forward = 1 / len(A) = 1  => [0, "a"]
        At parameter "B", index = 1 % len(B) = 1, carry-forward = 1 / len(B) = 0  => [1, "c"]
        At parameter "C", index = 0 % len(C) = 0, carry-forward = 0 / len(C) = 0  => [0, "d"]

    For index = 2,
        At parameter "A", index = 2 % len(A) = 0, carry-forward = 2 / len(A) = 2  => [0, "a"]
        At parameter "B", index = 2 % len(B) = 0, carry-forward = 2 / len(B) = 1  => [0, "b"]
        At parameter "C", index = 1 % len(C) = 1, carry-forward = 1 / len(C) = 0  => [1, "e"]

    For index = 3,
        At parameter "A", index = 3 % len(A) = 0, carry-forward = 3 / len(A) = 3  => [0, "a"]
        At parameter "B", index = 3 % len(B) = 1, carry-forward = 3 / len(B) = 1  => [1, "c"]
        At parameter "C", index = 1 % len(C) = 1, carry-forward = 1 / len(C) = 0  => [1, "e"]

    For index = 4,
        At parameter "A", index = 4 % len(A) = 0, carry-forward = 4 / len(A) = 4  => [0, "a"]
        At parameter "B", index = 4 % len(B) = 0, carry-forward = 4 / len(B) = 2  => [0, "b"]
        At parameter "C", index = 2 % len(C) = 2, carry-forward = 2 / len(C) = 0  => [2, "f"]

    For index = 5,
        At parameter "A", index = 5 % len(A) = 0, carry-forward = 5 / len(A) = 5  => [0, "a"]
        At parameter "B", index = 5 % len(B) = 1, carry-forward = 5 / len(B) = 2  => [1, "c"]
        At parameter "C", index = 2 % len(C) = 2, carry-forward = 2 / len(C) = 0  => [2, "f"]

    The logic can be extrapolated to any number of parameters each with any number of choices
    as long the number of choices remains a constant at any given parameter.
    """

    def __init__(self, parameters: List[Tuple[str, Union[SearchParameter, "SearchSpace"]]]):
        assert len(parameters) == len(
            {name for name, _ in parameters}
        ), "Parameter name in search space should be unique."

        self._parameters = self._order_search_space(parameters)

        # Consider the following basic search space scenarios -
        #
        # [["a"]] => len = 1
        # [["a"], [1]] => len = 1
        # [["a", "b"], [1]] => len = 2
        # [["a", "b"], [1, 2]] => len = 4
        # [["a", "b"], [1, 2, 3]] => len = 6
        #
        # Extrapolating to n parameters with Mi search points each, the total
        # search points would be the product of number of search points in each.
        # So, for a search space with n parameters, [M1, M2, M3, ... Mn],
        # len(SearchSpace) = len(M1) * len(M2) * len(M3) * .... * len(Mn)

        self._length = 1
        for _, param in self._parameters:
            self._length *= SearchSpace.get_param_length(param)

    @property
    def parameters(self) -> List[Tuple[str, Union[SearchParameter, "SearchSpace"]]]:
        """Return the parameters of this search space."""
        return self._parameters

    def __repr__(self):
        """Return the string representation of this search space."""
        return f"SearchSpace({self._parameters}, {self._length})"

    def __len__(self) -> int:
        """Return the length i.e. total number of possible search points in this search space."""
        return self._length

    def __iter__(self) -> Generator[SearchPoint, None, None]:
        """Iterate search points in this search space."""
        for index in range(self._length):
            yield self[index]

    def __getitem__(self, index: int) -> SearchPoint:
        """Return search point by index."""
        assert index < self._length
        return SearchPoint(index, self.get_sample_point_values(index))

    def _order_search_space(self, parameters: List[Tuple[str, SearchParameter]]) -> List[Tuple[str, SearchParameter]]:
        """Order the search space by topological order of parameters for each pass_id/space_name."""
        unordered_nodes = dict(parameters)
        ordered_nodes = order_search_parameters(unordered_nodes)
        return [(name, unordered_nodes[name]) for name in ordered_nodes]

    def get_sample_point_values(self, index: int) -> Dict[str, Tuple[int, Any]]:
        """Iterate parameters of this search space to generate a search point."""
        assert index < self._length

        values = OrderedDict()
        for name, param in self._parameters:
            index, values[name] = SearchSpace.get_suggestion(param, index, values)
        return values

    @staticmethod
    def get_param_length(param: Any) -> int:
        """Return the length (computed recursively) of the input parameter."""
        if isinstance(param, SearchParameter):
            if isinstance(param, Categorical):
                return sum(
                    (
                        SearchSpace.get_param_length(suggestion)
                        if isinstance(suggestion, (SearchParameter, SearchSpace))
                        else 1
                    )
                    for suggestion in param.get_support()
                )

            elif isinstance(param, Conditional):
                # For conditional search parameters, length is computed based on the support
                # that has the most choices. See explanation above.
                return max(SearchSpace.get_param_length(support) for support in param.support.values())

        elif isinstance(param, SearchSpace):
            return len(param)

        return 0

    @staticmethod
    def get_param_suggestions(param: Any, values: Dict[str, Any]) -> Union[List[Any], "SearchSpace"]:
        """Return the suggestions for the input param based on the values chosen so far."""
        if isinstance(param, SearchParameter):
            if isinstance(param, Categorical):
                return param.get_support()

            elif isinstance(param, Conditional):
                parent_values = {k: values[k][1] for k in param.parents}
                suggestions = param.get_support_with_args(parent_values)
                # Pad the suggestions to maximum length using the default value of the param.
                max_length = max(SearchSpace.get_param_length(support) for support in param.support.values())
                suggestions += param.default.get_support() * (max_length - len(suggestions))
                return suggestions

        elif isinstance(param, SearchSpace):
            return param

        return []

    @staticmethod
    def get_suggestion(param: Any, index: int, values: Dict[str, Any]) -> Tuple[int, Tuple[int, Any]]:
        """Recursively, compute the values for the input param based on the index.

        Each entry is a tuple of the index in the list of suggestions for that param and the corresponding choice.
        """
        length = SearchSpace.get_param_length(param)

        if index < length:
            if isinstance(param, SearchParameter):
                suggestions = SearchSpace.get_param_suggestions(param, values)

                for i, suggestion in enumerate(suggestions):
                    if isinstance(suggestion, (SearchParameter, SearchSpace)):
                        suggestion_length = SearchSpace.get_param_length(suggestion)
                        if index < suggestion_length:
                            _, (_, i_suggestion) = SearchSpace.get_suggestion(suggestion, index, values)
                            return 0, (i, i_suggestion)
                        else:
                            index -= suggestion_length
                    elif index > 0:
                        index -= 1
                    else:
                        return 0, (i, suggestion)

            elif isinstance(param, SearchSpace):
                return 0, (index, param.get_sample_point_values(index))

            else:
                return index, param

        _, suggestion = SearchSpace.get_suggestion(param, index % length, values)
        return index // length, suggestion
