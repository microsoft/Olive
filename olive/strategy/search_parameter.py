# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from olive.common.utils import flatten_dict, unflatten_dict


class SearchParameter(ABC):
    """
    Base class for search elements.
    Each search element should derive its own class.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_support(self) -> List[Any]:
        """
        get the support for the search parameter
        """
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError()

    @abstractmethod
    def to_json(self):
        raise NotImplementedError()


class Categorical(SearchParameter):
    """
    Search parameter that supports a list of values
    """

    def __init__(self, support: Union[List[str], List[int], List[float], List[bool]]):
        self.support = support

    def get_support(self) -> Union[List[str], List[int], List[float], List[bool]]:
        """
        get the support for the search parameter
        """
        return self.support

    def __repr__(self):
        return f"Categorical({self.support})"

    def to_json(self):
        return {"olive_parameter_type": "SearchParameter", "type": "Categorical", "support": self.support}


class Boolean(Categorical):
    """
    Search parameter that supports a boolean value
    """

    def __init__(self):
        super().__init__([True, False])


class Conditional(SearchParameter):
    """
    Conditional search parameter
    """

    def __init__(
        self,
        parents: Tuple[str],
        support: Dict[Tuple[Any], SearchParameter],
        default: SearchParameter = None,
    ):
        self.parents = parents
        self.support = support
        self.default = default or Categorical([None])

    def get_support(self, parent_values: Dict[str, Any]) -> Union[List[str], List[int], List[float], List[bool]]:
        """
        get the support for the search parameter for a given parent value
        """
        assert parent_values.keys() == set(self.parents), "parent values keys do not match the parents"
        parent_values = tuple([parent_values[parent] for parent in self.parents])
        return self.support.get(parent_values, self.default).get_support()

    def condition(self, parent_values: Dict[str, Any]) -> SearchParameter:
        """
        Fix the parent value and return a new search parameter
        """
        assert set(parent_values.keys()).issubset(set(self.parents)), "parent values keys not a subset of the parents"

        # if there is only one parent, return the support for the given parent value
        if len(self.parents) == 1:
            parent_values = (parent_values[self.parents[0]],)
            return self.support.get(parent_values, self.default)

        # condition the first parent and create a new conditional
        for parent_idx, parent in enumerate(self.parents):
            if parent in parent_values:
                parent_value = parent_values[parent]
                break
        new_parents = self.parents[:parent_idx] + self.parents[parent_idx + 1 :]  # noqa: E203
        new_support = {
            key[:parent_idx] + key[parent_idx + 1 :]: value  # noqa: E203
            for key, value in self.support.items()
            if key[parent_idx] == parent_value
        }
        # if there is no support for the given parent value, return the default
        if new_support == {}:
            return self.default
        # create a new conditional
        new_conditional = Conditional(new_parents, new_support, self.default)

        # condition the new conditional if there are more parents to condition, else return the new conditional
        del parent_values[parent]
        if len(parent_values) == 0:
            return new_conditional
        return new_conditional.condition(parent_values)

    def __repr__(self):
        return f"Conditional(parents: {self.parents}, support: {self.support}, default: {self.default})"

    def to_json(self):
        support = {}
        for key, value in self.support.items():
            support[key] = value.to_json()
        support = unflatten_dict(support)

        return {
            "olive_parameter_type": "SearchParameter",
            "type": "Conditional",
            "parents": self.parents,
            "support": support,
            "default": self.default.to_json(),
        }


def json_to_search_parameter(json: Dict[str, Any]) -> SearchParameter:
    """
    Convert a json to a search parameter
    """
    assert json["olive_parameter_type"] == "SearchParameter", "Not a search parameter"
    search_parameter_type = json["type"]
    if search_parameter_type == "Categorical":
        return Categorical(json["support"])
    if search_parameter_type == "Conditional":
        stop_condition = lambda x: (  # noqa: E731
            isinstance(x, dict) and x.get("olive_parameter_type") == "SearchParameter"
        )
        support = flatten_dict(json["support"], stop_condition=stop_condition)
        for key, value in support.items():
            support[key] = json_to_search_parameter(value)
        return Conditional(json["parents"], support, json_to_search_parameter(json["default"]))
    raise ValueError(f"Unknown search parameter type {search_parameter_type}")
