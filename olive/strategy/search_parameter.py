# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import ABC, abstractmethod
from enum import Enum
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


class SpecialParamValue(str, Enum):
    """
    Special values for parameters.
    """

    IGNORED = "OLIVE_IGNORED_PARAM_VALUE"
    INVALID = "OLIVE_INVALID_PARAM_VALUE"


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
        assert isinstance(parents, tuple), "parents must be a tuple"
        for key in support:
            assert isinstance(key, tuple), "support key must be a tuple"
            assert len(key) == len(parents), "support key length must match the number of parents"

        self.parents = parents
        self.support = support
        self.default = default or self.get_invalid_choice()

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

    @staticmethod
    def get_invalid_choice():
        """
        Return a categorical search parameter with the invalid choice
        """
        return Categorical([SpecialParamValue.INVALID])

    @staticmethod
    def get_ignored_choice():
        """
        Return a categorical search parameter with the ignored choice
        """
        return Categorical([SpecialParamValue.IGNORED])


class ConditionalDefault(Conditional):
    """
    Parameter with conditional default value
    """

    def __init__(self, parents: Tuple[str], support: Dict[Tuple[Any], Any], default: Any = SpecialParamValue.INVALID):
        support = {key: Categorical([value]) for key, value in support.items()}
        default = Categorical([default])
        super().__init__(parents, support, default)

    def get_support(self, parent_values: Dict[str, Any]) -> Union[bool, int, float, str]:
        """
        get the support for the search parameter for a given parent value
        """
        return super().get_support(parent_values)[0]

    def condition(self, parent_values: Dict[str, Any]) -> Union[bool, int, float, str, "ConditionalDefault"]:
        """
        Fix the parent value and return a new search parameter
        """
        value = super().condition(parent_values)
        if isinstance(value, Categorical):
            return value.get_support()[0]
        if isinstance(value, Conditional):
            return self.conditional_to_conditional_default(value)

    @staticmethod
    def conditional_to_conditional_default(conditional: Conditional) -> "ConditionalDefault":
        """
        Convert a conditional to a conditional default
        """
        support = {}
        for key, value in conditional.support.items():
            assert isinstance(value, Categorical), "Conditional support must be categorical"
            assert len(value.get_support()) == 1, "Conditional support must have only one value"
            support[key] = value.get_support()[0]
        assert isinstance(conditional.default, Categorical), "Conditional default must be categorical"
        assert len(conditional.default.get_support()) == 1, "Conditional default must have only one value"
        return ConditionalDefault(conditional.parents, support, conditional.default.get_support()[0])

    @staticmethod
    def conditional_default_to_conditional(conditional_default: "ConditionalDefault") -> Conditional:
        """
        Convert a conditional default to a conditional
        """
        return Conditional(conditional_default.parents, conditional_default.support, conditional_default.default)

    def __repr__(self):
        support = {key: value.get_support()[0] for key, value in self.support.items()}
        default = self.default.get_support()[0]
        return f"ConditionalDefault(parents: {self.parents}, support: {support}, default: {default})"

    def to_json(self):
        json_data = super().to_json()
        json_data["type"] = "ConditionalDefault"
        return json_data

    @staticmethod
    def get_invalid_choice():
        """
        Return a categorical search parameter with the invalid choice
        """
        return SpecialParamValue.INVALID

    @staticmethod
    def get_ignored_choice():
        """
        Return a categorical search parameter with the ignored choice
        """
        return SpecialParamValue.IGNORED


def json_to_search_parameter(json: Dict[str, Any]) -> SearchParameter:
    """
    Convert a json to a search parameter
    """
    assert json["olive_parameter_type"] == "SearchParameter", "Not a search parameter"
    search_parameter_type = json["type"]
    if search_parameter_type == "Categorical":
        return Categorical(json["support"])
    if search_parameter_type == "Conditional" or search_parameter_type == "ConditionalDefault":
        stop_condition = lambda x: (  # noqa: E731
            isinstance(x, dict) and x.get("olive_parameter_type") == "SearchParameter"
        )
        support = flatten_dict(json["support"], stop_condition=stop_condition)
        for key, value in support.items():
            support[key] = json_to_search_parameter(value)
        conditional = Conditional(json["parents"], support, json_to_search_parameter(json["default"]))
        if search_parameter_type == "ConditionalDefault":
            return ConditionalDefault.conditional_to_conditional_default(conditional)
        return conditional
    raise ValueError(f"Unknown search parameter type {search_parameter_type}")
