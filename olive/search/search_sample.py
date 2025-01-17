# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List

from olive.search.search_parameter import SpecialParamValue
from olive.search.search_point import SearchPoint

# ruff: noqa: PD011


@dataclass
class SearchSample:
    """Search step result from search strategy.

    Includes the search point and the input model ids to use to process the search point.
    """

    def __init__(self, search_point: SearchPoint, model_ids: List[str]):
        self.search_point = search_point
        self.model_ids = model_ids

    def __repr__(self):
        """Return the string representation."""
        return f"SearchSample({self.search_point.index}, {self.passes_configs}, {self.model_ids})"

    @property
    def passes_configs(self) -> Dict[str, Any]:
        """Return the pass config that can be merged with the workflow config.

        If any value in the hierarchy is SearchParameter.INVALID, return value would be None.
        If any value in the hierarchy is SearchParameter.IGNORED, return value exclude these parameters.
        """
        passes_configs = OrderedDict()
        for pass_name, (pass_index, params) in self.search_point.values.items():
            passes_configs[pass_name] = OrderedDict(
                [
                    ("index", pass_index),
                    ("params", OrderedDict()),
                ]
            )

            for param_name, (_, param_value) in params.items():
                if param_value == SpecialParamValue.INVALID:
                    return None  # Prune out invalid configurations
                elif param_value != SpecialParamValue.IGNORED:
                    passes_configs[pass_name]["params"][param_name] = param_value

        return passes_configs

    def to_json(self):
        """Return a json representation."""
        return {"search_point": self.search_point.to_json(), "model_ids": self.model_ids}

    @classmethod
    def from_json(cls, json_dict):
        """Create a SearchSample object from a json representation."""
        return cls(SearchPoint.from_json(json_dict["search_point"]), json_dict["model_ids"])
