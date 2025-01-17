# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from olive.search.search_parameter import SpecialParamValue

# ruff: noqa: PD011


@dataclass
class SearchPoint:
    """Search point from a search space.

    A search point is uniquely identified by an index and contains the corresponding values.
    Each value is a tuple of integer and some value (the choice). The "index" is the index of
    choice in all the possible choices.

    For example, for choices of ["a", "b", "c"] and value of [0, "a"], 0 is the index,
    and "a" is the corresponding choice. For value of [2, "c"], 2 would be the index and
    "c" would be the corresponding choice.
    """

    def __init__(self, index: int, values: Dict[str, Tuple[int, Any]]):
        self.index = index
        self.values = values

    def _format(self, arg: Any) -> str:
        """Return a string representation of the input arg. Builds the representation recursively."""
        return (
            "{" + ", ".join(f"{k}[{i}]: {self._format(v)}" for k, (i, v) in arg.items()) + "}"
            if isinstance(arg, OrderedDict)
            else str(arg)
        )

    def __repr__(self):
        """Return a string representation."""
        return f"SearchPoint({self.index}, {self._format(self.values)})"

    def __eq__(self, other):
        """Return true if this instance is the same as the input one."""
        return (
            (self.index == other.index) and (self.values == other.values) if isinstance(other, SearchPoint) else False
        )

    def is_valid(self) -> bool:
        """Return true if none of the value in the hierarchy is invalid."""

        def _is_valid(values: Dict[str, Tuple[int, Any]]) -> bool:
            for v in values.values():
                if isinstance(v, OrderedDict):
                    if not _is_valid(v):
                        return False
                elif v == SpecialParamValue.INVALID:
                    return False
            return True

        return _is_valid(self.values)

    def to_json(self):
        """Return a json representation."""
        return {"index": self.index, "values": self.values}

    @classmethod
    def from_json(cls, json_dict):
        """Create a SearchPoint object from a json representation."""
        return cls(json_dict["index"], json_dict["values"])
