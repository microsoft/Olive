# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict

from olive.search.search_parameter import SpecialParamValue
from olive.search.search_point import SearchPoint


class TestSearchPoint:
    def _make_point(self, index=0, values=None):
        if values is None:
            values = OrderedDict(
                {
                    "pass1": (
                        0,
                        OrderedDict(
                            {
                                "param1": (0, "a"),
                                "param2": (1, 10),
                            }
                        ),
                    )
                }
            )
        return SearchPoint(index=index, values=values)

    def test_creation(self):
        point = self._make_point()
        assert point.index == 0

    def test_repr(self):
        point = self._make_point()
        result = repr(point)
        assert "SearchPoint" in result
        assert "0" in result

    def test_equality_same(self):
        point1 = self._make_point()
        point2 = self._make_point()
        assert point1 == point2

    def test_equality_different_index(self):
        point1 = self._make_point(index=0)
        point2 = self._make_point(index=1)
        assert point1 != point2

    def test_equality_different_type(self):
        point = self._make_point()
        assert point != "not a search point"

    def test_is_valid_true(self):
        point = self._make_point()
        assert point.is_valid() is True

    def test_is_valid_false_with_invalid(self):
        # is_valid checks for OrderedDict values recursively, and checks
        # non-OrderedDict values against SpecialParamValue.INVALID
        values = OrderedDict(
            {
                "pass1": OrderedDict(
                    {
                        "param1": SpecialParamValue.INVALID,
                    }
                )
            }
        )
        point = SearchPoint(index=0, values=values)
        assert point.is_valid() is False

    def test_to_json(self):
        point = self._make_point(index=5)
        result = point.to_json()
        assert result["index"] == 5
        assert "values" in result

    def test_from_json_roundtrip(self):
        point = self._make_point(index=3)
        json_data = point.to_json()
        restored = SearchPoint.from_json(json_data)
        assert restored.index == 3
        assert restored == point
